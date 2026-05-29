#!/usr/bin/env python3
"""Build real-text per-layer MoE-input calibration hidden states.

The script streams text on the remote server, tokenizes it with the model
tokenizer, runs the model once with pre-hooks on selected MoE layers, and saves
the hidden state entering each MoE block. It never consumes a local dataset copy.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import load_dataset
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_layers(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def local_doc_texts(max_chars: int) -> list[str]:
    roots = [
        Path("/usr/share/doc"),
        Path.home() / "workspace" / "venv-dsv2" / "lib" / "python3.12" / "site-packages",
    ]
    suffixes = {".txt", ".md", ".rst", ""}
    texts: list[str] = []
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            if path.name.endswith((".py", ".so", ".bin", ".json")):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = " ".join(text.split())
            if len(text) < 80:
                continue
            texts.append(text)
            total += len(text)
            if total >= max_chars:
                return texts
    return texts


def stream_texts(preferred: str, max_chars: int) -> tuple[str, list[str]]:
    if preferred == "local":
        texts = local_doc_texts(max_chars)
        if not texts:
            raise RuntimeError("local text fallback found no usable English text")
        return "local_docs", texts

    attempts = []
    if preferred == "c4":
        attempts.append(("c4", lambda: load_dataset("allenai/c4", "en", split="train", streaming=True)))
    attempts.append(("wikitext", lambda: load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)))
    attempts.append(("wikitext2", lambda: load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)))

    errors: list[str] = []
    for name, fn in attempts:
        try:
            ds = fn()
            texts: list[str] = []
            total = 0
            for ex in ds:
                text = (ex.get("text") or "").strip()
                if len(text) < 40:
                    continue
                texts.append(text)
                total += len(text)
                if total >= max_chars:
                    break
            if texts:
                return name, texts
            errors.append(f"{name}: empty")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    texts = local_doc_texts(max_chars)
    if texts:
        return "local_docs_after_hf_failure", texts
    raise RuntimeError("No text dataset could be streamed: " + " | ".join(errors))


def get_decoder_layers(model):
    obj = model
    for attr in ("model", "base_model"):
        if hasattr(obj, "layers"):
            break
        obj = getattr(obj, attr, obj)
    if not hasattr(obj, "layers") and hasattr(model, "model") and hasattr(model.model, "layers"):
        obj = model.model
    if not hasattr(obj, "layers"):
        raise RuntimeError("Could not locate decoder layers on model")
    return obj.layers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--dataset", default="c4", choices=["c4", "wikitext", "local"])
    ap.add_argument("--output-dir", default="outputs_c4_512/data/calibration_c4")
    ap.add_argument("--offload-dir", default="outputs_c4_512/offload")
    ap.add_argument("--max-gpu-memory", default="26GiB")
    ap.add_argument("--max-cpu-memory", default="96GiB")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    layers_to_capture = parse_layers(args.layers)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.offload_dir).mkdir(parents=True, exist_ok=True)

    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    source, texts = stream_texts(args.dataset, max_chars=max(args.tokens * 12, 20000))
    log(f"streamed {len(texts)} text chunks from {source}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.repo,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    joined = "\n\n".join(texts)
    encoded = tokenizer(joined, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded.input_ids[0]
    if token_ids.numel() < args.tokens:
        raise RuntimeError(f"Only collected {token_ids.numel()} tokens, need {args.tokens}")
    token_ids = token_ids[: args.tokens]

    log("loading full model with device_map=auto")
    max_memory = {"cpu": args.max_cpu_memory}
    if torch.cuda.is_available():
        max_memory[0] = args.max_gpu_memory
    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        offload_folder=args.offload_dir,
        max_memory=max_memory,
        local_files_only=args.local_files_only,
    )
    model.eval()
    decoder_layers = get_decoder_layers(model)

    captures: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers_to_capture}
    hooks = []

    def make_hook(layer_idx: int):
        def hook(_module, inputs):
            hidden = inputs[0].detach().to("cpu", dtype=torch.bfloat16)
            captures[layer_idx].append(hidden.reshape(-1, hidden.shape[-1]))
        return hook

    for layer in layers_to_capture:
        hooks.append(decoder_layers[layer].mlp.register_forward_pre_hook(make_hook(layer)))
    try:
        input_device = next(model.parameters()).device
        for start in range(0, args.tokens, args.seq_len):
            end = min(start + args.seq_len, args.tokens)
            input_ids = token_ids[start:end].unsqueeze(0).to(input_device)
            attention_mask = torch.ones_like(input_ids).to(input_device)
            log(f"forward tokens {start}:{end}")
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    finally:
        for h in hooks:
            h.remove()

    meta = {
        "repo": args.repo,
        "dataset_requested": args.dataset,
        "dataset_used": source,
        "tokens": args.tokens,
        "seq_len": args.seq_len,
        "layers": layers_to_capture,
        "hf_endpoint": os.environ.get("HF_ENDPOINT"),
    }
    for layer, chunks in captures.items():
        if not chunks:
            raise RuntimeError(f"No captures for layer {layer}")
        hidden = torch.cat(chunks, dim=0)[: args.tokens].contiguous()
        path = out_dir / f"layer{layer:02d}_hidden.pt"
        torch.save({"hidden": hidden, "metadata": meta | {"layer": layer}}, path)
        log(f"saved {path} shape={tuple(hidden.shape)}")
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log("done")


if __name__ == "__main__":
    main()
