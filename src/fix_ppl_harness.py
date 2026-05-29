#!/usr/bin/env python3
"""Repair and validate the FANS-MoE PPL harness.

This script intentionally separates harness validation from compression:

1. Run a GPT-2 WikiText-2 sanity check to validate strided PPL logic.
2. Load DeepSeek-V2-Lite across visible GPUs without disk offload.
3. Evaluate dense baseline on WikiText-2 with the same strided PPL logic.

Compression is blocked until the dense baseline is credible.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_wikitext_text(dataset_name: str = "wikitext-2-raw-v1") -> tuple[str, str]:
    if dataset_name not in {"wikitext-2-raw-v1", "wikitext-103-raw-v1"}:
        raise ValueError(f"Unsupported eval dataset for harness repair: {dataset_name}")
    ds = load_dataset("wikitext", dataset_name, split="test")
    text = "\n\n".join(t for t in ds["text"] if t and t.strip())
    return dataset_name, text


def model_input_device(model: torch.nn.Module) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def compute_strided_ppl(
    model: torch.nn.Module,
    tokenizer,
    text: str,
    *,
    max_length: int,
    stride: int,
    max_tokens: int,
    label: str,
) -> dict[str, float | int]:
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids_full = enc.input_ids
    total_tokens = int(input_ids_full.size(1))
    seq_len = min(total_tokens, int(max_tokens))
    if seq_len < 128:
        raise RuntimeError(f"{label}: too few eval tokens: {seq_len}")

    input_ids_full = input_ids_full[:, :seq_len]
    dev = model_input_device(model)
    nll_sum = 0.0
    n_tokens = 0
    prev_end = 0
    model.eval()

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        if trg_len <= 0:
            break

        input_ids = input_ids_full[:, begin:end].to(dev)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, labels=target_ids, use_cache=False)

        valid = int((target_ids[..., 1:] != -100).sum().item())
        loss = float(out.loss.detach().float().cpu().item())
        if not math.isfinite(loss):
            raise RuntimeError(f"{label}: non-finite loss at window {begin}:{end}: {loss}")

        nll_sum += loss * valid
        n_tokens += valid
        log(f"{label}: window {begin}:{end} trg_len={trg_len} valid={valid} loss={loss:.4f}")

        prev_end = end
        if end == seq_len:
            break

    mean_nll = nll_sum / max(n_tokens, 1)
    ppl = math.exp(mean_nll)
    return {
        "ppl": float(ppl),
        "mean_nll": float(mean_nll),
        "tokens_scored": int(n_tokens),
        "tokens_available": int(total_tokens),
        "tokens_used": int(seq_len),
        "max_length": int(max_length),
        "stride": int(stride),
    }


def probe_next_token(model: torch.nn.Module, tokenizer, text: str, *, max_prompt_tokens: int = 64) -> dict[str, Any]:
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[:, :max_prompt_tokens]
    ids = ids.to(model_input_device(model))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False)
    logits = out.logits[:, -1, :].detach().float().cpu()[0]
    probs = torch.softmax(logits, dim=-1)
    top = torch.topk(probs, k=10)
    decoded = []
    for idx, prob in zip(top.indices.tolist(), top.values.tolist(), strict=False):
        decoded.append({"id": int(idx), "prob": float(prob), "text": tokenizer.decode([idx])})
    entropy = float(-(probs * torch.log(probs.clamp_min(1e-30))).sum().item())
    return {
        "prompt": tokenizer.decode(ids.detach().cpu()[0].tolist()),
        "logits_mean": float(logits.mean().item()),
        "logits_std": float(logits.std().item()),
        "logits_min": float(logits.min().item()),
        "logits_max": float(logits.max().item()),
        "entropy": entropy,
        "top10": decoded,
    }


def device_map_summary(model: torch.nn.Module) -> dict[str, Any]:
    hf_map = getattr(model, "hf_device_map", None)
    if not hf_map:
        return {"available": False, "counts": {}, "raw": {}}
    raw = {str(k): str(v) for k, v in hf_map.items()}
    counts = Counter(raw.values())
    return {"available": True, "counts": dict(counts), "raw": raw}


def get_nested_attr(root: Any, dotted_name: str) -> Any:
    obj = root
    for part in dotted_name.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)
    return obj


def checkpoint_tensor_stats(snapshot_dir: Path, key: str) -> dict[str, Any]:
    index_path = snapshot_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    filename = index["weight_map"][key]
    with safe_open(snapshot_dir / filename, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(key).float()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def model_tensor_stats(model: torch.nn.Module, key: str) -> dict[str, Any]:
    tensor = get_nested_attr(model, key).detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std().item()),
        "min": float(tensor.min().item()),
        "max": float(tensor.max().item()),
    }


def compare_selected_weight_stats(model: torch.nn.Module, snapshot_dir: Path) -> dict[str, Any]:
    keys = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.0.gate_proj.weight",
    ]
    stats: dict[str, Any] = {}
    for key in keys:
        try:
            model_stats = model_tensor_stats(model, key)
            ckpt_stats = checkpoint_tensor_stats(snapshot_dir, key)
            stats[key] = {
                "model": model_stats,
                "checkpoint": ckpt_stats,
                "std_ratio_model_over_checkpoint": (
                    model_stats["std"] / ckpt_stats["std"] if ckpt_stats["std"] else None
                ),
            }
        except Exception as exc:
            stats[key] = {"error": repr(exc)}
    return stats


def force_reload_safetensors(model: torch.nn.Module, repo: str, *, local_files_only: bool) -> dict[str, Any]:
    """Overwrite model parameters from safetensors one tensor at a time.

    The remote DeepSeek-V2 code is not fully compatible with newer Transformers
    loaders in this environment: some nn.Linear/Embedding tensors remain
    initialized even though loading_info reports no missing keys. This explicit
    reload is intentionally conservative and avoids holding a full state_dict in
    CPU or GPU memory.
    """

    snapshot_dir = Path(snapshot_download(repo_id=repo, local_files_only=local_files_only))
    index_path = snapshot_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing safetensors index: {index_path}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    filenames = sorted(set(index["weight_map"].values()))
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())

    copied = 0
    missing_targets: list[str] = []
    mismatched: list[dict[str, Any]] = []

    for filename in filenames:
        shard_path = snapshot_dir / filename
        log(f"force-reloading checkpoint shard {filename}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                target = params.get(key, buffers.get(key))
                if target is None:
                    missing_targets.append(key)
                    continue
                tensor = handle.get_tensor(key)
                if tuple(tensor.shape) != tuple(target.shape):
                    mismatched.append(
                        {
                            "key": key,
                            "checkpoint_shape": list(tensor.shape),
                            "model_shape": list(target.shape),
                        }
                    )
                    continue
                with torch.no_grad():
                    target.copy_(tensor.to(device=target.device, dtype=target.dtype))
                copied += 1
                del tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if mismatched:
        raise RuntimeError(f"shape mismatches while force-reloading checkpoint: {mismatched[:5]}")

    return {
        "snapshot_dir": str(snapshot_dir),
        "shards": filenames,
        "copied_tensors": copied,
        "missing_targets_count": len(missing_targets),
        "missing_targets_first20": missing_targets[:20],
    }


def cuda_memory_summary() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        str(i): {
            "allocated_gib": float(torch.cuda.memory_allocated(i) / (1024**3)),
            "reserved_gib": float(torch.cuda.memory_reserved(i) / (1024**3)),
            "max_allocated_gib": float(torch.cuda.max_memory_allocated(i) / (1024**3)),
            "max_reserved_gib": float(torch.cuda.max_memory_reserved(i) / (1024**3)),
        }
        for i in range(torch.cuda.device_count())
    }


def assert_no_disk_offload(model: torch.nn.Module, allow_cpu: bool) -> None:
    hf_map = getattr(model, "hf_device_map", {})
    bad_disk = [k for k, v in hf_map.items() if "disk" in str(v).lower()]
    if bad_disk:
        raise RuntimeError(f"disk offload detected: {bad_disk[:10]}")
    if not allow_cpu:
        bad_cpu = [k for k, v in hf_map.items() if str(v).lower() == "cpu"]
        if bad_cpu:
            raise RuntimeError(f"cpu offload detected while allow_cpu=false: {bad_cpu[:10]}")


def load_causal_lm(
    repo: str,
    *,
    torch_dtype,
    device_map,
    max_memory: dict[int | str, str] | None,
    trust_remote_code: bool,
    local_files_only: bool,
):
    kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "local_files_only": local_files_only,
    }
    if max_memory:
        kwargs["max_memory"] = max_memory
    return AutoModelForCausalLM.from_pretrained(repo, **kwargs)


def run_gpt2_sanity(text: str, args: argparse.Namespace) -> dict[str, Any]:
    log("loading GPT-2 sanity model")
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to("cuda:0").eval()
    result = compute_strided_ppl(
        model,
        tok,
        text,
        max_length=min(args.gpt2_max_length, args.max_length),
        stride=min(args.gpt2_stride, args.stride),
        max_tokens=args.gpt2_eval_tokens,
        label="gpt2",
    )
    del model
    torch.cuda.empty_cache()
    log(f"GPT2 sanity PPL={result['ppl']:.3f}")
    if result["ppl"] >= args.gpt2_max_ppl:
        raise RuntimeError(f"GPT2 sanity failed: ppl={result['ppl']:.3f}")
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--eval-dataset", default="wikitext-2-raw-v1")
    ap.add_argument("--eval-tokens", type=int, default=20000)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--max-memory-per-gpu", default="26GiB")
    ap.add_argument("--gpt2-sanity", action="store_true")
    ap.add_argument("--gpt2-eval-tokens", type=int, default=20000)
    ap.add_argument("--gpt2-max-length", type=int, default=1024)
    ap.add_argument("--gpt2-stride", type=int, default=512)
    ap.add_argument("--gpt2-max-ppl", type=float, default=50.0)
    ap.add_argument("--dense-max-ppl", type=float, default=15.0)
    ap.add_argument("--allow-cpu-offload", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--force-safetensors-reload", action="store_true")
    ap.add_argument("--output", default="outputs_realtext_512/ppl/ppl_results_v3_baseline.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} cuda_count={torch.cuda.device_count()}")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required")

    dataset_used, text = load_wikitext_text(args.eval_dataset)
    deepseek_tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    deepseek_token_count = int(deepseek_tok(text, return_tensors="pt", add_special_tokens=False).input_ids.size(1))
    if deepseek_token_count < 50000:
        raise RuntimeError(f"eval token count too small: {deepseek_token_count}")
    log(f"eval dataset={dataset_used} deepseek_tokens={deepseek_token_count}")

    results: dict[str, Any] = {
        "model": args.model,
        "eval_dataset": dataset_used,
        "deepseek_eval_tokens_available": deepseek_token_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "max_memory_per_gpu": args.max_memory_per_gpu,
        "gpt2_sanity": None,
        "dense_baseline": None,
        "device_map": None,
        "weight_stats_before_reload": None,
        "force_safetensors_reload": None,
        "weight_stats_after_reload": None,
        "cuda_memory": None,
        "passed_dense_guard": False,
    }

    if args.gpt2_sanity:
        results["gpt2_sanity"] = run_gpt2_sanity(text, args)

    visible = torch.cuda.device_count()
    max_memory: dict[int | str, str] = {i: args.max_memory_per_gpu for i in range(visible)}
    log(f"loading dense model across visible GPUs with max_memory={max_memory}")
    model = load_causal_lm(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    ).eval()
    results["device_map"] = device_map_summary(model)
    log(f"DEVICE MAP COUNTS: {results['device_map']['counts']}")
    assert_no_disk_offload(model, allow_cpu=args.allow_cpu_offload)
    snapshot_dir = Path(snapshot_download(repo_id=args.model, local_files_only=args.local_files_only))
    results["weight_stats_before_reload"] = compare_selected_weight_stats(model, snapshot_dir)
    if args.force_safetensors_reload:
        results["force_safetensors_reload"] = force_reload_safetensors(
            model,
            args.model,
            local_files_only=args.local_files_only,
        )
        results["weight_stats_after_reload"] = compare_selected_weight_stats(model, snapshot_dir)
    results["cuda_memory"] = {"after_load": cuda_memory_summary()}

    results["dense_baseline"] = compute_strided_ppl(
        model,
        deepseek_tok,
        text,
        max_length=args.max_length,
        stride=args.stride,
        max_tokens=args.eval_tokens,
        label="deepseek_dense",
    )
    results["next_token_probe"] = probe_next_token(
        model,
        deepseek_tok,
        "The capital of France is",
    )
    results["cuda_memory"]["after_dense_eval"] = cuda_memory_summary()
    dense_ppl = float(results["dense_baseline"]["ppl"])
    results["passed_dense_guard"] = dense_ppl < args.dense_max_ppl
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"wrote {out}")
    if dense_ppl >= args.dense_max_ppl:
        raise RuntimeError(f"DENSE BASELINE STILL BROKEN: ppl={dense_ppl:.3f}")
    log(f"DENSE BASELINE PASSED: ppl={dense_ppl:.3f}")


if __name__ == "__main__":
    main()
