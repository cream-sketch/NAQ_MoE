#!/usr/bin/env python3
"""Fallback BF16 reconstruction and PPL evaluation for FANS-MoE.

This script intentionally avoids monkey-patching DeepSeek MoE forward. It takes
the tier maps produced by fans_moe_lite.py, compresses selected expert weights
slot-by-slot, immediately dequantizes/reconstructs them back into the ordinary
model parameters, and evaluates a standard causal-LM loss.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_dataset
from huggingface_hub import snapshot_download
from safetensors import safe_open
from sklearn.cluster import KMeans
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM, AutoTokenizer

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False


TIER_NAMES = {0: "universal", 1: "group", 2: "specialist"}


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_layers(text: str | None, weights_dir: Path) -> list[int]:
    if text:
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
    summary = weights_dir / "summaries" / "run_summary.json"
    if summary.exists():
        return [int(x) for x in json.loads(summary.read_text(encoding="utf-8"))["layers"]]
    layers = set()
    for path in (weights_dir / "tier_maps").glob("layer*_tiers_B*.npy"):
        layers.add(int(path.name[len("layer") : len("layer") + 2]))
    if not layers:
        raise FileNotFoundError(f"No tier maps found under {weights_dir}")
    return sorted(layers)


def budget_tag(budget: float | str) -> str:
    return str(budget).replace(".", "p")


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def local_doc_texts(max_chars: int) -> tuple[str, str]:
    roots = [
        Path("/usr/share/doc"),
        Path.home() / "workspace" / "venv-dsv2" / "lib" / "python3.12" / "site-packages",
    ]
    chunks: list[str] = []
    total = 0
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".txt", ".md", ".rst", ""}:
                continue
            if path.name.endswith((".py", ".so", ".bin", ".json", ".pt", ".npy")):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            text = " ".join(text.split())
            if len(text) < 80:
                continue
            chunks.append(text)
            total += len(text)
            if total >= max_chars:
                return "local_docs", "\n\n".join(chunks)
    if not chunks:
        raise RuntimeError("local text fallback found no usable English text")
    return "local_docs", "\n\n".join(chunks)


def load_eval_text(dataset_name: str, max_chars: int) -> tuple[str, str, list[str]]:
    errors: list[str] = []
    dataset_path = Path(dataset_name)
    if dataset_path.exists() and dataset_path.is_file():
        text = dataset_path.read_text(encoding="utf-8", errors="ignore")
        return str(dataset_path), text, errors
    if dataset_name in {"wikitext-2-raw-v1", "wikitext2"}:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t and t.strip())
        return "wikitext-2-raw-v1", text, errors
    if dataset_name in {"wikitext-103-raw-v1", "wikitext103"}:
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        text = "\n\n".join(t for t in ds["text"] if t and t.strip())
        return "wikitext-103-raw-v1", text, errors
    if dataset_name in {"ptb", "ptb-test", "penn-treebank", "penn_treebank"}:
        local_ptb = Path("data/ptb/ptb.test.txt")
        if local_ptb.exists():
            text = local_ptb.read_text(encoding="utf-8", errors="ignore")
            return "ptb.test.txt", text, errors
        try:
            ds = load_dataset("ptb_text_only", "penn_treebank", split="test")
            text = "\n\n".join(
                (ex.get("sentence") or ex.get("text") or "").strip()
                for ex in ds
                if (ex.get("sentence") or ex.get("text") or "").strip()
            )
            return "ptb_text_only/penn_treebank", text, errors
        except Exception as exc:
            errors.append(f"ptb_text_only/penn_treebank: {type(exc).__name__}: {exc}")

    attempts: list[tuple[str, Any]] = []
    if dataset_name == "local":
        name, text = local_doc_texts(max_chars)
        return name, text, errors
    else:
        attempts.append((dataset_name, lambda: load_dataset(dataset_name, split="test", streaming=True)))

    for name, fn in attempts:
        try:
            ds = fn()
            chunks: list[str] = []
            total = 0
            for ex in ds:
                text = (ex.get("text") or ex.get("sentence") or "").strip()
                if len(text) < 20:
                    continue
                chunks.append(text)
                total += len(text)
                if total >= max_chars:
                    break
            if chunks:
                return name, "\n\n".join(chunks), errors
            errors.append(f"{name}: empty")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    name, text = local_doc_texts(max_chars)
    return f"{name}_after_dataset_failure", text, errors


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


def get_experts(model, layer: int):
    mlp = get_decoder_layers(model)[layer].mlp
    if not hasattr(mlp, "experts"):
        raise RuntimeError(f"Layer {layer} MLP has no experts attribute")
    return list(mlp.experts)


def input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def evaluate_ppl(
    model,
    tokenizer,
    text: str,
    eval_tokens: int,
    max_length: int,
    stride: int,
    label: str,
) -> dict[str, float | int]:
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    tokens_available = int(encoded.size(1))
    if tokens_available < 50000:
        raise RuntimeError(f"Evaluation text produced too few tokens: {tokens_available}")
    total = min(int(eval_tokens), tokens_available)
    if total < 32:
        raise RuntimeError("Evaluation text produced too few tokens")
    encoded = encoded[:, :total]
    device = input_device(model)
    nll_sum = 0.0
    token_count = 0
    prev_end = 0
    model.eval()
    for start in range(0, total, stride):
        end = min(start + max_length, total)
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        input_ids = encoded[:, start:end].to(device)
        labels = input_ids.clone()
        labels[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
        valid = int((labels[..., 1:] != -100).sum().item())
        loss = float(out.loss.detach().float().cpu().item())
        nll_sum += loss * valid
        token_count += valid
        log(f"{label}: window {start}:{end} trg_len={trg_len} valid={valid} loss={loss:.4f}")
        prev_end = end
        if end == total:
            break
    mean_nll = nll_sum / max(token_count, 1)
    return {
        "ppl": float(math.exp(mean_nll)),
        "mean_nll": float(mean_nll),
        "tokens": int(token_count),
        "tokens_available": tokens_available,
        "tokens_used": int(total),
        "max_length": int(max_length),
        "stride": int(stride),
    }


def evict_checkpoint_cache(repo: str, *, local_files_only: bool) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_download(repo_id=repo, local_files_only=local_files_only))
    evicted: list[str] = []
    unsupported = not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED")
    if unsupported:
        return {"supported": False, "snapshot_dir": str(snapshot_dir), "evicted_files": []}
    for path in sorted(snapshot_dir.glob("*.safetensors")):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            evicted.append(path.name)
        finally:
            os.close(fd)
    return {"supported": True, "snapshot_dir": str(snapshot_dir), "evicted_files": evicted}


def force_reload_safetensors(model: torch.nn.Module, repo: str, *, local_files_only: bool) -> dict[str, Any]:
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
    reload_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    for filename in filenames:
        log(f"force-reloading checkpoint shard {filename}")
        with safe_open(snapshot_dir / filename, framework="pt", device=reload_device) as handle:
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
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if mismatched:
        raise RuntimeError(f"shape mismatches while force-reloading checkpoint: {mismatched[:5]}")
    return {
        "snapshot_dir": str(snapshot_dir),
        "shards": filenames,
        "copied_tensors": copied,
        "reload_device": reload_device,
        "missing_targets_count": len(missing_targets),
        "missing_targets_first20": missing_targets[:20],
    }


def device_map_summary(model: torch.nn.Module) -> dict[str, Any]:
    hf_map = getattr(model, "hf_device_map", None)
    if not hf_map:
        return {"available": False, "counts": {}, "raw": {}}
    raw = {str(k): str(v) for k, v in hf_map.items()}
    counts: dict[str, int] = {}
    for value in raw.values():
        counts[value] = counts.get(value, 0) + 1
    return {"available": True, "counts": counts, "raw": raw}


def assert_no_offload(model: torch.nn.Module, allow_cpu: bool) -> None:
    hf_map = getattr(model, "hf_device_map", {})
    disk = [k for k, v in hf_map.items() if "disk" in str(v).lower()]
    if disk:
        raise RuntimeError(f"disk offload detected: {disk[:10]}")
    if not allow_cpu:
        cpu = [k for k, v in hf_map.items() if str(v).lower() == "cpu"]
        if cpu:
            raise RuntimeError(f"cpu offload detected: {cpu[:10]}")


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


def linear_weight_tensor(linear) -> torch.Tensor:
    weight = linear.weight
    if not getattr(weight, "is_meta", False):
        return weight
    hook = getattr(linear, "_hf_hook", None)
    weights_map = getattr(hook, "weights_map", None)
    if weights_map is not None and "weight" in weights_map:
        return weights_map["weight"]
    raise RuntimeError(f"Could not resolve offloaded weight for {linear}")


def expert_weight_triplet(expert) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        linear_weight_tensor(expert.gate_proj),
        linear_weight_tensor(expert.up_proj),
        linear_weight_tensor(expert.down_proj),
    )


def original_layer_path(cache_dir: Path, layer: int) -> Path:
    return cache_dir / f"layer{layer:02d}_original_expert_weights.pt"


def save_original_layers(model, layers: list[int], cache_dir: Path, refresh: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for layer in layers:
        out = original_layer_path(cache_dir, layer)
        if out.exists() and not refresh:
            log(f"layer {layer}: original cache exists")
            continue
        experts = get_experts(model, layer)
        payload = {"gate": [], "up": [], "down": []}
        for expert in experts:
            gate, up, down = expert_weight_triplet(expert)
            payload["gate"].append(gate.detach().to("cpu", dtype=torch.bfloat16).clone())
            payload["up"].append(up.detach().to("cpu", dtype=torch.bfloat16).clone())
            payload["down"].append(down.detach().to("cpu", dtype=torch.bfloat16).clone())
        torch.save(payload, out)
        log(f"layer {layer}: saved original expert weights to {out}")
        del payload
        gc.collect()


def quant_dequant_int4(x: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    max_abs = xf.abs().max()
    if float(max_abs) == 0.0:
        return x.clone()
    scale = max_abs / 7.0
    q = torch.round(xf / scale).clamp(-8, 7)
    return (q * scale).to(dtype=x.dtype)


def storage_ratio(tiers: np.ndarray, group_count: int) -> float:
    u = float((tiers == 0).mean())
    g = float((tiers == 1).mean())
    s = float((tiers == 2).mean())
    return (2.0 * u + float(group_count) * g + 32.0 * s) / 128.0


def load_layer_activations(weights_dir: Path, layer: int, perms: np.ndarray) -> list[torch.Tensor]:
    acts = []
    for expert in range(perms.shape[0]):
        path = weights_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing activation signature: {path}")
        acts.append(torch.load(path, map_location="cpu"))
    return acts


def group_labels_for_slot(
    activations: list[torch.Tensor],
    perms: np.ndarray,
    slot: int,
    group_count: int,
    seed: int,
) -> np.ndarray:
    x = torch.stack(
        [activations[expert][:, int(perms[expert, slot])].float() for expert in range(len(activations))],
        dim=0,
    ).numpy()
    km = KMeans(n_clusters=group_count, n_init=5, random_state=seed).fit(x)
    return km.labels_.astype(np.int16)


def apply_layer_from_tiers(
    model,
    layer: int,
    weights_dir: Path,
    original_cache_dir: Path,
    budget: float,
    group_count: int,
    seed: int,
    details_dir: Path,
) -> dict[str, Any]:
    tag = budget_tag(budget)
    tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
    perms = np.load(weights_dir / "alignments" / f"layer{layer:02d}_perms.npy")
    orig = torch.load(original_layer_path(original_cache_dir, layer), map_location="cpu")
    new = {kind: [tensor.clone() for tensor in orig[kind]] for kind in ("gate", "up", "down")}
    activations = load_layer_activations(weights_dir, layer, perms) if np.any(tiers == 1) else []
    group_labels = np.full((tiers.shape[0], perms.shape[0]), -1, dtype=np.int16)

    for slot, tier in enumerate(tiers.tolist()):
        cols = [int(perms[expert, slot]) for expert in range(perms.shape[0])]
        if tier == 0:
            avg_gate = torch.stack([orig["gate"][e][cols[e], :].float() for e in range(len(cols))]).mean(0).to(torch.bfloat16)
            avg_up = torch.stack([orig["up"][e][cols[e], :].float() for e in range(len(cols))]).mean(0).to(torch.bfloat16)
            avg_down = torch.stack([orig["down"][e][:, cols[e]].float() for e in range(len(cols))]).mean(0).to(torch.bfloat16)
            for e, col in enumerate(cols):
                new["gate"][e][col, :] = avg_gate
                new["up"][e][col, :] = avg_up
                new["down"][e][:, col] = avg_down
        elif tier == 1:
            labels = group_labels_for_slot(activations, perms, slot, group_count, seed)
            group_labels[slot] = labels
            for group in range(group_count):
                members = np.where(labels == group)[0].tolist()
                if not members:
                    continue
                avg_gate = torch.stack([orig["gate"][e][cols[e], :].float() for e in members]).mean(0).to(torch.bfloat16)
                avg_up = torch.stack([orig["up"][e][cols[e], :].float() for e in members]).mean(0).to(torch.bfloat16)
                avg_down = torch.stack([orig["down"][e][:, cols[e]].float() for e in members]).mean(0).to(torch.bfloat16)
                for e in members:
                    col = cols[e]
                    new["gate"][e][col, :] = avg_gate
                    new["up"][e][col, :] = avg_up
                    new["down"][e][:, col] = avg_down
        else:
            for e, col in enumerate(cols):
                new["gate"][e][col, :] = quant_dequant_int4(orig["gate"][e][col, :])
                new["up"][e][col, :] = quant_dequant_int4(orig["up"][e][col, :])
                new["down"][e][:, col] = quant_dequant_int4(orig["down"][e][:, col])

        if slot % 128 == 0:
            log(f"layer {layer} B={budget}: reconstructed slot {slot}/{len(tiers) - 1}")

    experts = get_experts(model, layer)
    with torch.no_grad():
        for e, expert in enumerate(experts):
            gate, up, down = expert_weight_triplet(expert)
            gate.copy_(new["gate"][e].to(device=gate.device, dtype=gate.dtype))
            up.copy_(new["up"][e].to(device=up.device, dtype=up.dtype))
            down.copy_(new["down"][e].to(device=down.device, dtype=down.dtype))

    details_dir.mkdir(parents=True, exist_ok=True)
    np.save(details_dir / f"layer{layer:02d}_group_labels_B{tag}.npy", group_labels)
    counts = {TIER_NAMES[i]: int((tiers == i).sum()) for i in range(3)}
    result = {
        "layer": layer,
        "budget": budget,
        "counts": counts,
        "actual_storage_ratio": storage_ratio(tiers, group_count),
    }
    del orig, new, activations, group_labels
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def apply_uniform_int4_layer(model, layer: int, original_cache_dir: Path) -> None:
    orig = torch.load(original_layer_path(original_cache_dir, layer), map_location="cpu")
    experts = get_experts(model, layer)
    with torch.no_grad():
        for e, expert in enumerate(experts):
            gate, up, down = expert_weight_triplet(expert)
            gate.copy_(quant_dequant_int4(orig["gate"][e]).to(device=gate.device, dtype=gate.dtype))
            up.copy_(quant_dequant_int4(orig["up"][e]).to(device=up.device, dtype=up.dtype))
            down.copy_(quant_dequant_int4(orig["down"][e]).to(device=down.device, dtype=down.dtype))
    log(f"layer {layer}: applied uniform INT4 dequant baseline")
    del orig
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_metrics_csv(weights_dir: Path, out_path: Path, layers: list[int], budgets: list[float], group_count: int) -> None:
    rows = []
    for layer in layers:
        d_path = weights_dir / "distances" / f"layer{layer:02d}_D.npy"
        sanity_path = weights_dir / "alignments" / f"layer{layer:02d}_sanity.json"
        if not d_path.exists() or not sanity_path.exists():
            continue
        d = np.load(d_path)
        sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
        vals = list(sanity.get("experts", {}).values())
        align_before = float(np.mean([x["mean_diag_unaligned"] for x in vals])) if vals else float("nan")
        align_after = float(np.mean([x["mean_diag_aligned"] for x in vals])) if vals else float("nan")
        for budget in budgets:
            tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{budget_tag(budget)}.npy")
            rows.append(
                {
                    "layer": layer,
                    "budget": budget,
                    "D_min": float(d.min()),
                    "D_mean": float(d.mean()),
                    "D_p25": float(np.quantile(d, 0.25)),
                    "D_p50": float(np.quantile(d, 0.50)),
                    "D_p90": float(np.quantile(d, 0.90)),
                    "alignment_before": align_before,
                    "alignment_after": align_after,
                    "alignment_delta": align_after - align_before,
                    "universal": int((tiers == 0).sum()),
                    "group": int((tiers == 1).sum()),
                    "specialist": int((tiers == 2).sum()),
                    "storage_ratio": storage_ratio(tiers, group_count),
                }
            )
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deepseek_v2_lite.yaml")
    ap.add_argument("--weights-dir", required=True)
    ap.add_argument("--budgets", default="0.08,0.12,0.18")
    ap.add_argument("--layers", default=None)
    ap.add_argument("--reconstruct", default="fallback_bf16", choices=["fallback_bf16"])
    ap.add_argument("--ppl-dataset", default="wikitext-2-raw-v1")
    ap.add_argument("--eval-tokens", type=int, default=2048)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=1024)
    ap.add_argument("--seq-len", type=int, default=None, help="Legacy alias; use --max-length/--stride.")
    ap.add_argument("--max-gpu-memory", default="30GiB")
    ap.add_argument("--max-cpu-memory", default=None)
    ap.add_argument("--offload-dir", default=None)
    ap.add_argument("--original-cache-dir", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--force-safetensors-reload", action="store_true")
    ap.add_argument("--allow-cpu-offload", action="store_true")
    ap.add_argument("--skip-uniform-int4", action="store_true")
    ap.add_argument("--refresh-original-cache", action="store_true")
    ap.add_argument("--evict-checkpoint-cache", action="store_true")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.seq_len is not None:
        args.max_length = args.seq_len
        args.stride = args.seq_len

    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    cfg = load_config(Path(args.config))
    repo = cfg["model"]["repo"]
    group_count = int(cfg["experiment"].get("group_count", 4))
    seed = int(cfg["experiment"].get("seed", 0))
    weights_dir = Path(args.weights_dir).resolve()
    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    layers = parse_layers(args.layers, weights_dir)
    out_dir = weights_dir / "ppl"
    out_dir.mkdir(parents=True, exist_ok=True)
    original_cache_dir = Path(args.original_cache_dir or out_dir / "original_layers").resolve()
    output_path = Path(args.output).resolve() if args.output else out_dir / "ppl_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT')}")
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    evict_info = None
    if args.evict_checkpoint_cache:
        evict_info = evict_checkpoint_cache(repo, local_files_only=args.local_files_only)
        log(f"evicted checkpoint cache: {evict_info}")
    log(f"loading tokenizer/model: {repo}")
    tokenizer = AutoTokenizer.from_pretrained(repo, trust_remote_code=True, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_memory: dict[int | str, str] = {}
    if torch.cuda.is_available():
        max_memory.update({i: args.max_gpu_memory for i in range(torch.cuda.device_count())})
    if args.max_cpu_memory:
        max_memory["cpu"] = args.max_cpu_memory
    model = AutoModelForCausalLM.from_pretrained(
        repo,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        max_memory=max_memory,
        local_files_only=args.local_files_only,
    )
    model.eval()
    device_map = device_map_summary(model)
    log(f"DEVICE MAP COUNTS: {device_map['counts']}")
    assert_no_offload(model, allow_cpu=args.allow_cpu_offload)
    reload_info = None
    if args.force_safetensors_reload:
        reload_info = force_reload_safetensors(model, repo, local_files_only=args.local_files_only)

    text_source, eval_text, dataset_errors = load_eval_text(args.ppl_dataset, max(args.eval_tokens * 8, 20000))
    log(f"evaluation text source: {text_source}")

    save_original_layers(model, layers, original_cache_dir, args.refresh_original_cache)
    write_metrics_csv(weights_dir, weights_dir / "metrics_summary.csv", layers, budgets, group_count)

    results: dict[str, Any] = {
        "repo": repo,
        "layers": layers,
        "reconstruct": args.reconstruct,
        "ppl_dataset_requested": args.ppl_dataset,
        "ppl_dataset_used": text_source,
        "dataset_errors": dataset_errors,
        "eval_tokens_requested": args.eval_tokens,
        "max_length": args.max_length,
        "stride": args.stride,
        "gpu_binding": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
        "max_memory": max_memory,
        "evict_checkpoint_cache": evict_info,
        "device_map": device_map,
        "force_safetensors_reload": reload_info,
        "cuda_memory_after_load": cuda_memory_summary(),
        "dense_baseline": None,
        "uniform_int4_allexpert": None,
        "fans_moe": {},
        "notes": [
            "PPL is measured after BF16 reconstruction in the ordinary model forward path.",
            "Compression is applied to the selected representative layers listed in `layers`.",
        ],
    }

    log("evaluating dense baseline")
    results["dense_baseline"] = evaluate_ppl(
        model, tokenizer, eval_text, args.eval_tokens, args.max_length, args.stride, "dense"
    )

    if not args.skip_uniform_int4:
        log("evaluating uniform INT4 baseline on selected layers")
        for layer in layers:
            apply_uniform_int4_layer(model, layer, original_cache_dir)
        results["uniform_int4_allexpert"] = evaluate_ppl(
            model, tokenizer, eval_text, args.eval_tokens, args.max_length, args.stride, "uniform_int4"
        )

    for budget in budgets:
        log(f"evaluating FANS-MoE budget={budget}")
        layer_meta = []
        for layer in layers:
            layer_meta.append(
                apply_layer_from_tiers(
                    model,
                    layer,
                    weights_dir,
                    original_cache_dir,
                    budget,
                    group_count,
                    seed,
                    out_dir / "compression_details",
                )
            )
        ppl = evaluate_ppl(
            model, tokenizer, eval_text, args.eval_tokens, args.max_length, args.stride, f"fans_moe_B{budget}"
        )
        results["fans_moe"][str(budget)] = {"ppl": ppl, "layers": layer_meta}
        output_path.with_suffix(".partial.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    results["cuda_memory_final"] = cuda_memory_summary()
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(f"wrote {output_path}")


if __name__ == "__main__":
    main()
