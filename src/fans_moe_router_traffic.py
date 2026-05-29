#!/usr/bin/env python3
"""Router-trace and traffic diagnostics for FANS-MoE.

This script uses saved real-text pre-MoE hidden states and checkpoint router
weights to avoid loading the full language model. It records top-k router traces
and estimates tier-aware weight traffic for existing FANS-MoE tier maps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_list(text: str, cast):
    out = []
    for part in text.split(","):
        part = part.strip()
        if part:
            out.append(cast(part))
    return out


def budget_tag(budget: float | str) -> str:
    return str(budget).replace(".", "p")


def load_index(snapshot_dir: Path) -> dict[str, str]:
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    return dict(index["weight_map"])


def load_tensor(snapshot_dir: Path, weight_map: dict[str, str], key: str) -> torch.Tensor:
    shard = snapshot_dir / weight_map[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def load_hidden(weights_dir: Path, layer: int) -> torch.Tensor:
    path = weights_dir / "data" / "calibration_realtext" / f"layer{layer:02d}_hidden.pt"
    obj = torch.load(path, map_location="cpu")
    hidden = obj["hidden"] if isinstance(obj, dict) else obj
    return hidden.float().contiguous()


def storage_ratio(tiers: np.ndarray, group_count: int, experts: int) -> float:
    u = float((tiers == 0).mean())
    g = float((tiers == 1).mean())
    s = float((tiers == 2).mean())
    return (2.0 * u + float(group_count) * g + 0.5 * float(experts) * s) / (2.0 * float(experts))


def pct(vals: np.ndarray, q: float) -> float:
    if vals.size == 0:
        return float("nan")
    return float(np.quantile(vals, q))


def compute_router_trace(
    hidden: torch.Tensor,
    gate: torch.Tensor,
    *,
    top_k: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    top_ids = []
    top_weights = []
    gate = gate.to(device=device, dtype=torch.float32)
    for start in range(0, hidden.shape[0], batch_size):
        h = hidden[start : start + batch_size].to(device=device, dtype=torch.float32)
        logits = h @ gate.T
        probs = torch.softmax(logits, dim=-1)
        vals, ids = torch.topk(probs, k=top_k, dim=-1)
        top_ids.append(ids.cpu().numpy().astype(np.int16))
        top_weights.append(vals.cpu().numpy().astype(np.float32))
    return np.concatenate(top_ids, axis=0), np.concatenate(top_weights, axis=0)


def layer_traffic(
    *,
    layer: int,
    budget: float,
    weights_dir: Path,
    top_ids: np.ndarray,
    d_model: int,
    d_ff: int,
    experts: int,
    top_k: int,
    group_count: int,
) -> dict[str, Any]:
    tag = budget_tag(budget)
    tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
    labels_path = weights_dir / "ppl" / "compression_details" / f"layer{layer:02d}_group_labels_B{tag}.npy"
    group_labels = np.load(labels_path) if labels_path.exists() else None

    n_u = int((tiers == 0).sum())
    n_g = int((tiers == 1).sum())
    n_s = int((tiers == 2).sum())
    triplet_params = 3 * int(d_model)
    dense_bf16_bytes = float(top_k * d_ff * triplet_params * 2)
    uniform_int4_bytes = float(top_k * d_ff * triplet_params * 0.5)
    universal_sb_bytes = float(n_u * triplet_params * 2)
    specialist_hbm_bytes = float(top_k * n_s * triplet_params * 0.5)

    active_group_counts = []
    if n_g > 0 and group_labels is not None:
        group_slots = np.where(tiers == 1)[0]
        for experts_for_token in top_ids:
            counts = []
            for slot in group_slots:
                labs = group_labels[int(slot), experts_for_token]
                labs = labs[labs >= 0]
                counts.append(len(set(int(x) for x in labs.tolist())))
            active_group_counts.append(float(np.mean(counts)) if counts else 0.0)
        active_group_counts_np = np.asarray(active_group_counts, dtype=np.float32)
        mean_active_groups = float(active_group_counts_np.mean())
        p50_active_groups = pct(active_group_counts_np, 0.50)
        p90_active_groups = pct(active_group_counts_np, 0.90)
    else:
        mean_active_groups = float(group_count if n_g else 0)
        p50_active_groups = mean_active_groups
        p90_active_groups = mean_active_groups
        active_group_counts_np = np.full((top_ids.shape[0],), mean_active_groups, dtype=np.float32)

    group_l2_bytes_per_token = active_group_counts_np * float(n_g * triplet_params * 1.0)
    group_l2_bytes = float(group_l2_bytes_per_token.mean())
    fans_total_bytes = universal_sb_bytes + group_l2_bytes + specialist_hbm_bytes

    return {
        "layer": int(layer),
        "budget": float(budget),
        "tokens": int(top_ids.shape[0]),
        "top_k": int(top_k),
        "n_universal": n_u,
        "n_group": n_g,
        "n_specialist": n_s,
        "storage_ratio": storage_ratio(tiers, group_count, experts),
        "mean_active_groups": mean_active_groups,
        "p50_active_groups": p50_active_groups,
        "p90_active_groups": p90_active_groups,
        "dense_bf16_hbm_bytes_per_token": dense_bf16_bytes,
        "uniform_int4_hbm_bytes_per_token": uniform_int4_bytes,
        "fans_universal_sb_bytes_per_token": universal_sb_bytes,
        "fans_group_l2_bytes_per_token": group_l2_bytes,
        "fans_specialist_hbm_bytes_per_token": specialist_hbm_bytes,
        "fans_total_bytes_per_token": fans_total_bytes,
        "fans_total_reduction_vs_dense": dense_bf16_bytes / fans_total_bytes if fans_total_bytes else math.nan,
        "fans_hbm_reduction_vs_dense": dense_bf16_bytes / specialist_hbm_bytes if specialist_hbm_bytes else math.inf,
        "fans_hbm_reduction_vs_uniform_int4": uniform_int4_bytes / specialist_hbm_bytes if specialist_hbm_bytes else math.inf,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--weights-dir", default="outputs_realtext_512")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--budgets", default="0.08,0.12,0.18")
    ap.add_argument("--output-dir", default="outputs_realtext_512/hw_router_traffic")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_list(args.layers, int)
    budgets = parse_list(args.budgets, float)

    snapshot_dir = Path(snapshot_download(repo_id=args.model, local_files_only=args.local_files_only))
    config = json.loads((snapshot_dir / "config.json").read_text(encoding="utf-8"))
    weight_map = load_index(snapshot_dir)
    d_model = int(config["hidden_size"])
    d_ff = int(config["moe_intermediate_size"])
    experts = int(config["n_routed_experts"])
    top_k = int(config["num_experts_per_tok"])
    group_count = 4

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    log(f"device={device} layers={layers} budgets={budgets} top_k={top_k}")

    all_rows: list[dict[str, Any]] = []
    trace_summary: dict[str, Any] = {
        "model": args.model,
        "weights_dir": str(weights_dir),
        "layers": layers,
        "budgets": budgets,
        "d_model": d_model,
        "d_ff": d_ff,
        "experts": experts,
        "top_k": top_k,
        "group_count": group_count,
        "layers_detail": {},
    }

    for layer in layers:
        log(f"layer {layer}: loading hidden states and router gate")
        hidden = load_hidden(weights_dir, layer)
        gate_key = f"model.layers.{layer}.mlp.gate.weight"
        gate = load_tensor(snapshot_dir, weight_map, gate_key)
        top_ids, top_weights = compute_router_trace(
            hidden,
            gate,
            top_k=top_k,
            device=device,
            batch_size=args.batch_size,
        )
        np.savez_compressed(
            out_dir / f"layer{layer:02d}_router_trace.npz",
            top_ids=top_ids,
            top_weights=top_weights,
        )
        counts = np.bincount(top_ids.reshape(-1), minlength=experts)
        trace_summary["layers_detail"][str(layer)] = {
            "tokens": int(top_ids.shape[0]),
            "expert_selection_min": int(counts.min()),
            "expert_selection_max": int(counts.max()),
            "expert_selection_mean": float(counts.mean()),
            "expert_selection_p90": pct(counts.astype(np.float32), 0.90),
        }

        for budget in budgets:
            row = layer_traffic(
                layer=layer,
                budget=budget,
                weights_dir=weights_dir,
                top_ids=top_ids,
                d_model=d_model,
                d_ff=d_ff,
                experts=experts,
                top_k=top_k,
                group_count=group_count,
            )
            all_rows.append(row)
            log(
                "layer {layer} B={budget}: total={total:.2f}MB/token "
                "HBM={hbm:.2f}MB/token activeG={ag:.2f}".format(
                    layer=layer,
                    budget=budget,
                    total=row["fans_total_bytes_per_token"] / 1e6,
                    hbm=row["fans_specialist_hbm_bytes_per_token"] / 1e6,
                    ag=row["mean_active_groups"],
                )
            )

    csv_path = out_dir / "traffic_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary: dict[str, Any] = trace_summary | {
        "traffic_rows": all_rows,
        "csv": str(csv_path),
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    (out_dir / "traffic_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {csv_path}")
    log(f"max_rss_kb={summary['max_rss_kb']}")


if __name__ == "__main__":
    main()
