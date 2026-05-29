#!/usr/bin/env python3
"""Routed-dispersion diagnostic for FANS-MoE.

Uses existing bypass activation signatures plus router traces. For each aligned
slot, it estimates dispersion on co-routed token subsets and compares it with
the bypass dispersion used for tier allocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(x), dtype=np.float64)
    return ranks


def corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    xx = x[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    denom = np.sqrt((xx * xx).sum() * (yy * yy).sum())
    return float((xx * yy).sum() / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return corr(rankdata(x[mask]), rankdata(y[mask]))


def pct(x: np.ndarray, q: float) -> float:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def load_aligned_activations(weights_dir: Path, layer: int) -> list[torch.Tensor]:
    perms = np.load(weights_dir / "alignments" / f"layer{layer:02d}_perms.npy")
    acts = []
    for expert in range(perms.shape[0]):
        path = weights_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt"
        raw = torch.load(path, map_location="cpu").float()
        aligned = raw[:, perms[expert].astype(np.int64)].contiguous()
        acts.append(aligned)
    return acts


def routed_dispersion_for_layer(
    weights_dir: Path,
    trace_dir: Path,
    layer: int,
    min_pair_tokens: int,
) -> dict[str, Any]:
    log(f"layer {layer}: loading aligned activations")
    acts = load_aligned_activations(weights_dir, layer)
    num_experts = len(acts)
    num_slots = acts[0].shape[1]
    trace = np.load(trace_dir / f"layer{layer:02d}_router_trace.npz")
    top_ids = trace["top_ids"]
    token_masks = []
    for expert in range(num_experts):
        token_masks.append(np.any(top_ids == expert, axis=1))

    d_sum = np.zeros((num_slots,), dtype=np.float64)
    d_count = np.zeros((num_slots,), dtype=np.int32)
    pair_token_counts = []

    for e1 in range(num_experts):
        a1 = acts[e1]
        m1 = token_masks[e1]
        for e2 in range(e1 + 1, num_experts):
            idx = np.where(m1 & token_masks[e2])[0]
            pair_token_counts.append(int(idx.size))
            if idx.size < min_pair_tokens:
                continue
            x = a1[idx, :]
            y = acts[e2][idx, :]
            num = (x * y).sum(dim=0)
            den = torch.linalg.vector_norm(x, dim=0) * torch.linalg.vector_norm(y, dim=0)
            cos = (num / den.clamp_min(1e-12)).cpu().numpy()
            dist = 1.0 - cos
            valid = np.isfinite(dist)
            d_sum[valid] += dist[valid]
            d_count[valid] += 1

    d_routed = np.full((num_slots,), np.nan, dtype=np.float32)
    valid_slots = d_count > 0
    d_routed[valid_slots] = (d_sum[valid_slots] / d_count[valid_slots]).astype(np.float32)
    d_bypass = np.load(weights_dir / "distances" / f"layer{layer:02d}_D.npy").astype(np.float32)

    out = {
        "layer": int(layer),
        "tokens": int(top_ids.shape[0]),
        "num_experts": int(num_experts),
        "num_slots": int(num_slots),
        "min_pair_tokens": int(min_pair_tokens),
        "valid_slots": int(valid_slots.sum()),
        "valid_slot_fraction": float(valid_slots.mean()),
        "mean_valid_pairs_per_slot": float(d_count[valid_slots].mean()) if valid_slots.any() else 0.0,
        "pair_tokens_mean": float(np.mean(pair_token_counts)),
        "pair_tokens_p50": float(np.quantile(pair_token_counts, 0.50)),
        "pair_tokens_p90": float(np.quantile(pair_token_counts, 0.90)),
        "D_bypass_mean": float(np.nanmean(d_bypass)),
        "D_routed_mean": float(np.nanmean(d_routed)),
        "D_bypass_p50": pct(d_bypass, 0.50),
        "D_routed_p50": pct(d_routed, 0.50),
        "pearson_D_B_vs_D_R": corr(d_bypass, d_routed),
        "spearman_D_B_vs_D_R": spearman(d_bypass, d_routed),
    }
    return {"summary": out, "D_bypass": d_bypass, "D_routed": d_routed, "D_routed_pair_count": d_count}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="outputs_realtext_512")
    ap.add_argument("--trace-dir", default="outputs_realtext_512/hw_router_traffic")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--min-pair-tokens", type=int, default=3)
    ap.add_argument("--output-dir", default="outputs_realtext_512/routed_dispersion")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for layer in parse_layers(args.layers):
        result = routed_dispersion_for_layer(weights_dir, trace_dir, layer, args.min_pair_tokens)
        rows.append(result["summary"])
        np.savez_compressed(
            out_dir / f"layer{layer:02d}_routed_dispersion.npz",
            D_bypass=result["D_bypass"],
            D_routed=result["D_routed"],
            D_routed_pair_count=result["D_routed_pair_count"],
        )
        log(
            "layer {layer}: valid={valid:.3f} pearson={pearson:.3f} spearman={spearman:.3f}".format(
                layer=layer,
                valid=result["summary"]["valid_slot_fraction"],
                pearson=result["summary"]["pearson_D_B_vs_D_R"],
                spearman=result["summary"]["spearman_D_B_vs_D_R"],
            )
        )

    with (out_dir / "routed_dispersion_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "weights_dir": str(weights_dir),
        "trace_dir": str(trace_dir),
        "rows": rows,
        "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    (out_dir / "routed_dispersion_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {out_dir / 'routed_dispersion_summary.csv'}")
    log(f"max_rss_kb={summary['max_rss_kb']}")


if __name__ == "__main__":
    main()
