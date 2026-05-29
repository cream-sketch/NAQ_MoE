#!/usr/bin/env python3
"""Dataflow traffic ablations for FANS-MoE tier maps."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import time
from pathlib import Path
from typing import Any

import numpy as np


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_list(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def budget_tag(budget: float | str) -> str:
    return str(budget).replace(".", "p")


def active_group_mean(top_ids: np.ndarray, labels: np.ndarray, group_slots: np.ndarray) -> float:
    vals = []
    for experts_for_token in top_ids:
        per_slot = []
        for slot in group_slots:
            labs = labels[int(slot), experts_for_token]
            labs = labs[labs >= 0]
            per_slot.append(len(set(int(x) for x in labs.tolist())))
        if per_slot:
            vals.append(float(np.mean(per_slot)))
    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="outputs_realtext_512")
    ap.add_argument("--trace-dir", default="outputs_realtext_512/hw_router_traffic")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--budgets", default="0.08,0.12,0.18")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--d-ff", type=int, default=1408)
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--group-count", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", default="outputs_realtext_512/dataflow_ablation")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_list(args.layers, int)
    budgets = parse_list(args.budgets, float)
    rng = np.random.default_rng(args.seed)
    triplet = 3 * args.d_model

    rows: list[dict[str, Any]] = []
    for layer in layers:
        top_ids = np.load(trace_dir / f"layer{layer:02d}_router_trace.npz")["top_ids"]
        dense_bf16 = float(args.top_k * args.d_ff * triplet * 2)
        uniform_int4 = float(args.top_k * args.d_ff * triplet * 0.5)
        for budget in budgets:
            tag = budget_tag(budget)
            tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
            labels = np.load(weights_dir / "ppl" / "compression_details" / f"layer{layer:02d}_group_labels_B{tag}.npy")
            group_slots = np.where(tiers == 1)[0]
            n_u = int((tiers == 0).sum())
            n_g = int((tiers == 1).sum())
            n_s = int((tiers == 2).sum())

            real_active_g = active_group_mean(top_ids, labels, group_slots)
            random_labels = np.full_like(labels, -1)
            if group_slots.size:
                random_labels[group_slots, :] = rng.integers(0, args.group_count, size=(group_slots.size, labels.shape[1]))
            random_active_g = active_group_mean(top_ids, random_labels, group_slots)

            u_shared = float(n_u * triplet * 2)
            g_real = float(real_active_g * n_g * triplet)
            g_random = float(random_active_g * n_g * triplet)
            g_no_reuse = float(args.top_k * n_g * triplet)
            s_stream = float(args.top_k * n_s * triplet * 0.5)

            fans = u_shared + g_real + s_stream
            no_group_reuse = u_shared + g_no_reuse + s_stream
            no_universal_reuse = float(args.top_k * n_u * triplet * 2) + g_real + s_stream
            tiered_streaming = float(args.top_k * (n_u * 2 + n_g * 1 + n_s * 0.5) * triplet)
            random_group = u_shared + g_random + s_stream

            row = {
                "layer": layer,
                "budget": budget,
                "n_universal": n_u,
                "n_group": n_g,
                "n_specialist": n_s,
                "active_groups_real": real_active_g,
                "active_groups_random": random_active_g,
                "dense_bf16_bytes": dense_bf16,
                "uniform_int4_bytes": uniform_int4,
                "fans_bytes": fans,
                "no_group_reuse_bytes": no_group_reuse,
                "no_universal_reuse_bytes": no_universal_reuse,
                "tiered_streaming_bytes": tiered_streaming,
                "random_group_bytes": random_group,
                "fans_vs_dense_reduction": dense_bf16 / fans,
                "no_group_reuse_over_fans": no_group_reuse / fans,
                "no_universal_reuse_over_fans": no_universal_reuse / fans,
                "random_group_over_fans": random_group / fans,
            }
            rows.append(row)
            log(
                f"layer {layer} B={budget}: fans={fans/1e6:.2f}MB "
                f"noG/FANS={row['no_group_reuse_over_fans']:.2f} "
                f"noU/FANS={row['no_universal_reuse_over_fans']:.2f} "
                f"random/FANS={row['random_group_over_fans']:.2f}"
            )

    csv_path = out_dir / "dataflow_ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"rows": rows, "max_rss_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
    (out_dir / "dataflow_ablation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"wrote {csv_path}")
    log(f"max_rss_kb={summary['max_rss_kb']}")


if __name__ == "__main__":
    main()
