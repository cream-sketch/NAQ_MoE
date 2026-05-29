#!/usr/bin/env python3
"""Software-side ablations for FANS-MoE tier allocation."""

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


TIER_NAMES = ["universal", "group", "specialist"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_list(text: str, cast):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def budget_tag(budget: float | str) -> str:
    return str(budget).replace(".", "p")


def normalize_columns(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return torch.nn.functional.normalize(x, p=2, dim=0, eps=1e-12)


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
    xx -= xx.mean()
    yy -= yy.mean()
    denom = np.sqrt((xx * xx).sum() * (yy * yy).sum())
    return float((xx * yy).sum() / denom) if denom > 0 else float("nan")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return float("nan")
    return corr(rankdata(x[mask]), rankdata(y[mask]))


def storage_ratio(tiers: np.ndarray, group_count: int, experts: int = 64) -> float:
    u = float((tiers == 0).mean())
    g = float((tiers == 1).mean())
    s = float((tiers == 2).mean())
    return (2.0 * u + float(group_count) * g + 0.5 * float(experts) * s) / (2.0 * float(experts))


def data_driven_tiers(metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(metric.reshape(-1, 1))
    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    relabel = {int(old): int(new) for new, old in enumerate(order)}
    tiers = np.array([relabel[int(x)] for x in km.labels_], dtype=np.int32)
    return tiers, centers[order]


def enforce_budget_by_metric(tiers: np.ndarray, metric: np.ndarray, budget: float, group_count: int) -> np.ndarray:
    tiers = tiers.copy()
    if storage_ratio(tiers, group_count) <= budget:
        return tiers
    spec_order = np.where(tiers == 2)[0]
    spec_order = spec_order[np.argsort(metric[spec_order])]
    for j in spec_order:
        tiers[j] = 1
        if storage_ratio(tiers, group_count) <= budget:
            return tiers
    group_order = np.where(tiers == 1)[0]
    group_order = group_order[np.argsort(metric[group_order])]
    for j in group_order:
        tiers[j] = 0
        if storage_ratio(tiers, group_count) <= budget:
            return tiers
    return tiers


def metric_tiers(metric: np.ndarray, budget: float, group_count: int) -> tuple[np.ndarray, dict[str, Any]]:
    natural, centers = data_driven_tiers(metric)
    final = enforce_budget_by_metric(natural, metric, budget, group_count)
    return final, {
        "centers": [float(x) for x in centers],
        "natural_storage_ratio": float(storage_ratio(natural, group_count)),
        "actual_storage_ratio": float(storage_ratio(final, group_count)),
        "budget_binding": bool(storage_ratio(natural, group_count) > budget),
        "moves_S_to_G": int(np.sum((natural == 2) & (final == 1))),
        "moves_G_to_U": int(np.sum((natural == 1) & (final == 0))),
    }


def load_raw_signatures(weights_dir: Path, layer: int) -> list[torch.Tensor]:
    out = []
    for expert in range(64):
        out.append(torch.load(weights_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt", map_location="cpu"))
    return out


def functional_dispersion_from_acts(acts: list[torch.Tensor], perms: np.ndarray | None, chunk: int = 64) -> np.ndarray:
    experts = len(acts)
    d_ff = acts[0].shape[1]
    triu = torch.triu_indices(experts, experts, offset=1)
    out = np.empty((d_ff,), dtype=np.float32)
    normed = []
    for expert, sig in enumerate(acts):
        if perms is None:
            aligned = sig
        else:
            aligned = sig[:, perms[expert].astype(np.int64)]
        normed.append(normalize_columns(aligned))
    for start in range(0, d_ff, chunk):
        end = min(start + chunk, d_ff)
        block = torch.stack([x[:, start:end].T for x in normed], dim=1)
        gram = torch.bmm(block, block.transpose(1, 2))
        avg_sim = gram[:, triu[0], triu[1]].mean(dim=1)
        out[start:end] = (1.0 - avg_sim).numpy()
    return out


def weight_dispersion(original_cache_dir: Path, layer: int, perms: np.ndarray, chunk: int = 32) -> np.ndarray:
    orig = torch.load(original_cache_dir / f"layer{layer:02d}_original_expert_weights.pt", map_location="cpu")
    experts = len(orig["gate"])
    d_ff = perms.shape[1]
    triu = torch.triu_indices(experts, experts, offset=1)
    out = np.empty((d_ff,), dtype=np.float32)
    for start in range(0, d_ff, chunk):
        end = min(start + chunk, d_ff)
        slot_count = end - start
        per_expert = []
        for e in range(experts):
            cols = torch.as_tensor(perms[e, start:end].astype(np.int64), dtype=torch.long)
            gate = orig["gate"][e][cols, :].float()
            up = orig["up"][e][cols, :].float()
            down = orig["down"][e][:, cols].T.float()
            vec = torch.cat([gate, up, down], dim=1)
            vec = torch.nn.functional.normalize(vec, p=2, dim=1, eps=1e-12)
            per_expert.append(vec)
        block = torch.stack(per_expert, dim=1).contiguous()
        gram = torch.bmm(block, block.transpose(1, 2))
        avg_sim = gram[:, triu[0], triu[1]].mean(dim=1)
        out[start:end] = (1.0 - avg_sim).numpy()
        log(f"layer {layer}: weight dispersion slots {start}-{end - 1}")
    return out


def tier_stats(metric: np.ndarray, tiers: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for idx, name in enumerate(TIER_NAMES):
        vals = metric[tiers == idx]
        out[f"{name}_count"] = int(vals.size)
        out[f"{name}_mean_D"] = float(vals.mean()) if vals.size else float("nan")
        out[f"{name}_min_D"] = float(vals.min()) if vals.size else float("nan")
        out[f"{name}_max_D"] = float(vals.max()) if vals.size else float("nan")
    out["separation_S_minus_U"] = out["specialist_mean_D"] - out["universal_mean_D"]
    return out


def random_tier_baseline(metric: np.ndarray, counts: list[int], seeds: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    sep = []
    u_mean = []
    g_mean = []
    s_mean = []
    base = np.array([0] * counts[0] + [1] * counts[1] + [2] * counts[2], dtype=np.int32)
    for _ in range(seeds):
        rng.shuffle(base)
        stats = tier_stats(metric, base)
        sep.append(stats["separation_S_minus_U"])
        u_mean.append(stats["universal_mean_D"])
        g_mean.append(stats["group_mean_D"])
        s_mean.append(stats["specialist_mean_D"])
    return {
        "random_trials": int(seeds),
        "random_universal_mean_D": float(np.mean(u_mean)),
        "random_group_mean_D": float(np.mean(g_mean)),
        "random_specialist_mean_D": float(np.mean(s_mean)),
        "random_separation_mean": float(np.mean(sep)),
        "random_separation_std": float(np.std(sep)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="outputs_realtext_512")
    ap.add_argument("--original-cache-dir", default="outputs_realtext_512/ppl/original_layers")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--budgets", default="0.08,0.12,0.18")
    ap.add_argument("--group-count", type=int, default=4)
    ap.add_argument("--random-trials", type=int, default=100)
    ap.add_argument("--output-dir", default="outputs_realtext_512/software_ablation")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    original_cache_dir = Path(args.original_cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_list(args.layers, int)
    budgets = parse_list(args.budgets, float)

    alignment_rows = []
    budget_rows = []
    all_payload: dict[str, Any] = {"layers": {}, "budgets": [float(x) for x in budgets]}

    for layer in layers:
        log(f"layer {layer}: loading signatures/permutations")
        perms = np.load(weights_dir / "alignments" / f"layer{layer:02d}_perms.npy")
        d_func = np.load(weights_dir / "distances" / f"layer{layer:02d}_D.npy").astype(np.float32)
        acts = load_raw_signatures(weights_dir, layer)
        d_raw = functional_dispersion_from_acts(acts, perms=None)
        d_aligned_recomputed = functional_dispersion_from_acts(acts, perms=perms)
        del acts
        d_weight = weight_dispersion(original_cache_dir, layer, perms)
        np.savez_compressed(
            out_dir / f"layer{layer:02d}_metrics.npz",
            D_functional=d_func,
            D_raw_index=d_raw,
            D_aligned_recomputed=d_aligned_recomputed,
            D_weight=d_weight,
        )

        align_row = {
            "layer": int(layer),
            "D_raw_mean": float(d_raw.mean()),
            "D_aligned_mean": float(d_func.mean()),
            "D_raw_p50": float(np.quantile(d_raw, 0.50)),
            "D_aligned_p50": float(np.quantile(d_func, 0.50)),
            "alignment_reduces_D_by": float(d_raw.mean() - d_func.mean()),
            "weight_vs_functional_pearson": corr(d_weight, d_func),
            "weight_vs_functional_spearman": spearman(d_weight, d_func),
            "raw_vs_functional_spearman": spearman(d_raw, d_func),
        }
        alignment_rows.append(align_row)
        layer_payload: dict[str, Any] = {"alignment": align_row, "budgets": {}}
        log(
            f"layer {layer}: rawD={align_row['D_raw_mean']:.4f} "
            f"alignedD={align_row['D_aligned_mean']:.4f} "
            f"weight_spearman={align_row['weight_vs_functional_spearman']:.4f}"
        )

        for budget in budgets:
            tag = budget_tag(budget)
            actual_tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
            functional_tiers, functional_meta = metric_tiers(d_func, budget, args.group_count)
            noalign_tiers, noalign_meta = metric_tiers(d_raw, budget, args.group_count)
            weight_tiers, weight_meta = metric_tiers(d_weight, budget, args.group_count)
            counts = [int((actual_tiers == idx).sum()) for idx in range(3)]

            actual_stats = tier_stats(d_func, actual_tiers)
            noalign_stats = tier_stats(d_func, noalign_tiers)
            weight_stats = tier_stats(d_func, weight_tiers)
            random_stats = random_tier_baseline(d_func, counts, args.random_trials)

            row = {
                "layer": int(layer),
                "budget": float(budget),
                "actual_storage_ratio": storage_ratio(actual_tiers, args.group_count),
                "actual_U": counts[0],
                "actual_G": counts[1],
                "actual_S": counts[2],
                "actual_U_mean_D": actual_stats["universal_mean_D"],
                "actual_G_mean_D": actual_stats["group_mean_D"],
                "actual_S_mean_D": actual_stats["specialist_mean_D"],
                "actual_separation": actual_stats["separation_S_minus_U"],
                "random_separation": random_stats["random_separation_mean"],
                "noalign_separation": noalign_stats["separation_S_minus_U"],
                "weight_separation": weight_stats["separation_S_minus_U"],
                "noalign_tier_overlap": float(np.mean(noalign_tiers == actual_tiers)),
                "weight_tier_overlap": float(np.mean(weight_tiers == actual_tiers)),
                "functional_recomputed_overlap": float(np.mean(functional_tiers == actual_tiers)),
                "functional_centers": ";".join(f"{x:.6f}" for x in functional_meta["centers"]),
                "functional_natural_ratio": functional_meta["natural_storage_ratio"],
                "functional_moves_S_to_G": functional_meta["moves_S_to_G"],
                "functional_moves_G_to_U": functional_meta["moves_G_to_U"],
                "cut_universal_max_D": actual_stats["universal_max_D"],
                "cut_group_min_D": actual_stats["group_min_D"],
                "cut_group_max_D": actual_stats["group_max_D"],
                "cut_specialist_min_D": actual_stats["specialist_min_D"],
            }
            budget_rows.append(row)
            layer_payload["budgets"][str(budget)] = {
                "actual": actual_stats,
                "random": random_stats,
                "no_alignment_tiers_evaluated_on_functional_D": noalign_stats,
                "weight_tiers_evaluated_on_functional_D": weight_stats,
                "functional_meta": functional_meta,
                "noalign_meta": noalign_meta,
                "weight_meta": weight_meta,
                "row": row,
            }
            log(
                f"layer {layer} B={budget}: sep actual={row['actual_separation']:.4f} "
                f"random={row['random_separation']:.4f} noalign={row['noalign_separation']:.4f} "
                f"weight={row['weight_separation']:.4f}"
            )
        all_payload["layers"][str(layer)] = layer_payload

    with (out_dir / "alignment_distance_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(alignment_rows[0].keys()))
        writer.writeheader()
        writer.writerows(alignment_rows)
    with (out_dir / "tier_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(budget_rows[0].keys()))
        writer.writeheader()
        writer.writerows(budget_rows)
    all_payload["max_rss_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (out_dir / "software_ablation_summary.json").write_text(json.dumps(all_payload, indent=2), encoding="utf-8")
    log(f"wrote {out_dir}")
    log(f"max_rss_kb={all_payload['max_rss_kb']}")


if __name__ == "__main__":
    main()
