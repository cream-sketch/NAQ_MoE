#!/usr/bin/env python3
"""Prepare B=0.12 PPL ablation tier-map directories for FANS-MoE."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np


def budget_tag(budget: float | str) -> str:
    return str(budget).replace(".", "p")


def parse_layers(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


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


def enforce_budget(tiers: np.ndarray, metric: np.ndarray, budget: float, group_count: int) -> np.ndarray:
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


def symlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst, target_is_directory=src.is_dir())
    except OSError:
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def prepare_base(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["activations", "data"]:
        symlink_or_copy(src / name, dst / name)
    for name in ["alignments", "distances", "tier_maps", "summaries"]:
        (dst / name).mkdir(parents=True, exist_ok=True)


def write_variant(
    *,
    src: Path,
    dst: Path,
    variant: str,
    layers: list[int],
    budget: float,
    group_count: int,
    metrics_dir: Path,
) -> None:
    prepare_base(src, dst)
    tag = budget_tag(budget)
    rng = np.random.default_rng(0)

    for layer in layers:
        metrics = np.load(metrics_dir / f"layer{layer:02d}_metrics.npz")
        if variant == "noalign":
            metric = metrics["D_raw_index"].astype(np.float32)
            perms = np.tile(np.arange(metric.shape[0], dtype=np.int32), (64, 1))
        elif variant == "weight":
            metric = metrics["D_weight"].astype(np.float32)
            perms = np.load(src / "alignments" / f"layer{layer:02d}_perms.npy")
        elif variant == "random":
            actual = np.load(src / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
            counts = [int((actual == i).sum()) for i in range(3)]
            tiers = np.array([0] * counts[0] + [1] * counts[1] + [2] * counts[2], dtype=np.int32)
            rng.shuffle(tiers)
            metric = metrics["D_functional"].astype(np.float32)
            centers = [float(np.mean(metric[tiers == i])) for i in range(3)]
            perms = np.load(src / "alignments" / f"layer{layer:02d}_perms.npy")
            natural_ratio = storage_ratio(tiers, group_count)
            final_tiers = tiers
        else:
            raise ValueError(variant)

        if variant != "random":
            natural, centers_np = data_driven_tiers(metric)
            final_tiers = enforce_budget(natural, metric, budget, group_count)
            centers = [float(x) for x in centers_np]
            natural_ratio = storage_ratio(natural, group_count)

        np.save(dst / "alignments" / f"layer{layer:02d}_perms.npy", perms)
        sanity_src = src / "alignments" / f"layer{layer:02d}_sanity.json"
        if sanity_src.exists() and variant != "noalign":
            shutil.copy2(sanity_src, dst / "alignments" / f"layer{layer:02d}_sanity.json")
        else:
            (dst / "alignments" / f"layer{layer:02d}_sanity.json").write_text(
                json.dumps({"layer": layer, "method": variant, "note": "PPL ablation permutation map"}, indent=2),
                encoding="utf-8",
            )

        np.save(dst / "distances" / f"layer{layer:02d}_D.npy", metric)
        (dst / "distances" / f"layer{layer:02d}_clusters.json").write_text("[]\n", encoding="utf-8")
        np.save(dst / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy", final_tiers)
        counts = {name: int((final_tiers == idx).sum()) for idx, name in enumerate(["universal", "group", "specialist"])}
        meta = {
            "variant": variant,
            "layer": layer,
            "budget": budget,
            "tier_method": variant,
            "centers": centers,
            "natural_storage_ratio": natural_ratio,
            "actual_storage_ratio": storage_ratio(final_tiers, group_count),
            "group_count": group_count,
            "counts": counts,
        }
        (dst / "tier_maps" / f"layer{layer:02d}_meta_B{tag}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (dst / "ABLATION_VARIANT.json").write_text(
        json.dumps({"variant": variant, "source": str(src), "budget": budget, "layers": layers}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="outputs_realtext_512")
    ap.add_argument("--metrics-dir", default="outputs_realtext_512/software_ablation")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--budget", type=float, default=0.12)
    ap.add_argument("--group-count", type=int, default=4)
    ap.add_argument("--output-root", default="outputs_realtext_512/ppl_ablation_inputs")
    args = ap.parse_args()

    src = Path(args.source).resolve()
    metrics_dir = Path(args.metrics_dir).resolve()
    root = Path(args.output_root).resolve()
    layers = parse_layers(args.layers)
    for variant in ["noalign", "weight", "random"]:
        dst = root / variant
        write_variant(
            src=src,
            dst=dst,
            variant=variant,
            layers=layers,
            budget=args.budget,
            group_count=args.group_count,
            metrics_dir=metrics_dir,
        )
        print(f"prepared {variant}: {dst}", flush=True)


if __name__ == "__main__":
    main()
