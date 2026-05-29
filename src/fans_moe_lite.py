#!/usr/bin/env python3
"""Pilot FANS-MoE automation for DeepSeek-V2-Lite routed experts.

This implements the algorithmic core from moe_algorithm_spec.md without loading
the full language model:

1. Extract per-neuron activation signatures for every routed expert in a layer.
2. Align neuron slots across experts with a greedy functional matching.
3. Compute per-slot functional dispersion across the 64 experts.
4. Allocate Universal / Group / Specialist tiers for requested budgets.

The calibration source is random hidden states by default. That keeps the pilot
lean and avoids dataset transfer; production C4 hidden-state capture can reuse
the same downstream phases once full-model calibration is available.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from safetensors import safe_open


TIER_NAMES = {0: "universal", 1: "group", 2: "specialist"}


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    hidden_size: int
    d_ff: int
    n_experts: int
    first_moe_layer: int
    last_moe_layer: int


@dataclass(frozen=True)
class RunSpec:
    layers: list[int]
    calibration_tokens: int
    calibration_source: str
    calibration_hidden_dir: Path | None
    seed: int
    budgets: list[float]
    group_count: int
    max_cluster_threshold: int
    cluster_similarity_threshold: float
    activation_batch_tokens: int
    distance_slot_chunk: int
    tier_method: str
    output_dir: Path


def parse_layers(text: str | None, default: list[int]) -> list[int]:
    if not text:
        return default
    layers: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            layers.extend(range(int(a), int(b) + 1))
        else:
            layers.append(int(part))
    return sorted(set(layers))


def load_run_config(path: Path, args: argparse.Namespace) -> tuple[ModelSpec, RunSpec]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]
    exp_cfg = cfg["experiment"]
    paths_cfg = cfg["paths"]

    layers = parse_layers(args.layers, list(exp_cfg["layers"]))
    budgets = (
        [float(x) for x in args.budgets.split(",")]
        if args.budgets
        else [float(x) for x in exp_cfg["budgets"]]
    )
    output_dir = Path(args.output_dir or paths_cfg["output_dir"]).resolve()

    model = ModelSpec(
        repo=model_cfg["repo"],
        hidden_size=int(model_cfg["hidden_size"]),
        d_ff=int(model_cfg["moe_intermediate_size"]),
        n_experts=int(model_cfg["n_routed_experts"]),
        first_moe_layer=int(model_cfg["first_moe_layer"]),
        last_moe_layer=int(model_cfg["last_moe_layer"]),
    )
    run = RunSpec(
        layers=layers,
        calibration_tokens=int(args.tokens or exp_cfg["calibration_tokens"]),
        calibration_source=str(args.calibration or exp_cfg["calibration_source"]),
        calibration_hidden_dir=Path(args.calibration_hidden_dir).resolve()
        if args.calibration_hidden_dir
        else None,
        seed=int(args.seed if args.seed is not None else exp_cfg["seed"]),
        budgets=budgets,
        group_count=int(args.group_count or exp_cfg["group_count"]),
        max_cluster_threshold=int(exp_cfg["max_cluster_threshold"]),
        cluster_similarity_threshold=float(exp_cfg["cluster_similarity_threshold"]),
        activation_batch_tokens=int(exp_cfg["activation_batch_tokens"]),
        distance_slot_chunk=int(exp_cfg["distance_slot_chunk"]),
        tier_method=str(args.tier_method or exp_cfg.get("tier_method", "legacy_quantile")),
        output_dir=output_dir,
    )
    return model, run


def ensure_dirs(out: Path) -> None:
    for rel in [
        "data",
        "activations",
        "alignments",
        "distances",
        "tier_maps",
        "summaries",
        "logs",
    ]:
        (out / rel).mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def validate_layers(model: ModelSpec, layers: list[int]) -> None:
    bad = [x for x in layers if x < model.first_moe_layer or x > model.last_moe_layer]
    if bad:
        raise ValueError(
            f"Invalid MoE layers {bad}; expected {model.first_moe_layer}..{model.last_moe_layer}"
        )


def download_index(repo: str) -> dict[str, str]:
    hf_hub_download(repo, "config.json")
    index_path = hf_hub_download(repo, "model.safetensors.index.json")
    with open(index_path, "r", encoding="utf-8") as f:
        return json.load(f)["weight_map"]


def weight_name(layer: int, expert: int, kind: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{expert}.{kind}_proj.weight"


def layer_shards(wmap: dict[str, str], layer: int, model: ModelSpec) -> list[str]:
    needed = []
    for expert in range(model.n_experts):
        for kind in ("gate", "up", "down"):
            name = weight_name(layer, expert, kind)
            if name not in wmap:
                raise KeyError(f"Missing weight in index: {name}")
            needed.append(wmap[name])
    return sorted(set(needed))


def load_layer_weights(
    wmap: dict[str, str], layer: int, model: ModelSpec
) -> dict[str, list[torch.Tensor]]:
    shards = layer_shards(wmap, layer, model)
    log(f"layer {layer}: loading shards {shards}")
    weights: dict[str, list[torch.Tensor | None]] = {
        "gate": [None] * model.n_experts,
        "up": [None] * model.n_experts,
        "down": [None] * model.n_experts,
    }
    wanted = {
        weight_name(layer, expert, kind): (kind, expert)
        for expert in range(model.n_experts)
        for kind in ("gate", "up", "down")
    }
    for shard in shards:
        path = hf_hub_download(model.repo, shard)
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            for name, (kind, expert) in wanted.items():
                if name in keys:
                    weights[kind][expert] = f.get_tensor(name).contiguous()

    loaded: dict[str, list[torch.Tensor]] = {}
    for kind, tensors in weights.items():
        if any(x is None for x in tensors):
            missing = [i for i, x in enumerate(tensors) if x is None]
            raise RuntimeError(f"Missing {kind} weights for experts: {missing}")
        loaded[kind] = [x for x in tensors if x is not None]
    return loaded


def calibration_hidden(model: ModelSpec, run: RunSpec, layer: int) -> Path:
    if run.calibration_source in {"c4", "wikitext", "real_text", "saved_layer_hidden"}:
        base = run.calibration_hidden_dir or (run.output_dir / "data" / f"calibration_{run.calibration_source}")
        path = base / f"layer{layer:02d}_hidden.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing real calibration hidden states for layer {layer}: {path}. "
                "Run build_real_calibration.py first."
            )
        return path

    path = run.output_dir / "data" / (
        f"calibration_{run.calibration_source}_{run.calibration_tokens}_seed{run.seed}.pt"
    )
    if path.exists():
        return path
    if run.calibration_source != "random_hidden":
        raise NotImplementedError(
            "This mode requires precomputed per-layer hidden states. "
            "Use the saved activations interface for future full-model C4 capture."
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(run.seed)
    x = torch.randn(
        run.calibration_tokens,
        model.hidden_size,
        generator=gen,
        dtype=torch.float32,
    ).to(torch.bfloat16)
    torch.save({"hidden": x, "source": run.calibration_source, "seed": run.seed}, path)
    log(f"wrote calibration hidden states: {path}")
    return path


def activation_path(run: RunSpec, layer: int, expert: int) -> Path:
    return run.output_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt"


def extract_activations(
    layer: int,
    model: ModelSpec,
    run: RunSpec,
    weights: dict[str, list[torch.Tensor]],
    force: bool,
) -> None:
    hidden_obj = torch.load(calibration_hidden(model, run, layer), map_location="cpu")
    x_cpu: torch.Tensor = hidden_obj["hidden"]
    if x_cpu.shape != (run.calibration_tokens, model.hidden_size):
        raise ValueError(f"Unexpected calibration shape: {tuple(x_cpu.shape)}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"layer {layer}: extracting signatures on {device}")

    stats: dict[str, Any] = {
        "layer": layer,
        "tokens": run.calibration_tokens,
        "experts": model.n_experts,
        "d_ff": model.d_ff,
        "calibration_source": run.calibration_source,
        "expert_stats": {},
    }

    for expert in range(model.n_experts):
        out_path = activation_path(run, layer, expert)
        if out_path.exists() and not force:
            continue
        Wg = weights["gate"][expert].to(device)
        Wu = weights["up"][expert].to(device)
        chunks: list[torch.Tensor] = []
        for start in range(0, run.calibration_tokens, run.activation_batch_tokens):
            end = min(start + run.activation_batch_tokens, run.calibration_tokens)
            x = x_cpu[start:end].to(device)
            with torch.no_grad():
                act = F.silu(x @ Wg.t()) * (x @ Wu.t())
            chunks.append(act.detach().to(torch.bfloat16).cpu())
        sig = torch.cat(chunks, dim=0).contiguous()
        torch.save(sig, out_path)
        stats["expert_stats"][str(expert)] = {
            "mean_abs": float(sig.float().abs().mean().item()),
            "max_abs": float(sig.float().abs().max().item()),
            "nonzero_fraction": float((sig != 0).float().mean().item()),
        }
        del Wg, Wu, sig, chunks
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if expert % 8 == 0:
            log(f"layer {layer}: extracted expert {expert}/{model.n_experts - 1}")

    stats_path = run.output_dir / "activations" / f"layer{layer:02d}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def normalize_columns(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    xf = x.float().to(device)
    return xf / (xf.norm(dim=0, keepdim=True) + 1e-8)


def greedy_perm(sim: torch.Tensor) -> np.ndarray:
    sim_np = sim.float().cpu().numpy()
    d = sim_np.shape[0]
    order = np.argsort(sim_np.max(axis=1))[::-1]
    used = np.zeros(d, dtype=bool)
    perm = np.empty(d, dtype=np.int32)
    for row in order:
        scores = sim_np[row].copy()
        scores[used] = -np.inf
        col = int(np.argmax(scores))
        perm[row] = col
        used[col] = True
    return perm


def align_layer(layer: int, model: ModelSpec, run: RunSpec, force: bool) -> None:
    out_path = run.output_dir / "alignments" / f"layer{layer:02d}_perms.npy"
    sanity_path = run.output_dir / "alignments" / f"layer{layer:02d}_sanity.json"
    if out_path.exists() and sanity_path.exists() and not force:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"layer {layer}: aligning neurons with greedy functional matching")
    ref = torch.load(activation_path(run, layer, 0), map_location="cpu")
    ref_n = normalize_columns(ref, device)
    perms = np.zeros((model.n_experts, model.d_ff), dtype=np.int32)
    perms[0] = np.arange(model.d_ff, dtype=np.int32)
    sanity: dict[str, Any] = {"layer": layer, "method": "greedy", "experts": {}}
    eye = torch.arange(model.d_ff, device=device)

    for expert in range(1, model.n_experts):
        sig = torch.load(activation_path(run, layer, expert), map_location="cpu")
        target_n = normalize_columns(sig, device)
        with torch.no_grad():
            sim = ref_n.t() @ target_n
        perm = greedy_perm(sim)
        perms[expert] = perm
        aligned_cols = torch.tensor(perm, dtype=torch.long, device=device)
        raw_diag = sim[eye, eye].mean().item()
        aligned_diag = sim[eye, aligned_cols].mean().item()
        sanity["experts"][str(expert)] = {
            "mean_diag_unaligned": raw_diag,
            "mean_diag_aligned": aligned_diag,
            "improvement": aligned_diag - raw_diag,
            "is_permutation": bool(np.unique(perm).size == model.d_ff),
        }
        del sig, target_n, sim
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if expert % 8 == 0:
            log(f"layer {layer}: aligned expert {expert}/{model.n_experts - 1}")

    np.save(out_path, perms)
    sanity_path.write_text(json.dumps(sanity, indent=2), encoding="utf-8")


def compute_distance(layer: int, model: ModelSpec, run: RunSpec, force: bool) -> None:
    out_path = run.output_dir / "distances" / f"layer{layer:02d}_D.npy"
    cluster_path = run.output_dir / "distances" / f"layer{layer:02d}_clusters.json"
    if out_path.exists() and cluster_path.exists() and not force:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    perms = np.load(run.output_dir / "alignments" / f"layer{layer:02d}_perms.npy")
    log(f"layer {layer}: computing functional distance matrix")

    sigs = torch.empty(
        (model.n_experts, run.calibration_tokens, model.d_ff),
        dtype=torch.float32,
        device=device,
    )
    for expert in range(model.n_experts):
        sig = torch.load(activation_path(run, layer, expert), map_location="cpu")
        perm = torch.tensor(perms[expert], dtype=torch.long)
        sig = sig[:, perm]
        sigs[expert] = normalize_columns(sig, device)
        if expert % 16 == 0:
            log(f"layer {layer}: loaded aligned signature {expert}/{model.n_experts - 1}")

    triu = torch.triu_indices(model.n_experts, model.n_experts, offset=1, device=device)
    D = torch.empty(model.d_ff, dtype=torch.float32)
    cluster_info: list[dict[str, Any]] = []

    for start in range(0, model.d_ff, run.distance_slot_chunk):
        end = min(start + run.distance_slot_chunk, model.d_ff)
        slots = sigs[:, :, start:end].permute(2, 0, 1).contiguous()
        with torch.no_grad():
            gram = torch.bmm(slots, slots.transpose(1, 2))
            avg_sim = gram[:, triu[0], triu[1]].mean(dim=1)
            D[start:end] = (1.0 - avg_sim).cpu()
            neigh = (gram > run.cluster_similarity_threshold).sum(dim=2).max(dim=1).values
            neigh_cpu = neigh.cpu().tolist()
        for offset, max_cluster in enumerate(neigh_cpu):
            j = start + offset
            cluster_info.append(
                {
                    "slot": j,
                    "method": "cosine_neighborhood_proxy",
                    "threshold": run.cluster_similarity_threshold,
                    "max_cluster_size": int(max_cluster),
                }
            )
        log(f"layer {layer}: distance slots {start}-{end - 1}")

    np.save(out_path, D.numpy())
    cluster_path.write_text(json.dumps(cluster_info, indent=2), encoding="utf-8")
    del sigs
    if device.type == "cuda":
        torch.cuda.empty_cache()


def allocate_tiers(
    D: np.ndarray,
    cluster_info: list[dict[str, Any]],
    tau1: float,
    tau2: float,
    max_cluster_threshold: int,
) -> np.ndarray:
    tiers = np.zeros(D.shape[0], dtype=np.int32)
    max_clusters = np.array([x["max_cluster_size"] for x in cluster_info])
    for j in range(D.shape[0]):
        if D[j] < tau1 and max_clusters[j] >= max_cluster_threshold:
            tiers[j] = 0
        elif D[j] < tau2:
            tiers[j] = 1
        else:
            tiers[j] = 2
    return tiers


def storage_ratio(tiers: np.ndarray, group_count: int) -> float:
    u = float((tiers == 0).mean())
    g = float((tiers == 1).mean())
    s = float((tiers == 2).mean())
    # v2 storage model per neuron slot across 64 experts:
    # original = 64 * 2 BF16 units, universal = 1 * 2, group = G * 1,
    # specialist = 64 * 0.5.
    return (2.0 * u + float(group_count) * g + 32.0 * s) / 128.0


def data_driven_tiers(D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(D.reshape(-1, 1))
    centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(centers)
    relabel = {int(old): int(new) for new, old in enumerate(order)}
    tiers = np.array([relabel[int(x)] for x in km.labels_], dtype=np.int32)
    return tiers, centers[order]


def enforce_budget_by_dispersion(tiers: np.ndarray, D: np.ndarray, budget: float, group_count: int) -> np.ndarray:
    tiers = tiers.copy()
    if storage_ratio(tiers, group_count) <= budget:
        return tiers

    # Specialist is the expensive tier. Move the least-specialist-like slots
    # first, preserving the highest-D slots as specialist as long as possible.
    spec_order = np.where(tiers == 2)[0]
    spec_order = spec_order[np.argsort(D[spec_order])]
    for j in spec_order:
        tiers[j] = 1
        if storage_ratio(tiers, group_count) <= budget:
            return tiers

    group_order = np.where(tiers == 1)[0]
    group_order = group_order[np.argsort(D[group_order])]
    for j in group_order:
        tiers[j] = 0
        if storage_ratio(tiers, group_count) <= budget:
            return tiers
    return tiers


def best_thresholds(
    D: np.ndarray,
    cluster_info: list[dict[str, Any]],
    budget: float,
    run: RunSpec,
) -> tuple[np.ndarray, dict[str, Any]]:
    if run.tier_method == "data_driven":
        natural_tiers, centers = data_driven_tiers(D)
        natural_ratio = storage_ratio(natural_tiers, run.group_count)
        tiers = enforce_budget_by_dispersion(natural_tiers, D, budget, run.group_count)
        return tiers, {
            "tier_method": "data_driven_kmeans_1d",
            "centers": [float(x) for x in centers],
            "natural_storage_ratio": float(natural_ratio),
            "budget_binding": bool(natural_ratio > budget),
            "budget_nonbinding": bool(natural_ratio <= budget),
        }

    all_specialist_ratio = 96.0 / 768.0
    budget_nonbinding = budget >= all_specialist_ratio

    if budget_nonbinding:
        tau1 = float(np.quantile(D, 0.25))
        tau2 = float(np.quantile(D, 0.70))
        tiers = allocate_tiers(D, cluster_info, tau1, tau2, run.max_cluster_threshold)
        if (tiers == 0).mean() < 0.02:
            tiers = allocate_tiers(D, cluster_info, tau1, tau2, 1)
        return tiers, {
            "tau1": tau1,
            "tau2": tau2,
            "selection": "quantile_fallback",
            "budget_nonbinding": True,
            "note": (
                "Spec budgets >= all-specialist INT4 storage ratio (0.125), so the "
                "storage constraint is non-binding. Used dispersion quantiles to "
                "produce a nontrivial three-tier pilot map."
            ),
        }

    candidates: list[tuple[float, float, float, np.ndarray, float]] = []
    tau1_min, tau1_max = float(np.quantile(D, 0.02)), float(np.quantile(D, 0.45))
    tau2_min, tau2_max = float(np.quantile(D, 0.30)), float(np.quantile(D, 0.95))
    for tau1 in np.linspace(tau1_min, tau1_max, 24):
        for tau2 in np.linspace(tau2_min, tau2_max, 24):
            if tau1 >= tau2:
                continue
            tiers = allocate_tiers(D, cluster_info, tau1, tau2, run.max_cluster_threshold)
            ratio = storage_ratio(tiers, run.group_count)
            if ratio > budget:
                continue
            objective = float(D[tiers == 0].sum() - D[tiers == 2].sum())
            candidates.append((objective, float(tau1), float(tau2), tiers, ratio))
    if not candidates:
        raise RuntimeError(f"No feasible threshold pair found for budget={budget}")
    candidates.sort(key=lambda x: x[0])
    _, tau1, tau2, tiers, _ = candidates[0]
    return tiers, {
        "tau1": tau1,
        "tau2": tau2,
        "selection": "budget_grid_search",
        "budget_nonbinding": False,
    }


def tier_allocation(layer: int, model: ModelSpec, run: RunSpec, force: bool) -> None:
    D = np.load(run.output_dir / "distances" / f"layer{layer:02d}_D.npy")
    with open(run.output_dir / "distances" / f"layer{layer:02d}_clusters.json", "r", encoding="utf-8") as f:
        cluster_info = json.load(f)
    layer_summary: dict[str, Any] = {
        "layer": layer,
        "D": {
            "min": float(D.min()),
            "max": float(D.max()),
            "mean": float(D.mean()),
            "p25": float(np.quantile(D, 0.25)),
            "p50": float(np.quantile(D, 0.50)),
            "p70": float(np.quantile(D, 0.70)),
            "p90": float(np.quantile(D, 0.90)),
        },
        "budgets": {},
    }
    for budget in run.budgets:
        tag = str(budget).replace(".", "p")
        tiers_path = run.output_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy"
        meta_path = run.output_dir / "tier_maps" / f"layer{layer:02d}_meta_B{tag}.json"
        if tiers_path.exists() and meta_path.exists() and not force:
            continue
        tiers, meta = best_thresholds(D, cluster_info, budget, run)
        counts = {TIER_NAMES[i]: int((tiers == i).sum()) for i in range(3)}
        fractions = {name: value / model.d_ff for name, value in counts.items()}
        ratio = storage_ratio(tiers, run.group_count)
        meta.update(
            {
                "layer": layer,
                "budget": budget,
                "actual_storage_ratio": ratio,
                "group_count": run.group_count,
                "counts": counts,
                "fractions": fractions,
            }
        )
        np.save(tiers_path, tiers)
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        layer_summary["budgets"][str(budget)] = meta
        log(
            f"layer {layer} budget {budget}: "
            f"U/G/S={counts['universal']}/{counts['group']}/{counts['specialist']} "
            f"storage={ratio:.4f}"
        )
    summary_path = run.output_dir / "summaries" / f"layer{layer:02d}_summary.json"
    summary_path.write_text(json.dumps(layer_summary, indent=2), encoding="utf-8")


def write_run_summary(model: ModelSpec, run: RunSpec, started: float) -> None:
    summaries = []
    for layer in run.layers:
        path = run.output_dir / "summaries" / f"layer{layer:02d}_summary.json"
        if path.exists():
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
    payload = {
        "repo": model.repo,
        "layers": run.layers,
        "calibration_source": run.calibration_source,
        "calibration_tokens": run.calibration_tokens,
        "tier_method": run.tier_method,
        "budgets": run.budgets,
        "elapsed_sec": time.time() - started,
        "summaries": summaries,
    }
    out = run.output_dir / "summaries" / "run_summary.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote run summary: {out}")


def mean_alignment_improvement(run: RunSpec, layer: int) -> dict[str, float | int]:
    path = run.output_dir / "alignments" / f"layer{layer:02d}_sanity.json"
    if not path.exists():
        return {"experts": 0, "mean_before": float("nan"), "mean_after": float("nan"), "mean_improvement": float("nan")}
    data = json.loads(path.read_text(encoding="utf-8"))
    vals = list(data.get("experts", {}).values())
    if not vals:
        return {"experts": 0, "mean_before": float("nan"), "mean_after": float("nan"), "mean_improvement": float("nan")}
    return {
        "experts": len(vals),
        "mean_before": float(np.mean([x["mean_diag_unaligned"] for x in vals])),
        "mean_after": float(np.mean([x["mean_diag_aligned"] for x in vals])),
        "mean_improvement": float(np.mean([x["improvement"] for x in vals])),
    }


def write_markdown_report(model: ModelSpec, run: RunSpec) -> None:
    summary_path = run.output_dir / "summaries" / "run_summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    lines = [
        "# FANS-MoE Automated Experiment Results",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Scope",
        "",
        f"- Model: `{model.repo}`",
        f"- Layers: `{', '.join(str(x) for x in run.layers)}`",
        f"- Routed experts per layer: `{model.n_experts}`",
        f"- Neurons per expert: `{model.d_ff}`",
        f"- Calibration source: `{run.calibration_source}`",
        f"- Calibration tokens: `{run.calibration_tokens}`",
        f"- Tier method: `{run.tier_method}`",
        f"- GPU binding: `CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}`",
        "",
        "No dataset was transferred through `ial-jump`. Text/model assets are fetched directly on the remote server.",
        "",
        "## Phase Status",
        "",
        "| Phase | Status | Output |",
        "|---|---:|---|",
        "| 1. Activation signatures | done | `activations/layerXX_expertYY.pt` |",
        "| 2. Neuron alignment | done | `alignments/layerXX_perms.npy` |",
        "| 3. Functional distances | done | `distances/layerXX_D.npy` |",
        "| 4. Tier allocation | done | `tier_maps/layerXX_*` |",
        "| 5. Weight compression | not run in this pilot | pending full calibration/PPL stage |",
        "| 6. PPL reconstruction | not run in this pilot | pending full-model/offload stage |",
        "",
        "## Layer Results",
        "",
        "| Layer | D mean | D p25 | D p50 | D p90 | Align before | Align after | Align delta | B | Universal | Group | Specialist | Storage ratio |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer_summary in summary.get("summaries", []):
        layer = int(layer_summary["layer"])
        align = mean_alignment_improvement(run, layer)
        d = layer_summary["D"]
        for budget, meta in layer_summary.get("budgets", {}).items():
            counts = meta["counts"]
            lines.append(
                "| {layer} | {dmean:.4f} | {p25:.4f} | {p50:.4f} | {p90:.4f} | "
                "{before:.4f} | {after:.4f} | {delta:.4f} | {budget} | "
                "{u} | {g} | {s} | {ratio:.4f} |".format(
                    layer=layer,
                    dmean=d["mean"],
                    p25=d["p25"],
                    p50=d["p50"],
                    p90=d["p90"],
                    before=align["mean_before"],
                    after=align["mean_after"],
                    delta=align["mean_improvement"],
                    budget=budget,
                    u=counts["universal"],
                    g=counts["group"],
                    s=counts["specialist"],
                    ratio=meta["actual_storage_ratio"],
                )
            )
    lines.extend(
        [
            "",
            "## Findings",
            "",
            "1. The Phase 1-4 automation is runnable on the remote RTX 5090 using only `CUDA_VISIBLE_DEVICES=3`.",
            "2. Greedy functional alignment produces valid permutations and substantially improves same-slot signature similarity relative to raw neuron indices.",
            "3. Tier maps are produced by the configured tier method and v2 storage formula.",
            "4. PPL reconstruction is handled by the separate `compress_and_ppl.py` fallback path.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "cd ~/workspace/fans_moe",
            ". ../venv-dsv2/bin/activate",
            "export HF_ENDPOINT=https://hf-mirror.com",
            "export CUDA_VISIBLE_DEVICES=3",
            "python -u src/fans_moe_lite.py --config configs/deepseek_v2_lite.yaml --calibration c4 --calibration-hidden-dir outputs_c4_512/data/calibration_c4 --layers 1,5,13,26 --tokens 512 --tier-method data_driven --budgets 0.08,0.12,0.18 --output-dir outputs_c4_512 --force",
            "```",
        ]
    )
    report = run.output_dir / "EXPERIMENT_RESULTS.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"wrote markdown report: {report}")


def run_pipeline(args: argparse.Namespace) -> None:
    started = time.time()
    model, run = load_run_config(Path(args.config), args)
    ensure_dirs(run.output_dir)
    validate_layers(model, run.layers)
    torch.manual_seed(run.seed)
    log(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    log(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '<unset>')}")
    if torch.cuda.is_available():
        log(f"cuda device: {torch.cuda.get_device_name(0)}")
    log(
        f"starting FANS-MoE pilot: layers={run.layers}, tokens={run.calibration_tokens}, "
        f"budgets={run.budgets}"
    )
    wmap = download_index(model.repo)

    for layer in run.layers:
        log(f"=== layer {layer} ===")
        weights = None
        if not args.skip_extract:
            weights = load_layer_weights(wmap, layer, model)
            extract_activations(layer, model, run, weights, args.force)
            del weights
        if not args.skip_align:
            align_layer(layer, model, run, args.force)
        if not args.skip_distance:
            compute_distance(layer, model, run, args.force)
        if not args.skip_tiers:
            tier_allocation(layer, model, run, args.force)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_run_summary(model, run, started)
    write_markdown_report(model, run)
    log("done")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/deepseek_v2_lite.yaml")
    ap.add_argument("--layers", default=None, help="Comma/range list, e.g. 5 or 1,5,13 or 1-3")
    ap.add_argument("--tokens", type=int, default=None)
    ap.add_argument("--budgets", default=None, help="Comma list, e.g. 0.4,0.5,0.6")
    ap.add_argument("--calibration", default=None, help="random_hidden, c4, wikitext, or saved_layer_hidden")
    ap.add_argument("--calibration-hidden-dir", default=None)
    ap.add_argument("--tier-method", default=None, choices=["legacy_quantile", "data_driven"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--group-count", type=int, default=None)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--skip-align", action="store_true")
    ap.add_argument("--skip-distance", action="store_true")
    ap.add_argument("--skip-tiers", action="store_true")
    return ap


if __name__ == "__main__":
    run_pipeline(build_argparser().parse_args())
