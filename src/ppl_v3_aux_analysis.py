#!/usr/bin/env python3
"""Auxiliary diagnostics for the FANS-MoE PPL v3 run."""

from __future__ import annotations

import argparse
import binascii
import json
import struct
import zlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open


def budget_tag(budget: str) -> str:
    return str(budget).replace(".", "p")


def load_indexed_tensor(snapshot_dir: Path, key: str) -> torch.Tensor:
    index = json.loads((snapshot_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shard = snapshot_dir / index["weight_map"][key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        return handle.get_tensor(key)


def write_png_rgb(path: Path, width: int, height: int, pixels: bytearray) -> None:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", binascii.crc32(kind + data) & 0xFFFFFFFF)

    rows = []
    stride = width * 3
    for y in range(height):
        rows.append(b"\x00" + bytes(pixels[y * stride : (y + 1) * stride]))
    raw = b"".join(rows)
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(raw, level=6))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def set_pixel(pixels: bytearray, width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if 0 <= x < width and 0 <= y < height:
        idx = (y * width + x) * 3
        pixels[idx : idx + 3] = bytes(color)


def fill_rect(
    pixels: bytearray,
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    x0, x1 = max(0, min(x0, x1)), min(width, max(x0, x1))
    y0, y1 = max(0, min(y0, y1)), min(height, max(y0, y1))
    for y in range(y0, y1):
        row = (y * width + x0) * 3
        pixels[row : row + (x1 - x0) * 3] = bytes(color) * (x1 - x0)


def draw_vline(
    pixels: bytearray,
    width: int,
    height: int,
    x: int,
    y0: int,
    y1: int,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    for dx in range(-(thickness // 2), thickness // 2 + 1):
        fill_rect(pixels, width, height, x + dx, y0, x + dx + 1, y1, color)


def write_hist_png(path: Path, values: np.ndarray, markers: list[tuple[float, tuple[int, int, int]]]) -> None:
    width, height = 1080, 648
    pixels = bytearray([255] * width * height * 3)
    plot_x0, plot_y0 = 80, 40
    plot_w, plot_h = 920, 540
    plot_x1, plot_y1 = plot_x0 + plot_w, plot_y0 + plot_h
    fill_rect(pixels, width, height, plot_x0, plot_y0, plot_x1, plot_y1, (248, 249, 250))

    counts, edges = np.histogram(values, bins=30)
    max_count = max(int(counts.max()), 1)
    for i, count in enumerate(counts.tolist()):
        x0 = plot_x0 + int(i * plot_w / len(counts)) + 1
        x1 = plot_x0 + int((i + 1) * plot_w / len(counts)) - 1
        bar_h = int((count / max_count) * (plot_h - 16))
        fill_rect(pixels, width, height, x0, plot_y1 - bar_h, x1, plot_y1, (79, 124, 172))

    xmin, xmax = float(edges[0]), float(edges[-1])
    for value, color in markers:
        if xmin <= value <= xmax:
            x = plot_x0 + int((value - xmin) / max(xmax - xmin, 1e-12) * plot_w)
            draw_vline(pixels, width, height, x, plot_y0, plot_y1, color, thickness=3)

    fill_rect(pixels, width, height, plot_x0, plot_y1, plot_x1 + 1, plot_y1 + 2, (20, 20, 20))
    fill_rect(pixels, width, height, plot_x0, plot_y0, plot_x0 + 2, plot_y1, (20, 20, 20))
    write_png_rgb(path, width, height, pixels)


def plot_d_histograms(weights_dir: Path, layers: list[int], budget: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = budget_tag(budget)
    summary: dict[str, Any] = {}
    for layer in layers:
        d = np.load(weights_dir / "distances" / f"layer{layer:02d}_D.npy")
        tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
        meta_path = weights_dir / "tier_maps" / f"layer{layer:02d}_meta_B{tag}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

        cut_info: dict[str, float | None] = {}
        if np.any(tiers == 0):
            cut_info["universal_max_D"] = float(d[tiers == 0].max())
        else:
            cut_info["universal_max_D"] = None
        if np.any(tiers == 1):
            cut_info["group_max_D"] = float(d[tiers == 1].max())
            cut_info["group_min_D"] = float(d[tiers == 1].min())
        else:
            cut_info["group_max_D"] = None
            cut_info["group_min_D"] = None
        if np.any(tiers == 2):
            cut_info["specialist_min_D"] = float(d[tiers == 2].min())
        else:
            cut_info["specialist_min_D"] = None

        counts = {name: int((tiers == idx).sum()) for idx, name in enumerate(["U", "G", "S"])}
        out_path = out_dir / f"D_hist_layer{layer:02d}_B{tag}.png"
        markers = []
        if cut_info["universal_max_D"] is not None:
            markers.append((float(cut_info["universal_max_D"]), (27, 158, 119)))
        if cut_info["group_max_D"] is not None:
            markers.append((float(cut_info["group_max_D"]), (217, 95, 2)))
        for center in meta.get("centers", []):
            markers.append((float(center), (70, 70, 70)))
        write_hist_png(out_path, d, markers)

        summary[str(layer)] = {
            "D_min": float(d.min()),
            "D_mean": float(d.mean()),
            "D_p25": float(np.quantile(d, 0.25)),
            "D_p50": float(np.quantile(d, 0.50)),
            "D_p90": float(np.quantile(d, 0.90)),
            "counts": counts,
            "cuts": cut_info,
            "figure": str(out_path),
        }
    return summary


def normalized_rows(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), p=2, dim=1, eps=1e-12)


def mean_max_cosine(x: torch.Tensor, y_norm: torch.Tensor, batch: int = 2048) -> tuple[float, int]:
    if x.numel() == 0:
        return float("nan"), 0
    total = 0.0
    count = 0
    for start in range(0, x.shape[0], batch):
        xb = normalized_rows(x[start : start + batch])
        vals = (xb @ y_norm.T).max(dim=1).values
        total += float(vals.sum().item())
        count += int(vals.numel())
    return total / max(count, 1), count


def shared_signatures(snapshot_dir: Path, weights_dir: Path, layer: int) -> torch.Tensor:
    hidden_obj = torch.load(
        weights_dir / "data" / "calibration_realtext" / f"layer{layer:02d}_hidden.pt",
        map_location="cpu",
    )
    hidden = (hidden_obj["hidden"] if isinstance(hidden_obj, dict) else hidden_obj).float()
    gate = load_indexed_tensor(snapshot_dir, f"model.layers.{layer}.mlp.shared_experts.gate_proj.weight").float()
    up = load_indexed_tensor(snapshot_dir, f"model.layers.{layer}.mlp.shared_experts.up_proj.weight").float()
    sig = F.silu(hidden @ gate.T) * (hidden @ up.T)
    tokens, width = sig.shape
    if width % 2 != 0:
        raise RuntimeError(f"expected two shared experts in layer {layer}, got width={width}")
    return sig.reshape(tokens, 2, width // 2).permute(1, 2, 0).reshape(width, tokens).contiguous()


def routed_slot_centroids(weights_dir: Path, layer: int, slots: np.ndarray) -> torch.Tensor:
    if slots.size == 0:
        return torch.empty(0, 0)
    perms = np.load(weights_dir / "alignments" / f"layer{layer:02d}_perms.npy")
    acts = [
        torch.load(weights_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt", map_location="cpu").float()
        for expert in range(perms.shape[0])
    ]
    rows = []
    for slot in slots.tolist():
        vectors = [acts[expert][:, int(perms[expert, slot])] for expert in range(perms.shape[0])]
        rows.append(torch.stack(vectors, dim=0).mean(dim=0))
    return torch.stack(rows, dim=0)


def routed_expert_vectors(
    weights_dir: Path,
    layer: int,
    slots: np.ndarray,
    y_norm: torch.Tensor,
    batch_slots: int = 64,
) -> tuple[float, int]:
    if slots.size == 0:
        return float("nan"), 0
    perms = np.load(weights_dir / "alignments" / f"layer{layer:02d}_perms.npy")
    acts = [
        torch.load(weights_dir / "activations" / f"layer{layer:02d}_expert{expert:02d}.pt", map_location="cpu").float()
        for expert in range(perms.shape[0])
    ]
    total = 0.0
    count = 0
    for start in range(0, len(slots), batch_slots):
        rows = []
        for slot in slots[start : start + batch_slots].tolist():
            for expert in range(perms.shape[0]):
                rows.append(acts[expert][:, int(perms[expert, slot])])
        x = torch.stack(rows, dim=0)
        mean, n = mean_max_cosine(x, y_norm)
        total += mean * n
        count += n
    return total / max(count, 1), count


def shared_alignment(weights_dir: Path, layers: list[int], budget: str, model_repo: str) -> dict[str, Any]:
    snapshot_dir = Path(snapshot_download(repo_id=model_repo, local_files_only=True))
    tag = budget_tag(budget)
    per_layer: dict[str, Any] = {}
    weighted_uni_centroid = []
    weighted_uni_expert = []
    weighted_spec_expert = []

    for layer in layers:
        tiers = np.load(weights_dir / "tier_maps" / f"layer{layer:02d}_tiers_B{tag}.npy")
        universal_slots = np.where(tiers == 0)[0]
        specialist_slots = np.where(tiers == 2)[0]
        shared = shared_signatures(snapshot_dir, weights_dir, layer)
        shared_norm = normalized_rows(shared)

        uni_centroids = routed_slot_centroids(weights_dir, layer, universal_slots)
        uni_centroid_mean, uni_centroid_n = mean_max_cosine(uni_centroids, shared_norm)
        uni_expert_mean, uni_expert_n = routed_expert_vectors(weights_dir, layer, universal_slots, shared_norm)
        spec_expert_mean, spec_expert_n = routed_expert_vectors(weights_dir, layer, specialist_slots, shared_norm)

        per_layer[str(layer)] = {
            "universal_slots": int(universal_slots.size),
            "specialist_slots": int(specialist_slots.size),
            "shared_neurons": int(shared.shape[0]),
            "universal_centroid_vs_shared_mean_max_cos": uni_centroid_mean,
            "universal_expert_neuron_vs_shared_mean_max_cos": uni_expert_mean,
            "specialist_expert_neuron_vs_shared_mean_max_cos": spec_expert_mean,
            "universal_centroid_vectors": uni_centroid_n,
            "universal_expert_vectors": uni_expert_n,
            "specialist_expert_vectors": spec_expert_n,
        }
        weighted_uni_centroid.append((uni_centroid_mean, uni_centroid_n))
        weighted_uni_expert.append((uni_expert_mean, uni_expert_n))
        weighted_spec_expert.append((spec_expert_mean, spec_expert_n))

    def weighted_mean(items: list[tuple[float, int]]) -> float:
        num = sum(v * n for v, n in items if n > 0)
        den = sum(n for _, n in items if n > 0)
        return num / max(den, 1)

    return {
        "model_repo": model_repo,
        "budget": budget,
        "layers": layers,
        "method": "mean max cosine to any shared-expert SwiGLU intermediate neuron signature",
        "overall": {
            "universal_centroid_vs_shared": weighted_mean(weighted_uni_centroid),
            "universal_expert_neuron_vs_shared": weighted_mean(weighted_uni_expert),
            "specialist_expert_neuron_vs_shared": weighted_mean(weighted_spec_expert),
        },
        "per_layer": per_layer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-dir", default="outputs_realtext_512")
    ap.add_argument("--layers", default="1,5,13,26")
    ap.add_argument("--budget", default="0.12")
    ap.add_argument("--model-repo", default="deepseek-ai/DeepSeek-V2-Lite")
    args = ap.parse_args()

    weights_dir = Path(args.weights_dir)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    figs_dir = weights_dir / "figs"
    d_summary = plot_d_histograms(weights_dir, layers, args.budget, figs_dir)
    shared = shared_alignment(weights_dir, layers, args.budget, args.model_repo)

    (weights_dir / "D_hist_summary.json").write_text(json.dumps(d_summary, indent=2), encoding="utf-8")
    (weights_dir / "shared_alignment.json").write_text(json.dumps(shared, indent=2), encoding="utf-8")
    print(json.dumps({"D_hist_summary": d_summary, "shared_alignment": shared["overall"]}, indent=2))


if __name__ == "__main__":
    main()
