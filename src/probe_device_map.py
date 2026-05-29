#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

import torch
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="deepseek-ai/DeepSeek-V2-Lite")
    ap.add_argument("--max-gpu-memory", default="18GiB")
    ap.add_argument("--max-cpu-memory", default="96GiB")
    ap.add_argument("--offload-dir", default="outputs_realtext_512/ppl_safe/device_map_probe_offload")
    args = ap.parse_args()

    os.makedirs(args.offload_dir, exist_ok=True)
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
        offload_buffers=True,
        max_memory=max_memory,
        local_files_only=True,
    )
    print("loaded")
    for key, value in sorted(model.hf_device_map.items()):
        if key in {"", "model", "lm_head", "model.embed_tokens", "model.norm"} or key.startswith("model.layers."):
            print(f"{key or '<root>'}: {value}")


if __name__ == "__main__":
    main()
