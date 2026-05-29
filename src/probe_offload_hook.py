#!/usr/bin/env python3
from __future__ import annotations

import torch
import transformers.utils.import_utils as transformers_import_utils
from transformers import AutoModelForCausalLM

if not hasattr(transformers_import_utils, "is_torch_fx_available"):
    transformers_import_utils.is_torch_fx_available = lambda: False

model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-V2-Lite",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    offload_folder="outputs_realtext_512/ppl_safe/hook_probe_offload",
    offload_buffers=True,
    max_memory={0: "18GiB", "cpu": "96GiB"},
    local_files_only=True,
)
expert = model.model.layers[26].mlp.experts[0]
for module_name, module in [
    ("layer", model.model.layers[26]),
    ("mlp", model.model.layers[26].mlp),
    ("expert", expert),
    ("gate", expert.gate_proj),
]:
    print("MODULE", module_name, type(module))
    hook = getattr(module, "_hf_hook", None)
    print(" hook", type(hook), hook)
    if hook is not None:
        for attr in ["weights_map", "offload", "execution_device", "io_same_device", "place_submodules"]:
            value = getattr(hook, attr, None)
            if attr == "weights_map":
                print(" ", attr, type(value), None if value is None else list(value.keys())[:5])
            else:
                print(" ", attr, type(value), value)

param = expert.gate_proj.weight
print("param device", param.device, "shape", tuple(param.shape), "is_meta", param.is_meta)
gate_hook = getattr(expert.gate_proj, "_hf_hook", None)
if gate_hook is not None and getattr(gate_hook, "weights_map", None) is not None:
    wm = gate_hook.weights_map
    print("weights_map dir", [x for x in dir(wm) if not x.startswith("__")])
    for attr in ["dataset", "prefix"]:
        value = getattr(wm, attr, None)
        print("weights_map", attr, type(value), value)
        if attr == "dataset" and value is not None:
            print("dataset dir", [x for x in dir(value) if not x.startswith("__")][:30])
    print("gate weights keys sample", list(wm.keys())[:20])
    for key in ["weight", "model.layers.26.mlp.experts.0.gate_proj.weight"]:
        if key in wm:
            t = wm[key]
            print("wm", key, type(t), getattr(t, "device", None), getattr(t, "shape", None), getattr(t, "dtype", None))
