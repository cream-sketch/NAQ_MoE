# FANS-MoE Automated Experiment Results

Generated: 2026-05-21 22:58:44

## Scope

- Model: `deepseek-ai/DeepSeek-V2-Lite`
- Layers: `1, 5, 13, 26`
- Routed experts per layer: `64`
- Neurons per expert: `1408`
- Calibration source: `saved_layer_hidden`
- Calibration tokens: `512`
- Tier method: `data_driven`
- GPU binding: `CUDA_VISIBLE_DEVICES=3`

No dataset was transferred through `ial-jump`. Text/model assets are fetched directly on the remote server.

## Phase Status

| Phase | Status | Output |
|---|---:|---|
| 1. Activation signatures | done | `activations/layerXX_expertYY.pt` |
| 2. Neuron alignment | done | `alignments/layerXX_perms.npy` |
| 3. Functional distances | done | `distances/layerXX_D.npy` |
| 4. Tier allocation | done | `tier_maps/layerXX_*` |
| 5. Weight compression | done | `ppl/original_layers`, `ppl/compression_details` |
| 6. PPL reconstruction | done | `ppl/ppl_results.json` |

## Layer Results

| Layer | D mean | D p25 | D p50 | D p90 | Align before | Align after | Align delta | B | Universal | Group | Specialist | Storage ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7782 | 0.6875 | 0.8096 | 0.9324 | -0.0000 | 0.3386 | 0.3386 | 0.08 | 231 | 847 | 330 | 0.0800 |
| 1 | 0.7782 | 0.6875 | 0.8096 | 0.9324 | -0.0000 | 0.3386 | 0.3386 | 0.12 | 231 | 590 | 587 | 0.1199 |
| 1 | 0.7782 | 0.6875 | 0.8096 | 0.9324 | -0.0000 | 0.3386 | 0.3386 | 0.18 | 231 | 484 | 693 | 0.1364 |
| 5 | 0.8546 | 0.8032 | 0.8751 | 0.9559 | -0.0003 | 0.2628 | 0.2631 | 0.08 | 215 | 864 | 329 | 0.0800 |
| 5 | 0.8546 | 0.8032 | 0.8751 | 0.9559 | -0.0003 | 0.2628 | 0.2631 | 0.12 | 215 | 607 | 586 | 0.1199 |
| 5 | 0.8546 | 0.8032 | 0.8751 | 0.9559 | -0.0003 | 0.2628 | 0.2631 | 0.18 | 215 | 502 | 691 | 0.1362 |
| 13 | 0.8029 | 0.7214 | 0.8338 | 0.9497 | -0.0002 | 0.3018 | 0.3020 | 0.08 | 242 | 835 | 331 | 0.0800 |
| 13 | 0.8029 | 0.7214 | 0.8338 | 0.9497 | -0.0002 | 0.3018 | 0.3020 | 0.12 | 242 | 578 | 588 | 0.1199 |
| 13 | 0.8029 | 0.7214 | 0.8338 | 0.9497 | -0.0002 | 0.3018 | 0.3020 | 0.18 | 242 | 482 | 684 | 0.1348 |
| 26 | 0.8411 | 0.7812 | 0.8701 | 0.9574 | -0.0003 | 0.2693 | 0.2696 | 0.08 | 184 | 898 | 326 | 0.0799 |
| 26 | 0.8411 | 0.7812 | 0.8701 | 0.9574 | -0.0003 | 0.2693 | 0.2696 | 0.12 | 184 | 640 | 584 | 0.1199 |
| 26 | 0.8411 | 0.7812 | 0.8701 | 0.9574 | -0.0003 | 0.2693 | 0.2696 | 0.18 | 184 | 475 | 749 | 0.1456 |

## Findings

1. The Phase 1-4 automation is runnable on the remote RTX 5090 using only `CUDA_VISIBLE_DEVICES=3`.
2. Greedy functional alignment produces valid permutations and substantially improves same-slot signature similarity relative to raw neuron indices.
3. Tier maps are produced by the configured tier method and v2 storage formula.
4. Safe PPL reconstruction completed with BF16 fallback, local-doc evaluation text, `eval_tokens=512`, `seq_len=256`, and uniform INT4 skipped to reduce station memory pressure.

## PPL Results

| Variant | PPL | Mean NLL | Tokens |
|---|---:|---:|---:|
| Dense baseline | 132327.5421 | 11.7930 | 510 |
| FANS-MoE B=0.08 | 130589.3326 | 11.7798 | 510 |
| FANS-MoE B=0.12 | 131001.4607 | 11.7830 | 510 |
| FANS-MoE B=0.18 | 130229.1633 | 11.7771 | 510 |

## Reproduction

```bash
cd ~/workspace/fans_moe
. ../venv-dsv2/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=3
python -u src/fans_moe_lite.py --config configs/deepseek_v2_lite.yaml --calibration saved_layer_hidden --calibration-hidden-dir outputs_realtext_512/data/calibration_realtext --layers 1,5,13,26 --tokens 512 --tier-method data_driven --budgets 0.08,0.12,0.18 --output-dir outputs_realtext_512 --force
python -u src/compress_and_ppl.py --config configs/deepseek_v2_lite.yaml --weights-dir outputs_realtext_512 --layers 1,5,13,26 --budgets 0.08,0.12,0.18 --ppl-dataset local --eval-tokens 512 --seq-len 256 --max-gpu-memory 18GiB --max-cpu-memory 96GiB --offload-dir outputs_realtext_512/ppl_safe/offload --original-cache-dir outputs_realtext_512/ppl/original_layers --local-files-only --skip-uniform-int4
```
