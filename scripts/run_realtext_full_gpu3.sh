#!/usr/bin/env bash
set -euo pipefail

BASE="${NAQ_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_ACTIVATE="${NAQ_MOE_VENV:-$BASE/../venv-dsv2/bin/activate}"
OUT="$BASE/outputs_realtext_512"
LOG_DIR="$OUT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_realtext_full_gpu3_${STAMP}.log"

mkdir -p "$LOG_DIR" "$OUT/offload" "$OUT/ppl/offload"

exec > >(tee -a "$MASTER_LOG") 2>&1

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"
}

cd "$BASE"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  . "$VENV_ACTIVATE"
fi
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

log "start realtext full pipeline"
log "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
log "HF_ENDPOINT=$HF_ENDPOINT"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true

log "phase 1: build real-text calibration hidden states"
python -u src/build_real_calibration.py \
  --layers 1,5,13,26 \
  --tokens 512 \
  --seq-len 256 \
  --dataset local \
  --output-dir outputs_realtext_512/data/calibration_realtext \
  --offload-dir outputs_realtext_512/offload \
  --max-gpu-memory 20GiB \
  --max-cpu-memory 96GiB \
  --local-files-only

log "phase 2: run FANS-MoE phases 1-4 on real-text calibration"
python -u src/fans_moe_lite.py \
  --config configs/deepseek_v2_lite.yaml \
  --calibration saved_layer_hidden \
  --calibration-hidden-dir outputs_realtext_512/data/calibration_realtext \
  --layers 1,5,13,26 \
  --tokens 512 \
  --tier-method data_driven \
  --budgets 0.08,0.12,0.18 \
  --output-dir outputs_realtext_512 \
  --force

log "phase 3: run BF16 reconstruction and PPL evaluation"
python -u src/compress_and_ppl.py \
  --config configs/deepseek_v2_lite.yaml \
  --weights-dir outputs_realtext_512 \
  --layers 1,5,13,26 \
  --budgets 0.08,0.12,0.18 \
  --ppl-dataset local \
  --eval-tokens 2048 \
  --seq-len 512 \
  --max-gpu-memory 20GiB \
  --max-cpu-memory 96GiB \
  --offload-dir outputs_realtext_512/ppl/offload \
  --local-files-only

log "done realtext full pipeline"
