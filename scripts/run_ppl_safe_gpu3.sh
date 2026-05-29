#!/usr/bin/env bash
set -euo pipefail

BASE="${NAQ_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_ACTIVATE="${NAQ_MOE_VENV:-$BASE/../venv-dsv2/bin/activate}"
OUT="$BASE/outputs_realtext_512"
LOG_DIR="$OUT/logs"
STAMP="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOG_DIR/run_ppl_safe_gpu3_${STAMP}.log"

mkdir -p "$LOG_DIR" "$OUT/ppl_safe/offload"
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

log "start safe PPL continuation"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
free -h || true

python -u src/compress_and_ppl.py \
  --config configs/deepseek_v2_lite.yaml \
  --weights-dir outputs_realtext_512 \
  --layers 1,5,13,26 \
  --budgets 0.08,0.12,0.18 \
  --ppl-dataset local \
  --eval-tokens 512 \
  --seq-len 256 \
  --max-gpu-memory 18GiB \
  --max-cpu-memory 96GiB \
  --offload-dir outputs_realtext_512/ppl_safe/offload \
  --original-cache-dir outputs_realtext_512/ppl/original_layers \
  --local-files-only \
  --skip-uniform-int4

log "done safe PPL continuation"
