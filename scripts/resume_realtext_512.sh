#!/usr/bin/env bash
set -uo pipefail

BASE="${NAQ_MOE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_ACTIVATE="${NAQ_MOE_VENV:-$BASE/../venv-dsv2/bin/activate}"
CACHE="/home/ial-lvyx/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite"
REV="604d5664dddd88a0433dbae533b7fe9472482de0"
BLOB_SHA="843ec689624f3a520526e040f0326c4dc9865e8172942ca98a084fe136fdb21a"
BLOB="$CACHE/blobs/$BLOB_SHA"
SNAP="$CACHE/snapshots/$REV/model-00003-of-000004.safetensors"
EXPECTED_SIZE="8590718520"
URL="https://hf-mirror.com/deepseek-ai/DeepSeek-V2-Lite/resolve/main/model-00003-of-000004.safetensors"

WATCH_LOG="$BASE/outputs_realtext_512/logs/resume_realtext_512_watcher.log"
WGET_LOG="$BASE/outputs_realtext_512/logs/wget_shard_00003_resume.log"
RUN_LOG="$BASE/outputs_realtext_512/logs/build_real_calibration_resume.log"

mkdir -p "$(dirname "$WATCH_LOG")" "$CACHE/snapshots/$REV"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" >> "$WATCH_LOG"
}

blob_size() {
  stat -c%s "$BLOB" 2>/dev/null || printf '0'
}

log "watcher start pid=$$"

while true; do
  size="$(blob_size)"
  if [[ "$size" == "$EXPECTED_SIZE" ]]; then
    log "shard3 complete size=$size"
    break
  fi
  if [[ "$size" =~ ^[0-9]+$ ]] && (( size > EXPECTED_SIZE )); then
    log "shard3 size exceeds expected: size=$size expected=$EXPECTED_SIZE"
    exit 2
  fi
  if pgrep -af '^wget .*model-00003-of-000004\.safetensors' >/dev/null; then
    log "waiting active shard3 download size=$size/$EXPECTED_SIZE"
    sleep 60
  else
    log "resuming shard3 download size=$size/$EXPECTED_SIZE"
    wget -c --tries=20 --timeout=60 --read-timeout=60 --waitretry=15 -O "$BLOB" "$URL" >> "$WGET_LOG" 2>&1
    rc=$?
    log "wget exited rc=$rc size=$(blob_size)/$EXPECTED_SIZE"
    sleep 15
  fi
done

rm -f "$BLOB.incomplete"
ln -sfn "../../blobs/$BLOB_SHA" "$SNAP"
log "snapshot link ready: $SNAP"

while true; do
  mem="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3 | tr -d ' ')"
  util="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 3 | tr -d ' ')"
  if [[ "$mem" =~ ^[0-9]+$ ]] && (( mem < 500 )); then
    log "gpu3 free mem=${mem}MiB util=${util}%"
    break
  fi
  log "waiting gpu3 mem=${mem}MiB util=${util}%"
  sleep 120
done

cd "$BASE" || exit 3
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck source=/dev/null
  . "$VENV_ACTIVATE"
fi
export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

log "launch build_real_calibration local-files-only"
python -u src/build_real_calibration.py \
  --layers 1,5,13,26 \
  --tokens 512 \
  --seq-len 256 \
  --dataset local \
  --output-dir outputs_realtext_512/data/calibration_realtext \
  --offload-dir outputs_realtext_512/offload \
  --max-gpu-memory 26GiB \
  --max-cpu-memory 96GiB \
  --local-files-only > "$RUN_LOG" 2>&1
rc=$?
log "build_real_calibration exit=$rc"
exit "$rc"
