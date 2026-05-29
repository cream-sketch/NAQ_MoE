#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
cd ..
mkdir -p outputs/logs

python -u src/fans_moe_lite.py \
  --config configs/deepseek_v2_lite.yaml \
  "$@" 2>&1 | tee "outputs/logs/run_$(date +%Y%m%d_%H%M%S).log"
