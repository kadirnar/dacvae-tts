#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/train_2gpu.sh configs/nano_tr.yaml data/tr/merged runs/tr-nano [extra train flags]
config_path=${1:?config YAML required}
cache_path=${2:?merged cache directory required}
output_path=${3:?output directory required}
shift 3
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec "$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/torchrun" --standalone --nnodes=1 --nproc_per_node="${TRAIN_GPUS:-2}" \
  -m dacvae_tts train \
  --config "$config_path" --cache "$cache_path" --output "$output_path" --workers "${TRAIN_WORKERS:-4}" --worker-threads 1 \
  --prefetch-factor "${TRAIN_PREFETCH:-4}" --loader-start-method spawn --cuda-prefetch "$@"
