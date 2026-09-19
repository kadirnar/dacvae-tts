#!/usr/bin/env bash
set -euo pipefail
# CPU DDP/Gloo; each rank trains on a separate shard. Set threads to fit the host.
config_path=${1:?config YAML required}
cache_path=${2:?merged cache directory required}
output_path=${3:?output directory required}
shift 3
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
exec torchrun --standalone --nnodes=1 --nproc_per_node="${TRAIN_PROCESSES:-2}" -m dacvae_tts train \
  --config "$config_path" --cache "$cache_path" --output "$output_path" --device cpu --precision fp32 \
  --workers "${TRAIN_WORKERS:-2}" --worker-threads 1 --prefetch-factor "${TRAIN_PREFETCH:-2}" \
  --loader-start-method spawn --no-cuda-prefetch "$@"
