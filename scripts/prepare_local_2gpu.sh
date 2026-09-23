#!/usr/bin/env bash
# Encode locally downloaded Parquet shards, one partition per shard, one encoder process per GPU.
# Usage: bash scripts/prepare_local_2gpu.sh RAW_DIR TOTAL_SHARDS OUTPUT_DIR [extra prepare_hf_shards flags]
set -euo pipefail
raw=${1:?raw parquet directory required}
total=${2:?number of shards required}
output=${3:?output directory required}
shift 3
gpus=${PREPARE_GPUS:-2}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$output"
pids=()
for ((gpu = 0; gpu < gpus; gpu++)); do
  CUDA_VISIBLE_DEVICES=$gpu "$(dirname "${BASH_SOURCE[0]}")/../.venv/bin/python" "$(dirname "${BASH_SOURCE[0]}")/prepare_hf_shards.py" \
    --repo local --local-dir "$raw" --total "$total" \
    --shards "0-$((total - 1))" --every "$gpus" --offset "$gpu" --output "$output" --workers "${PREPARE_WORKERS:-8}" "$@" \
    > "$output/prepare-gpu$gpu.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "an encoder failed; see $output/prepare-gpu*.log" >&2; exit 1; }
done
echo "all partitions encoded: $output/parts"
