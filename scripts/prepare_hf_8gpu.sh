#!/usr/bin/env bash
# Encode a sharded Hugging Face dataset with one encoder process per GPU, then merge.
# Usage: HF_TOKEN=... bash scripts/prepare_hf_8gpu.sh ORG/DATASET TOTAL_SHARDS OUTPUT_DIR [extra prepare_hf_shards flags]
# Example (4M-row podcast corpus, same columns as Vyvo/en-dataset-3):
#   bash scripts/prepare_hf_8gpu.sh Vyvo/en-dataset-4m 1500 data/corpus \
#     --speaker-column speaker --quality-column quality_score --min-quality 55 --reject-digits \
#     --loudness -16 --max-seconds 20
set -euo pipefail
repo=${1:?dataset repo required}
total=${2:?number of shards required}
output=${3:?output directory required}
shift 3
gpus=${PREPARE_GPUS:-8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$output"
pids=()
for ((gpu = 0; gpu < gpus; gpu++)); do
  CUDA_VISIBLE_DEVICES=$gpu python scripts/prepare_hf_shards.py --repo "$repo" --total "$total" \
    --shards "0-$((total - 1))" --every "$gpus" --offset "$gpu" --output "$output" --workers "${PREPARE_WORKERS:-6}" "$@" \
    > "$output/prepare-gpu$gpu.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || { echo "an encoder failed; see $output/prepare-gpu*.log" >&2; exit 1; }
done
dacvae-tts merge --inputs "$output"/parts/part-* --output "$output/merged" --keep-singletons --drop-conflicting-duplicates
