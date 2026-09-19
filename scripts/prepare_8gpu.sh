#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/prepare_8gpu.sh /dataset/manifest.jsonl /cache [extra prepare flags]
manifest_path=${1:?manifest required}
cache_root=${2:?cache directory required}
shift 2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
mkdir -p "$cache_root"
processes=()
for rank in {0..7}; do
  CUDA_VISIBLE_DEVICES="$rank" dacvae-tts prepare --manifest "$manifest_path" \
    --output "$cache_root/part-$rank" --shard-index "$rank" --num-shards 8 "$@" \
    > "$cache_root/prepare-$rank.log" 2>&1 &
  processes+=("$!")
done
failed=0
for process in "${processes[@]}"; do
  wait "$process" || failed=1
done
if ((failed)); then
  echo "At least one encoder failed; inspect preparation logs." >&2
  exit 1
fi
dacvae-tts merge --inputs "$cache_root"/part-{0..7} --output "$cache_root/merged"
