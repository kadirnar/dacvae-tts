#!/usr/bin/env bash
set -euo pipefail
# Usage: bash scripts/prepare_8gpu.sh /dataset/manifest.jsonl /cache [extra prepare flags]
manifest_path=${1:?manifest required}
cache_root=${2:?cache directory required}
shift 2
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
mkdir -p "$cache_root"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  if ((${#devices[@]} != 8)); then
    echo "CUDA_VISIBLE_DEVICES must expose exactly eight devices for this launcher." >&2
    exit 1
  fi
else
  devices=(0 1 2 3 4 5 6 7)
fi
processes=()
cleanup() {
  local running
  running=$(jobs -pr)
  if [[ -n "$running" ]]; then
    # Only this launcher's active jobs; do not signal completed/reused PIDs.
    kill $running 2>/dev/null || true
    wait || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for rank in {0..7}; do
  CUDA_VISIBLE_DEVICES="${devices[$rank]}" dacvae-tts prepare --manifest "$manifest_path" \
    --output "$cache_root/part-$rank" --shard-index "$rank" --num-shards 8 "$@" \
    > "$cache_root/prepare-$rank.log" 2>&1 &
  processes+=("$!")
done
for process in "${processes[@]}"; do
  if ! wait "$process"; then
    echo "An encoder failed; stopping remaining workers. Inspect preparation logs." >&2
    exit 1
  fi
done
dacvae-tts merge --inputs "$cache_root"/part-{0..7} --output "$cache_root/merged"
