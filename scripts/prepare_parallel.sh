#!/usr/bin/env bash
set -euo pipefail
# Generic partition launcher; prefer prepare_8gpu.sh or prepare_cpu.sh.
manifest_path=${1:?manifest required}
cache_root=${2:?cache directory required}
shift 2
device=${PREPARE_DEVICE:-cuda}
count=${PREPARE_PROCESSES:-8}
if [[ ! "$count" =~ ^[1-9][0-9]*$ ]] || [[ "$device" != cuda && "$device" != cpu ]]; then
  echo "PREPARE_PROCESSES must be positive; PREPARE_DEVICE must be cuda or cpu" >&2
  exit 1
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
mkdir -p "$cache_root"
if [[ "$device" == cpu ]]; then
  devices=()
  for ((rank=0; rank<count; rank++)); do devices+=(""); done
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a devices <<< "$CUDA_VISIBLE_DEVICES"
  if ((${#devices[@]} != count)); then
    echo "CUDA_VISIBLE_DEVICES must expose exactly $count devices for this launcher." >&2
    exit 1
  fi
else
  devices=()
  for ((rank=0; rank<count; rank++)); do devices+=("$rank"); done
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
partitions=()
for ((rank=0; rank<count; rank++)); do
  partitions+=("$cache_root/part-$rank")
  CUDA_VISIBLE_DEVICES="${devices[$rank]}" dacvae-tts prepare --manifest "$manifest_path" \
    --output "$cache_root/part-$rank" --shard-index "$rank" --num-shards "$count" \
    --device "$device" --codec-backend fast "$@" \
    > "$cache_root/prepare-$rank.log" 2>&1 &
  processes+=("$!")
done
for process in "${processes[@]}"; do
  if ! wait "$process"; then
    echo "An encoder failed; stopping remaining workers. Inspect preparation logs." >&2
    exit 1
  fi
done
dacvae-tts merge --inputs "${partitions[@]}" --output "$cache_root/merged"
