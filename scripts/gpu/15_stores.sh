#!/usr/bin/env bash
# Step 1b: the sidecar stores some A/B arms read next to the training cache (the cache itself is never modified).
# Every store is split over all GPUs of $GPUS and merged; finished stores and finished rows are skipped.
#   bash scripts/gpu/15_stores.sh [frames] [speakers] [tempo]      (default: all three)
#   frames    CACHE/teacher/mhubert147-l12-pca256  speech-REPA targets        (repa, repa-tla arms; #10)
#   speakers  CACHE/teacher/ecapa-speechbrain      SpeechBrain ECAPA vectors  (tla, repa-tla, speaker-condition arms)
#   tempo     CACHE/tempo/wsola-v1                 WSOLA tempo variants       (tempo-prompts arm; ~5x the latents)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
need "$CACHE/metadata.json"

stores=("$@")
((${#stores[@]})) || stores=(frames speakers tempo)

complete() {  # complete STORE: the store has a merged, complete index
  "$PY" - "$1/metadata.json" <<'EOF'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
meta = json.loads(path.read_text()) if path.exists() else {}
sys.exit(0 if meta.get("complete") and meta.get("merged") else 1)
EOF
}

merge_teacher() { "$PY" scripts/extract_teacher_features.py merge --output "$1" --cache "$CACHE"; }
merge_tempo() { "$PY" scripts/build_tempo_variants.py merge --output "$1" --cache "$CACHE"; }

# sharded STORE MERGE_FUNCTION BUILD_COMMAND... : BUILD_COMMAND --shard-index i --num-shards N on the i-th GPU of
# $GPUS, then MERGE_FUNCTION STORE. One GPU writes the store root directly (no partitions, nothing to merge).
sharded() {
  local store=$1 merge=$2
  shift 2
  local -a gpus=($GPUS) pids=()
  local i status=0
  if ((NGPU == 1)); then
    CUDA_VISIBLE_DEVICES=${gpus[0]} "$@" --shard-index 0 --num-shards 1 > "$store.log" 2>&1
    return
  fi
  for i in "${!gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=${gpus[$i]} "$@" --shard-index "$i" --num-shards "$NGPU" > "$store.part-$i.log" 2>&1 &
    pids+=("$!")
  done
  for i in "${pids[@]}"; do wait "$i" || status=1; done
  ((status == 0)) || { log "a partition of $store failed; see $store.part-*.log"; return 1; }
  "$merge" "$store"
}

for store in "${stores[@]}"; do
  case $store in
    frames)
      dir=$CACHE/teacher/mhubert147-l12-pca256
      complete "$dir" && { log "exists: $dir"; continue; }
      mkdir -p "$dir"
      if [[ ! -f $dir/pca.pt ]]; then
        log "fit the 768 -> 256 PCA of mHuBERT-147 layer 12 once"
        CUDA_VISIBLE_DEVICES=${GPUS%% *} "$PY" scripts/extract_teacher_features.py fit-pca --cache "$CACHE" \
          --output "$dir" --layer 12 --pca-dim 256 --device cuda
      fi
      log "speech-REPA frames -> $dir"
      sharded "$dir" merge_teacher \
        "$PY" scripts/extract_teacher_features.py frames --cache "$CACHE" --output "$dir" --layer 12 --device cuda --quiet
      ;;
    speakers)
      dir=$CACHE/teacher/ecapa-speechbrain
      complete "$dir" && { log "exists: $dir"; continue; }
      mkdir -p "$dir"
      log "speaker embeddings (train and val: the speaker condition also conditions validation) -> $dir"
      sharded "$dir" merge_teacher \
        "$PY" scripts/extract_teacher_features.py speakers --cache "$CACHE" --output "$dir" --splits train,val \
        --device cuda --quiet
      ;;
    tempo)
      dir=$CACHE/tempo/wsola-v1
      complete "$dir" && { log "exists: $dir"; continue; }
      mkdir -p "$dir"
      log "tempo variants x0.8/0.9/1.0/1.111/1.25 -> $dir"
      sharded "$dir" merge_tempo \
        "$PY" scripts/build_tempo_variants.py build --cache "$CACHE" --output "$dir" --device cuda --quiet
      ;;
    *)
      log "unknown store: $store (frames, speakers or tempo)"
      exit 1
      ;;
  esac
  complete "$dir" || { log "store not complete: $dir"; exit 1; }
done
log "stores done"
