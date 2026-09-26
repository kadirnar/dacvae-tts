#!/usr/bin/env bash
# Stream a sharded Hugging Face Parquet corpus through the DACVAE encoder and the corpus scorer in one pass, for
# corpora that do not fit on disk next to their caches (Codyfederer/tr-combined: 23 shards, 85 GB, 277 h).
# Shard i+1 downloads while shard i is encoded (prepare_hf_shards.py -> CORPUS/parts/part-i) and re-transcribed with
# Whisper large-v3 + DNSMOS (transcribe_corpus.py -> SCORES/scores.jsonl) on the same GPU; the raw shard is deleted
# once both finished. Restartable: finished shards (CORPUS/stream-done/i) are skipped, the two consumers skip their
# own finished work.
#   HF_TOKEN=... bash scripts/data/stream_encode_score.sh Codyfederer/tr-combined 23 /workspace/data/trc \
#       /workspace/outputs/trc-scores /workspace/models/sig_bak_ovr.onnx [extra prepare_hf_shards.py flags]
# The row uids of both outputs are "data/train-XXXXX-of-YYYYY.parquet:<row>", so make_drop_list.py's drop lists
# apply to the merged cache directly.
set -euo pipefail
repo=${1:?dataset repo}
total=${2:?number of shards}
corpus=${3:?cache root}
scores=${4:?scores directory}
dnsmos=${5:?DNSMOS onnx path}
shift 5
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
py=${PY:-$here/.venv/bin/python}
raw=$corpus/stream-raw
mkdir -p "$raw" "$corpus/stream-done" "$scores"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

name_of() { printf 'train-%05d-of-%05d.parquet' "$1" "$total"; }

fetch() {  # fetch INDEX: raw/shard-INDEX/data/<name> (the layout that gives "data/<name>:<row>" uids)
  local i=$1 name dest
  name=$(name_of "$i")
  dest=$raw/shard-$i
  [[ -f $dest/data/$name ]] && return 0
  mkdir -p "$dest/data"
  "$py" - "$repo" "$name" "$dest/download" <<'EOF'
import os, sys, time
from huggingface_hub import hf_hub_download
repo, name, dest = sys.argv[1:]
for attempt in range(6):
    try:
        hf_hub_download(repo, name, repo_type="dataset", local_dir=dest, token=os.environ.get("HF_TOKEN"))
        break
    except Exception as error:
        if attempt == 5:
            raise
        print(f"retry {attempt + 1} {name}: {error}", flush=True)
        time.sleep(15 * (attempt + 1))
EOF
  mv "$dest/download/$name" "$dest/data/$name"
  rm -rf "$dest/download"
}

pending=()
for ((i = 0; i < total; i++)); do
  [[ -f $corpus/stream-done/$i ]] || pending+=("$i")
done
echo "$(date '+%F %T') ${#pending[@]} shards pending" >&2
prefetch_pid=
for position in "${!pending[@]}"; do
  i=${pending[$position]}
  if [[ -n $prefetch_pid ]]; then
    wait "$prefetch_pid"
  else
    fetch "$i"
  fi
  prefetch_pid=
  next=${pending[$((position + 1))]:-}
  if [[ -n $next ]]; then
    fetch "$next" > "$corpus/stream-fetch-$next.log" 2>&1 &
    prefetch_pid=$!
  fi
  started=$(date +%s)
  "$py" "$here/scripts/prepare_hf_shards.py" --repo "$repo" --local-dir "$raw/shard-$i" --total "$total" --shards "$i" \
    --pattern "data/train-{index:05d}-of-{total:05d}.parquet" --output "$corpus" "$@" \
    >> "$corpus/stream-encode.log" 2>&1 &
  encode_pid=$!
  "$py" "$here/scripts/transcribe_corpus.py" --raw "$raw/shard-$i/data" --output "$scores" --device cuda \
    --dnsmos "$dnsmos" --dnsmos-device "${DNSMOS_DEVICE:-cuda}" --metric-normalization "${METRIC_NORM:-turkish-v2}" --batch-size "${ASR_BATCH:-48}" \
    --workers "${SCORE_WORKERS:-10}" >> "$scores/stream-score.log" 2>&1 &
  score_pid=$!
  status=0
  wait "$encode_pid" || status=1
  wait "$score_pid" || status=1
  if ((status)); then
    echo "$(date '+%F %T') shard $i failed; see $corpus/stream-encode.log and $scores/stream-score.log" >&2
    exit 1
  fi
  rm -rf "${raw:?}/shard-$i"
  touch "$corpus/stream-done/$i"
  echo "$(date '+%F %T') shard $i done in $(( $(date +%s) - started )) s" >&2
done
echo "$(date '+%F %T') all shards done" >&2
