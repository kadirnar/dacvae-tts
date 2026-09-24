#!/usr/bin/env bash
# Step 1: the Turkish training caches with run C's recipe, the corpus scores behind the clean/hq filters, the
# silence latent, the speaker-leakage report (#6) and the leak-free Common Voice prompt set. Idempotent.
#   HF_TOKEN=... bash scripts/gpu/10_data.sh
# Recipe (checked against run C's cache metadata): quality_score >= 55, 1-20 s, -16 LUFS, all language tags,
# conflicting duplicates dropped, single-recording speakers kept; clean = CER <= 0.10, DNSMOS OVRL >= 2.8,
# >= 2 words; hq = CER <= 0.05, OVRL >= 3.0, quality >= 70, >= 3 words. Text normalization: $TEXT_NORM.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
need "$DNSMOS" "$FREYA"
mkdir -p "$RAW" "$CORPUS" "$OUT" "$(dirname "$SCORES")"

shards=$(compgen -G "$RAW/data/train-*-of-*.parquet" | wc -l || true)
if ((shards < DATASET_SHARDS)); then
  log "download $DATASET ($DATASET_SHARDS shards) -> $RAW"
  hf_fetch "$DATASET" dataset "$RAW" "data/*.parquet" README.md
fi

log "encode: one DACVAE encoder per GPU ($NGPU), $CORPUS/parts (finished shards are skipped)"
PREPARE_GPUS=$NGPU bash scripts/prepare_local_2gpu.sh "$RAW" "$DATASET_SHARDS" "$CORPUS" \
  --text-normalization "$TEXT_NORM" --languages any --speaker-column speaker \
  --quality-column quality_score --min-quality 55 --min-seconds 1 --max-seconds 20 --loudness -16

merge() {  # merge OUTPUT [DROP_LIST]
  local output=$1 drop=${2:-}
  [[ -f $output/metadata.json ]] && { log "exists: $output"; return; }
  log "merge -> $output"
  "$PY" -m dacvae_tts merge --inputs "$CORPUS"/parts/part-* --output "$output" --keep-singletons \
    --drop-conflicting-duplicates ${drop:+--drop-uids "$drop"}
}
merge "$CORPUS/merged"

log "re-transcribe every clip (Whisper large-v3) and score DNSMOS -> $SCORES (resumable)"
CUDA_VISIBLE_DEVICES=${GPUS%% *} "$PY" scripts/transcribe_corpus.py --raw "$RAW/data" --output "$(dirname "$SCORES")" \
  --device cuda --dnsmos "$DNSMOS"

"$PY" scripts/make_drop_list.py --scores "$SCORES" --output "$DATA/drop-clean.json" \
  --max-cer 0.10 --min-ovrl 2.8 --min-words 2
"$PY" scripts/make_drop_list.py --scores "$SCORES" --output "$DATA/drop-hq.json" \
  --max-cer 0.05 --min-ovrl 3.0 --min-quality 70 --min-words 3
merge "$CACHE" "$DATA/drop-clean.json"
merge "$HQ_CACHE" "$DATA/drop-hq.json"

for cache in "$CACHE" "$HQ_CACHE"; do  # tail-silence pairs and skip negatives (#8, #11)
  [[ -f $cache/silence.pt ]] || CUDA_VISIBLE_DEVICES=${GPUS%% *} "$PY" scripts/silence_latent.py --cache "$cache" --device cuda
done

if [[ ! -f $CLUSTERS/summary.json ]]; then
  log "speaker leakage (#6): cross-episode voice clusters -> $CLUSTERS"
  CUDA_VISIBLE_DEVICES=${GPUS%% *} "$PY" scripts/speaker_clusters.py --cache "$CORPUS/merged" --output "$CLUSTERS" \
    --per-label 8 --threshold 0.65 --split-key '^(.+)_speaker_\d+$' --device cuda
fi

if [[ ! -f $PROMPTS ]]; then
  log "leak-free prompt set: 48 Common Voice test speakers -> $PROMPT_SET"
  "$PY" scripts/data/make_prompt_set.py --parquet "$CV17/tr/test" --exclude-sentences "$FREYA" --speakers 48 \
    --dnsmos "$DNSMOS" --output "$PROMPT_SET"
fi
log "data done: $CACHE, $HQ_CACHE, $CLUSTERS/leakage.json, $PROMPTS; next: 15_stores.sh or 20_eval_inference.sh"
