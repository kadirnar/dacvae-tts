#!/usr/bin/env bash
# Teacher stores of the clean cache from the ORIGINAL audio (ParquetAudio), for #10 (speech-REPA, TLA-SA) and the
# speaker-condition arm. The mHuBERT PCA basis is fitted on 1000 decoded rows (random access), then applied.
set -euo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
C=/workspace/data/trc/clean
PQ=(--audio-source parquet --parquet-repo Codyfederer/tr-combined)
(
  .venv/bin/python scripts/extract_teacher_features.py speakers --cache $C --output $C/teacher/ecapa-speechbrain \
    --splits train,val --device cuda --quiet "${PQ[@]}" --parquet-dir /workspace/data/trc/parquet-spk
  echo "$(date '+%F %T') speakers done"
) > /workspace/logs/store-speakers.log 2>&1 &
(
  .venv/bin/python scripts/extract_teacher_features.py fit-pca --cache $C --output $C/teacher/mhubert147-l12-pca256 \
    --layer 12 --pca-dim 256 --device cuda
  .venv/bin/python scripts/extract_teacher_features.py frames --cache $C --output $C/teacher/mhubert147-l12-pca256 \
    --layer 12 --device cuda --quiet "${PQ[@]}" --parquet-dir /workspace/data/trc/parquet-frames
  echo "$(date '+%F %T') frames done"
) > /workspace/logs/store-frames.log 2>&1 &
wait
