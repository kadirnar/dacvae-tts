#!/usr/bin/env bash
# GPU 1 during the pilot: codec/ASR ceiling, checkpoint monitor (Whisper large-v3, Turkish), corpus re-transcription.
set -euo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
mkdir -p /workspace/outputs
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/eval_ceiling.py --cache /workspace/data/tr55/pilot --cases 48 \
  --output /workspace/outputs/ceiling-pilot --dnsmos /workspace/models/sig_bak_ovr.onnx > /workspace/outputs/ceiling-pilot.log 2>&1 &
echo "ceiling pid $!"
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/monitor.py --run runs/tr-pilot --cache /workspace/data/tr55/pilot --cases 48 \
  --language tr --asr-model large-v3 --asr-device cuda --steps 32 --guidance 3.0 --poll 120 \
  --wandb-project dacvae-tts-tr --wandb-id tr-pilot-monitor > runs/tr-pilot-monitor.log 2>&1 &
echo "monitor pid $!"
CUDA_VISIBLE_DEVICES=1 nohup .venv/bin/python scripts/transcribe_corpus.py --raw /workspace/data/raw/data \
  --output /workspace/outputs/corpus-scores --workers 24 --dnsmos /workspace/models/sig_bak_ovr.onnx > /workspace/outputs/corpus-scores.log 2>&1 &
echo "transcribe pid $!"
