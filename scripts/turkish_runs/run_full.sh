#!/usr/bin/env bash
# Full run: 17 shards (quality>=55, ~77 h train), 2x RTX 4090 DDP, configs/nano_tr.yaml, 60k updates.
# Usage: bash /workspace/run_full.sh RUN_NAME [extra train flags]
set -euo pipefail
name=${1:?run name}; shift || true
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
TRAIN_GPUS=2 nohup bash scripts/train_2gpu.sh configs/nano_tr.yaml /workspace/data/tr55/merged "runs/$name" \
  --frame-budget "${FRAME_BUDGET:-10000}" --wandb-project dacvae-tts-tr "$@" > "runs/$name.log" 2>&1 &
echo "training pid $!"
# Monitor on GPU 1 (synthesis on GPU, Whisper large-v3 fp16 on GPU), 48 unseen-speaker cases, guidance 3 / 32 steps.
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 nohup .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged --cases 48 \
  --language tr --asr-model large-v3 --asr-device cuda --steps 32 --guidance 3.0 --poll 180 \
  --wandb-project dacvae-tts-tr --wandb-id "$name-monitor" > "runs/$name-monitor.log" 2>&1 &
echo "monitor pid $!"
