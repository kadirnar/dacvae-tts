#!/usr/bin/env bash
# Pilot: 4 shards (~21 h, quality>=55), single GPU 0, 12k updates. Monitor + ceiling run on GPU 1.
set -euo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
CUDA_VISIBLE_DEVICES=0 TRAIN_GPUS=1 nohup bash scripts/train_2gpu.sh configs/nano_tr_pilot.yaml /workspace/data/tr55/pilot runs/tr-pilot \
  --frame-budget 12000 --wandb-project dacvae-tts-tr > runs/tr-pilot.log 2>&1 &
echo "pilot training pid $!"
