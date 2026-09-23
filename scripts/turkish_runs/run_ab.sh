#!/usr/bin/env bash
# Two single-GPU full-data runs: A (nano_tr, budget 12000) on GPU 0, B (nano_tr_ke4, budget 7000) on GPU 1.
# Usage: bash /workspace/run_ab.sh A|B RUN_NAME
set -euo pipefail
which=${1:?A or B}; name=${2:?run name}
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs
if [[ "$which" == A ]]; then gpu=0; cfg=configs/nano_tr.yaml; budget=12000; else gpu=1; cfg=configs/nano_tr_ke4.yaml; budget=7000; fi
CUDA_VISIBLE_DEVICES=$gpu TRAIN_GPUS=1 nohup bash scripts/train_2gpu.sh "$cfg" /workspace/data/tr55/merged "runs/$name" \
  --frame-budget "$budget" --wandb-project dacvae-tts-tr > "runs/$name.log" 2>&1 &
echo "training $name pid $! on gpu $gpu"
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged --cases 48 \
  --language tr --asr-model large-v3 --asr-device cpu --steps 32 --guidance 3.0 --poll 180 \
  --wandb-project dacvae-tts-tr --wandb-id "$name-monitor" > "runs/$name-monitor.log" 2>&1 &
echo "monitor $name pid $!"
