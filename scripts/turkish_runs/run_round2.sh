#!/usr/bin/env bash
# Round 2 launchers. Usage:
#   bash /workspace/run_round2.sh w512 tr-w512-clean            (GPU 0, from scratch, clean cache, 60k)
#   bash /workspace/run_round2.sh stage2 tr-stage2 runs/X/step-0040000.pt   (GPU 1, warm start, clean cache, 20k)
set -euo pipefail
kind=${1:?w512|stage2}; name=${2:?run name}; init=${3:-}
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [[ "$kind" == w512 ]]; then gpu=0; cfg=configs/nano_tr_w512.yaml; budget=6000; extra=(); else gpu=1; cfg=configs/nano_tr_stage2.yaml; budget=7000; extra=(--init-from "$init"); fi
CUDA_VISIBLE_DEVICES=$gpu TRAIN_GPUS=1 nohup bash scripts/train_2gpu.sh "$cfg" /workspace/data/tr55/clean "runs/$name" \
  --frame-budget "$budget" --wandb-project dacvae-tts-tr "${extra[@]}" > "runs/$name.log" 2>&1 &
echo "training $name pid $! on gpu $gpu"
CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged --cases 48 \
  --language tr --asr-model large-v3 --asr-device cpu --steps 32 --guidance 3.0 --poll 180 \
  --wandb-project dacvae-tts-tr --wandb-id "$name-monitor" > "runs/$name-monitor.log" 2>&1 &
echo "monitor $name pid $!"
