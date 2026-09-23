#!/usr/bin/env bash
set -uo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
until [ -f runs/tr-w512-clean/step-0060000.pt ] && ! pgrep -f "runs/tr-w512-clean --workers" >/dev/null; do sleep 60; done
sleep 30
echo "$(date +%H:%M) C finished; launching tr-w512-stage2-hq on GPU 1"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 TRAIN_GPUS=1 nohup bash scripts/train_2gpu.sh configs/nano_tr_w512_stage2.yaml /workspace/data/tr55/hq runs/tr-w512-stage2-hq \
  --frame-budget 6000 --wandb-project dacvae-tts-tr --init-from runs/tr-w512-clean/step-0060000.pt > runs/tr-w512-stage2-hq.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 nohup .venv/bin/python scripts/monitor.py --run runs/tr-w512-stage2-hq --cache /workspace/data/tr55/merged --cases 48 \
  --language tr --asr-model large-v3 --asr-device cpu --steps 32 --guidance 3.0 --poll 180 --wandb-project dacvae-tts-tr --wandb-id tr-w512-stage2-hq-monitor > runs/tr-w512-stage2-hq-monitor.log 2>&1 &
nohup .venv/bin/python scripts/push_checkpoint_audio.py --run runs/tr-w512-stage2-hq --repo VoiceHub/dacvae-tts-tr-w512-stage2-hq \
  --title "DACVAE-TTS Turkish run C stage 2 (width 512, high-quality subset)" \
  --notes "Warm start from VoiceHub/dacvae-tts-tr-w512-clean (step 60k, 66.5M parameters), 10k more updates on the high-quality subset (Whisper CER <= 0.05, DNSMOS OVRL >= 3.0, quality_score >= 70; 21k rows) at LR 2e-4, batch expansion 4, frame budget 6000, one RTX 4090." \
  --watch 300 > runs/tr-w512-stage2-hq-push.log 2>&1 &
nohup bash /workspace/finish_run.sh tr-w512-stage2-hq 10000 1 tr-w512-stage2-hq > /workspace/outputs/finish-tr-w512-stage2-hq.log 2>&1 &
echo "$(date +%H:%M) chain launched"
