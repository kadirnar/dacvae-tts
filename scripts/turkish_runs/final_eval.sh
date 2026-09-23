#!/usr/bin/env bash
# Final evaluation of one checkpoint: guidance sweep with DNSMOS on the 48 held-out cases, Freya-TR-Eval (495 sentences)
# with the chosen setting, then push everything to VoiceHub.
# Usage: GPU=0 bash /workspace/final_eval.sh RUN_NAME CHECKPOINT GUIDANCE
set -euo pipefail
name=${1:?}; ckpt=${2:?}; g=${3:-5.0}
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for gg in 3.0 4.0 5.0 6.0; do
  .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged --cases 48 --language tr \
    --asr-model large-v3 --asr-device cpu --checkpoint "$ckpt" --guidance "$gg" --steps 32 --dnsmos /workspace/models/sig_bak_ovr.onnx 2>&1 | grep '^{"checkpoint"' | cut -c1-160 || true
done
.venv/bin/python scripts/eval_sentences.py --checkpoint "$ckpt" --cache /workspace/data/tr55/merged --sentences /workspace/data/eval/freya_tr_eval.jsonl \
  --output "/workspace/outputs/freya-$name" --prompts 24 --guidance "$g" --steps 32 --asr-device cpu --dnsmos /workspace/models/sig_bak_ovr.onnx 2>&1 | tail -25
