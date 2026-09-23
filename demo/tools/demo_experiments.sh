#!/usr/bin/env bash
# Freya-TR-Eval experiments for the demo defaults (published model tr-w512-clean step 60k, guidance 5, 32 steps).
# Usage: GPU=0 bash /workspace/demo_experiments.sh NAME [NAME ...]   (runs the named experiments one after another)
set -uo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
declare -A EXTRA=(
  [clamp]="--duration-mode clamp"
  [predictor]="--duration-mode predictor"
  [syllable]="--duration-mode syllable"
  [rescale07]="--cfg-rescale 0.7"
  [apg05]="--apg-eta 0.5 --apg-momentum -0.3"
  [until07]="--guidance-until 0.7"
  [spk3]="--speaker-guidance 3.0"
  [spk7]="--speaker-guidance 7.0"
  [bo3]="--candidates 3"
  [base-s1000]="--seed 1000"
  [clamp-s1000]="--seed 1000 --duration-mode clamp"
  [predictor-s1000]="--seed 1000 --duration-mode predictor"
)
for name in "$@"; do
  out=/workspace/outputs/demo-$name
  if [ -f "$out/summary.json" ]; then echo "$(date +%H:%M) skip $name (done)"; continue; fi
  echo "$(date +%H:%M) start $name on GPU $CUDA_VISIBLE_DEVICES: ${EXTRA[$name]}"
  .venv/bin/python scripts/eval_sentences.py --checkpoint runs/tr-w512-clean/step-0060000.pt --cache /workspace/data/tr55/merged \
    --sentences /workspace/data/eval/freya_tr_eval.jsonl --output "$out" --prompts 24 --guidance 5.0 --steps 32 \
    --asr-backend faster-whisper --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx ${EXTRA[$name]} \
    > "$out.log" 2>&1
  echo "$(date +%H:%M) end $name: $(.venv/bin/python -c "import json; s=json.load(open('$out/summary.json')); print({k: round(s[k],4) for k in ('wer','cer','speaker_similarity','dnsmos_ovrl','files_clipping','median_lufs')})" 2>&1 | tail -1)"
done
