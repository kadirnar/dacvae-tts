#!/usr/bin/env bash
# One Freya-TR-Eval run of the published checkpoint with arbitrary extra options.
# Usage: GPU=0 bash /workspace/demo_run.sh NAME [eval_sentences.py options...]
set -uo pipefail
name=${1:?name}; shift
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
export CUDA_VISIBLE_DEVICES=${GPU:-0} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
out=/workspace/outputs/demo-$name
if [ -f "$out/summary.json" ]; then echo "$(date +%H:%M) skip $name (done)"; exit 0; fi
echo "$(date +%H:%M) start $name on GPU $CUDA_VISIBLE_DEVICES: $*"
.venv/bin/python scripts/eval_sentences.py --checkpoint runs/tr-w512-clean/step-0060000.pt --cache /workspace/data/tr55/merged \
  --sentences /workspace/data/eval/freya_tr_eval.jsonl --output "$out" --prompts 24 --guidance 5.0 --steps 32 \
  --asr-backend faster-whisper --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx "$@" > "$out.log" 2>&1
echo "$(date +%H:%M) end $name: $(.venv/bin/python -c "import json; s=json.load(open('$out/summary.json')); print({k: round(s[k],4) for k in ('wer','cer','speaker_similarity','dnsmos_ovrl','files_clipping','median_lufs')})" 2>&1 | tail -1)"
