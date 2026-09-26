#!/usr/bin/env bash
# Quality-condition controllability: the x-quality-cond 20k checkpoint asked for different DNSMOS targets.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
CK=/workspace/runs/trc-x-quality-cond/step-0020000.pt
until [[ -f $CK ]]; do sleep 60; done
for spec in "unknown 3,3,3" "high 4.0,4.5,3.8"; do
  set -- $spec
  out=/workspace/outputs/trc/x-quality-cond/step-0020000-q$1
  [[ -f $out/summary.json ]] && continue
  mkdir -p "$out"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint $CK \
    --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl \
    --output "$out" --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
    --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric \
    --quality-target "$2" > "$out.log" 2>&1 && echo "$(date '+%F %T') done $1"
done
