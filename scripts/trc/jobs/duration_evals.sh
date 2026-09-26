#!/usr/bin/env bash
# The 60k model with the duration predictor refit on tr-combined (single sample, seeds 42 and 1000).
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
CK=/workspace/runs/trc-full-cross/step-0060000.pt; DM=/workspace/outputs/trc/duration_trc.json
for spec in "predictor-trc predictor 42" "auto-trc auto 42" "predictor-trc-s1000 predictor 1000" "auto-trc-s1000 auto 1000"; do
  set -- $spec
  out=/workspace/outputs/trc/systems/new-$1
  [[ -f $out/summary.json ]] && continue
  mkdir -p "$out"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint $CK \
    --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl \
    --output "$out" --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
    --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric \
    --duration-mode $2 --duration-model $DM --seed $3 > "$out.log" 2>&1 && echo "$(date '+%F %T') done $1"
done
