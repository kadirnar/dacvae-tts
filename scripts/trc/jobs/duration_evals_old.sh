#!/usr/bin/env bash
# Run C with the duration predictor refit on tr-combined: separates the model's gain from the predictor's.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
CK=/workspace/models/run-c/model.pt; DM=/workspace/outputs/trc/duration_trc.json
for seed in 42 1000; do
  out=/workspace/outputs/trc/systems/old-predictor-trc$([[ $seed == 1000 ]] && echo -s1000 || true)
  [[ -f $out/summary.json ]] && continue
  mkdir -p "$out"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint $CK \
    --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl \
    --output "$out" --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
    --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric \
    --duration-mode predictor --duration-model $DM --seed $seed > "$out.log" 2>&1 && echo "$(date '+%F %T') done $seed"
done
