#!/usr/bin/env bash
# Run C at 20k and 40k updates on the leak-free protocol: separates the step effect and the data effect.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
lock=/workspace/outputs/trc/.evaluation.lock
for step in 0020000 0040000; do
  for seed in 42 1000; do
    out=/workspace/outputs/trc/run-c-steps/step-$step$([[ $seed == 1000 ]] && echo -s1000)
    [[ -f $out/summary.json ]] && continue
    mkdir -p "$out"
    flock "$lock" .venv/bin/python scripts/eval_sentences.py --checkpoint /workspace/models/run-c-steps/checkpoints/step-$step.pt \
      --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl \
      --output "$out" --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
      --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric \
      --seed $seed > "$out.log" 2>&1 && echo "$(date '+%F %T') done $step s$seed" || echo "$(date '+%F %T') FAILED $step s$seed"
  done
done
