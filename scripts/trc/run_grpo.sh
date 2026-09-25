#!/usr/bin/env bash
# Flow-GRPO post-training (#16) of the full-length model, run as an exclusive queue job: it holds the evaluation lock
# while training (the rollouts, four reward judges and the policy do not fit next to a training arm or an evaluation),
# then evaluates the final policy with the A/B protocol (seeds 42 and 1000) and pushes it to VoiceHub.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
set -a; . /workspace/.env; set +a
RUN=${GRPO_RUN:-/workspace/runs/trc-grpo}
OUT=${GRPO_OUT:-/workspace/outputs/trc/grpo}
STEPS=${GRPO_STEPS:-600}
INIT=${GRPO_INIT:-/workspace/runs/trc-full-cross/step-0060000.pt}
FINAL=$RUN/grpo-$(printf %06d "$STEPS").pt
mkdir -p "$RUN" "$OUT"
if [[ ! -f $FINAL ]]; then
  exec 9> /workspace/outputs/trc/.evaluation.lock
  flock 9
  rm -f "$RUN"/grpo-log.jsonl
  .venv/bin/dacvae-tts post-train --mode grpo --checkpoint "$INIT" --cache /workspace/data/trc/clean --output "$RUN" \
    --dnsmos-model /workspace/models/sig_bak_ovr.onnx --steps "$STEPS" --save-every 100 --monitor-every 100 \
    --group-size 8 --prompts-per-step 4 --sample-steps 16 --guidance 5 --learning-rate 1e-5 --language tr \
    --metric-normalization turkish-v2 --codec-backend fast >> "$RUN.log" 2>&1
  flock -u 9
fi
# push_snapshot.py and the evaluation layout name snapshots step-NNNNNNN.pt
ln -f "$FINAL" "$RUN/step-$(printf %07d "$STEPS").pt"
COMMON=(--prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl
  --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx
  --protocol-v2 --metric-normalization turkish-v2 --freya-metric)
for seed in 42 1000; do
  dir=$OUT/step-$(printf %07d "$STEPS")
  if [[ $seed == 1000 ]]; then dir=$dir-s1000; fi  # not `$([[ ]] && echo)`: its status 1 ends the script under set -e
  [[ -f $dir/summary.json ]] && continue
  mkdir -p "$dir"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint "$FINAL" \
    "${COMMON[@]}" --output "$dir" --seed "$seed" > "$dir.log" 2>&1
done
.venv/bin/python scripts/trc/push_snapshot.py --run "$RUN" --repo VoiceHub/dacvae-tts-trc-grpo --step "$STEPS" \
  --eval "$OUT/step-$(printf %07d "$STEPS")" --title "DACVAE-TTS tr-combined: Flow-GRPO post-training (#16)" \
  --notes "Flow-GRPO ($STEPS updates, group 8 x 4 prompts, 16 steps, SDE window 2, sigma 0.5, KL 0.04, LR 1e-5; reward CER 1.0 + SIM 0.5 + DNSMOS 0.4 + UTMOS 0.4) from full-cross step 60000." > /dev/null
touch "$OUT/done"
