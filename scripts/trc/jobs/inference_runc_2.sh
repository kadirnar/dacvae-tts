#!/usr/bin/env bash
# Remaining run C inference jobs, serialized with the arm evaluations through the evaluation lock.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
dir=/workspace/outputs/trc/inference-runc
lock=/workspace/outputs/trc/.evaluation.lock
COMMON=(--checkpoint /workspace/models/run-c/model.pt --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json
  --sentences /workspace/data/eval/freya_tr_eval.jsonl --guidance 5.0 --steps 32 --asr-backend faster-whisper
  --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric)
declare -A ARM=(
  [bo3-composite]="--candidates 3 --select-by cer:10,dnsmos:1" [predictor]="--duration-mode predictor"
  [until05-g6]="--guidance-until 0.5 --guidance 6.0" [until05-g7]="--guidance-until 0.5 --guidance 7.0"
  [moment-std]="--moment-match std" [speaker-g3]="--speaker-guidance 3.0"
)
for name in bo3-composite predictor until05-g6 until05-g7 moment-std speaker-g3; do
  [[ -f $dir/$name/summary.json ]] && continue
  rm -rf "$dir/$name"
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  flock "$lock" .venv/bin/python scripts/eval_sentences.py "${COMMON[@]}" --output "$dir/$name" ${ARM[$name]} > "$dir/$name.log" 2>&1 \
    && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name"
done
# #13: pre-tanh gain vs base at equal loudness (-16 LUFS), rescored with the same prompts.
for name in base pretanh-auto; do
  out=$dir/$name-lufs16
  [[ -f $out/summary.json ]] && continue
  .venv/bin/python scripts/trc/level_match.py "$dir/$name" "$out"
  echo "$(date '+%F %T') start $name-lufs16"
  flock "$lock" .venv/bin/python scripts/eval_sentences.py "${COMMON[@]}" --output "$out" --rescore > "$out.log" 2>&1 \
    && echo "$(date '+%F %T') done $name-lufs16" || echo "$(date '+%F %T') FAILED $name-lufs16"
done
