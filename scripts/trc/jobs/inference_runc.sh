#!/usr/bin/env bash
# Inference-only experiments on run C (#12 duration, #13 sampler/output), leak-free CV prompt set, protocol v2.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
dir=/workspace/outputs/trc/inference-runc
COMMON=(--checkpoint /workspace/models/run-c/model.pt --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json
  --sentences /workspace/data/eval/freya_tr_eval.jsonl --guidance 5.0 --steps 32 --asr-backend faster-whisper
  --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric)
declare -A ARM=(
  [seed1000]="--seed 1000" [auto]="--duration-mode auto" [clamp]="--duration-mode clamp"
  [articulation]="--duration-mode articulation" [predictor]="--duration-mode predictor"
  [until05]="--guidance-until 0.5" [until05-g6]="--guidance-until 0.5 --guidance 6.0" [until05-g7]="--guidance-until 0.5 --guidance 7.0"
  [late-apg]="--guidance-split 0.5 --apg-eta-late 0.5" [late-g2]="--guidance-split 0.5 --guidance-late 2.0"
  [pretanh-auto]="--pre-tanh-gain auto" [moment-std]="--moment-match std" [speaker-g3]="--speaker-guidance 3.0"
  [bo3]="--candidates 3" [bo3-duration]="--candidates 3 --duration-factors 1.0,0.9,1.1"
  [bo3-composite]="--candidates 3 --select-by cer:10,dnsmos:1"
)
for name in seed1000 auto clamp articulation until05 pretanh-auto late-apg late-g2 bo3 bo3-duration bo3-composite predictor until05-g6 until05-g7 moment-std speaker-g3; do
  [[ -f $dir/$name/summary.json ]] && continue
  echo "$(date '+%F %T') start $name"
  # shellcheck disable=SC2086
  .venv/bin/python scripts/eval_sentences.py "${COMMON[@]}" --output "$dir/$name" ${ARM[$name]} > "$dir/$name.log" 2>&1 \
    && echo "$(date '+%F %T') done $name" || echo "$(date '+%F %T') FAILED $name"
done
