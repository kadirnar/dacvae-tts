#!/usr/bin/env bash
# Deployable-system evaluations: the new 60k model with the inference settings that won on run C (#12/#13), and run C
# with the same, so the comparison is system to system.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
COMMON=(--prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl
  --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx
  --protocol-v2 --metric-normalization turkish-v2 --freya-metric)
NEW=/workspace/runs/trc-full-cross/step-0060000.pt; OLD=/workspace/models/run-c/model.pt
run() {  # run NAME CHECKPOINT OPTIONS...
  local name=$1 ckpt=$2; shift 2
  local out=/workspace/outputs/trc/systems/$name
  [[ -f $out/summary.json ]] && return
  mkdir -p "$out"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint "$ckpt" "${COMMON[@]}" \
    --output "$out" "$@" > "$out.log" 2>&1
  echo "$(date '+%F %T') done $name"
}
run new-auto $NEW --duration-mode auto
run new-auto-s1000 $NEW --duration-mode auto --seed 1000
run old-auto-s1000 $OLD --duration-mode auto --seed 1000
run new-auto-bo3c $NEW --duration-mode auto --candidates 3 --select-by cer:10,dnsmos:1
run old-auto-bo3c $OLD --duration-mode auto --candidates 3 --select-by cer:10,dnsmos:1
