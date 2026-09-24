#!/usr/bin/env bash
# Step 4: the longer runs that start from or replace run C, each evaluated like step 2 and compared with run C's
# `base` arm of 20_eval_inference.sh (same prompts, sentences and scorer). One job per GPU; finished jobs are skipped.
#   mg-w07, mg-w05  #14 (c): model-guidance fine-tune of run C, 8k updates, w = 0.7 (~CFG 3.3) and 0.5 (~CFG 2);
#                   sampled with guidance 1 (one forward per step instead of two)
#   wsd             #14 (a): a single 60k WSD run from scratch whose cooldown (last 20 %) runs on the hq cache,
#                   against run C 60k (+ its separate hq stage 2); ~10 GPU-hours on an RTX 4090
#   grpo            #16: 1,000 Flow-GRPO updates of run C under the composite reward (CER, SIM, DNSMOS, UTMOS); read
#                   the oracle-ode / oracle-sde jobs of step 2 first: they bound what reweighting can reach
#   bash scripts/gpu/40_posttrain.sh [mg-w07] [mg-w05] [wsd] [grpo]      (default: all)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
need "$CKPT_C" "$CACHE/metadata.json" "$PROMPTS" "$FREYA" "$DNSMOS"
dir=$OUT/posttrain
generated=$RUNS/configs
mkdir -p "$dir" "$generated" "$RUNS"
JOBS=(mg-w07 mg-w05 wsd grpo)

evaluate() {  # evaluate GPU NAME CHECKPOINT [eval_sentences options]
  local gpu=$1 name=$2 checkpoint=$3 out=$dir/$2
  shift 3
  [[ -f $out/summary.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_sentences.py --checkpoint "$checkpoint" --prompt-set "$PROMPTS" \
    --sentences "$FREYA" --output "$out" --guidance "$GUIDANCE" --steps "$SAMPLE_STEPS" --asr-backend faster-whisper \
    --asr-device cuda --dnsmos "$DNSMOS" --protocol-v2 --metric-normalization "$METRIC_NORM" --freya-metric "$@" \
    > "$out.log" 2>&1
}

train() {  # train GPU RUN CONFIG FINAL [train options]: one-GPU run, resumed from last.pt when interrupted
  local gpu=$1 run=$2 config=$3 final=$4
  shift 4
  [[ -f $final ]] && return 0
  local -a extra=(--frame-budget "$FRAME_BUDGET" "$@")
  if [[ -f $run/last.pt ]]; then
    extra+=(--resume "$run/last.pt")
  fi
  [[ -n $WANDB_PROJECT ]] && extra+=(--wandb-project "$WANDB_PROJECT" --wandb-group posttrain)
  CUDA_VISIBLE_DEVICES=$gpu TRAIN_GPUS=1 bash scripts/train_2gpu.sh "$config" "$CACHE" "$run" "${extra[@]}" \
    >> "$run.log" 2>&1
  [[ -f $final ]]
}

job() {  # job GPU NAME
  local gpu=$1 name=$2 config run=$RUNS/$2
  case $name in
    mg-w07 | mg-w05)
      config=$generated/$name.yaml
      "$PY" scripts/gpu/derive_config.py configs/experiments/tr_w512_model_guidance_ft.yaml "$config" \
        "train.model_guidance_weight=0.${name#mg-w0}" > /dev/null
      # --init-from is a warm start (fresh schedule); a resumed run continues from its own last.pt instead.
      local -a init=()
      [[ -f $run/last.pt ]] || init=(--init-from "$CKPT_C")
      train "$gpu" "$run" "$config" "$run/step-0008000.pt" "${init[@]}" &&
        evaluate "$gpu" "$name" "$run/step-0008000.pt" --guidance 1.0
      ;;
    wsd)
      [[ -f $HQ_CACHE/metadata.json ]] || { log "wsd needs the hq cache $HQ_CACHE (10_data.sh)"; return 1; }
      config=$generated/wsd.yaml
      "$PY" scripts/gpu/derive_config.py configs/experiments/tr_w512_wsd.yaml "$config" \
        "train.decay_cache=$HQ_CACHE" > /dev/null
      train "$gpu" "$run" "$config" "$run/step-0060000.pt" &&
        evaluate "$gpu" "$name" "$run/step-0060000.pt"
      ;;
    grpo)
      if [[ ! -f $run/grpo-001000.pt ]]; then
        CUDA_VISIBLE_DEVICES=$gpu "$PY" -m dacvae_tts post-train --mode grpo --checkpoint "$CKPT_C" --cache "$CACHE" \
          --output "$run" --dnsmos-model "$DNSMOS" --steps 1000 --group-size 8 --prompts-per-step 4 \
          --sample-steps 16 --guidance "$GUIDANCE" --learning-rate 1e-5 --save-every 100 > "$run.log" 2>&1 || return 1
      fi
      evaluate "$gpu" "$name" "$run/grpo-001000.pt"
      ;;
    *)
      log "unknown job: $name (${JOBS[*]})"
      return 1
      ;;
  esac
}

jobs=("$@")
((${#jobs[@]})) || jobs=("${JOBS[@]}")
status=0
run_queue job "${jobs[@]}" || status=1

base=$OUT/inference/base
runs=()
for name in "${JOBS[@]}"; do
  [[ -f $dir/$name/results.jsonl ]] && runs+=("$name=$dir/$name")
done
if [[ -f $base/results.jsonl ]] && ((${#runs[@]})); then
  "$PY" scripts/compare_evals.py "run-c=$base" "${runs[@]}" --stratify length --markdown "$dir/compare.md" \
    --output "$dir/compare.json" > /dev/null
  log "paired comparison against run C: $dir/compare.md"
else
  log "no comparison: run C's base arm ($base) comes from 20_eval_inference.sh"
fi
"$PY" scripts/gpu/summarize.py "$dir" > "$dir/summary.tsv"
exit "$status"
