#!/usr/bin/env bash
# Step 3: the training A/Bs (#7-#11, #14 and the four options of the second round). Every arm changes one thing
# against run C's recipe (configs/nano_tr_w512.yaml), trains on one GPU on the clean cache with the same frame
# budget and the unchanged 60k LR schedule, stops at 20k updates (train --steps 60000 --stop-after 20000) and is
# evaluated like step 2 (Freya-TR-Eval, leak-free prompt set, protocol v2). The baseline runs with two seeds: their
# difference is the noise floor an option has to beat. One arm per GPU at a time; an interrupted arm resumes from
# its last.pt, finished arms and evaluations are skipped.
#   bash scripts/gpu/30_ab_training.sh                        (all arms)
#   bash scripts/gpu/30_ab_training.sh base-s42 base-s43 swiglu
#   bash scripts/gpu/30_ab_training.sh benchmark              (#7 step-time grid; run it alone on an idle machine)
# Prerequisites: 10_data.sh (cache, silence.pt, prompt set) and 15_stores.sh for the arms that read a store.
# Results: $RUNS/ab-<arm>/, $OUT/ab/<arm>/, $OUT/ab/compare.md (paired jackknife-t vs base-s42; base-s43's row is
# the seed noise floor), compare-duration-<scale>.md (tempo-prompts vs base-s42 at duration scale 0.8 and 1.2),
# $OUT/ab/summary.tsv and $OUT/ab/training.tsv (s/update, memory, validation flow).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
need "$CACHE/metadata.json" "$PROMPTS" "$FREYA" "$DNSMOS"
dir=$OUT/ab
generated=$RUNS/configs
mkdir -p "$dir" "$generated" "$RUNS"
((AB_STOP % 5000 == 0)) || { log "AB_STOP must be a multiple of keep_every (5000): step-N.pt is the evaluated file"; exit 1; }

X=configs/experiments
declare -A CONFIG=(
  [base-s42]=configs/nano_tr_w512.yaml
  [base-s43]=$X/tr_w512_seed43.yaml
  # second round: new architecture and data-pipeline options
  [char-units]=$X/tr_w512_char_units.yaml
  [tempo-prompts]=$X/tr_w512_tempo_prompts.yaml
  [speaker-condition]=$X/tr_w512_speaker_condition.yaml
  [speaker-context]=$X/tr_w512_speaker_context.yaml
  # 8: latent negatives instead of the transcript hinge, and no negatives at all
  [latent-negatives]=$X/tr_w512_latent_negatives.yaml
  [no-negatives]="derive configs/nano_tr_w512.yaml train.contrastive_mode=none"
  # 9: DiT block options
  [long-skip]=$X/tr_w512_long_skip.yaml
  [value-residual]=$X/tr_w512_value_residual.yaml
  [ffn-conv]=$X/tr_w512_ffn_conv.yaml
  [attn-gate]=$X/tr_w512_attn_gate.yaml
  [swiglu]=$X/tr_w512_swiglu.yaml
  [final-adaln]=$X/tr_w512_final_adaln.yaml
  [cond-text-pool]=$X/tr_w512_cond_text_pool.yaml
  # 10: teacher alignment
  [repa]=$X/tr_w512_repa.yaml
  [tla]=$X/tr_w512_tla.yaml
  [repa-tla]=$X/tr_w512_repa_tla.yaml
  # 11: training pairs, all together and one group at a time
  [pairs]=$X/tr_w512_pairs.yaml
  [pairs-cross]="derive configs/nano_tr_w512.yaml train.cross_prompt_prob=0.4 train.cross_prompt_max_utterances=3 train.cross_prompt_max_seconds=12.0"
  [pairs-tail]="derive configs/nano_tr_w512.yaml train.tail_silence_prob=0.3 train.tail_silence_max_seconds=0.8 train.long_prompt_prob=0.25 train.prompt_fraction_long_max=0.85 train.prompt_cut=quiet"
  [pairs-char-ctc]="derive configs/nano_tr_w512.yaml model.ctc_targets=chars"
  # 14: regularization (WSD and the model-guidance fine-tune are longer runs: 40_posttrain.sh)
  [regularized]=$X/tr_w512_regularized.yaml
  # 7: the speed options together; quality must not move, s/update should fall (training.tsv)
  [speed]="derive configs/nano_tr_w512.yaml train.grad_checkpoint=selective train.compile=blocks train.compile_dynamic=batch train.strict_checks=false train.pad_multiple=64 train.text_pad_multiple=32 train.loader_negatives=true"
)
ARMS=(base-s42 base-s43 char-units tempo-prompts speaker-condition speaker-context latent-negatives no-negatives
  long-skip value-residual ffn-conv attn-gate swiglu final-adaln cond-text-pool repa tla repa-tla pairs pairs-cross
  pairs-tail pairs-char-ctc regularized speed)

declare -A NEEDS=(  # stores and files an arm reads next to the cache
  [tempo-prompts]=tempo/wsola-v1
  [speaker-condition]=teacher/ecapa-speechbrain
  [tla]=teacher/ecapa-speechbrain
  [repa]=teacher/mhubert147-l12-pca256
  [repa-tla]="teacher/mhubert147-l12-pca256 teacher/ecapa-speechbrain"
  [latent-negatives]=silence.pt
  [pairs]=silence.pt
  [pairs-tail]=silence.pt
)

config_of() {  # config_of ARM: the arm's config file (derived ones are written under $RUNS/configs)
  local spec=${CONFIG[$1]}
  if [[ $spec == derive\ * ]]; then
    read -r -a words <<<"$spec"
    "$PY" scripts/gpu/derive_config.py "${words[1]}" "$generated/$1.yaml" "${words[@]:2}" > /dev/null
    echo "$generated/$1.yaml"
  else
    echo "$spec"
  fi
}

final_of() { printf '%s/ab-%s/step-%07d.pt' "$RUNS" "$1" "$AB_STOP"; }

train_arm() {  # train_arm GPU ARM
  local gpu=$1 arm=$2 run=$RUNS/ab-$2 final config item
  final=$(final_of "$arm")
  [[ -f $final ]] && { log "exists: $final"; return 0; }
  for item in ${NEEDS[$arm]:-}; do
    [[ -e $CACHE/$item ]] || { log "$arm needs $CACHE/$item (see 10_data.sh / 15_stores.sh)"; return 1; }
  done
  config=$(config_of "$arm") || return 1
  local -a extra=(--frame-budget "$FRAME_BUDGET" --steps "$AB_STEPS" --stop-after "$AB_STOP")
  [[ -f $run/last.pt ]] && extra+=(--resume "$run/last.pt")
  [[ -n $WANDB_PROJECT ]] && extra+=(--wandb-project "$WANDB_PROJECT" --wandb-group ab)
  CUDA_VISIBLE_DEVICES=$gpu TRAIN_GPUS=1 bash scripts/train_2gpu.sh "$config" "$CACHE" "$run" "${extra[@]}" \
    >> "$run.log" 2>&1
  [[ -f $final ]]
}

eval_arm() {  # eval_arm GPU ARM
  local gpu=$1 arm=$2 out=$dir/$2 final
  final=$(final_of "$arm")
  [[ -f $out/summary.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_sentences.py --checkpoint "$final" --prompt-set "$PROMPTS" \
    --sentences "$FREYA" --output "$out" --guidance "$GUIDANCE" --steps "$SAMPLE_STEPS" --asr-backend faster-whisper \
    --asr-device cuda --dnsmos "$DNSMOS" --protocol-v2 --metric-normalization "$METRIC_NORM" --freya-metric \
    > "$out.log" 2>&1
}

arm_job() { train_arm "$@" && eval_arm "$@"; }

# Tempo prompts should make the model robust to a prompt/target rate mismatch: the rate sweep of their config.
DURATION_SCALES=(0.8 1.2)
duration_job() {  # duration_job GPU ARM.durSCALE
  local gpu=$1 job=$2 arm=${2%.dur*} scale=${2##*.dur} out=$dir/$2
  [[ -f $out/summary.json ]] && return 0
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/eval_sentences.py --checkpoint "$(final_of "$arm")" --prompt-set "$PROMPTS" \
    --sentences "$FREYA" --output "$out" --guidance "$GUIDANCE" --steps "$SAMPLE_STEPS" --asr-backend faster-whisper \
    --asr-device cuda --dnsmos "$DNSMOS" --protocol-v2 --metric-normalization "$METRIC_NORM" --freya-metric \
    --duration-scale "$scale" > "$out.log" 2>&1
}

benchmark() {
  local gpu=${GPUS%% *}
  log "training-step benchmark (#7) on GPU $gpu -> $dir/benchmark-train-step.jsonl"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" scripts/benchmark_train_step.py --config configs/nano_tr_w512.yaml \
    --frame-budget "$FRAME_BUDGET" --output "$dir/benchmark-train-step.jsonl"
}

arms=("$@")
if [[ ${arms[*]:-} == benchmark ]]; then
  benchmark
  exit
fi
((${#arms[@]})) || arms=("${ARMS[@]}")
for arm in "${arms[@]}"; do
  [[ -n ${CONFIG[$arm]+set} ]] || { log "unknown arm: $arm (${ARMS[*]})"; exit 1; }
  config_of "$arm" > /dev/null  # fail early on a config that does not load
done
status=0
run_queue arm_job "${arms[@]}" || status=1

sweep=()
for scale in "${DURATION_SCALES[@]}"; do
  [[ -f $(final_of base-s42) && -f $(final_of tempo-prompts) ]] && sweep+=("base-s42.dur$scale" "tempo-prompts.dur$scale")
done
if ((${#sweep[@]})); then
  run_queue duration_job "${sweep[@]}" || status=1
  for scale in "${DURATION_SCALES[@]}"; do
    [[ -f $dir/base-s42.dur$scale/results.jsonl && -f $dir/tempo-prompts.dur$scale/results.jsonl ]] || continue
    "$PY" scripts/compare_evals.py "base=$dir/base-s42.dur$scale" "tempo-prompts=$dir/tempo-prompts.dur$scale" \
      --markdown "$dir/compare-duration-$scale.md" > /dev/null
  done
fi

runs=()
for arm in "${ARMS[@]}"; do
  [[ -f $dir/$arm/results.jsonl ]] && runs+=("$arm=$dir/$arm")
done
if [[ -f $dir/base-s42/results.jsonl ]] && ((${#runs[@]} > 1)); then
  "$PY" scripts/compare_evals.py "${runs[@]}" --baseline base-s42 --stratify length --markdown "$dir/compare.md" \
    --output "$dir/compare.json" > /dev/null
  log "paired comparison: $dir/compare.md"
fi
"$PY" scripts/gpu/summarize.py "$dir" > "$dir/summary.tsv"
"$PY" scripts/gpu/train_stats.py "$RUNS"/ab-* > "$dir/training.tsv"
log "tables: $dir/summary.tsv, $dir/training.tsv"
exit "$status"
