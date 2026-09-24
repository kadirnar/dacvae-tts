#!/usr/bin/env bash
# Step 2: everything that needs run C but no training (#3, #6, #12, #13, #16's oracle). Every arm synthesizes
# Freya-TR-Eval (495 sentences) with the leak-free Common Voice prompt set and scores it with protocol v2
# (deterministic Whisper large-v3, SIM-o + a SpeechBrain SV, UTMOS, DNSMOS, clipping/loudness/bandwidth), the
# turkish-v2 metric and the Freya-convention columns; one arm per GPU at a time, finished arms are skipped.
#   bash scripts/gpu/20_eval_inference.sh             (all jobs)
#   bash scripts/gpu/20_eval_inference.sh base clamp  (named jobs only)
# Results: $OUT/inference/<job>/{results.jsonl,summary.json}, compare.md (paired, speaker-clustered jackknife-t
# intervals against `base`) and summary.tsv (every job's headline numbers, podcast and ceiling jobs included).
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
need "$CKPT_C" "$PROMPTS" "$FREYA" "$DNSMOS" "$CACHE/metadata.json"
dir=$OUT/inference
mkdir -p "$dir"

# Sampler/duration/output options, each against `base` (run C's deployed settings: rule duration, CFG 5, 32 steps).
declare -A ARM=(
  [base]=""
  [seed1000]="--seed 1000"                                   # generation-noise floor of a paired comparison
  [clamp]="--duration-mode clamp"
  [auto]="--duration-mode auto"
  [predictor]="--duration-mode predictor"
  [articulation]="--duration-mode articulation"               # 12
  [bo3]="--candidates 3"                                      # Whisper selects, Whisper judges: an upper bound
  [bo3-duration]="--candidates 3 --duration-factors 1.0,0.9,1.1"   # 12: duration-diverse reranking
  [bo3-composite]="--candidates 3 --select-by cer:10,dnsmos:1"     # 13: composite selector
  [until05]="--guidance-until 0.5"                            # CFG only in the noisy half
  [until05-g6]="--guidance-until 0.5 --guidance 6.0"
  [until05-g7]="--guidance-until 0.5 --guidance 7.0"
  [late-apg]="--guidance-split 0.5 --apg-eta-late 0.5"        # 13: CFG early, APG late
  [late-g2]="--guidance-split 0.5 --guidance-late 2.0"        # 13: weaker late guidance
  [pretanh-auto]="--pre-tanh-gain auto"                       # 13: compare at equal loudness (summary.tsv LUFS)
  [moment-std]="--moment-match std"                           # 13
  [speaker-g3]="--speaker-guidance 3.0"
)
PROMPT_SET_JOBS=(base seed1000 clamp auto predictor articulation bo3 bo3-duration bo3-composite until05 until05-g6
  until05-g7 late-apg late-g2 pretanh-auto moment-std speaker-g3)
# podcast-*: the old cache prompts (held-out podcast labels, 6/10 of their voices also in train) with and without
# the leaking labels (#6); ceiling-*: codec resynthesis and the original recordings of 200 held-out pairs (#3);
# oracle-*: best-of-8 under the GRPO reward with the deployed ODE sampler and with the SDE rollout policy (#16).
OTHER_JOBS=(podcast-base podcast-noleak ceiling-codec ceiling-real oracle-ode oracle-sde)

COMMON=(--sentences "$FREYA" --guidance "$GUIDANCE" --steps "$SAMPLE_STEPS" --asr-backend faster-whisper
  --asr-device cuda --dnsmos "$DNSMOS" --protocol-v2 --metric-normalization "$METRIC_NORM" --freya-metric)
AUDIO=$dir/podcast-audio  # original recordings of the podcast prompts and ceiling targets (SIM-o, real-audio ceiling)

podcast_cases() {  # podcast_cases OUTPUT COUNT [EXCLUDE]: the cache cases eval_sentences / eval_ceiling will select
  "$PY" - "$CACHE" "$@" <<'EOF'
import json
import sys

sys.path.insert(0, "scripts")
from monitor import select_cases  # noqa: E402

from dacvae_tts.speakers import read_speaker_list  # noqa: E402

cache, output, count, *exclude = sys.argv[1:]
_, cases = select_cases(cache, int(count), 42, exclude=read_speaker_list(exclude[0]) if exclude else ())
with open(output, "w") as stream:
    json.dump(cases, stream, indent=1, ensure_ascii=False)
EOF
}

export_originals() {
  [[ -f $AUDIO/.complete ]] && return
  need "$CLUSTERS/leakage.json"
  podcast_cases "$dir/cases-podcast.json" 24
  podcast_cases "$dir/cases-noleak.json" 24 "$CLUSTERS/leakage.json"
  podcast_cases "$dir/cases-ceiling.json" 200
  "$PY" -c "import json, sys; json.dump(sum((json.load(open(p)) for p in sys.argv[2:]), []), open(sys.argv[1], 'w'))" \
    "$dir/cases-all.json" "$dir/cases-podcast.json" "$dir/cases-noleak.json" "$dir/cases-ceiling.json"
  log "export the original podcast recordings of those cases (downloads the shards of $DATASET) -> $AUDIO"
  "$PY" scripts/export_case_audio.py --repo "$DATASET" --cases "$dir/cases-all.json" --cache "$CACHE" --output "$AUDIO"
  touch "$AUDIO/.complete"
}

job() {  # job GPU NAME
  local gpu=$1 name=$2 out=$dir/$2
  [[ -f $out/summary.json ]] && { log "exists: $out"; return 0; }
  export CUDA_VISIBLE_DEVICES=$gpu
  case $name in
    podcast-base)
      "$PY" scripts/eval_sentences.py --checkpoint "$CKPT_C" --cache "$CACHE" --prompts 24 --prompt-audio "$AUDIO" \
        --output "$out" "${COMMON[@]}" ;;
    podcast-noleak)
      "$PY" scripts/eval_sentences.py --checkpoint "$CKPT_C" --cache "$CACHE" --prompts 24 --prompt-audio "$AUDIO" \
        --exclude-speakers "$CLUSTERS/leakage.json" --output "$out" "${COMMON[@]}" ;;
    ceiling-codec | ceiling-real)
      "$PY" scripts/eval_ceiling.py --cache "$CACHE" --cases 200 --output "$out" --dnsmos "$DNSMOS" --protocol-v2 \
        --metric-normalization "$METRIC_NORM" --prompt-audio "$AUDIO" $([[ $name == ceiling-real ]] && echo --real-audio) ;;
    oracle-ode)
      "$PY" scripts/oracle_best_of_n.py --checkpoint "$CKPT_C" --cache "$CACHE" --output "$out" --split val \
        --limit 64 --candidates 8 --sample-steps 32 --guidance "$GUIDANCE" --dnsmos-model "$DNSMOS" --save-audio 8 ;;
    oracle-sde)
      "$PY" scripts/oracle_best_of_n.py --checkpoint "$CKPT_C" --cache "$CACHE" --output "$out" --split val \
        --limit 64 --candidates 8 --sampler sde --sample-steps 16 --guidance "$GUIDANCE" --dnsmos-model "$DNSMOS" ;;
    *)
      [[ -n ${ARM[$name]+set} ]] || { log "unknown job: $name"; return 1; }
      # shellcheck disable=SC2086  # the arm's options are separate words
      "$PY" scripts/eval_sentences.py --checkpoint "$CKPT_C" --prompt-set "$PROMPTS" --output "$out" \
        "${COMMON[@]}" ${ARM[$name]} ;;
  esac > "$out.log" 2>&1
}

jobs=("$@")
((${#jobs[@]})) || jobs=("${PROMPT_SET_JOBS[@]}" "${OTHER_JOBS[@]}")
for name in "${jobs[@]}"; do
  if [[ $name == podcast-* || $name == ceiling-* ]]; then
    export_originals
    break
  fi
done
status=0
run_queue job "${jobs[@]}" || status=1

# Paired comparison of the prompt-set arms (same 48 voices and 495 sentences) against base.
runs=()
for name in "${PROMPT_SET_JOBS[@]}"; do
  [[ -f $dir/$name/results.jsonl ]] && runs+=("$name=$dir/$name")
done
if [[ -f $dir/base/results.jsonl ]] && ((${#runs[@]} > 1)); then
  "$PY" scripts/compare_evals.py "${runs[@]}" --baseline base --stratify length --markdown "$dir/compare.md" \
    --output "$dir/compare.json" > /dev/null
  log "paired comparison: $dir/compare.md"
fi
"$PY" "$REPO/scripts/gpu/summarize.py" "$dir" > "$dir/summary.tsv" && log "headline table: $dir/summary.tsv"
exit "$status"
