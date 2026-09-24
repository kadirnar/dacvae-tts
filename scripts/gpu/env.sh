# Shared settings and helpers of the GPU runbook (scripts/gpu/README.md); sourced by every step.
# Every variable can be overridden from the environment or from $WORK/.env (secrets: HF_TOKEN, WANDB_API_KEY).
# shellcheck shell=bash

: "${WORK:=/workspace}"
if [[ -f $WORK/.env ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$WORK/.env"
  set +a
fi

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
PY=${PY:-$REPO/.venv/bin/python}
DATA=${DATA:-$WORK/data}
MODELS=${MODELS:-$WORK/models}
OUT=${OUT:-$WORK/outputs}
RUNS=${RUNS:-$WORK/runs}

# Training corpus (Vyvo/tr-dataset-12: 17 Parquet shards, 42,591 podcast segments) and its caches.
DATASET=${DATASET:-Vyvo/tr-dataset-12}
DATASET_SHARDS=${DATASET_SHARDS:-17}
RAW=${RAW:-$DATA/raw/tr-dataset-12}            # holds data/train-XXXXX-of-00017.parquet
CORPUS=${CORPUS:-$DATA/tr55}                    # parts/, merged/, clean/, hq/
CACHE=${CACHE:-$CORPUS/clean}                   # training cache of every A/B (run C's cache)
HQ_CACHE=${HQ_CACHE:-$CORPUS/hq}                # stage-2 / WSD-cooldown cache
TEXT_NORM=${TEXT_NORM:-turkish-v2}              # run C was prepared with turkish-v1 (138 of 42,591 lines differ)
SCORES=${SCORES:-$OUT/corpus-scores/scores.jsonl}
CLUSTERS=${CLUSTERS:-$OUT/speaker-clusters}

# Evaluation assets.
FREYA=${FREYA:-$DATA/eval/freya_tr_eval.jsonl}
CV17=${CV17:-$DATA/cv17}                        # fixie-ai/common_voice_17_0, tr/test Parquet
PROMPT_SET=${PROMPT_SET:-$DATA/eval/cv-tr-prompts}
PROMPTS=${PROMPTS:-$PROMPT_SET/prompts.json}
DNSMOS=${DNSMOS:-$MODELS/sig_bak_ovr.onnx}
CKPT_C=${CKPT_C:-$MODELS/run-c/model.pt}        # published run C (VoiceHub/dacvae-tts-tr-w512, 60k updates)

# Protocol shared by every arm; changing any of these between arms invalidates the comparison.
FRAME_BUDGET=${FRAME_BUDGET:-6000}              # run C: one 24 GB RTX 4090
AB_STEPS=${AB_STEPS:-60000}                     # LR schedule length of the 20k A/B arms ...
AB_STOP=${AB_STOP:-20000}                       # ... stopped here (train --stop-after)
GUIDANCE=${GUIDANCE:-5.0}
SAMPLE_STEPS=${SAMPLE_STEPS:-32}
METRIC_NORM=${METRIC_NORM:-turkish-v2}
WANDB_PROJECT=${WANDB_PROJECT:-}

if [[ -z ${GPUS:-} ]]; then
  GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr '\n' ' ' || true)
fi
GPUS=${GPUS:-0}
NGPU=$(wc -w <<<"$GPUS" | tr -d ' ')

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

log() { printf '%s %s\n' "$(date '+%F %T')" "$*" >&2; }

# need PATH... : stop with a hint when a prerequisite of this step is missing.
need() {
  local path
  for path in "$@"; do
    [[ -e $path ]] || { log "missing: $path (run the earlier steps of scripts/gpu/README.md first)"; exit 1; }
  done
}

# run_queue FUNCTION JOB... : call `FUNCTION GPU JOB` for every job, one job at a time per GPU of $GPUS.
# Each GPU worker claims the next unclaimed job (mkdir is atomic), so long and short jobs balance out.
# Job names must be valid directory names. Returns non-zero when any job failed; the others still run.
run_queue() {
  local fn=$1
  shift
  local claims gpu pid status=0
  local -a pids=()
  claims=$(mktemp -d)
  for gpu in $GPUS; do
    (
      set +e
      failed=0
      for job in "$@"; do
        mkdir "$claims/$job" 2>/dev/null || continue
        log "start $job on GPU $gpu"
        if "$fn" "$gpu" "$job"; then
          log "done  $job"
        else
          log "FAILED $job on GPU $gpu"
          failed=1
        fi
      done
      exit "$failed"
    ) &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || status=1
  done
  rm -rf "$claims"
  return "$status"
}

# hf_fetch REPO TYPE DEST [PATTERN...] : download (part of) a Hugging Face repository into DEST.
hf_fetch() {
  local repo=$1 type=$2 dest=$3
  shift 3
  "$PY" - "$repo" "$type" "$dest" "$@" <<'EOF'
import os
import sys

from huggingface_hub import snapshot_download

repo, kind, dest, *patterns = sys.argv[1:]
snapshot_download(repo, repo_type=kind, local_dir=dest, allow_patterns=patterns or None,
                  token=os.environ.get("HF_TOKEN") or None)
EOF
}
