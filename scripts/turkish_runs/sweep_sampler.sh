#!/usr/bin/env bash
# Sampler sweep on one checkpoint with the monitor's 48 held-out cases (Whisper on CPU, synthesis on GPU).
# Usage: GPU=1 bash /workspace/sweep_sampler.sh RUN_NAME CHECKPOINT_PATH
set -euo pipefail
name=${1:?run name}; ckpt=${2:?checkpoint}
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
run() {  # guidance guidance_until noise_scale sway duration_scale steps
  CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged \
    --cases 48 --language tr --asr-model large-v3 --asr-device cpu --checkpoint "$ckpt" \
    --guidance "$1" --guidance-until "$2" --noise-scale "$3" --sway "$4" --duration-scale "$5" --steps "$6" 2>&1 | grep '^{"checkpoint"' | cut -c1-200
}
run 2.0 1.0 1.0 -1.0 1.0 32
run 4.0 1.0 1.0 -1.0 1.0 32
run 3.0 0.5 1.0 -1.0 1.0 32
run 3.0 1.0 0.9 -1.0 1.0 32
run 3.0 1.0 1.0 0.0 1.0 32
run 3.0 1.0 1.0 -1.0 0.9 32
run 3.0 1.0 1.0 -1.0 1.0 16
