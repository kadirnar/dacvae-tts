#!/usr/bin/env bash
set -euo pipefail
name=${1:?}; ckpt=${2:?}
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
run() {
  CUDA_VISIBLE_DEVICES=${GPU:-1} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged \
    --cases 48 --language tr --asr-model large-v3 --asr-device cpu --checkpoint "$ckpt" \
    --guidance "$1" --guidance-until "$2" --noise-scale "$3" --sway "$4" --duration-scale "$5" --steps "$6" 2>&1 | grep -c '^{"checkpoint"'
}
run 4.0 1.0 1.0 -1.0 1.0 32
run 5.0 1.0 1.0 -1.0 1.0 32
run 4.0 0.5 1.0 -1.0 1.0 16
run 6.0 1.0 1.0 -1.0 1.0 32
