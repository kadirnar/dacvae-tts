#!/usr/bin/env bash
# Queue full-v3 behind round 4: launched once y-attn-gate (the last round-4 arm) has started, so the two queues never
# start trainers at the same time; arm_queue.py then waits for the free training slot.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
until grep -q "start y-attn-gate" /workspace/outputs/trc/queue-round4.log; do sleep 30; done
sleep 60
exec .venv/bin/python scripts/trc/arm_queue.py full-v3 --parallel 1 >> /workspace/outputs/trc/queue-round5.log 2>&1
