#!/usr/bin/env bash
# Move the runs that still pushed to their former repos into the experiments repo once their runners have ended.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
wait_for() { until grep -q "$1" "$2"; do sleep 60; done; }
wait_for "finish full-v2" /workspace/outputs/trc/queue-finish.log
.venv/bin/python scripts/trc/migrate_hub.py --runs full-v2 && echo "$(date '+%F %T') migrated full-v2"
wait_for "finish x-swiglu" /workspace/outputs/trc/queue-finish.log
.venv/bin/python scripts/trc/migrate_hub.py --runs x-swiglu && echo "$(date '+%F %T') migrated x-swiglu"
until grep -q "done x-attn-gate\|FAILED x-attn-gate" /workspace/outputs/trc/queue-round3.log && ! pgrep -f "run_arm.py --arm x-attn-gate" > /dev/null; do sleep 60; done
.venv/bin/python scripts/trc/migrate_hub.py --runs x-attn-gate && echo "$(date '+%F %T') migrated x-attn-gate"
