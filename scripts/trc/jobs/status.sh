#!/usr/bin/env bash
# One-screen status of the tr-combined experiments.
date '+%F %T'
for run in $(ls -dt /workspace/runs/trc-*/ 2>/dev/null | head -2); do
  last=$(grep '"flow"' "$run/train.jsonl" 2>/dev/null | tail -1 | python3 -c "import sys,json; r=json.loads(sys.stdin.read() or '{}'); print(r.get('step'), round(r.get('flow',0),4), round(r.get('elapsed_seconds',0)/100,3),'s/upd')" 2>/dev/null)
  val=$(grep validation_flow "$run/train.jsonl" 2>/dev/null | tail -1 | python3 -c "import sys,json; r=json.loads(sys.stdin.read() or '{}'); print('val', round(r.get('validation_flow',0),4))" 2>/dev/null)
  echo "$(basename $run): step $last $val"
done
tail -2 /workspace/outputs/trc/queue-round1c.log
tail -1 /workspace/outputs/trc/inference-runc/queue.log
for d in /workspace/outputs/trc/*/; do a=$(basename $d); [ -f $d/step-0020000/summary.json ] && python3 -c "
import json,os; s=json.load(open('$d/step-0020000/summary.json')); t='$d/step-0020000-s1000/summary.json'; u=json.load(open(t)) if os.path.exists(t) else {}
print('  $a 20k: WER %.2f CER %.2f SIM-o %.3f DNSMOS %.3f UTMOS %.3f' % (100*s['wer'],100*s['cer'],s.get('sim_o',0),s.get('dnsmos_ovrl',0),s.get('utmos',0)), '| s1000 WER %.2f CER %.2f' % (100*u['wer'],100*u['cer']) if u else '')"; done 2>/dev/null
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; df -h / | tail -1 | awk '{print "disk free", $4}'
