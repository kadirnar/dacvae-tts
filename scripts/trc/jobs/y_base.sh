#!/usr/bin/env bash
# y-base = full-v2's 20k snapshot (same recipe, seed and data order as a fresh v2 arm stopped at 20k): full 495-sentence
# evaluations with sampling seeds 42 and 1000, its quick checks copied from full-v2, pushed as the y-base folder
# (scores and audio; the checkpoint is full-v2/checkpoints/step-0020000.pt).
set -euo pipefail
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
RUN=/workspace/runs/trc-y-base; OUT=/workspace/outputs/trc/y-base; V2=/workspace/outputs/trc/full-v2
mkdir -p "$RUN" "$OUT"
[[ -f $RUN/step-0020000.pt ]] || .venv/bin/python -c "
from huggingface_hub import hf_hub_download; import shutil
shutil.copy(hf_hub_download('VoiceHub/dacvae-tts-tr-combined', 'full-v2/checkpoints/step-0020000.pt'), '$RUN/step-0020000.pt')"
cp /workspace/runs/trc-full-v2/config.json "$RUN/config.json"
.venv/bin/python -c "
import json
rows = [l for l in open('/workspace/runs/trc-full-v2/train.jsonl') if l.strip() and json.loads(l)['step'] <= 20000]
open('$RUN/train.jsonl', 'w').writelines(rows)"
for step in 0005000 0010000 0015000; do
  mkdir -p "$OUT/step-$step"; cp "$V2/step-$step/summary.json" "$V2/step-$step/results.jsonl" "$OUT/step-$step/"
done
COMMON=(--prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl
  --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx
  --protocol-v2 --metric-normalization turkish-v2 --freya-metric)
for seed in 42 1000; do
  dir=$OUT/step-0020000; if [[ $seed == 1000 ]]; then dir=$dir-s1000; fi
  [[ -f $dir/summary.json ]] && continue
  mkdir -p "$dir"
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint "$RUN/step-0020000.pt" \
    "${COMMON[@]}" --output "$dir" --seed "$seed" > "$dir.log" 2>&1
done
NOTES="The full-v2 recipe at 20k updates: full-v2's own 20k snapshot (identical training up to there), baseline of the round-3 A/Bs (y-*). Checkpoint: full-v2/checkpoints/step-0020000.pt."
for step in 5000 10000 15000; do
  .venv/bin/python scripts/trc/push_snapshot.py --run "$RUN" --repo VoiceHub/dacvae-tts-tr-combined --subdir y-base --public \
    --no-checkpoint --step $step --eval "$OUT/step-$(printf %07d $step)" --title "DACVAE-TTS tr-combined: y-base (full-v2 recipe at 20k)" --notes "$NOTES" > /dev/null
done
.venv/bin/python scripts/trc/push_snapshot.py --run "$RUN" --repo VoiceHub/dacvae-tts-tr-combined --subdir y-base --public \
  --no-checkpoint --step 20000 --eval "$OUT/step-0020000" --second-eval "$OUT/step-0020000-s1000" \
  --title "DACVAE-TTS tr-combined: y-base (full-v2 recipe at 20k)" --notes "$NOTES" > /dev/null
rm -f "$RUN/step-0020000.pt"
.venv/bin/python scripts/trc/experiment_log.py --runs y-base > /dev/null
echo "$(date '+%F %T') y-base done"
