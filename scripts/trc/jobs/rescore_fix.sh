#!/usr/bin/env bash
# Rescore evaluations whose scoring rows failed with CUDA OOM, then refresh their Hub summaries.
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
for spec in "base-s42 0020000 0" "pairs-cross 0010000 96"; do
  set -- $spec
  out=/workspace/outputs/trc/$1/step-$2
  limit=(); [[ $3 != 0 ]] && limit=(--limit $3)
  .venv/bin/python scripts/trc/eval_slot.py -- .venv/bin/python scripts/eval_sentences.py --checkpoint /workspace/runs/trc-$1/step-$2.pt \
    --prompt-set /workspace/data/eval/cv-tr-prompts/prompts.json --sentences /workspace/data/eval/freya_tr_eval.jsonl \
    --output "$out" --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
    --dnsmos /workspace/models/sig_bak_ovr.onnx --protocol-v2 --metric-normalization turkish-v2 --freya-metric --seed 42 \
    "${limit[@]}" --rescore >> "$out.rescore.log" 2>&1
  errors=$(python3 -c "import json; print(sum(str(json.loads(l).get('error','')).startswith('score: ') for l in open('$out/results.jsonl')))")
  echo "$(date '+%F %T') $1 $2 rescored, remaining scoring errors: $errors"
  rm -f /workspace/runs/trc-$1/pushed-dacvae-tts-trc-$1.json.tmp
  .venv/bin/python scripts/trc/push_snapshot.py --run /workspace/runs/trc-$1 --repo VoiceHub/dacvae-tts-trc-$1 --step $((10#$2)) \
    --eval "$out" --no-checkpoint --title "DACVAE-TTS tr-combined A/B arm \`$1\`" > /dev/null 2>&1 && echo "  pushed"
done
