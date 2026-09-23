#!/usr/bin/env bash
# After a run's training ends: guidance sweep with DNSMOS (GPU Whisper), 5 custom sentences, Freya-TR-Eval (beam-5 Whisper),
# push everything (audio + texts + final checkpoint) to the run's VoiceHub dataset.
# Usage: bash /workspace/finish_run.sh RUN_NAME FINAL_STEP GPU REPO_SUFFIX
set -uo pipefail
name=${1:?}; step=${2:?}; gpu=${3:?}; repo="VoiceHub/dacvae-tts-${4:-$name}"
cd /workspace/dacvae-tts
set -a; . /workspace/.env; set +a
ckpt=$(printf "runs/%s/step-%07d.pt" "$name" "$step")
until [ -f "$ckpt" ] && ! pgrep -f "runs/$name --workers" >/dev/null; do sleep 60; done
sleep 30
echo "$(date +%H:%M) training of $name finished; evaluating $ckpt on GPU $gpu"
export CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
for g in 4.0 5.0 6.0; do
  .venv/bin/python scripts/monitor.py --run "runs/$name" --cache /workspace/data/tr55/merged --cases 48 --language tr \
    --asr-model large-v3 --asr-device cuda --checkpoint "$ckpt" --guidance "$g" --steps 32 --dnsmos /workspace/models/sig_bak_ovr.onnx 2>&1 | grep '^{"checkpoint"' | cut -c1-160
done
.venv/bin/python scripts/eval_sentences.py --checkpoint "$ckpt" --cache /workspace/data/tr55/merged --sentences /workspace/data/eval/custom5.txt \
  --output "/workspace/outputs/custom5-$name-$step" --prompts 5 --guidance 5.0 --steps 32 --asr-device cuda --dnsmos /workspace/models/sig_bak_ovr.onnx 2>&1 | grep -E '"wer"|"cer"' | head -2
.venv/bin/python scripts/push_custom_audio.py --folder "/workspace/outputs/custom5-$name-$step" --repo "$repo" --name "custom-sentences-step-$(printf %07d $step)" \
  --checkpoint "$ckpt" --title "5 unseen Turkish sentences, $name checkpoint $step (guidance 5)" 2>&1 | tail -1
.venv/bin/python scripts/eval_sentences.py --checkpoint "$ckpt" --cache /workspace/data/tr55/merged --sentences /workspace/data/eval/freya_tr_eval.jsonl \
  --output "/workspace/outputs/freya-$name-$step" --prompts 24 --guidance 5.0 --steps 32 --asr-backend faster-whisper --asr-device cuda \
  --dnsmos /workspace/models/sig_bak_ovr.onnx 2>&1 | grep -E '"wer"|"cer"|"speaker_similarity"|"dnsmos_ovrl"' | head -4
.venv/bin/python scripts/push_custom_audio.py --folder "/workspace/outputs/freya-$name-$step" --repo "$repo" --name "freya-tr-eval-step-$(printf %07d $step)" \
  --checkpoint "$ckpt" --title "Freya-TR-Eval (495 sentences), $name checkpoint $step, guidance 5, 24 unseen prompt voices, faster-whisper large-v3 beam 5" 2>&1 | tail -1
.venv/bin/python scripts/push_checkpoint_audio.py --run "runs/$name" --repo "$repo" --checkpoints "$ckpt" 2>&1 | grep -E "pushed|https" | tail -2
echo "$(date +%H:%M) finish_run $name done"
