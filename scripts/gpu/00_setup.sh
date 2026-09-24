#!/usr/bin/env bash
# Step 0: Python environment, evaluation assets and the published run C checkpoint. Idempotent: re-running skips
# whatever is already in place. Needs network access; HF_TOKEN only if a repository is gated for your account.
#   bash scripts/gpu/00_setup.sh            (CUDA=cu126 for another CUDA build of PyTorch, see scripts/setup.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
cd "$REPO"
mkdir -p "$DATA/eval" "$MODELS" "$OUT" "$RUNS"

log "environment: $REPO/.venv"
bash scripts/setup.sh --dev
export PATH="$HOME/.local/bin:$PATH"
# SpeechBrain ECAPA: speaker clusters, the TLA-SA / speaker-condition stores and protocol v2's second SV metric.
uv pip install --python "$PY" speechbrain

if [[ ! -s $DNSMOS ]]; then
  log "DNSMOS P.835 model -> $DNSMOS"
  curl -fL --retry 3 -o "$DNSMOS" https://github.com/microsoft/DNS-Challenge/raw/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
fi

if [[ ! -s $FREYA ]]; then
  log "Freya-TR-Eval sentences -> $FREYA"
  hf_fetch freyavoice/freya-tr-eval dataset "$(dirname "$FREYA")" freya_tr_eval.jsonl
  [[ $(basename "$FREYA") == freya_tr_eval.jsonl ]] || mv "$(dirname "$FREYA")/freya_tr_eval.jsonl" "$FREYA"
fi

if [[ ! -s $CKPT_C ]]; then
  log "run C checkpoint (VoiceHub/dacvae-tts-tr-w512) -> $CKPT_C"
  hf_fetch VoiceHub/dacvae-tts-tr-w512 model "$(dirname "$CKPT_C")" model.pt config.json README.md
fi

if ! compgen -G "$CV17/tr/test/*/*.parquet" >/dev/null && ! compgen -G "$CV17/tr/test/*.parquet" >/dev/null; then
  log "Common Voice 17 Turkish test split (leak-free prompt voices) -> $CV17"
  hf_fetch fixie-ai/common_voice_17_0 dataset "$CV17" "tr/test/*"
fi

log "checks"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || log "nvidia-smi not found: no GPU visible"
"$PY" -m dacvae_tts inspect --config configs/nano_tr_w512.yaml | "$PY" -c "import json, sys; print('run C parameters', json.load(sys.stdin)['parameters'])"
"$PY" - "$CKPT_C" <<'EOF'
import sys

import torch

saved = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print("run C checkpoint: step", saved.get("step"), "stage", saved.get("stage"))
EOF
log "setup done; next: bash scripts/gpu/10_data.sh"
