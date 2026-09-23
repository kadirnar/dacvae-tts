#!/usr/bin/env bash
# Copy the dacvae_tts package (with the Turkish frontend, duration predictor and sampler options) into the Space
# and the model repository bundles.
set -euo pipefail
for target in /workspace/release/space/dacvae_tts /workspace/release/model/dacvae_tts; do
  mkdir -p "$target"
  rsync -a --delete --exclude "__pycache__" /workspace/dacvae-tts/src/dacvae_tts/ "$target/"
done
echo "synced: $(ls /workspace/release/space/dacvae_tts | wc -l) files"
