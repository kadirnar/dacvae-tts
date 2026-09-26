#!/usr/bin/env bash
# Assemble the v2 demo Space (https://huggingface.co/spaces/Vyvo/dacvae-tts-tr-v2-demo) in /workspace/release/space-v2:
# demo/space-v2 (app, engine, card, requirements), the example prompts of the v1 demo and a copy of the package.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
out=${1:-/workspace/release/space-v2}
mkdir -p "$out"
cp demo/space-v2/app.py demo/space-v2/engine.py demo/space-v2/README.md demo/space-v2/requirements.txt "$out/"
rsync -a --delete demo/space/examples/ "$out/examples/"
rsync -a --delete --exclude "__pycache__" src/dacvae_tts/ "$out/dacvae_tts/"
echo "assembled $out"
