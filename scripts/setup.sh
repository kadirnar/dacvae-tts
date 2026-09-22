#!/usr/bin/env bash
# One-step environment setup: creates .venv, installs PyTorch for your CUDA build and everything else.
#   bash scripts/setup.sh              # CUDA 12.8 wheels (default)
#   CUDA=cu126 bash scripts/setup.sh   # other CUDA build, or CUDA=cpu
#   bash scripts/setup.sh --dev        # also pytest/ruff
set -euo pipefail
cd "$(dirname "$0")/.."
cuda=${CUDA:-cu128}
torch_version=${TORCH_VERSION:-2.8.0}
extras=""
[[ "${1:-}" == "--dev" ]] && extras="[dev]"
if ! command -v uv >/dev/null; then
  echo "installing uv (https://docs.astral.sh/uv/)" >&2
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[[ -d .venv ]] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python "torch==${torch_version}" "torchaudio==${torch_version}" \
  --index-url "https://download.pytorch.org/whl/${cuda}"
uv pip install --python .venv/bin/python -e ".${extras}"
.venv/bin/dacvae-tts inspect --config configs/nano.yaml >/dev/null && echo "ok: source .venv/bin/activate"
