#!/usr/bin/env bash
set -euo pipefail
export PREPARE_DEVICE=cpu PREPARE_PROCESSES=${PREPARE_PROCESSES:-2}
exec bash "$(dirname "${BASH_SOURCE[0]}")/prepare_parallel.sh" "$@"
