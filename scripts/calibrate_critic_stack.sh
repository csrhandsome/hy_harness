#!/usr/bin/env bash
# Calibrate frozen critic checkpoints and emit a deployable stack config.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec uv run python sde_harness/training/calibrate_critic_stack.py "$@"

