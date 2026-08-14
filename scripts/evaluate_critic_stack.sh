#!/usr/bin/env bash
# Evaluate per-level and routed Critic Stack metrics. Pass all CLI arguments through.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec uv run python sde_harness/training/evaluate_critic_stack.py "$@"
