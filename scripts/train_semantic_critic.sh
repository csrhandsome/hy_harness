#!/usr/bin/env bash
# Offline slow Semantic Critic training. Pass all CLI arguments through.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec uv run python sde_harness/training/train_semantic_critic.py "$@"

