#!/usr/bin/env bash
# Build versioned Fast/Semantic Critic dataset manifests from Hook event logs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
exec uv run python sde_harness/training/build_critic_dataset.py "$@"
