#!/usr/bin/env bash
set -euo pipefail

SERVING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${SERVING_PYTHON:-}" ]]; then
  PYTHON="${SERVING_PYTHON}"
elif [[ -x "${SERVING_ROOT}/.venv/bin/python" ]]; then
  PYTHON="${SERVING_ROOT}/.venv/bin/python"
else
  PYTHON="$(command -v python3)"
fi
export PYTHONPATH="${SERVING_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m vla_serving.cli "$@"
