#!/usr/bin/env bash
# Start Hy-Embodied-VLM for the embedded Harness / CodeBuddy planner.
#
# Unlike `serve.sh`, this launcher keeps raw vLLM on a private adjacent port
# and exposes a public OpenAI-compatible endpoint through the local
# tool-choice compatibility proxy.
set -euo pipefail

SERVING_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Reuse the common serving environment, including SETUPTOOLS_USE_DISTUTILS,
# PYTHONPATH, and the SERVING_PYTHON-aware interpreter selection.
# shellcheck source=_run.sh
source "${SERVING_ROOT}/scripts/_run.sh"

# CodeBuddy supplies the Harness MCP schemas but leaves ``tool_choice`` at its
# OpenAI default ("auto"). Hy-Embodied-VLM reliably emits a parseable HYV3
# tool call only for ``required``. Keep vLLM on a private adjacent port and
# expose a tiny local proxy on the public planner port; it forces ``required``
# only for requests containing a Harness mcp__*__* tool.
PUBLIC_PORT="${PORT:-8080}"
VLLM_BACKEND_PORT="${VLLM_BACKEND_PORT:-$((PUBLIC_PORT + 1))}"

cleanup() {
  if [[ -n "${VLLM_PID:-}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null || true
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

PORT="${VLLM_BACKEND_PORT}" \
  bash "${SERVING_ROOT}/scripts/serve.sh" --model hy-embodied-vlm-1.0 &
VLLM_PID=$!

echo "[serving] tool-choice proxy: public=${PUBLIC_PORT} backend=${VLLM_BACKEND_PORT}"
"${PYTHON}" -m vla_serving.runtimes.vllm.tool_choice_proxy \
  --host "${HOST:-0.0.0.0}" \
  --port "${PUBLIC_PORT}" \
  --backend-url "http://127.0.0.1:${VLLM_BACKEND_PORT}"