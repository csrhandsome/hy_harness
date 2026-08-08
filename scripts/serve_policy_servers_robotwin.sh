#!/usr/bin/env bash
# Start one Hy-VLA RoboTwin action-policy RPC server per GPU.
#
# This runs in the root Hy-VLA environment. serving/.venv belongs to the
# planner VLM (vLLM), while vla-policy-server owns the action model.
set -euo pipefail

HY_HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${HY_HARNESS_ROOT}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
CKPT_PATH="${CKPT_PATH:-${HY_HARNESS_ROOT}/models/Hy-Embodied-0.5-VLA-RoboTwin}"
NORM_PATH="${NORM_PATH:-}"
SERVER_START_TIMEOUT_S="${SERVER_START_TIMEOUT_S:-1800}"

BLEND_MODE="${BLEND_MODE:-rel_abs}"
EXC_ACTION_SIZE="${EXC_ACTION_SIZE:-7}"
IMG_HISTORY_SIZE="${IMG_HISTORY_SIZE:-6}"
IMG_HISTORY_INTERVAL="${IMG_HISTORY_INTERVAL:-5}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/hy-harness-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# This container defines proxy variables but leaves no_proxy empty, which sends
# loopback RPC through the corporate squid proxy and returns HTML instead of JSON.
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
export NO_PROXY="${no_proxy}"

if [[ ! -d "${CKPT_PATH}" ]]; then
  echo "CKPT_PATH does not exist: ${CKPT_PATH}" >&2
  exit 2
fi
if [[ -z "${NORM_PATH}" ]]; then
  NORM_PATH="${CKPT_PATH}/norm_stats.pkl"
fi
if [[ ! -f "${NORM_PATH}" ]]; then
  echo "NORM_PATH does not exist: ${NORM_PATH}" >&2
  exit 2
fi

read -r -a GPU_ARR <<< "${GPUS//,/ }"
NUM_GPUS="${#GPU_ARR[@]}"
if (( NUM_GPUS == 0 )); then
  echo "GPUS is empty" >&2
  exit 2
fi

LOG_DIR="${HY_HARNESS_ROOT}/logs/policy_servers_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "checkpoint: ${CKPT_PATH}"
echo "gpus:       ${GPUS} (${NUM_GPUS} servers)"
echo "ports:      ${BASE_PORT}..$((BASE_PORT + NUM_GPUS - 1))"
echo "logs:       ${LOG_DIR}"
echo
echo "Each server loads its own copy of the weights, so expect this to take"
echo "several minutes and to use ${NUM_GPUS}x the single-server memory."
echo

PIDS=()
cleanup() {
  echo
  echo "shutting down ${#PIDS[@]} policy servers..."
  local pid
  for pid in "${PIDS[@]:-}"; do
    [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  port=$((BASE_PORT + i))
  echo "launching: gpu=${gpu} port=${port} log=${LOG_DIR}/gpu${gpu}.log"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  uv run --locked --no-sync vla-policy-server \
    --benchmark robotwin \
    --checkpoint "${CKPT_PATH}" \
    --norm-path "${NORM_PATH}" \
    --blend-mode "${BLEND_MODE}" \
    --exc-action-size "${EXC_ACTION_SIZE}" \
    --img-history-size "${IMG_HISTORY_SIZE}" \
    --img-history-interval "${IMG_HISTORY_INTERVAL}" \
    --host "${HOST}" \
    --port "${port}" \
    > "${LOG_DIR}/gpu${gpu}.log" 2>&1 &
  PIDS+=("$!")
done

echo
echo "waiting for readiness (up to ${SERVER_START_TIMEOUT_S}s)..."
deadline=$((SECONDS + SERVER_START_TIMEOUT_S))
for i in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$i]}"
  port=$((BASE_PORT + i))
  until curl -sS --noproxy '*' --max-time 5 \
        -X POST "http://${HOST}:${port}/call" \
        -H 'Content-Type: application/json' \
        -d '{"id":"probe","method":"metadata","args":[],"kwargs":{}}' \
        2>/dev/null | grep -q '"ok": *true'; do
    if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
      echo "  gpu=${gpu} port=${port}: FAILED. Tail of its log:" >&2
      tail -25 "${LOG_DIR}/gpu${gpu}.log" >&2
      exit 3
    fi
    if (( SECONDS > deadline )); then
      echo "  gpu=${gpu} port=${port}: not ready within ${SERVER_START_TIMEOUT_S}s" >&2
      exit 3
    fi
    sleep 5
  done
  echo "  gpu=${gpu} port=${port}: ready"
done

echo
echo "all ${NUM_GPUS} policy servers ready."
echo "Now run the evaluation in another terminal:"
echo "  GPUS=${GPUS} BASE_PORT=${BASE_PORT} bash robotwin_eval/run_eval.sh"
echo "Press Ctrl-C here to stop them."
echo

# Hold the servers. If any one dies, report it and shut the rest down rather
# than leaving the evaluator talking to a half-dead fleet.
wait -n
echo "a policy server exited unexpectedly; see ${LOG_DIR}/" >&2
exit 3