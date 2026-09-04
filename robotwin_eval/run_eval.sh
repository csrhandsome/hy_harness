#!/usr/bin/env bash
# Terminal 3: evaluate many RoboTwin tasks in parallel, one worker per GPU.
#
# script/eval_policy.py is strictly serial (a single `while succ_seed <
# test_num` loop), so one simulator cannot saturate a GPU. The parallel axis is
# independent tasks: this hands the task list to N workers, each pinned to one
# GPU and talking to that GPU's action server.
#
# This does NOT start the action servers. Run
# `bash scripts/serve_policy_servers_robotwin.sh` first; it owns the
# server fleet and uses the same GPU->port mapping as this script.
#
# Usage:
#   bash robotwin_eval/run_eval.sh
#   TASKS="adjust_bottle lift_pot" TEST_NUM=20 bash robotwin_eval/run_eval.sh
#   GPUS="0,1,2,3" bash robotwin_eval/run_eval.sh
#   DRY_RUN=1 bash robotwin_eval/run_eval.sh  # validate only
#
# Use GPUS=0 and one task in TASKS for a single-task debug run.
set -euo pipefail

HY_HARNESS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_EVAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${HY_HARNESS_ROOT}"

ROBOTWIN_DIR="${ROBOTWIN_DIR:-"${HY_HARNESS_ROOT}/../RoboTwin"}"

# Must match scripts/serve_policy_servers_robotwin.sh: GPU i uses port BASE_PORT + i.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BASE_PORT="${BASE_PORT:-8001}"
HOST="${HOST:-127.0.0.1}"

TEST_NUM="${TEST_NUM:-10}"
TASK_CONFIG="${TASK_CONFIG:-demo_clean}"
SEED="${SEED:-10000}"
POLICY_TIMEOUT_S="${POLICY_TIMEOUT_S:-120}"
HARNESS_ENABLED="${HARNESS_ENABLED:-1}"
HARNESS_PLANNER="${HARNESS_PLANNER:-codebuddy}"
HARNESS_MODEL="${HARNESS_MODEL:-hy_a3b}"
HARNESS_MAX_TURNS="${HARNESS_MAX_TURNS:-100}"
HARNESS_TIMEOUT_S="${HARNESS_TIMEOUT_S:-1200}"

export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/hy-harness-robotwin-venv}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplconfig-robotwin}"
mkdir -p "${MPLCONFIGDIR}"

# shellcheck source=internal/docker_env.sh
source "${ROBOTWIN_EVAL_ROOT}/internal/docker_env.sh"

export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost,::1"
export NO_PROXY="${no_proxy}"
# The CodeBuddy headless binary may otherwise inherit an ASCII-compatible
# locale from batch launchers. Its OpenAI-compatible stream can contain
# non-ASCII text, so force UTF-8 for both the simulator and SDK child CLI.
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONUTF8=1

if [[ ! -d "${ROBOTWIN_DIR}/envs" ]]; then
  echo "ROBOTWIN_DIR is not an upstream RoboTwin checkout: ${ROBOTWIN_DIR}" >&2
  exit 2
fi

# Task names must match envs/<task_name>.py, which is what --task_name selects.
if [[ -z "${TASKS:-}" ]]; then
  TASKS="$(cd "${ROBOTWIN_DIR}" && ls envs/ 2>/dev/null \
           | grep -E '\.py$' \
           | sed 's/\.py$//' \
           | grep -v -E '^(_|base_task$|utils$)' \
           | tr '\n' ' ')" || true
fi
if [[ -z "${TASKS// /}" ]]; then
  echo "Could not discover any task from ${ROBOTWIN_DIR}/envs/." >&2
  echo "Pass them explicitly, e.g. TASKS=\"adjust_bottle lift_pot\"" >&2
  exit 2
fi

read -r -a GPU_ARR <<< "${GPUS//,/ }"
read -r -a TASK_ARR <<< "${TASKS}"
NUM_GPUS="${#GPU_ARR[@]}"
NUM_TASKS="${#TASK_ARR[@]}"

if (( NUM_GPUS == 0 || NUM_TASKS == 0 )); then
  echo "nothing to do (gpus=${NUM_GPUS} tasks=${NUM_TASKS})" >&2
  exit 2
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${HY_HARNESS_ROOT}/logs/parallel_eval_${RUN_ID}"

echo "run id:     ${RUN_ID}"
echo "gpus:       ${GPUS} (${NUM_GPUS} workers)"
echo "ports:      ${BASE_PORT}..$((BASE_PORT + NUM_GPUS - 1))"
echo "tasks:      ${NUM_TASKS}"
echo "test_num:   ${TEST_NUM} episodes per task"
echo "logs:       ${LOG_DIR}"
echo

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  for i in "${!GPU_ARR[@]}"; do
    echo "worker: gpu=${GPU_ARR[$i]} -> http://${HOST}:$((BASE_PORT + i))"
  done
  printf 'task: %s\n' "${TASK_ARR[@]}"
  echo "DRY_RUN=1: no GPU work performed."
  exit 0
fi

echo "verifying RoboTwin evaluation environment..."
uv --project "${ROBOTWIN_EVAL_ROOT}" run --locked --no-sync \
  python "${ROBOTWIN_EVAL_ROOT}/internal/setup_sim.py" \
  --robotwin-dir "${ROBOTWIN_DIR}"

mkdir -p "${LOG_DIR}"

# The Docker graphics workaround is per-container; do it once so N workers do
# not race to download the same package.
if [[ ! -f /tmp/hy-harness-nvidia-gfx/lib/libGLX_nvidia.so.0 ]]; then
  bash "${ROBOTWIN_EVAL_ROOT}/internal/docker_env.sh" install
fi

# Fail fast and with a clear reason if the server fleet is not up: otherwise
# every worker would independently fail deep inside RoboTwin startup.
echo "checking policy servers..."
for i in "${!GPU_ARR[@]}"; do
  port=$((BASE_PORT + i))
  if ! curl -sS --noproxy '*' --max-time 5 \
       -X POST "http://${HOST}:${port}/call" \
       -H 'Content-Type: application/json' \
       -d '{"id":"probe","method":"metadata","args":[],"kwargs":{}}' \
       2>/dev/null | grep -q '"ok": *true'; then
    echo "  gpu=${GPU_ARR[$i]} port=${port}: NOT responding" >&2
    echo >&2
    echo "Start the server fleet first, in another terminal:" >&2
    echo "  GPUS=${GPUS} BASE_PORT=${BASE_PORT} bash scripts/serve_policy_servers_robotwin.sh" >&2
    echo >&2
    echo "It must use the same GPUS/BASE_PORT as this run" >&2
    echo "(GPUS=${GPUS} BASE_PORT=${BASE_PORT})." >&2
    exit 3
  fi
  echo "  gpu=${GPU_ARR[$i]} port=${port}: ok"
done

# All workers share one symlink; create it once rather than racing on it.
# remote_policy.py resolves vla_protocol from the adjacent root checkout if it
# is not installed in the simulator environment.
POLICY_LINK="${ROBOTWIN_DIR}/policy/hy_vla"
if [[ -L "${POLICY_LINK}" || ! -e "${POLICY_LINK}" ]]; then
  ln -sfn "${HY_HARNESS_ROOT}/robotwin_eval" "${POLICY_LINK}"
else
  echo "Refusing to replace non-symlink policy path: ${POLICY_LINK}" >&2
  exit 2
fi

# ---------------------------------------------------------------------------
# Workers pull from a shared queue, so a slow task does not idle other GPUs
# the way a fixed round-robin split would.
#
# The queue lives on local disk, not the shared filesystem: LOG_DIR is on
# fuseblk (cephfs), where advisory locking across processes is not dependable.
# Each pop runs inside `flock <lockfile> bash -c ...` so the read-and-truncate
# is one atomic critical section -- an inherited-fd `flock 9` is not reliable
# across `&`-forked subshells.
# ---------------------------------------------------------------------------
QUEUE="/tmp/hy-harness-eval-queue-${RUN_ID}"
LOCK="${QUEUE}.lock"
printf '%s\n' "${TASK_ARR[@]}" > "${QUEUE}"
: > "${LOCK}"
# Keep a copy next to the logs for post-mortem, but never read it back.
cp "${QUEUE}" "${LOG_DIR}/queue.initial"

cleanup_queue() { rm -f "${QUEUE}" "${QUEUE}.tmp" "${LOCK}"; }
trap cleanup_queue EXIT

pop_task() {
  flock "${LOCK}" bash -c '
    q="$1"
    [[ -s "${q}" ]] || exit 0
    head -1 "${q}"
    tail -n +2 "${q}" > "${q}.tmp"
    mv "${q}.tmp" "${q}"
  ' _ "${QUEUE}"
}

run_worker() {
  local gpu="$1" port="$2"
  local task log
  while :; do
    task="$(pop_task)"
    [[ -z "${task}" ]] && break

    echo "[gpu${gpu}] start ${task}"
    log="${LOG_DIR}/task_${task}.log"
    if (
      cd "${ROBOTWIN_DIR}"
      CUDA_VISIBLE_DEVICES="${gpu}" \
      POLICY_ENDPOINT="http://${HOST}:${port}" \
      SDE_HARNESS_ROOT="${HY_HARNESS_ROOT}/sde_harness" \
      HY_VLA_ROOT="${HY_HARNESS_ROOT}" \
      ROBOTWIN_HARNESS="${HARNESS_ENABLED}" \
      ROBOTWIN_HARNESS_PLANNER="${HARNESS_PLANNER}" \
      ROBOTWIN_HARNESS_MODEL="${HARNESS_MODEL}" \
      ROBOTWIN_HARNESS_MAX_TURNS="${HARNESS_MAX_TURNS}" \
      ROBOTWIN_HARNESS_TIMEOUT_S="${HARNESS_TIMEOUT_S}" \
      CODEBUDDY_TIMEOUT_S="${HARNESS_TIMEOUT_S}" \
      PYTHONUTF8="${PYTHONUTF8}" \
      LANG="${LANG}" \
      LC_ALL="${LC_ALL}" \
      PYTHONWARNINGS="ignore::UserWarning" \
      uv --project "${ROBOTWIN_EVAL_ROOT}" run --locked --no-sync \
        python -u script/eval_policy.py \
          --config policy/hy_vla/deploy_policy.yaml \
          --overrides \
            --task_name "${task}" \
            --task_config "${TASK_CONFIG}" \
            --ckpt_setting "Hy-VLA-RoboTwin" \
            --instruction_type "unseen" \
            --seed "${SEED}" \
            --test_num "${TEST_NUM}" \
            --policy_name "policy.hy_vla" \
            --policy_endpoint "http://${HOST}:${port}" \
            --policy_timeout_s "${POLICY_TIMEOUT_S}" \
            --harness.enabled "${HARNESS_ENABLED}" \
            --harness.planner "${HARNESS_PLANNER}" \
            --harness.model "${HARNESS_MODEL}" \
            --harness.max_turns "${HARNESS_MAX_TURNS}" \
            --harness.planner_timeout_s "${HARNESS_TIMEOUT_S}"
    ) > "${log}" 2>&1; then
      echo "[gpu${gpu}] done  ${task}"
      echo "${task} OK" >> "${LOG_DIR}/status.txt"
    else
      echo "[gpu${gpu}] FAIL  ${task} (see ${log})"
      echo "${task} FAIL" >> "${LOG_DIR}/status.txt"
    fi
  done
}

WORKER_PIDS=()
for i in "${!GPU_ARR[@]}"; do
  run_worker "${GPU_ARR[$i]}" "$((BASE_PORT + i))" &
  WORKER_PIDS+=("$!")
done

echo
echo "${NUM_GPUS} workers running over ${NUM_TASKS} tasks."
echo "watch progress:  tail -f ${LOG_DIR}/task_*.log"
echo

for pid in "${WORKER_PIDS[@]}"; do
  wait "${pid}" || true
done

echo
echo "=== results ==="
SUMMARY="${LOG_DIR}/summary.txt"
: > "${SUMMARY}"
for task in "${TASK_ARR[@]}"; do
  rate="$(find "${ROBOTWIN_DIR}/eval_result/${task}" -name '_result.txt' \
          -newermt "-1 day" 2>/dev/null | sort | tail -1 | xargs -r tail -1)"
  printf '%-34s %s\n' "${task}" "${rate:-no result}" | tee -a "${SUMMARY}"
done
echo
echo "summary: ${SUMMARY}"
echo "logs:    ${LOG_DIR}"