#!/usr/bin/env bash
# Unified launcher for original LIBERO, LIBERO-plus, and LIBERO-Pro.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VARIANT="standard"
HARNESS=false
VARIANT_SELECTED=false
FORWARDED_ARGS=()

usage() {
  cat <<'EOF'
Usage: bash libero_eval/eval.sh [selector]... [-- evaluator-arguments...]

Selectors:
  --original, --standard  Evaluate original LIBERO (default).
  --plus                  Evaluate LIBERO-plus.
  --pro                   Evaluate LIBERO-Pro.
  --harness               Run through the SDE Harness instead of the direct evaluator.
  --no-harness            Use the direct evaluator (default).
  -h, --help              Show this help.

All unrecognized arguments are forwarded to the selected evaluator. Use `--`
before evaluator arguments when a forwarded argument could be mistaken for a
selector. Examples:
  POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/eval.sh --plus
  POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/eval.sh --pro -- --max-tasks 2
  CKPT_PATH=/path/to/hy-vla-libero bash libero_eval/eval.sh --pro --harness
EOF
}

select_variant() {
  local requested="$1"
  if [[ "${VARIANT_SELECTED}" == true && "${VARIANT}" != "${requested}" ]]; then
    echo "ERROR: choose only one of --original/--standard, --plus, or --pro" >&2
    exit 2
  fi
  VARIANT="${requested}"
  VARIANT_SELECTED=true
}

while (($#)); do
  case "$1" in
    --original|--standard) select_variant standard ;;
    --plus) select_variant plus ;;
    --pro) select_variant pro ;;
    --harness) HARNESS=true ;;
    --no-harness) HARNESS=false ;;
    -h|--help) usage; exit 0 ;;
    --)
      shift
      FORWARDED_ARGS+=("$@")
      break
      ;;
    *) FORWARDED_ARGS+=("$1") ;;
  esac
  shift
done

run_direct_evaluator() {
  local libero_root task_suite num_trials
  case "${VARIANT}" in
    standard) libero_root="${ROOT_DIR}/third_party/LIBERO" ;;
    plus) libero_root="${ROOT_DIR}/third_party/LIBERO-plus" ;;
    pro) libero_root="${ROOT_DIR}/third_party/LIBERO-PRO" ;;
  esac
  [[ -d "${libero_root}/libero/libero" ]] || {
    echo "ERROR: missing ${libero_root}" >&2
    exit 1
  }

  task_suite="${TASK_SUITE:-libero_spatial}"
  if [[ -n "${NUM_TRIALS:-}" ]]; then
    num_trials="${NUM_TRIALS}"
  elif [[ "${VARIANT}" == plus ]]; then
    num_trials=1
  else
    num_trials=20
  fi

  export PYTHONPATH="${libero_root}:${ROOT_DIR}:${ROOT_DIR}/packages/vla_protocol/src:${PYTHONPATH:-}"
  local args=(
    --policy-endpoint "${POLICY_ENDPOINT:-http://127.0.0.1:8001}"
    --task-suite-name "${task_suite}"
    --num-trials-per-task "${num_trials}"
  )
  [[ "${MAX_TASKS:-0}" != 0 ]] && args+=(--max-tasks "${MAX_TASKS}")
  [[ "${START_TASK_ID:-0}" != 0 ]] && args+=(--start-task-id "${START_TASK_ID}")
  [[ -n "${PERTURBATION_CATEGORY:-}" ]] && args+=(--perturbation-category "${PERTURBATION_CATEGORY}")
  [[ -n "${EVAL_CFG:-}" ]] && args+=(--evaluation-config-path "${EVAL_CFG}")
  [[ "${SAVE_VIDEOS:-0}" =~ ^(1|true|True)$ ]] && args+=(--save-videos)

  echo "Hy-VLA LIBERO eval: variant=${VARIANT} suite=${task_suite} policy=${POLICY_ENDPOINT:-http://127.0.0.1:8001}"
  exec uv run --project "${ROOT_DIR}/libero_eval" python -m libero_eval.runner \
    --variant "${VARIANT}" "${args[@]}" "${FORWARDED_ARGS[@]}"
}

run_harness_evaluator() {
  local task_suite
  case "${VARIANT}" in
    standard|plus) task_suite="${TASK_SUITE:-libero_spatial}" ;;
    pro) task_suite="${TASK_SUITE:-libero_spatial_swap}" ;;
  esac

  local policy_endpoint="${POLICY_ENDPOINT:-http://127.0.0.1:8001}"

  export PYTHONPATH="${ROOT_DIR}/sde_harness:${ROOT_DIR}:${PYTHONPATH:-}"
  export LIBERO_PYTHON="${LIBERO_PYTHON:-${ROOT_DIR}/libero_eval/.venv/bin/python}"
  export SDE_HARNESS_ROOT="${ROOT_DIR}/sde_harness" HY_VLA_ROOT="${ROOT_DIR}" VLA_ADAPTER_ROOT="${ROOT_DIR}"
  export LIBERO_TYPE="${VARIANT}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ACTION_DIM="${ACTION_DIM:-7}"
  export GRIPPER_MODE="${GRIPPER_MODE:-libero}" DEVICE="${DEVICE:-cuda}" DTYPE="${DTYPE:-bfloat16}"
  export NORM_PATH="${NORM_PATH:-}" VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"

  echo "Hy-VLA harness: variant=${VARIANT} suite=${task_suite} policy=${policy_endpoint}"
  exec uv run --project "${ROOT_DIR}/libero_eval" python -m rpent.cli.main \
    --env libero --libero-type "${VARIANT}" --suite "${task_suite}" \
    --task "${TASK:-0}" --seed "${SEED:-0}" \
    --vla-endpoint "${policy_endpoint}" \
    --planner "${PLANNER:-codebuddy}" --model "${MODEL:-hy_a3b}" "${FORWARDED_ARGS[@]}"
}

cd "${ROOT_DIR}"
if [[ "${HARNESS}" == true ]]; then
  run_harness_evaluator
else
  run_direct_evaluator
fi
