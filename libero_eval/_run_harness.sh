#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant is required}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HARNESS_DIR="${ROOT_DIR}/sde_harness"
cd "${ROOT_DIR}"

case "${VARIANT}" in
  standard) TASK_SUITE="${TASK_SUITE:-libero_spatial}" ;;
  plus) TASK_SUITE="${TASK_SUITE:-libero_spatial}" ;;
  pro) TASK_SUITE="${TASK_SUITE:-libero_spatial_swap}" ;;
  *) echo "ERROR: unknown LIBERO variant: ${VARIANT}" >&2; exit 2 ;;
esac

CKPT_PATH="${CKPT_PATH:-${HY_VLA_CHECKPOINT:-}}"
[[ -n "${CKPT_PATH}" ]] || { echo "ERROR: set CKPT_PATH or HY_VLA_CHECKPOINT" >&2; exit 1; }
export PYTHONPATH="${HARNESS_DIR}:${ROOT_DIR}:${PYTHONPATH:-}"
export LIBERO_PYTHON="${LIBERO_PYTHON:-${ROOT_DIR}/libero_eval/.venv/bin/python}"
export SDE_HARNESS_ROOT="${HARNESS_DIR}" HY_VLA_ROOT="${ROOT_DIR}" VLA_ADAPTER_ROOT="${ROOT_DIR}"
export HY_VLA_CHECKPOINT="${CKPT_PATH}" LIBERO_TYPE="${VARIANT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" ACTION_DIM="${ACTION_DIM:-7}"
export GRIPPER_MODE="${GRIPPER_MODE:-libero}" DEVICE="${DEVICE:-cuda}" DTYPE="${DTYPE:-bfloat16}"
export NORM_PATH="${NORM_PATH:-}" VLM_MODEL_PATH="${VLM_MODEL_PATH:-}"

echo "Hy-VLA harness: variant=${VARIANT} suite=${TASK_SUITE} checkpoint=${CKPT_PATH}"
uv run --extra harness python -m rpent.cli.main \
  --env libero --libero-type "${VARIANT}" --suite "${TASK_SUITE}" \
  --task "${TASK:-0}" --seed "${SEED:-0}" --adapter-checkpoint "${CKPT_PATH}" \
  --planner "${PLANNER:-codebuddy}" --model "${MODEL:-hy_a3b}" "${@:2}"
