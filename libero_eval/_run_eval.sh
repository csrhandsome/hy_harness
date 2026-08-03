#!/usr/bin/env bash
set -euo pipefail
VARIANT="${1:?variant is required}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
case "${VARIANT}" in
  standard) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO"; PY_ENTRY="libero_eval/run_libero_eval.py" ;;
  plus) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO-plus"; PY_ENTRY="libero_eval/run_libero_plus_eval.py" ;;
  pro) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO-PRO"; PY_ENTRY="libero_eval/run_libero_pro_eval.py" ;;
  *) echo "ERROR: unknown LIBERO variant: ${VARIANT}" >&2; exit 2 ;;
esac
[[ -d "${LIBERO_ROOT}/libero/libero" ]] || { echo "ERROR: missing ${LIBERO_ROOT}" >&2; exit 1; }
CKPT_PATH="${CKPT_PATH:-${HY_VLA_CHECKPOINT:-}}"
[[ -n "${CKPT_PATH}" ]] || { echo "ERROR: set CKPT_PATH or HY_VLA_CHECKPOINT" >&2; exit 1; }
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
if [[ -n "${NUM_TRIALS:-}" ]]; then NUM_TRIALS_VALUE="${NUM_TRIALS}"
elif [[ "${VARIANT}" == plus ]]; then NUM_TRIALS_VALUE=1
else NUM_TRIALS_VALUE=20; fi
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="${LIBERO_ROOT}:${ROOT_DIR}:${PYTHONPATH:-}"
ARGS=(--checkpoint "${CKPT_PATH}" --task-suite-name "${TASK_SUITE}"
  --num-trials-per-task "${NUM_TRIALS_VALUE}" --action-dim "${ACTION_DIM:-7}"
  --gripper-mode "${GRIPPER_MODE:-libero}" --device "${DEVICE:-cuda}"
  --dtype "${DTYPE:-bfloat16}")
[[ -n "${NORM_PATH:-}" ]] && ARGS+=(--norm-path "${NORM_PATH}")
[[ -n "${VLM_MODEL_PATH:-}" ]] && ARGS+=(--vlm-model-path "${VLM_MODEL_PATH}")
[[ -n "${NUM_OPEN_LOOP_STEPS:-}" ]] && ARGS+=(--num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}")
[[ -n "${IMG_HISTORY_SIZE:-}" ]] && ARGS+=(--img-history-size "${IMG_HISTORY_SIZE}")
[[ -n "${IMG_HISTORY_INTERVAL:-}" ]] && ARGS+=(--img-history-interval "${IMG_HISTORY_INTERVAL}")
[[ "${MAX_TASKS:-0}" != 0 ]] && ARGS+=(--max-tasks "${MAX_TASKS}")
[[ "${START_TASK_ID:-0}" != 0 ]] && ARGS+=(--start-task-id "${START_TASK_ID}")
[[ -n "${PERTURBATION_CATEGORY:-}" ]] && ARGS+=(--perturbation-category "${PERTURBATION_CATEGORY}")
[[ -n "${EVAL_CFG:-}" ]] && ARGS+=(--evaluation-config-path "${EVAL_CFG}")
[[ "${SAVE_VIDEOS:-0}" =~ ^(1|true|True)$ ]] && ARGS+=(--save-videos)
echo "Hy-VLA LIBERO eval: variant=${VARIANT} suite=${TASK_SUITE} checkpoint=${CKPT_PATH}"
uv run --extra libero python "${PY_ENTRY}" "${ARGS[@]}" "${@:2}"
