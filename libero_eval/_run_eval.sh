#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:?variant is required}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

case "${VARIANT}" in
  standard) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO"; PY_ENTRY="libero_eval/run_libero_eval.py" ;;
  plus) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO-plus"; PY_ENTRY="libero_eval/run_libero_plus_eval.py" ;;
  pro) LIBERO_ROOT="${ROOT_DIR}/third_party/LIBERO-PRO"; PY_ENTRY="libero_eval/run_libero_pro_eval.py" ;;
  *) echo "ERROR: unknown LIBERO variant: ${VARIANT}" >&2; exit 2 ;;
esac

[[ -d "${LIBERO_ROOT}/libero/libero" ]] || { echo "ERROR: missing ${LIBERO_ROOT}" >&2; exit 1; }
TASK_SUITE="${TASK_SUITE:-libero_spatial}"
POLICY_ENDPOINT="${POLICY_ENDPOINT:-http://127.0.0.1:8001}"
if [[ -n "${NUM_TRIALS:-}" ]]; then NUM_TRIALS_VALUE="${NUM_TRIALS}"
elif [[ "${VARIANT}" == plus ]]; then NUM_TRIALS_VALUE=1
else NUM_TRIALS_VALUE=20; fi

export PYTHONPATH="${LIBERO_ROOT}:${ROOT_DIR}:${ROOT_DIR}/packages/vla_protocol/src:${PYTHONPATH:-}"
ARGS=(--policy-endpoint "${POLICY_ENDPOINT}" --task-suite-name "${TASK_SUITE}"
  --num-trials-per-task "${NUM_TRIALS_VALUE}")
[[ "${MAX_TASKS:-0}" != 0 ]] && ARGS+=(--max-tasks "${MAX_TASKS}")
[[ "${START_TASK_ID:-0}" != 0 ]] && ARGS+=(--start-task-id "${START_TASK_ID}")
[[ -n "${PERTURBATION_CATEGORY:-}" ]] && ARGS+=(--perturbation-category "${PERTURBATION_CATEGORY}")
[[ -n "${EVAL_CFG:-}" ]] && ARGS+=(--evaluation-config-path "${EVAL_CFG}")
[[ "${SAVE_VIDEOS:-0}" =~ ^(1|true|True)$ ]] && ARGS+=(--save-videos)

echo "Hy-VLA LIBERO eval: variant=${VARIANT} suite=${TASK_SUITE} policy=${POLICY_ENDPOINT}"
uv run --project "${ROOT_DIR}/libero_eval" python "${PY_ENTRY}" "${ARGS[@]}" "${@:2}"
