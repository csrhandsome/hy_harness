#!/usr/bin/env bash
# =============================================================================
# Hy-VLA RoboTwin eval: regress on the first-6 tasks (10 rollouts each)
# on a single GPU.
#
# Usage:
#   bash scripts/eval_robotwin_test.sh
#
# Env overrides (with defaults):
#   TEST_NUM       10
#   CUDA_VISIBLE_DEVICES  0
#   CKPT_PATH      /path/to/Hy-VLA-RoboTwin
#   ROBOTWIN_DIR   /path/to/RoboTwin
#   HY_VLA_DIR     parent of this script
#   LOG_DIR        <HY_VLA_DIR>/eval_logs
# =============================================================================

set -euo pipefail

# --------- 1. defaults ---------
TEST_NUM=${TEST_NUM:-10}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/path/to/RoboTwin}
CKPT_PATH=${CKPT_PATH:-/path/to/Hy-VLA-RoboTwin}
LOG_DIR=${LOG_DIR:-"${HY_VLA_DIR}/eval_logs"}

# --------- 2. Fixed first-6 task list (regression subset) ---------
TASKS=(
    adjust_bottle
    beat_block_hammer
    blocks_ranking_rgb
    blocks_ranking_size
    click_alarmclock
    click_bell
)

# --------- 3. RoboTwin eval_policy.py invariants for Hy-VLA ---------
CKPT_SETTING=Hy-VLA-RoboTwin
TASK_CONFIG=demo_clean
INSTRUCTION_TYPE=unseen
SEED=10000

# --------- 4. Symlink robotwin_eval/ -> RoboTwin/policy/hy_vla (idempotent) ---
ln -sfn "${HY_VLA_DIR}/robotwin_eval" "${ROBOTWIN_DIR}/policy/hy_vla"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${HY_VLA_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# --------- 5. Banner ---------
echo "========================================================"
echo "Hy-VLA RoboTwin quick regression (single GPU)"
echo "Tasks (${#TASKS[@]})    : ${TASKS[*]}"
echo "Rollouts/task : ${TEST_NUM}"
echo "GPU           : ${CUDA_VISIBLE_DEVICES}"
echo "Ckpt path     : ${CKPT_PATH}"
echo "RoboTwin dir  : ${ROBOTWIN_DIR}"
echo "Log dir       : ${LOG_DIR}"
echo "========================================================"

# --------- 6. Sequential evaluation loop ---------
n_done=0
n_failed=0
for task in "${TASKS[@]}"; do
    local log="${LOG_DIR}/${task}.log"
    echo "[${task}] starting ...  (log: ${log})"
    local rc=0
    (
        cd "${ROBOTWIN_DIR}"
        PYTHONWARNINGS=ignore::UserWarning \
        python -u script/eval_policy.py \
            --config policy/hy_vla/deploy_policy.yaml \
            --overrides \
                --task_name "${task}" \
                --task_config "${TASK_CONFIG}" \
                --ckpt_setting "${CKPT_SETTING}" \
                --instruction_type "${INSTRUCTION_TYPE}" \
                --seed "${SEED}" \
                --test_num "${TEST_NUM}" \
                --policy_name policy.hy_vla \
                --ckpt_path "${CKPT_PATH}" \
            2>&1 \
            | sed -u 's/\r/\n/g' \
            | sed -u '/^\(\x1b\[[0-9;]*m\)*step: /d' \
            >"${log}"
    ) || rc=$?
    if [ $rc -eq 0 ]; then
        local rate
        rate=$(grep -a 'Success rate' "${log}" | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' || true)
        echo "[${task}] done.  ${rate}"
        n_done=$((n_done + 1))
    else
        echo "[${task}] FAILED (rc=${rc}); see ${log}"
        n_failed=$((n_failed + 1))
    fi
done

echo "--------------------------------------------------------"
echo "Finished: ${n_done}/${#TASKS[@]} tasks succeeded"
[ $n_failed -gt 0 ] && echo "WARNING: ${n_failed} task(s) failed — check logs in ${LOG_DIR}"
[ $n_failed -gt 0 ] && exit 1
exit 0
