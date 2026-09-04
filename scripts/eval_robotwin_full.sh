#!/usr/bin/env bash
# =============================================================================
# Hy-VLA RoboTwin eval: 50 tasks x {demo_clean, demo_randomized} = 100 sub-tasks
# with a FIFO work-stealing queue across N GPUs, sub-tasks pre-sorted by
# estimated duration (descending) so workers pick long jobs first (~LPT).
#
# Usage:
#   bash scripts/eval_robotwin_full.sh [MAX_GPUS]
#     defaults: MAX_GPUS=8  -> uses GPU ids 0..MAX_GPUS-1
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3 TEST_NUM=20 \
#       bash scripts/eval_robotwin_full.sh 4
#
# Env overrides (with defaults):
#   TEST_NUM              100
#   CKPT_PATH             /path/to/Hy-VLA-RoboTwin
#   ROBOTWIN_DIR          /path/to/RoboTwin
#   HY_VLA_DIR            parent of this script
#   LOG_DIR               <HY_VLA_DIR>/eval_logs_full_50task
#   BLEND_MODE            rel_abs
#   EXC_ACTION_SIZE       7
#   IMG_HISTORY_SIZE      6
#   IMG_HISTORY_INTERVAL  5
#   NORM_PATH             "" (auto: <CKPT_PATH>/norm_stats.pkl)
#   TASKS                 "" (override 50-task list, space-sep)
#   TASK_CONFIGS          "demo_clean demo_randomized" (space-sep)
# =============================================================================

set -euo pipefail

# --------- 1. CLI args + defaults ---------
MAX_GPUS=${1:-8}

if ! [[ "${MAX_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[error] MAX_GPUS must be a positive integer, got: '${MAX_GPUS}'" >&2
    exit 2
fi

TEST_NUM=${TEST_NUM:-100}
HY_VLA_DIR=${HY_VLA_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
ROBOTWIN_DIR=${ROBOTWIN_DIR:-/path/to/RoboTwin}
CKPT_PATH=${CKPT_PATH:-/path/to/Hy-VLA-RoboTwin}
LOG_DIR=${LOG_DIR:-"${HY_VLA_DIR}/eval_logs_full_50task"}

BLEND_MODE=${BLEND_MODE:-rel_abs}
EXC_ACTION_SIZE=${EXC_ACTION_SIZE:-7}
IMG_HISTORY_SIZE=${IMG_HISTORY_SIZE:-6}
IMG_HISTORY_INTERVAL=${IMG_HISTORY_INTERVAL:-5}

case "${BLEND_MODE}" in
    rel_abs|rel_only|abs_only) ;;
    *) echo "[error] BLEND_MODE must be one of rel_abs|rel_only|abs_only, got '${BLEND_MODE}'" >&2; exit 2 ;;
esac

NORM_PATH=${NORM_PATH:-""}
if [ -n "${NORM_PATH}" ] && [ ! -f "${NORM_PATH}" ]; then
    echo "[error] NORM_PATH set but file not found: ${NORM_PATH}" >&2
    exit 2
fi

# --------- 2. Task list and configs ---------
if [ -n "${TASKS:-}" ]; then
    # shellcheck disable=SC2206
    TASKS=( ${TASKS} )
else
    TASKS=(
        adjust_bottle beat_block_hammer blocks_ranking_rgb blocks_ranking_size
        click_alarmclock click_bell dump_bin_bigbin grab_roller
        handover_block handover_mic hanging_mug lift_pot
        move_can_pot move_pillbottle_pad move_playingcard_away move_stapler_pad
        open_laptop open_microwave pick_diverse_bottles pick_dual_bottles
        place_a2b_left place_a2b_right place_bread_basket place_bread_skillet
        place_burger_fries place_can_basket place_cans_plasticbox place_container_plate
        place_dual_shoes place_empty_cup place_fan place_mouse_pad
        place_object_basket place_object_scale place_object_stand place_phone_stand
        place_shoe press_stapler put_bottles_dustbin put_object_cabinet
        rotate_qrcode scan_object shake_bottle shake_bottle_horizontally
        stack_blocks_three stack_blocks_two stack_bowls_three stack_bowls_two
        stamp_seal turn_switch
    )
fi

if [ -n "${TASK_CONFIGS:-}" ]; then
    # shellcheck disable=SC2206
    TASK_CONFIGS=( ${TASK_CONFIGS} )
else
    TASK_CONFIGS=( demo_clean demo_randomized )
fi

# --------- 3. RoboTwin eval_policy.py invariants for Hy-VLA ---------
CKPT_SETTING=Hy-VLA-RoboTwin
INSTRUCTION_TYPE=unseen
SEED=10000

# --------- 4. Symlink robotwin_eval/ -> RoboTwin/policy/hy_vla (idempotent) ---
ln -sfn "${HY_VLA_DIR}/robotwin_eval" "${ROBOTWIN_DIR}/policy/hy_vla"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${HY_VLA_DIR}:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.4

# --------- 5. Build (task,cfg) sub-task list, sorted by est duration desc ---
QUEUE_FILE=$(mktemp -t hyvla_queue_50task.XXXXXX)
LOCK_FILE="${QUEUE_FILE}.lock"
: >"${LOCK_FILE}"
trap 'rm -f "${QUEUE_FILE}" "${QUEUE_FILE}.tmp" "${LOCK_FILE}"' EXIT

python3 - "${QUEUE_FILE}" "${TASKS[@]}" "--cfgs" "${TASK_CONFIGS[@]}" <<'PY_EOF'
import sys

argv = sys.argv[1:]
queue_file = argv[0]
rest = argv[1:]
sep = rest.index("--cfgs")
tasks = rest[:sep]
cfgs = rest[sep + 1:]

# Estimated n=10 wall-clock seconds (rounded to 100s).
EST_DUR_10 = {
    ("hanging_mug", "demo_randomized"): 6000,
    ("open_microwave", "demo_randomized"): 5000,
    ("hanging_mug", "demo_clean"): 4500,
    ("put_bottles_dustbin", "demo_randomized"): 3900,
    ("put_bottles_dustbin", "demo_clean"): 2900,
    ("stack_bowls_three", "demo_clean"): 2600,
    ("turn_switch", "demo_randomized"): 2400,
    ("put_object_cabinet", "demo_randomized"): 2100,
    ("place_object_basket", "demo_clean"): 2100,
    ("open_microwave", "demo_clean"): 2000,
    ("stack_bowls_three", "demo_randomized"): 2000,
    ("blocks_ranking_size", "demo_randomized"): 2000,
    ("handover_block", "demo_randomized"): 1800,
    ("blocks_ranking_size", "demo_clean"): 1800,
    ("handover_mic", "demo_clean"): 1700,
    ("scan_object", "demo_clean"): 1700,
    ("handover_mic", "demo_randomized"): 1600,
    ("place_can_basket", "demo_randomized"): 1600,
    ("place_object_basket", "demo_randomized"): 1400,
    ("place_bread_skillet", "demo_clean"): 1400,
    ("place_a2b_right", "demo_clean"): 1400,
    ("place_can_basket", "demo_clean"): 1400,
    ("blocks_ranking_rgb", "demo_randomized"): 1300,
    ("pick_diverse_bottles", "demo_clean"): 1300,
    ("turn_switch", "demo_clean"): 1300,
    ("handover_block", "demo_clean"): 1200,
    ("stack_blocks_three", "demo_randomized"): 1200,
    ("put_object_cabinet", "demo_clean"): 1200,
    ("blocks_ranking_rgb", "demo_clean"): 1100,
    ("stack_blocks_three", "demo_clean"): 1100,
    ("place_a2b_left", "demo_randomized"): 1100,
    ("place_a2b_right", "demo_randomized"): 1000,
    ("pick_dual_bottles", "demo_clean"): 1000,
    ("scan_object", "demo_randomized"): 1000,
    ("place_cans_plasticbox", "demo_randomized"): 1000,
    ("stack_bowls_two", "demo_randomized"): 1000,
    ("place_dual_shoes", "demo_randomized"): 900,
    ("stamp_seal", "demo_clean"): 900,
    ("place_object_scale", "demo_clean"): 900,
    ("move_pillbottle_pad", "demo_randomized"): 900,
    ("stack_blocks_two", "demo_randomized"): 900,
    ("place_cans_plasticbox", "demo_clean"): 900,
    ("place_phone_stand", "demo_clean"): 900,
    ("dump_bin_bigbin", "demo_clean"): 900,
    ("pick_diverse_bottles", "demo_randomized"): 900,
    ("stack_bowls_two", "demo_clean"): 800,
    ("place_dual_shoes", "demo_clean"): 800,
    ("pick_dual_bottles", "demo_randomized"): 800,
    ("place_burger_fries", "demo_randomized"): 800,
    ("stamp_seal", "demo_randomized"): 800,
    ("stack_blocks_two", "demo_clean"): 800,
    ("place_burger_fries", "demo_clean"): 800,
    ("move_playingcard_away", "demo_randomized"): 700,
    ("dump_bin_bigbin", "demo_randomized"): 700,
    ("place_fan", "demo_clean"): 700,
    ("place_bread_basket", "demo_randomized"): 700,
    ("rotate_qrcode", "demo_clean"): 700,
    ("move_stapler_pad", "demo_clean"): 700,
    ("place_bread_basket", "demo_clean"): 700,
    ("place_a2b_left", "demo_clean"): 700,
    ("press_stapler", "demo_randomized"): 700,
    ("move_can_pot", "demo_randomized"): 700,
    ("open_laptop", "demo_randomized"): 700,
    ("rotate_qrcode", "demo_randomized"): 600,
    ("move_can_pot", "demo_clean"): 600,
    ("place_bread_skillet", "demo_randomized"): 600,
    ("move_stapler_pad", "demo_randomized"): 600,
    ("place_shoe", "demo_randomized"): 600,
    ("place_object_scale", "demo_randomized"): 600,
    ("move_pillbottle_pad", "demo_clean"): 600,
    ("open_laptop", "demo_clean"): 600,
    ("adjust_bottle", "demo_randomized"): 600,
    ("lift_pot", "demo_randomized"): 600,
    ("place_empty_cup", "demo_randomized"): 600,
    ("place_fan", "demo_randomized"): 600,
    ("place_mouse_pad", "demo_clean"): 600,
    ("place_mouse_pad", "demo_randomized"): 600,
    ("place_object_stand", "demo_randomized"): 500,
    ("lift_pot", "demo_clean"): 500,
    ("place_shoe", "demo_clean"): 500,
    ("place_container_plate", "demo_randomized"): 500,
    ("adjust_bottle", "demo_clean"): 500,
    ("place_empty_cup", "demo_clean"): 500,
    ("beat_block_hammer", "demo_randomized"): 500,
    ("place_phone_stand", "demo_randomized"): 500,
    ("shake_bottle", "demo_randomized"): 500,
    ("shake_bottle_horizontally", "demo_randomized"): 500,
    ("place_object_stand", "demo_clean"): 500,
    ("press_stapler", "demo_clean"): 500,
    ("place_container_plate", "demo_clean"): 500,
    ("grab_roller", "demo_randomized"): 500,
    ("shake_bottle_horizontally", "demo_clean"): 500,
    ("shake_bottle", "demo_clean"): 500,
    ("beat_block_hammer", "demo_clean"): 400,
    ("move_playingcard_away", "demo_clean"): 400,
    ("grab_roller", "demo_clean"): 400,
    ("click_alarmclock", "demo_randomized"): 400,
    ("click_bell", "demo_randomized"): 400,
    ("click_alarmclock", "demo_clean"): 300,
    ("click_bell", "demo_clean"): 300,
}
DEFAULT_DUR = 1500

sub = [(t, c, EST_DUR_10.get((t, c), DEFAULT_DUR)) for t in tasks for c in cfgs]
sub.sort(key=lambda x: -x[2])

with open(queue_file, "w") as f:
    for t, c, d in sub:
        f.write(f"{t}\t{c}\t{d}\n")
PY_EOF

NUM_SUBTASKS=$(wc -l <"${QUEUE_FILE}")
TOTAL_EST_S=$(awk -F'\t' '{s+=$3} END{print s+0}' "${QUEUE_FILE}")
LONGEST_EST_S=$(head -n 1 "${QUEUE_FILE}" | awk -F'\t' '{print $3+0}')

pop_task() {
    local fd
    exec {fd}>"${LOCK_FILE}"
    flock "${fd}"
    local line=""
    if [ -s "${QUEUE_FILE}" ]; then
        line=$(head -n 1 "${QUEUE_FILE}")
        tail -n +2 "${QUEUE_FILE}" >"${QUEUE_FILE}.tmp"
        mv "${QUEUE_FILE}.tmp" "${QUEUE_FILE}"
    fi
    flock -u "${fd}"
    exec {fd}>&-
    printf '%s' "${line}"
}

# --------- 6. Per-GPU worker: drain the queue ---------
run_worker() {
    local gpu_id=$1
    local n_done=0
    while :; do
        local line
        line=$(pop_task)
        [ -z "${line}" ] && break

        local task task_config est_dur
        task=$(printf '%s' "${line}" | awk -F'\t' '{print $1}')
        task_config=$(printf '%s' "${line}" | awk -F'\t' '{print $2}')
        est_dur=$(printf '%s' "${line}" | awk -F'\t' '{print $3}')

        local log="${LOG_DIR}/${task}_${task_config}.log"
        echo "[GPU ${gpu_id}] >>> ${task} / ${task_config}  (est=${est_dur}s, log: ${log})"
        local rc=0

        local -a extra_overrides=()
        if [ -n "${NORM_PATH}" ]; then
            extra_overrides+=( --norm_path "${NORM_PATH}" )
        fi

        (
            cd "${ROBOTWIN_DIR}"
            CUDA_VISIBLE_DEVICES="${gpu_id}" \
            PYTHONWARNINGS=ignore::UserWarning \
            python -u script/eval_policy.py \
                --config policy/hy_vla/deploy_policy.yaml \
                --overrides \
                    --task_name "${task}" \
                    --task_config "${task_config}" \
                    --ckpt_setting "${CKPT_SETTING}" \
                    --instruction_type "${INSTRUCTION_TYPE}" \
                    --seed "${SEED}" \
                    --test_num "${TEST_NUM}" \
                    --policy_name policy.hy_vla \
                    --ckpt_path "${CKPT_PATH}" \
                    --blend_mode "${BLEND_MODE}" \
                    --exc_action_size "${EXC_ACTION_SIZE}" \
                    --img_history_size "${IMG_HISTORY_SIZE}" \
                    --img_history_interval "${IMG_HISTORY_INTERVAL}" \
                    "${extra_overrides[@]}" \
                2>&1 \
                | sed -u 's/\r/\n/g' \
                | sed -u '/^\(\x1b\[[0-9;]*m\)*step: /d' \
                >"${log}"
        ) || rc=$?
        if [ $rc -eq 0 ]; then
            local rate
            rate=$(grep -a 'Success rate' "${log}" | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' || true)
            echo "[GPU ${gpu_id}] <<< ${task} / ${task_config} done.  ${rate}"
            n_done=$((n_done + 1))
        else
            echo "[GPU ${gpu_id}] <<< ${task} / ${task_config} FAILED (rc=${rc}); see ${log}"
            return $rc
        fi
    done
    echo "[GPU ${gpu_id}] worker exiting (ran ${n_done} sub-tasks)."
}

# --------- 7. Banner ---------
EST_WALL_LB_H=$(awk -v g="${MAX_GPUS}" -v t="${TOTAL_EST_S}" -v tn="${TEST_NUM}" 'BEGIN{printf "%.2f", (t * (tn/10.0)) / g / 3600.0}')
EST_WALL_LONGEST_H=$(awk -v l="${LONGEST_EST_S}" -v tn="${TEST_NUM}" 'BEGIN{printf "%.2f", (l * (tn/10.0)) / 3600.0}')

echo "========================================================"
echo "Hy-VLA RoboTwin FULL 50-task eval (FIFO work-stealing)"
echo "Tasks                 : ${#TASKS[@]}"
echo "Task configs          : ${TASK_CONFIGS[*]}"
echo "Sub-tasks queued      : ${NUM_SUBTASKS}"
echo "Rollouts/sub-task     : ${TEST_NUM}"
echo "Max GPUs              : ${MAX_GPUS}  (ids 0..$((MAX_GPUS - 1)))"
echo "Ckpt path             : ${CKPT_PATH}"
echo "RoboTwin dir          : ${ROBOTWIN_DIR}"
echo "Log dir               : ${LOG_DIR}"
echo "blend_mode            : ${BLEND_MODE}"
echo "exc_action_size       : ${EXC_ACTION_SIZE}"
echo "img_history_size      : ${IMG_HISTORY_SIZE}"
echo "img_history_interval  : ${IMG_HISTORY_INTERVAL}"
if [ -n "${NORM_PATH}" ]; then
    echo "norm_path             : ${NORM_PATH}"
else
    echo "norm_path             : <auto: ${CKPT_PATH}/norm_stats.pkl>"
fi
echo "Wall (lower bound)    : ~${EST_WALL_LB_H}h on ${MAX_GPUS} GPU"
echo "Wall (longest single) : ~${EST_WALL_LONGEST_H}h"
echo "========================================================"

# --------- 8. Spawn worker pool + collect return codes ---------
declare -a WORKER_PIDS=()
for ((gpu_id = 0; gpu_id < MAX_GPUS; gpu_id++)); do
    run_worker "${gpu_id}" &
    WORKER_PIDS+=("$!")
    echo "[driver] GPU ${gpu_id} worker pid=$!"
done

echo "[driver] waiting for ${#WORKER_PIDS[@]} workers to drain ${NUM_SUBTASKS} sub-tasks..."

set +e
final_rc=0
for ((i = 0; i < ${#WORKER_PIDS[@]}; i++)); do
    wait "${WORKER_PIDS[$i]}"
    rc=$?
    echo "[driver] GPU ${i} rc=${rc}"
    [ $rc -ne 0 ] && final_rc=$rc
done
set -e

# --------- 9. Aggregate Success rates ---------
echo ""
echo "========================================================"
echo "Per-sub-task Success rate summary"
echo "========================================================"
{
    printf '%-40s %-18s %s\n' "task" "task_config" "success_rate"
    printf '%-40s %-18s %s\n' "----" "-----------" "------------"
    for log_file in "${LOG_DIR}"/*.log; do
        [ -f "${log_file}" ] || continue
        base=$(basename "${log_file}" .log)
        if [[ "${base}" == *_demo_clean ]]; then
            task="${base%_demo_clean}"; cfg="demo_clean"
        elif [[ "${base}" == *_demo_randomized ]]; then
            task="${base%_demo_randomized}"; cfg="demo_randomized"
        else
            task="${base}"; cfg="?"
        fi
        rate=$(grep -a 'Success rate' "${log_file}" | tail -1 | sed -E 's/\x1b\[[0-9;]*m//g' | sed -E 's/.*Success rate[^0-9.]*//' | head -c 80 || true)
        [ -z "${rate}" ] && rate="<no result>"
        printf '%-40s %-18s %s\n' "${task}" "${cfg}" "${rate}"
    done
} | tee "${LOG_DIR}/_summary.txt"
echo "Summary written to ${LOG_DIR}/_summary.txt"

exit $final_rc
