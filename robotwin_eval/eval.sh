#!/usr/bin/env bash
# Convenience launcher mirroring the upstream RoboTwin invocation.
#
# Usage (run from the RoboTwin repo root, with this Hy-VLA repo
# symlinked as ``policy/hy_vla``):
#
#   bash policy/hy_vla/eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
#
# Example:
#
#   bash policy/hy_vla/eval.sh beat_block_hammer demo_clean Hy-VLA-posttrain 10000 0
#
# The five positional arguments are forwarded to RoboTwin's
# script/eval_policy.py via --overrides; everything Hy-VLA-specific
# (ckpt_path, blend_mode, exc_action_size, MEM cadence, ...) lives in
# deploy_policy.yaml and can be overridden the same way.

set -euo pipefail

task_name=${1}
task_config=${2}
ckpt_setting=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/hy_vla/deploy_policy.yaml \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name policy.hy_vla
