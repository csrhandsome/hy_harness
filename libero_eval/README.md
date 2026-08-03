# LIBERO evaluation

This directory contains Hy-VLA evaluation entrypoints for standard LIBERO,
LIBERO-plus, and LIBERO-Pro, plus the matching SDE Harness launchers copied
from the working VLA-Adapter integration.

The low-level policy is now `HyVLALiberoPolicy`. A compatible checkpoint must
be fine-tuned for LIBERO's 8-D state and 7-D action protocol and ship a
`norm_stats.pkl` containing `qpos_mean`, `qpos_std`, `action_mean`, and
`action_std`.

```bash
uv sync --extra libero

CKPT_PATH=/path/to/hy-vla-libero \
  bash libero_eval/run_libero_eval.sh

CKPT_PATH=/path/to/hy-vla-libero MAX_TASKS=2 NUM_TRIALS=1 \
  bash libero_eval/run_libero_plus_eval.sh

CKPT_PATH=/path/to/hy-vla-libero MAX_TASKS=1 NUM_TRIALS=1 \
  bash libero_eval/run_libero_pro_eval.sh
```

Set `NORM_PATH` when the norm file is not next to the checkpoint. Other useful
overrides are `TASK_SUITE`, `ACTION_DIM`, `NUM_OPEN_LOOP_STEPS`,
`GRIPPER_MODE=libero|openvla|raw`, `SAVE_VIDEOS=1`, and `CUDA_VISIBLE_DEVICES`.

Harness variants use the same checkpoint and environment variables:

```bash
uv sync --extra harness
CKPT_PATH=/path/to/hy-vla-libero \
  bash libero_eval/run_libero_pro_eval_harness.sh
```

`download_libero_pro_ckpts.sh` is retained unchanged from VLA-Adapter for
reproducibility and downloads VLA-Adapter checkpoints. The Hy-VLA evaluators do
not consume those checkpoints: provide a LIBERO-fine-tuned Hy-VLA checkpoint
through `CKPT_PATH`, with `norm_stats.pkl` beside it or an explicit
`NORM_PATH`.

The vendored benchmark trees live under `third_party/LIBERO`,
`third_party/LIBERO-plus`, and `third_party/LIBERO-PRO`.
