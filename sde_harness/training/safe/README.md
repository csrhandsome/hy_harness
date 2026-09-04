# RobotWin SAFE pipeline

One Harness run writes one `episode.hdf5`.  The file follows the compatible
UMI core layout: `aligned_timestamp30`, `observations/qpos`, `action/qpos`,
and JPEG streams at `observations/images/{cam_high,cam_left_wrist,cam_right_wrist}`.
The RobotWin-only `annotations/outcome` and `harness/*` groups retain terminal
success/failure and the source of every executed action.

Keep raw episodes immutable.  The SAFE data flow uses versioned sidecars:

```bash
python -m training.safe.cache_prefix \
  --hdf5-dir runs --output prefix_cache --ckpt-path /path/to/hy-vla

python -m training.safe.build_manifest \
  --hdf5-dir runs --prefix-dir prefix_cache --output safe_manifest.jsonl

python -m training.safe.train \
  --manifest safe_manifest.jsonl --output safe_checkpoint
```

`cache_prefix` invokes `HyVLAPolicyWrapper.extract_prefix`, so offline feature
extraction shares the deployed model's RGB handling, resize/padding, text
tokenizer and optional MEM history.  It must use the same checkpoint and
history configuration as deployment. RobotWin recorder qpos/action streams
use the shared xyzw quaternion convention; the cache reader converts the
simulator rows back to the online wrapper's wxyz boundary.

The initial training head predicts outcome risk (`1 - task_correct`) when no
explicit `annotations/safety/risk_within_horizon` label is available.  Those
labels are distinct: progress and terminal task success alone do not establish
physical safety.  The training command refuses a train split without both risk
classes.
