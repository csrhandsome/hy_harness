# LIBERO evaluation

LIBERO, LIBERO-plus, and LIBERO-Pro run in the isolated
`libero_eval/.venv`. The Hy-VLA checkpoint is loaded by a Policy Server in the
repository root environment; the benchmark sends observations to that server
and receives executable action chunks.

## Setup

```bash
uv sync --extra harness
uv sync --project libero_eval
```

Prepare the benchmark trees under `third_party/LIBERO`,
`third_party/LIBERO-plus`, and `third_party/LIBERO-PRO` as described in the
root README.

## Start the action model

```bash
uv run --extra harness vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

Use `--norm-path` when `norm_stats.pkl` is not beside the checkpoint. Model
options such as `--action-dim`, `--gripper-mode`, `--dtype`, and image-history
cadence belong to this server command rather than the simulator command.

## Run evaluation

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/run_libero_eval.sh
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/run_libero_plus_eval.sh
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/run_libero_pro_eval.sh
```

Useful simulator-side overrides include `TASK_SUITE`, `NUM_TRIALS`,
`MAX_TASKS`, `START_TASK_ID`, `PERTURBATION_CATEGORY`, `SAVE_VIDEOS`, and
`EVAL_CFG`.

## Harness

```bash
CKPT_PATH=/absolute/path/to/hy-vla-libero \
  bash libero_eval/run_libero_pro_eval_harness.sh
```

The Harness and local VLA process use the root `.venv`; only the environment
server uses `libero_eval/.venv`. Set `LIBERO_PYTHON` to override that simulator
interpreter. See `docs/environment-layout.md` for the complete process layout.
