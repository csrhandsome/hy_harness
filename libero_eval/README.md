# LIBERO evaluation

LIBERO, LIBERO-plus, and LIBERO-Pro run in the isolated
`libero_eval/.venv`. The Hy-VLA checkpoint is loaded by a Policy Server in the
repository root environment; the benchmark sends observations to that server
and receives executable action chunks.

## Setup

```bash
uv sync
uv sync --project libero_eval
```

Prepare the benchmark trees under `third_party/LIBERO`,
`third_party/LIBERO-plus`, and `third_party/LIBERO-PRO` as described in the
root README.

Download the LIBERO-plus assets and LIBERO-Pro BDDL/init files with one command:

```bash
bash libero_eval/download.sh
```

Use `bash libero_eval/download.sh plus`, `pro`, or `pro-ckpts` to select a
single target. `pro-ckpts` is only for the legacy VLA-Adapter checkpoints and
does not download a Hy-VLA checkpoint. RPent memory belongs to Harness and is
downloaded separately with `bash sde_harness/scripts/download_rpent_memory.sh`.

## Start the action model

```bash
uv run vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

Use `--norm-path` when `norm_stats.pkl` is not beside the checkpoint. Model
options such as `--action-dim`, `--gripper-mode`, `--dtype`, and image-history
cadence belong to this server command rather than the simulator command.

## Run evaluation

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/eval.sh
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/eval.sh --plus
POLICY_ENDPOINT=http://127.0.0.1:8001 bash libero_eval/eval.sh --pro
```

Useful simulator-side overrides include `TASK_SUITE`, `NUM_TRIALS`,
`MAX_TASKS`, `START_TASK_ID`, `PERTURBATION_CATEGORY`, `SAVE_VIDEOS`, and
`EVAL_CFG`. Arguments that do not select a variant are forwarded to the
underlying evaluator; use `--` before them when needed, for example
`bash libero_eval/eval.sh --pro -- --max-tasks 2`.

## Harness

Start the root Policy Server shown above, then run:

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/eval.sh --pro --harness
```

The Planner, Harness, and environment server all use `libero_eval/.venv`; the
server process in the root `.venv` owns the checkpoint and model inference.
Set `LIBERO_PYTHON` to override the simulator interpreter. `--harness` enables
this mode; omit it (or pass `--no-harness`) to use the direct evaluator. See
`docs/environment-layout.md` for the complete process layout.
