# SDE Harness for Hy-VLA

`sde_harness` is a member of the root uv workspace. It shares the repository
root `.venv` and does not own a second environment:

```bash
uv sync --extra harness
```

Simulator dependencies are intentionally absent from this package. Create the
LIBERO environment separately:

```bash
uv sync --project libero_eval
```

Run through the stable wrappers:

```bash
CKPT_PATH=/path/to/hy-vla-libero \
TASK_SUITE=libero_spatial_swap TASK=0 SEED=0 \
PLANNER=codebuddy MODEL=hy_a3b \
  bash libero_eval/run_libero_pro_eval_harness.sh
```

The wrapper starts the LIBERO environment server with
`libero_eval/.venv/bin/python`, while the Planner and VLA model process use the
root interpreter. `LIBERO_PYTHON` can select another compatible simulator
environment. Planner VLM serving remains the separate service under
`serving/`; see `docs/environment-layout.md`.
