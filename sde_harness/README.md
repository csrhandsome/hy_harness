# SDE Harness for Hy-VLA

`sde_harness` is a shared package. It is installed into the virtual environment
of the simulator that runs it, rather than into the root Hy-VLA environment.

- LIBERO: `uv sync --project libero_eval` installs it into `libero_eval/.venv`.
- RoboTwin: `uv sync --project robotwin_eval` installs it into the RoboTwin evaluation environment.
- RoboDojo: `uv sync --project robodojo_eval` installs it into the RoboDojo evaluation environment.

The root `.venv` owns Hy-VLA training, inference, and `vla-policy-server`.
Start the Policy Server before a Harness run:

```bash
uv run vla-policy-server \
  --benchmark libero \
  --checkpoint /path/to/hy-vla-libero \
  --port 8001
```

Then launch LIBERO Harness from its own environment:

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
TASK_SUITE=libero_spatial_swap TASK=0 SEED=0 \
PLANNER=codebuddy MODEL=hy_a3b \
  bash libero_eval/eval.sh --pro --harness
```

The Harness connects to that endpoint; it does not spawn or load a VLA model.
Planner VLM serving remains the separate service under `serving/`; see
`docs/environment-layout.md`.
