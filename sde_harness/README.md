# SDE Harness for Hy-VLA

This directory is the RPent-style agent Harness copied from the working
VLA-Adapter setup and adapted to this repository.

- `rpent/`: planner, RPC, dashboard, memory, and common tools.
- `robots/libero/env_server.py`: standard/plus/pro LIBERO environment server.
- `robots/libero/vla_server.py`: local Hy-VLA action server.
- `robots/libero/sam3_server.py`: optional SAM3 perception server.

Install the Harness dependencies into the repository's root environment:

The default planner is the CodeBuddy Agent SDK (`codebuddy`), routed to the local vLLM service at `http://127.0.0.1:8080/v1`.

```bash
uv sync --extra harness
```

Run through the stable wrappers in `libero_eval/`:

```bash
CKPT_PATH=/path/to/hy-vla-libero \
TASK_SUITE=libero_spatial_swap TASK=0 SEED=0 \
PLANNER=codebuddy MODEL=hy_a3b \
  bash libero_eval/run_libero_pro_eval_harness.sh
```

The checkpoint must use LIBERO's state/action convention and provide
`norm_stats.pkl`; use `NORM_PATH` to override its location. The default CodeBuddy SDK route is the local vLLM service at `http://127.0.0.1:8080/v1`
with served model `hy_a3b`; override it through `CODEBUDDY_OPENAI_BASE_URL` or
place settings in the gitignored `.env.local` copied from `.env.local.example`.

The copied CLI still accepts the historical `--adapter-checkpoint` name for
compatibility, but it now launches `HyVLALiberoPolicy`, not VLA-Adapter.
