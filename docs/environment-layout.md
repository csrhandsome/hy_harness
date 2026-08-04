# Environment and policy-server layout

The repository uses three dependency boundaries. Simulator environments are a
logical group, but each benchmark keeps its own physical virtual environment so
MuJoCo, SAPIEN, Gym, and related pins cannot collide.

| Environment | Location | Owns |
| --- | --- | --- |
| Hy-VLA + Harness | `.venv/` | training/inference stack, RPent planners, action Policy Server |
| LIBERO simulator | `libero_eval/.venv/` | LIBERO, robosuite, MuJoCo, lightweight policy client |
| RoboTwin simulator | the upstream RoboTwin environment | RoboTwin/SAPIEN and lightweight policy client |
| Planner VLM serving | `serving/.venv/` | vLLM, its plugin, and OpenAI-compatible VLM serving |

## Create or refresh environments

From the repository root:

```bash
uv sync --extra harness
uv sync --project libero_eval
uv sync --project serving
```

The root workspace contains `sde_harness` and `packages/vla_protocol`; Harness
does not create a second environment. `libero_eval` and `serving` are explicitly
excluded from that workspace and therefore materialize their own `.venv`.

## Direct LIBERO evaluation

Start the action model in the root environment:

```bash
uv run --extra harness vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

In another shell, run the simulator-only evaluator:

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/run_libero_eval.sh
```

The same server works with the Plus and Pro launchers when the checkpoint is
compatible. The benchmark process sends lossless uint8 arrays and state vectors
over the local RPC protocol; it does not import Torch or `hy_vla`.

## Direct RoboTwin evaluation

Start one action server per concurrently evaluated GPU/worker:

```bash
uv run --extra harness vla-policy-server \
  --benchmark robotwin \
  --checkpoint /absolute/path/to/hy-vla-robotwin \
  --norm-path /absolute/path/to/norm_stats.pkl \
  --blend-mode rel_abs \
  --exc-action-size 7 \
  --img-history-size 6 \
  --img-history-interval 5 \
  --port 8001
```

Then launch RoboTwin in its own environment with `POLICY_ENDPOINT` set. The
policy symlink imports `robotwin_eval.remote_policy`; model weights, Torch, and
normalization remain on the server side.

The server deliberately protects its stateful image history and action cache
from interleaved sessions. A multi-GPU RoboTwin run must use one server and port
per worker rather than sharing one server across simultaneous episodes.

## Harness evaluation

The LIBERO Harness launcher remains a single command:

```bash
CKPT_PATH=/absolute/path/to/hy-vla-libero \
  bash libero_eval/run_libero_pro_eval_harness.sh
```

The launcher runs the Planner and model process with the root `.venv`; the
LIBERO environment server is spawned with `libero_eval/.venv/bin/python`.
Override that interpreter with `LIBERO_PYTHON` when needed.

The historical embedded RoboTwin Harness still runs in the RoboTwin process and
is not part of the isolated direct-evaluation path. Fully isolating that
experimental mode requires turning the live `TASK_ENV` object into a benchmark
RPC service; leave `harness.enabled: false` for the separated setup.

## Planner VLM serving

`serving/.venv` is unrelated to the action Policy Server. It owns the vLLM model
used by Harness planners:

```bash
bash serving/scripts/verify_flow.sh
bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0
```

By default the action Policy Server uses port 8001 and planner VLM serving uses
port 8080, which makes the two roles explicit.

