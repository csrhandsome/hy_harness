# Environment and policy-server layout

The repository has one model-serving environment and one Harness environment per
simulator. Hy-VLA weights are never loaded by a simulator process.

| Environment | Location | Owns |
| --- | --- | --- |
| Hy-VLA | `.venv/` | training, local inference, and the action Policy Server |
| LIBERO + Harness | `libero_eval/.venv/` | LIBERO/robosuite/MuJoCo, RPent Planner, and the lightweight policy client |
| RoboTwin + Harness | `robotwin_eval/.venv/` | RoboTwin/SAPIEN, RPent Planner, and the lightweight policy client |
| RoboDojo + Harness | `robodojo_eval/.venv/` | RoboDojo, RPent Planner, and the lightweight policy client |
| Planner VLM serving | `serving/.venv/` | vLLM, its plugin, and OpenAI-compatible VLM serving |

## Create or refresh environments

From this repository root:

```bash
# Training, local inference, and vla-policy-server only.
uv sync

# LIBERO simulator and its Harness/Planner dependencies.
uv sync --project libero_eval

# Planner VLM service.
uv sync --project serving
```

`robotwin_eval/pyproject.toml` and `robodojo_eval/pyproject.toml` own the
Harness-side dependencies for their respective simulators. Create or refresh
those environments through the corresponding manifest:

```bash
uv sync --project robotwin_eval
uv sync --project robodojo_eval
```

The simulator's upstream dependencies remain declared in those manifests.

## Root action Policy Server

Start the benchmark-specific server from the root environment. It loads the
checkpoint once and is the only process that imports Hy-VLA and Torch for
inference.

```bash
uv run vla-policy-server \
  --benchmark libero \
  --checkpoint /absolute/path/to/hy-vla-libero \
  --port 8001
```

The default endpoint is `http://127.0.0.1:8001`. Start one server per
concurrent stateful evaluation worker; do not share one server between
concurrent episodes.

## LIBERO

For direct evaluation, the simulator sends observations to the root server:

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/eval.sh --pro
```

For a Harness run, start the same root server first, then run the Planner in
the LIBERO environment:

```bash
POLICY_ENDPOINT=http://127.0.0.1:8001 \
  bash libero_eval/eval.sh --pro --harness
```

The Harness launcher passes `POLICY_ENDPOINT` to `--vla-endpoint`; it no longer
spawns a VLA model inside `libero_eval/.venv`.

## RoboTwin

Start one root server per worker:

```bash
uv run vla-policy-server \
  --benchmark robotwin \
  --checkpoint /absolute/path/to/hy-vla-robotwin \
  --norm-path /absolute/path/to/norm_stats.pkl \
  --blend-mode rel_abs \
  --exc-action-size 7 \
  --img-history-size 6 \
  --img-history-interval 5 \
  --port 8001
```

Launch RoboTwin in its own environment with `POLICY_ENDPOINT` set. The policy
symlink imports only the remote client. Enable its optional embedded Harness
only after installing `sde-harness` into the RoboTwin environment.

## RoboDojo

`robodojo_eval/pyproject.toml` owns RoboDojo's Harness dependencies. The
existing RoboDojo policy flow is unchanged; configure and launch it through
`robodojo_eval/deploy_policy.yml` and `robodojo_eval/eval.sh`.

## Planner VLM serving

`serving/.venv` is independent of the action Policy Server. It owns the vLLM
model used by Harness planners:

```bash
bash serving/scripts/verify_flow.sh
bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0
```

By default the action Policy Server uses port 8001 and Planner VLM serving uses
port 8080.
