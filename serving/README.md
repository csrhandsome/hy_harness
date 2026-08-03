# Serving

This directory uses a model registry and shared runtimes. A model directory is
configuration only; executable code belongs to a runtime or plugin.

```text
serving/
├── config.yaml                         # active model and global defaults
├── models/
│   ├── hy-embodied-0.5/model.yaml
│   └── hy-embodied-vlm-1.0/model.yaml
├── src/vla_serving/
│   ├── cli.py                          # list/show/serve/download/client/verify
│   └── runtimes/
│       ├── transformers/               # one shared OpenAI-compatible server
│       └── vllm/                       # one shared vLLM command builder
├── plugins/hy-embodied-vlm-1.0/        # vendored vLLM extension only
├── scripts/                            # stable shell entry points
└── cache/<model-id>/                   # ignored local weights and metadata
```

## Setup

The serving environment is isolated from the repository root because its
torch and vLLM pins differ.

```bash
cd serving
uv sync
```

## Commands

```bash
# Inspect the registry.
bash serving/scripts/servingctl list
bash serving/scripts/servingctl show --model hy-embodied-0.5

# Validate configuration and source syntax without loading weights or CUDA.
bash serving/scripts/verify_flow.sh

# Print commands without starting a server.
bash serving/scripts/serve.sh --dry-run
ACTIVE=hy-embodied-0.5 bash serving/scripts/serve.sh --dry-run

# Download into cache/<model-id>/ and then serve.
bash serving/scripts/download_weights.sh --model hy-embodied-vlm-1.0
bash serving/scripts/serve.sh --model hy-embodied-vlm-1.0

# Query a running OpenAI-compatible endpoint.
bash serving/scripts/client.sh --model hy-embodied-vlm-1.0
```

Environment variables such as `MODEL_PATH`, `SERVED_NAME`, `HOST`, `PORT`,
`TP`, `GPU_MEM_UTIL`, `MAX_MODEL_LEN`, `DEVICE`, `DTYPE`, and
`EXTRA_ARGS` override manifest defaults.

Use `verify --online` only when metadata network checks are wanted. Normal
verification is deliberately offline and does not treat an expected GPU OOM as
a test.

## Cache

All downloads use `cache/<model-id>/`. The cache is local, ignored by Git, and
can be deleted at any time; commands fall back to the configured repository id.
