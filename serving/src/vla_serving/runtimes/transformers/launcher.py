"""Build the command for the shared Transformers OpenAI-compatible server."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from vla_serving.config import ModelConfig, Registry


def build_command(registry: Registry, model: ModelConfig) -> list[str]:
    cache = registry.cache_dir(model)
    default_model = str(cache) if (cache / "config.json").is_file() else model.repository
    model_path = os.environ.get("MODEL_PATH", default_model)
    defaults = model.defaults
    server = Path(__file__).with_name("server.py")
    return [
        sys.executable,
        str(server),
        "--model",
        model_path,
        "--model-id",
        model.id,
        "--served-name",
        os.environ.get("SERVED_NAME", model.served_name),
        "--host",
        os.environ.get("HOST", str(registry.defaults.get("host", "0.0.0.0"))),
        "--port",
        os.environ.get("PORT", str(registry.defaults.get("port", 8080))),
        "--device",
        os.environ.get("DEVICE", str(defaults.get("device", "cuda"))),
        "--dtype",
        os.environ.get("DTYPE", str(defaults.get("dtype", "bfloat16"))),
    ]
