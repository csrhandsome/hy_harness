"""Build a vLLM command from a model manifest."""

from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

from vla_serving.config import ModelConfig, Registry


def _flag(args: list[str], name: str, value: object | None) -> None:
    if value is not None and value != "":
        args.extend((name, str(value)))


def build_command(registry: Registry, model: ModelConfig) -> list[str]:
    cache = registry.cache_dir(model)
    default_model = str(cache) if (cache / "config.json").is_file() else model.repository
    model_path = os.environ.get("MODEL_PATH", default_model)
    defaults = model.defaults
    options = model.runtime_options
    sibling_vllm = Path(sys.executable).with_name("vllm")
    executable = str(sibling_vllm) if sibling_vllm.is_file() else "vllm"
    args = [executable, "serve", model_path]
    _flag(args, "--served-model-name", os.environ.get("SERVED_NAME", model.served_name))
    _flag(args, "--host", os.environ.get("HOST", registry.defaults.get("host", "0.0.0.0")))
    _flag(args, "--port", os.environ.get("PORT", registry.defaults.get("port", 8080)))
    _flag(args, "--tensor-parallel-size", os.environ.get("TP", defaults.get("tensor_parallel_size", 1)))
    _flag(args, "--gpu-memory-utilization", os.environ.get("GPU_MEM_UTIL", defaults.get("gpu_memory_utilization", 0.9)))
    _flag(args, "--max-model-len", os.environ.get("MAX_MODEL_LEN", defaults.get("max_model_len", 2048)))
    _flag(args, "--limit-mm-per-prompt", options.get("limit_mm_per_prompt"))
    _flag(args, "--reasoning-parser", options.get("reasoning_parser"))
    _flag(args, "--tool-call-parser", options.get("tool_call_parser"))
    # Required for OpenAI-compatible clients that send ``tool_choice="auto"``
    # (including the CodeBuddy Agent SDK used by the embedded harness).
    if options.get("enable_auto_tool_choice"):
        args.append("--enable-auto-tool-choice")

    template_name = options.get("chat_template")
    if template_name and (cache / str(template_name)).is_file():
        _flag(args, "--chat-template", cache / str(template_name))
    if options.get("trust_remote_code"):
        args.append("--trust-remote-code")
    if options.get("enforce_eager"):
        args.append("--enforce-eager")
    args.extend(shlex.split(os.environ.get("EXTRA_ARGS", "")))
    return args
