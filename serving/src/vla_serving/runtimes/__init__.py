"""Runtime launcher registry."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from vla_serving.config import ConfigError

SUPPORTED_RUNTIMES = {"transformers", "vllm"}


def get_runtime(name: str) -> ModuleType:
    if name not in SUPPORTED_RUNTIMES:
        supported = ", ".join(sorted(SUPPORTED_RUNTIMES))
        raise ConfigError(f"unsupported runtime {name!r}; supported: {supported}")
    return import_module(f"vla_serving.runtimes.{name}.launcher")
