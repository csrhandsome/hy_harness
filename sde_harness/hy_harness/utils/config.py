"""Path resolution and environment-variable configuration."""

from __future__ import annotations

import os
from pathlib import Path

_LOCAL_ENV_LOADED = False


def get_repo_root() -> Path:
    for key in ("SDE_HARNESS_ROOT", "HYHARNESS_REPO_ROOT"):
        value = os.environ.get(key)
        if value:
            return Path(value).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def load_local_env(*, override: bool = False) -> Path | None:
    global _LOCAL_ENV_LOADED
    if _LOCAL_ENV_LOADED and not override:
        return None
    path = get_repo_root() / ".env.local"
    if not path.is_file():
        _LOCAL_ENV_LOADED = True
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key and (override or key not in os.environ):
            os.environ[key] = value
    _LOCAL_ENV_LOADED = True
    return path


def get_hy_vla_root() -> Path:
    value = os.environ.get("HY_VLA_ROOT") or os.environ.get("VLA_ADAPTER_ROOT")
    if value:
        return Path(value).expanduser().resolve()
    return get_repo_root().parent


def get_vla_adapter_root() -> Path:
    """Backward-compatible alias retained for copied Harness modules."""
    return get_hy_vla_root()


def get_resources_dir(env_name: str) -> Path:
    return get_repo_root() / "resources" / env_name


def get_memory_dir(env_name: str) -> Path:
    return get_resources_dir(env_name) / "memory"


def get_checkpoint_path() -> str:
    for key in (
        "HY_VLA_CHECKPOINT",
        "CKPT_PATH",
        "ADAPTER_CHECKPOINT_PATH",
        "VLA_ADAPTER_CHECKPOINT",
    ):
        value = os.environ.get(key)
        if value:
            return value
    return ""


def get_adapter_checkpoint_path() -> str:
    """Backward-compatible alias for the original Harness integration."""
    return get_checkpoint_path()


def get_libero_type() -> str:
    return os.environ.get("LIBERO_TYPE", "pro")


def get_libero_root(libero_type: str | None = None) -> Path:
    kind = (libero_type or get_libero_type()).lower()
    root = get_hy_vla_root() / "third_party"
    mapping = {
        "standard": root / "LIBERO",
        "pro": root / "LIBERO-PRO",
        "plus": root / "LIBERO-plus",
    }
    if kind not in mapping:
        raise ValueError(f"unknown LIBERO_TYPE={kind!r}; expected standard|pro|plus")
    return mapping[kind]


def get_libero_config_dir(libero_type: str | None = None) -> Path:
    kind = (libero_type or get_libero_type()).lower()
    root = get_hy_vla_root() / "libero_eval" / "configs"
    mapping = {
        "standard": root / ".libero_config",
        "pro": root / ".libero_pro_config",
        "plus": root / ".libero_plus_config",
    }
    if kind not in mapping:
        raise ValueError(f"unknown LIBERO_TYPE={kind!r}; expected standard|pro|plus")
    return mapping[kind]
