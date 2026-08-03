"""Load and validate the serving and per-model manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the serving registry is incomplete or inconsistent."""


def _parse_scalar(raw: str) -> object:
    if (raw.startswith('"') and raw.endswith('"')) or (
        raw.startswith("'") and raw.endswith("'")
    ):
        return raw[1:-1]
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "~", ""}:
        return None
    try:
        return int(raw) if "." not in raw else float(raw)
    except ValueError:
        return raw


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        key, separator, value = raw.strip().partition(":")
        if not separator or not key.strip():
            raise ConfigError(f"expected 'key: value', got {raw!r}")
        parent = stack[-1][1]
        if not value.strip():
            child: dict[str, Any] = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _parse_scalar(value.strip())
    return root


def load_yaml(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: root must be a mapping")
    return data


def _mapping(data: dict[str, Any], key: str, source: Path) -> dict[str, Any]:
    value = data.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{source}: {key} must be a mapping")
    return value


@dataclass(frozen=True)
class ModelConfig:
    id: str
    name: str
    status: str
    runtime: str
    repository: str
    served_name: str
    description: str
    defaults: dict[str, Any]
    runtime_options: dict[str, Any]
    manifest: Path


@dataclass(frozen=True)
class Registry:
    root: Path
    active: str
    defaults: dict[str, Any]
    cache_root: Path
    models: dict[str, ModelConfig]

    def select(self, model_id: str | None = None) -> ModelConfig:
        selected = model_id or self.active
        try:
            return self.models[selected]
        except KeyError as exc:
            known = ", ".join(sorted(self.models)) or "(none)"
            raise ConfigError(
                f"unknown model {selected!r}; known models: {known}"
            ) from exc

    def cache_dir(self, model: ModelConfig) -> Path:
        return self.cache_root / model.id


def load_registry(config_path: Path) -> Registry:
    config_path = config_path.resolve()
    root = config_path.parent
    data = load_yaml(config_path)
    active = str(data.get("active") or "")
    if not active:
        raise ConfigError(f"{config_path}: active is required")
    defaults = _mapping(data, "defaults", config_path)
    models_root = (root / str(data.get("models_dir", "models"))).resolve()
    cache_root = (root / str(data.get("cache_dir", "cache"))).resolve()
    models: dict[str, ModelConfig] = {}

    for manifest in sorted(models_root.glob("*/model.yaml")):
        raw = load_yaml(manifest)
        required = ("id", "name", "runtime", "repository", "served_name")
        missing = [key for key in required if not raw.get(key)]
        if missing:
            raise ConfigError(f"{manifest}: missing {', '.join(missing)}")
        model_id = str(raw["id"])
        if model_id != manifest.parent.name:
            raise ConfigError(
                f"{manifest}: id {model_id!r} must match directory "
                f"{manifest.parent.name!r}"
            )
        if model_id in models:
            raise ConfigError(f"duplicate model id: {model_id}")
        models[model_id] = ModelConfig(
            id=model_id,
            name=str(raw["name"]),
            status=str(raw.get("status", "ready")),
            runtime=str(raw["runtime"]),
            repository=str(raw["repository"]),
            served_name=str(raw["served_name"]),
            description=str(raw.get("description", "")),
            defaults=_mapping(raw, "defaults", manifest),
            runtime_options=_mapping(raw, "runtime_options", manifest),
            manifest=manifest,
        )

    registry = Registry(root, active, defaults, cache_root, models)
    registry.select()
    return registry
