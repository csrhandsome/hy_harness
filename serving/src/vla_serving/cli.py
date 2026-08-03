"""Command-line interface for all configured serving models."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from vla_serving.config import ConfigError, ModelConfig, Registry, load_registry
from vla_serving.runtimes import SUPPORTED_RUNTIMES, get_runtime


def default_config() -> Path:
    return Path(__file__).resolve().parents[2] / "config.yaml"


def select_model(registry: Registry, requested: str | None) -> ModelConfig:
    override = requested or os.environ.get("ACTIVE") or os.environ.get("SERVING_MODEL")
    return registry.select(override)


def validate_model(registry: Registry, model: ModelConfig) -> None:
    if model.runtime not in SUPPORTED_RUNTIMES:
        raise ConfigError(f"{model.manifest}: unsupported runtime {model.runtime!r}")
    plugin = model.runtime_options.get("plugin")
    if plugin:
        plugin_dir = registry.root / str(plugin)
        if not (plugin_dir / "pyproject.toml").is_file():
            raise ConfigError(f"{model.manifest}: plugin does not exist: {plugin_dir}")


def model_dict(registry: Registry, model: ModelConfig) -> dict[str, Any]:
    return {
        "id": model.id,
        "name": model.name,
        "status": model.status,
        "runtime": model.runtime,
        "repository": model.repository,
        "served_name": model.served_name,
        "cache_dir": str(registry.cache_dir(model)),
        "description": model.description,
        "defaults": model.defaults,
        "runtime_options": model.runtime_options,
    }


def command_list(registry: Registry, _: argparse.Namespace) -> int:
    print(f"{'MODEL':28} {'RUNTIME':14} {'STATUS':10} NAME")
    for model in registry.models.values():
        marker = "*" if model.id == registry.active else " "
        print(f"{marker} {model.id:26} {model.runtime:14} {model.status:10} {model.name}")
    return 0


def command_show(registry: Registry, args: argparse.Namespace) -> int:
    model = select_model(registry, args.model)
    print(json.dumps(model_dict(registry, model), indent=2, ensure_ascii=False))
    return 0


def command_serve(registry: Registry, args: argparse.Namespace) -> int:
    model = select_model(registry, args.model)
    validate_model(registry, model)
    if model.status != "ready":
        raise ConfigError(f"model {model.id!r} has status={model.status!r}")
    command = get_runtime(model.runtime).build_command(registry, model)
    print(f"[serving] model={model.id} runtime={model.runtime}", flush=True)
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return 0
    os.environ.setdefault("HF_ENDPOINT", str(registry.defaults.get("hf_endpoint", "")))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.execvpe(command[0], command, os.environ)
    return 0


def command_download(registry: Registry, args: argparse.Namespace) -> int:
    model = select_model(registry, args.model)
    target = Path(args.output).resolve() if args.output else registry.cache_dir(model)
    os.environ.setdefault("HF_ENDPOINT", str(registry.defaults.get("hf_endpoint", "")))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    print(f"[download] {model.repository} -> {target}")
    if args.dry_run:
        return 0
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ConfigError("huggingface-hub is missing; run: cd serving && uv sync") from exc
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model.repository, local_dir=target)
    return 0


def request_json(url: str, payload: dict[str, Any] | None = None) -> Any:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.load(response)
    except urllib.error.URLError as exc:
        raise ConfigError(f"request failed: {url}: {exc}") from exc


def command_client(registry: Registry, args: argparse.Namespace) -> int:
    model = select_model(registry, args.model)
    host = args.host or os.environ.get("HOST", "127.0.0.1")
    port = args.port or int(os.environ.get("PORT", registry.defaults.get("port", 8080)))
    base_url = f"http://{host}:{port}"
    print(json.dumps(request_json(f"{base_url}/v1/models"), indent=2, ensure_ascii=False))
    payload = {
        "model": os.environ.get("SERVED_NAME", model.served_name),
        "messages": [{"role": "user", "content": args.prompt}],
        "max_tokens": args.max_tokens,
        "enable_thinking": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    result = request_json(f"{base_url}/v1/chat/completions", payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def verify_sources(registry: Registry) -> None:
    for root in (registry.root / "src", registry.root / "plugins"):
        for source in root.rglob("*.py"):
            ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for script in sorted((registry.root / "scripts").glob("*.sh")):
        subprocess.run(["bash", "-n", str(script)], check=True)


def command_verify(registry: Registry, args: argparse.Namespace) -> int:
    for model in registry.models.values():
        validate_model(registry, model)
        command = get_runtime(model.runtime).build_command(registry, model)
        if not command:
            raise ConfigError(f"{model.id}: runtime produced an empty command")
        print(f"OK model {model.id}: runtime={model.runtime} cache={registry.cache_dir(model)}")
    verify_sources(registry)
    print("OK Python and shell syntax")
    if args.online:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ConfigError("huggingface-hub is missing; run: cd serving && uv sync") from exc
        for model in registry.models.values():
            target = registry.cache_dir(model)
            path = hf_hub_download(
                repo_id=model.repository,
                filename="config.json",
                local_dir=target,
            )
            print(f"OK metadata {model.id}: {path}")
    print("Verification completed without loading weights or initializing CUDA.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-serving")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(os.environ.get("SERVING_CONFIG", default_config())),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list configured models")
    show = subparsers.add_parser("show", help="show resolved model configuration")
    serve = subparsers.add_parser("serve", help="start the selected model runtime")
    download = subparsers.add_parser("download", help="download weights into the model cache")
    client = subparsers.add_parser("client", help="send a text request to a running server")
    verify = subparsers.add_parser("verify", help="run checks that do not load model weights")
    for child in (show, serve, download, client):
        child.add_argument("--model", help="model id; defaults to ACTIVE or config active")
    serve.add_argument("--dry-run", action="store_true", help="print the launch command only")
    download.add_argument("--output", help="override cache destination")
    download.add_argument("--dry-run", action="store_true", help="print destination only")
    client.add_argument("--host")
    client.add_argument("--port", type=int)
    client.add_argument("--prompt", default="How do you open a fridge?")
    client.add_argument("--max-tokens", type=int, default=128)
    verify.add_argument("--online", action="store_true", help="download config.json metadata")
    return parser


COMMANDS = {
    "list": command_list,
    "show": command_show,
    "serve": command_serve,
    "download": command_download,
    "client": command_client,
    "verify": command_verify,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        registry = load_registry(args.config)
        return COMMANDS[args.command](registry, args)
    except (ConfigError, OSError, subprocess.CalledProcessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
