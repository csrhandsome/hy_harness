"""LIBERO environment extension."""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import (
    system_prompt,
    user_prompt,
)
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from rpent.dashboard.state import State
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


def get_env_spec() -> EnvSpec:
    """Return the LIBERO env identity, prompt bundle, and runner hooks.

    Tool schemas, handlers, server lifecycle, and the MCP allowlist live on
    the LIBERO toolkit (see :func:`get_toolkit`).
    """
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    """Return the LIBERO toolkit (common tools + LIBERO primitives)."""
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
    )


def _base_suite_name(suite: str) -> str:
    """Map a LIBERO-Pro/Plus suite name to the base unnorm key."""
    for base in ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"):
        if suite == base or suite.startswith(base + "_"):
            return base
    return suite


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """Register LIBERO CLI flags on the shared ``parser``.

    When ``use_dashboard`` is True, ``--suite`` / ``--task`` are made optional
    because the dashboard launcher will fill them in before ``_parse_config``
    validates. Under CLI-only, they are required — argparse errors out early
    with the usual usage message.
    """
    required = not use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default=None,
                        choices=["standard", "pro", "plus"],
                        help="LIBERO variant (defaults to LIBERO_TYPE env / pro).")
    parser.add_argument("--suite", default=None, required=required,
                        help="e.g. libero_object_task, libero_spatial_swap")
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--adapter-checkpoint",
        default=None,
        help="VLA-Adapter checkpoint dir (defaults to ADAPTER_CHECKPOINT_PATH).",
    )
    parser.add_argument(
        "--unnorm-key",
        default=None,
        help="Action unnorm key (defaults to base suite of --suite).",
    )
    parser.add_argument(
        "--use-pro-version",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load VLA-Adapter Pro action-head / projector weights.",
    )
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local env_server is spawned.")
    parser.add_argument("--vla-endpoint", default=None,
                        help="[protocol://]host:port of the root Policy Server "
                             "(defaults to POLICY_ENDPOINT; protocol=http|socket).")
    parser.add_argument("--sam3-endpoint", default=None,
                        help="[protocol://]host:port of an existing SAM3 server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset and SAM3_CHECKPOINT_PATH is set, a local "
                             "SAM3 server is spawned; otherwise SAM3 is skipped.")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to pin LIBERO rendering/inference to.")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """Validate final ``args`` and derive per-run identifiers.

    Under ``--dashboard``, ``_add_cli_args`` left ``--suite`` / ``--task``
    optional so the dashboard could fill them; this is where we enforce
    they're set now that any overrides have been applied.
    """
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")

    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
    output_dir = Path(output_dir)

    dashboard_state = None
    if getattr(args, "dashboard", False):
        from rpent.dashboard.state import State
        dashboard_state = State(
            run_id=f"{args.suite}/{output_dir.name}",
            name=recipe_tag,
            suite=args.suite,
            task=args.task,
            seed=args.seed,
            output_dir=str(output_dir),
            video_path=str(output_dir / "episode.mp4"),
        )
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        dashboard_state=dashboard_state,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
    )


def _parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """Parse ``[protocol://]host:port`` into ``(protocol, host, port)``.

    Protocol defaults to ``http`` when the prefix is omitted.
    """
    if "://" in endpoint:
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


def _subprocess_env(
    cuda_device: int | None,
    *,
    clear_cuda_visible: bool = False,
    **extra: str,
) -> dict[str, str]:
    """Build a subprocess environment with optional CUDA pinning.

    Native LIBERO clears CUDA_VISIBLE_DEVICES and resolves its global CUDA
    ordinal to the matching EGL device in env_server. VLA/SAM3 retain the
    usual visibility-based isolation.
    """
    env = os.environ.copy()
    if clear_cuda_visible and cuda_device is not None:
        env.pop("CUDA_VISIBLE_DEVICES", None)
    elif cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.update(extra)
    return env


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """Spawn env + vla + SAM3 daemons and build clients for LIBERO.

    Each server can be spawned or attached-to independently: pass an
    endpoint to attach, or leave it unset to spawn a local subprocess.

    Heavy deps (rpc / vla / daemon / env_client) are imported lazily so
    that a bare ``import robots.libero`` (for ``get_env_spec`` /
    ``get_toolkit``) doesn't drag them in.
    """
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import (
        get_adapter_checkpoint_path,
        get_libero_type,
        get_vla_adapter_root,
    )
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    adapter_ckpt = args.adapter_checkpoint or get_adapter_checkpoint_path()
    unnorm_key = args.unnorm_key or _base_suite_name(args.suite)
    adapter_root = str(get_vla_adapter_root())
    vla_endpoint = args.vla_endpoint or os.environ.get("POLICY_ENDPOINT")
    if not vla_endpoint:
        raise RuntimeError(
            "Set POLICY_ENDPOINT or --vla-endpoint to a root vla-policy-server; "
            "the LIBERO Harness environment does not load Hy-VLA weights."
        )
    harness_root = str(get_repo_root())
    args.vla_endpoint = vla_endpoint

    def _pythonpath_env(
        *, clear_cuda_visible: bool = False, **extra: str
    ) -> dict[str, str]:
        """Ensure subprocesses can import ``rpent``, ``robots``, and VLA-Adapter."""
        merged = os.pathsep.join(
            p for p in (harness_root, adapter_root, os.environ.get("PYTHONPATH", "")) if p
        )
        return _subprocess_env(
            args.cuda_device,
            clear_cuda_visible=clear_cuda_visible,
            PYTHONPATH=merged,
            **extra,
        )

    cuda_args = (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
    )

    libero_python = os.environ.get("LIBERO_PYTHON")
    if not libero_python:
        libero_python = str(Path(adapter_root) / "libero_eval" / ".venv" / "bin" / "python")
    if not Path(libero_python).is_file():
        raise RuntimeError(
            f"LIBERO interpreter not found: {libero_python}. "
            "Run `uv sync --project libero_eval` or set LIBERO_PYTHON."
        )

    # --- env_server --------------------------------------------------------
    if args.env_endpoint is None:
        host, port = "127.0.0.1", pick_free_port()
        env_daemon = ProcessDaemon(
            name="env_server",
            cmd=[
                libero_python,
                str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                "--suite", args.suite,
                "--task", str(args.task),
                "--seed", str(args.seed),
                "--max-episode-steps", str(args.max_episode_steps),
                "--libero-type", libero_type,
                "--transport", "http",
                "--host", host,
                "--port", str(port),
                *cuda_args,
            ],
            env=_pythonpath_env(
                clear_cuda_visible=True,
                LIBERO_TYPE=libero_type,
                MUJOCO_GL="egl",
                ROBOT_PLATFORM="LIBERO",
                VLA_ADAPTER_ROOT=adapter_root,
                SDE_HARNESS_ROOT=harness_root,
            ),
            log_path=str(Path(output_dir) / "env_server.log"),
        )
        env_daemon.start()
        daemons.append(env_daemon)
        env_client: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(env_client, daemon=env_daemon)
    else:
        protocol, host, port = _parse_endpoint(args.env_endpoint)
        if protocol == "socket":
            env_client = SocketRpcClient(host, port)
        elif protocol == "http":
            env_client = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--env-endpoint protocol must be socket or http, got {protocol!r}"
            )

    # --- vla_server (VLA-Adapter weights; RLinf Pi0.5 removed) -------------
    if args.vla_endpoint is None:
        if not adapter_ckpt:
            raise RuntimeError(
                "ADAPTER_CHECKPOINT_PATH / --adapter-checkpoint is required "
                "to spawn vla_server (or pass --vla-endpoint)."
            )
        host, port = "127.0.0.1", pick_free_port()
        vla_cmd = [
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
            "--transport", "http",
            "--host", host,
            "--port", str(port),
            "--model-path", str(adapter_ckpt),
            "--unnorm-key", str(unnorm_key),
        ]
        if args.use_pro_version:
            vla_cmd.append("--use-pro-version")
        else:
            vla_cmd.append("--no-use-pro-version")
        vla_daemon = ProcessDaemon(
            name="vla_server",
            cmd=vla_cmd,
            env=_pythonpath_env(
                VLA_ADAPTER_ROOT=adapter_root,
                SDE_HARNESS_ROOT=harness_root,
                ADAPTER_CHECKPOINT_PATH=str(adapter_ckpt),
                ADAPTER_UNNORM_KEY=str(unnorm_key),
            ),
            log_path=str(Path(output_dir) / "vla_server.log"),
        )
        vla_daemon.start()
        daemons.append(vla_daemon)
        vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        # Adapter checkpoint load can exceed the default 300s on cold start.
        wait_for_ready(vla_rpc, timeout_s=600.0, daemon=vla_daemon)
    else:
        protocol, host, port = _parse_endpoint(args.vla_endpoint)
        if protocol == "socket":
            vla_rpc = SocketRpcClient(host, port)
        elif protocol == "http":
            vla_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
            )

    # --- sam3_server (optional) --------------------------------------------
    # Skip when neither an endpoint nor SAM3_CHECKPOINT_PATH is provided so
    # 8GB laptops can run VLA-only without loading SAM3 (~10GB+).
    from rpent.utils.logging import get_logger

    _log = get_logger("libero")
    sam3_ckpt = (os.environ.get("SAM3_CHECKPOINT_PATH") or "").strip()
    sam3_client: Sam3Client | None = None
    if args.sam3_endpoint is not None:
        protocol, host, port = _parse_endpoint(args.sam3_endpoint)
        if protocol == "socket":
            sam3_rpc: RpcClient = SocketRpcClient(host, port)
        elif protocol == "http":
            sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
        else:
            raise ValueError(
                f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
            )
        wait_for_ready(sam3_rpc)
        sam3_client = Sam3Client(sam3_rpc)
    elif sam3_ckpt:
        host, port = "127.0.0.1", pick_free_port()
        sam3_daemon = ProcessDaemon(
            name="sam3_server",
            cmd=[
                sys.executable,
                str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                "--transport", "http",
                "--host", host,
                "--port", str(port),
            ],
            env=_pythonpath_env(
                VLA_ADAPTER_ROOT=adapter_root,
                SDE_HARNESS_ROOT=harness_root,
                SAM3_CHECKPOINT_PATH=sam3_ckpt,
            ),
            log_path=str(Path(output_dir) / "sam3_server.log"),
        )
        sam3_daemon.start()
        daemons.append(sam3_daemon)
        sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
        wait_for_ready(sam3_rpc, daemon=sam3_daemon)
        sam3_client = Sam3Client(sam3_rpc)
    else:
        _log.warning(
            "SAM3 disabled (SAM3_CHECKPOINT_PATH unset and no --sam3-endpoint); "
            "segment tool will return an error if called"
        )

    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_client,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
        "sam3_client": sam3_client,
    }
    return daemons, primitives_kwargs
