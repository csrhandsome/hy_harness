"""Command-line launcher for the benchmark-facing Hy-VLA policy server."""

from __future__ import annotations

import argparse
import json
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vla-policy-server")
    parser.add_argument("--benchmark", required=True, choices=["libero", "robotwin"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--norm-path", default=None)
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--num-open-loop-steps", type=int, default=None)
    parser.add_argument("--gripper-mode", choices=["libero", "openvla", "raw"], default="libero")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--blend-mode", choices=["rel_abs", "rel_only", "abs_only"], default="rel_only")
    parser.add_argument("--exc-action-size", type=int, default=20)
    parser.add_argument("--img-history-size", type=int, default=None)
    parser.add_argument("--img-history-interval", type=int, default=1)
    parser.add_argument("--umi-coord-frame", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--umi-gripper-space", action=argparse.BooleanOptionalAction, default=False)
    return parser


def _adapter_config(args: argparse.Namespace) -> dict[str, Any]:
    if args.benchmark == "libero":
        return {
            "checkpoint": args.checkpoint,
            "norm_path": args.norm_path,
            "vlm_model_path": args.vlm_model_path,
            "action_dim": args.action_dim,
            "num_open_loop_steps": args.num_open_loop_steps,
            "gripper_mode": args.gripper_mode,
            "device": args.device,
            "dtype": args.dtype,
            "img_history_size": args.img_history_size,
            "img_history_interval": args.img_history_interval,
        }
    return {
        "ckpt_path": args.checkpoint,
        "norm_path": args.norm_path,
        "vlm_model_path": args.vlm_model_path,
        "blend_mode": args.blend_mode,
        "exc_action_size": args.exc_action_size,
        "img_history_size": args.img_history_size or 1,
        "img_history_interval": args.img_history_interval,
        "umi_coord_frame": args.umi_coord_frame,
        "umi_gripper_space": args.umi_gripper_space,
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _adapter_config(args)
    if args.dry_run:
        safe = {**config, "checkpoint": config.get("checkpoint") or config.get("ckpt_path")}
        print(json.dumps({"benchmark": args.benchmark, "host": args.host, "port": args.port, "config": safe}, indent=2))
        return 0

    from vla_protocol import RpcServer

    from .adapters import LiberoPolicyAdapter, RobotwinPolicyAdapter
    from .server import PolicyRpcService

    adapter_cls = LiberoPolicyAdapter if args.benchmark == "libero" else RobotwinPolicyAdapter
    print(f"Loading {args.benchmark} policy from {args.checkpoint}", flush=True)
    service = PolicyRpcService(adapter_cls(**config))
    server = RpcServer((args.host, args.port), service.dispatch)
    host, port = server.server_address
    print(f"Policy RPC server listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0

