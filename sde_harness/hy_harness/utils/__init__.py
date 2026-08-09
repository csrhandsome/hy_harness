"""Utility helpers: config, logging, path resolution, templates."""

from hy_harness.utils.logging import get_logger, get_output_dir, init_output_dir
from hy_harness.utils.rpc import RpcClient
from hy_harness.utils.socket_rpc import (
    RpcError,
    SocketRpcClient,
    SocketRpcServer,
)
from hy_harness.utils.templates import (
    default_variables,
    substitute,
    substitute_text,
)

__all__ = [
    "RpcClient",
    "RpcError",
    "SocketRpcClient",
    "SocketRpcServer",
    "default_variables",
    "get_logger",
    "get_output_dir",
    "init_output_dir",
    "substitute",
    "substitute_text",
]
