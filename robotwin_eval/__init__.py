# coding=utf-8
"""Hy-VLA RoboTwin evaluation hooks and lazy server-side policy exports."""

from __future__ import annotations

from typing import Any

from .deploy_policy import encode_obs, eval, get_model, reset_model

__all__ = [
    "encode_obs",
    "eval",
    "get_model",
    "reset_model",
    "RemoteRobotwinPolicy",
    "HyVLAPolicyWrapper",
    "build_policy",
]


def __getattr__(name: str) -> Any:
    if name == "RemoteRobotwinPolicy":
        from .remote_policy import RemoteRobotwinPolicy

        return RemoteRobotwinPolicy
    if name in {"HyVLAPolicyWrapper", "build_policy"}:
        from .policy_wrapper import HyVLAPolicyWrapper, build_policy

        return {"HyVLAPolicyWrapper": HyVLAPolicyWrapper, "build_policy": build_policy}[name]
    raise AttributeError(name)
