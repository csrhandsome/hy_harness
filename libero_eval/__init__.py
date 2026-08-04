"""Hy-VLA evaluation support for LIBERO, LIBERO-plus, and LIBERO-Pro."""

from __future__ import annotations

from typing import Any

__all__ = ["HyVLALiberoPolicy", "RemoteLiberoPolicy"]


def __getattr__(name: str) -> Any:
    if name == "HyVLALiberoPolicy":
        from .policy_wrapper import HyVLALiberoPolicy

        return HyVLALiberoPolicy
    if name == "RemoteLiberoPolicy":
        from .remote_policy import RemoteLiberoPolicy

        return RemoteLiberoPolicy
    raise AttributeError(name)
