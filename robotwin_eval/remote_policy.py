"""RoboTwin policy wrapper backed by the root Hy-VLA policy server."""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    from vla_protocol import RpcClient
except ImportError:
    # RoboTwin normally imports this directory through a policy symlink rather
    # than installing the repository. Make the dependency-free protocol source
    # visible without installing the root Hy-VLA package or its Torch stack.
    protocol_src = Path(__file__).resolve().parents[1] / "packages" / "vla_protocol" / "src"
    sys.path.insert(0, str(protocol_src))
    from vla_protocol import RpcClient


class RemoteRobotwinPolicy:
    """Expose RoboTwin's reset/get_action surface over RPC."""

    def __init__(self, config: dict[str, Any]):
        self.endpoint = str(
            config.get("policy_endpoint")
            or os.environ.get("POLICY_ENDPOINT")
            or "http://127.0.0.1:8001"
        ).rstrip("/")
        self.client = RpcClient(
            self.endpoint, timeout_s=float(config.get("policy_timeout_s", 120.0))
        )
        metadata = self.client.call("metadata", timeout_s=5.0)
        if metadata.get("benchmark") != "robotwin":
            raise RuntimeError(
                f"policy server at {self.endpoint} serves {metadata.get('benchmark')!r}, not 'robotwin'"
            )
        self.session_id = uuid.uuid4().hex

    def reset(self) -> str:
        self.session_id = uuid.uuid4().hex
        self.client.call("reset", kwargs={"session_id": self.session_id})
        return "remote Hy-VLA policy reset"

    def get_action(self, batch: dict[str, Any]) -> np.ndarray:
        state = np.asarray(batch["observation.state"], dtype=np.float32)
        if state.ndim == 2:
            state = state[0]
        result = self.client.call(
            "robotwin.step",
            kwargs={
                "session_id": self.session_id,
                "instruction": str((batch.get("task") or [""])[0]),
                "images": {
                    "top": np.asarray(batch["raw_images.top_head"], dtype=np.uint8),
                    "left": np.asarray(batch["raw_images.hand_left"], dtype=np.uint8),
                    "right": np.asarray(batch["raw_images.hand_right"], dtype=np.uint8),
                },
                "state": state[:16],
            },
        )
        action = np.asarray(result["action"], dtype=np.float32)
        if action.shape != (16,):
            raise ValueError(f"policy server returned invalid RoboTwin action: {action.shape}")
        return action

    def close(self) -> None:
        try:
            self.client.call("close", kwargs={"session_id": self.session_id})
        finally:
            self.client.close()


__all__ = ["RemoteRobotwinPolicy"]

