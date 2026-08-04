"""Lightweight LIBERO client for an out-of-process Hy-VLA policy server."""

from __future__ import annotations

import os
import uuid
from typing import Any

import numpy as np

from vla_protocol import RpcClient


class RemoteLiberoPolicy:
    def __init__(self, endpoint: str | None = None, *, timeout_s: float = 120.0):
        self.endpoint = (endpoint or os.environ.get("POLICY_ENDPOINT") or "http://127.0.0.1:8001").rstrip("/")
        self.client = RpcClient(self.endpoint, timeout_s=timeout_s)
        metadata = self.client.call("metadata", timeout_s=5.0)
        if metadata.get("benchmark") != "libero":
            raise RuntimeError(
                f"policy server at {self.endpoint} serves {metadata.get('benchmark')!r}, not 'libero'"
            )
        self.session_id = uuid.uuid4().hex

    def reset(self) -> None:
        self.session_id = uuid.uuid4().hex
        self.client.call("reset", kwargs={"session_id": self.session_id})

    def predict(
        self,
        instruction: str,
        main_image: np.ndarray,
        wrist_image: np.ndarray | None,
        state: np.ndarray,
    ) -> np.ndarray:
        images: dict[str, Any] = {"main": np.asarray(main_image, dtype=np.uint8)}
        if wrist_image is not None:
            images["wrist"] = np.asarray(wrist_image, dtype=np.uint8)
        result = self.client.call(
            "libero.predict",
            kwargs={
                "session_id": self.session_id,
                "instruction": instruction,
                "images": images,
                "state": np.asarray(state, dtype=np.float32),
            },
        )
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 3 or actions.shape[0] != 1:
            raise ValueError(f"policy server returned invalid LIBERO actions: {actions.shape}")
        return actions[0]

    def close(self) -> None:
        try:
            self.client.call("close", kwargs={"session_id": self.session_id})
        finally:
            self.client.close()


__all__ = ["RemoteLiberoPolicy"]

