"""Benchmark-specific observation and action adapters for the policy server."""

from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np


def _decode_image(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        image = value
    elif isinstance(value, dict) and isinstance(value.get("data"), str):
        import imageio.v2 as imageio

        image = imageio.imread(io.BytesIO(base64.b64decode(value["data"])))
    else:
        raise TypeError("image must be an ndarray or an encoded image block")
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 image, got {image.shape}")
    return image.astype(np.uint8, copy=False)


class LiberoPolicyAdapter:
    benchmark = "libero"

    def __init__(self, **config: Any):
        from libero_eval.policy_wrapper import HyVLALiberoPolicy

        self.policy = HyVLALiberoPolicy(**config)
        self.config = dict(config)

    def metadata(self) -> dict[str, Any]:
        return {
            "request_method": "libero.predict",
            "state_dim": 8,
            "action_dim": int(self.policy.action_dim),
            "checkpoint": str(self.policy.checkpoint),
        }

    def reset(self) -> str:
        self.policy.reset()
        return "LIBERO policy reset"

    def dispatch(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if method not in {"libero.predict", "predict"}:
            raise ValueError(f"unsupported LIBERO policy method: {method!r}")
        images = kwargs.get("images") or {}
        main = _decode_image(images.get("main"))
        wrist_value = images.get("wrist")
        wrist = _decode_image(wrist_value) if wrist_value is not None else None
        state = np.asarray(kwargs.get("state"), dtype=np.float32)
        if state.ndim == 2 and state.shape[0] == 1:
            state = state[0]
        actions = self.policy.predict(
            str(kwargs.get("instruction") or ""), main, wrist, state
        )
        return {
            "actions": np.asarray(actions, dtype=np.float32)[None],
            "shape": [1, *actions.shape],
            "dtype": "float32",
        }


class RobotwinPolicyAdapter:
    benchmark = "robotwin"

    def __init__(self, **config: Any):
        from robotwin_eval.policy_wrapper import build_policy

        self.policy = build_policy(config)
        self.config = dict(config)

    def metadata(self) -> dict[str, Any]:
        return {
            "request_method": "robotwin.chunk",
            "legacy_request_method": "robotwin.step",
            "state_dim": 16,
            "action_dim": 16,
            "action_chunk_size": int(getattr(self.policy, "action_chunk_size", 50)),
            "default_execute_steps": int(
                getattr(self.policy, "default_execute_steps", 30)
            ),
            "checkpoint": str(self.config["ckpt_path"]),
        }

    def reset(self) -> str:
        return self.policy.reset()

    @staticmethod
    def _chw(image: np.ndarray) -> np.ndarray:
        return image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

    def _batch(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Decode one wire observation into the model batch format."""
        images = kwargs.get("images") or {}
        top = _decode_image(images.get("top"))
        left = _decode_image(images.get("left"))
        right = _decode_image(images.get("right"))
        state16 = np.asarray(kwargs.get("state"), dtype=np.float32).reshape(-1)
        if state16.shape != (16,):
            raise ValueError(
                f"RoboTwin state must have shape (16,), got {state16.shape}"
            )
        state = np.zeros((1, 32), dtype=np.float32)
        state[0, :16] = state16
        return {
            "observation.images.top_head": self._chw(top),
            "observation.images.hand_left": self._chw(left),
            "observation.images.hand_right": self._chw(right),
            "observation.state": state,
            "task": [str(kwargs.get("instruction") or "")],
            "raw_images.top_head": top,
            "raw_images.hand_left": left,
            "raw_images.hand_right": right,
        }

    def dispatch(self, method: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        if method == "robotwin.invalidate_cache":
            self.policy.invalidate_action_cache()
            return {"ok": True}
        if method not in {"robotwin.step", "robotwin.chunk", "robotwin.observe"}:
            raise ValueError(f"unsupported RoboTwin policy method: {method!r}")
        batch = self._batch(kwargs)
        if method == "robotwin.observe":
            self.policy.observe(batch)
            return {"ok": True}
        if method == "robotwin.chunk":
            actions = np.asarray(
                self.policy.get_action_chunk(
                    batch, max_actions=kwargs.get("max_actions")
                ),
                dtype=np.float32,
            )
            if actions.ndim != 2 or actions.shape[1] != 16:
                raise ValueError(
                    f"RoboTwin policy returned invalid chunk {actions.shape}"
                )
            return {
                "actions": actions,
                "shape": list(actions.shape),
                "dtype": "float32",
                "fresh_forward": True,
            }
        action = np.asarray(self.policy.get_action(batch), dtype=np.float32)
        if action.shape != (16,):
            raise ValueError(f"RoboTwin policy returned {action.shape}, expected (16,)")
        return {"action": action, "shape": [16], "dtype": "float32"}
