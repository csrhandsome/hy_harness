"""Hy-VLA policy wrapper for the single-arm LIBERO action protocol.

The checkpoint is expected to have been trained on LIBERO-style observations:
two RGB views, an 8-D proprioceptive state, and a 7-D action
``[dx, dy, dz, dax, day, daz, gripper]``.  Hy-VLA internally pads state and
action tensors to its configured maximum dimensions, so checkpoints whose
action head still exposes the repository default width remain usable; only the
first ``action_dim`` channels are decoded for LIBERO.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hy_vla import HyVLA, HyVLAConfig


def _load_norm_stats(path: str | Path) -> dict[str, np.ndarray]:
    norm_path = Path(path)
    if norm_path.suffix.lower() == ".json":
        raw = json.loads(norm_path.read_text(encoding="utf-8"))
    else:
        with norm_path.open("rb") as handle:
            raw = pickle.load(handle)
    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    missing = [key for key in required if key not in raw]
    if missing:
        raise KeyError(f"{norm_path}: missing norm-stat keys: {', '.join(missing)}")
    return {key: np.asarray(raw[key], dtype=np.float32) for key in required}


def resolve_norm_path(checkpoint: str | Path, norm_path: str | Path | None) -> Path:
    if norm_path:
        resolved = Path(norm_path).expanduser()
    else:
        resolved = Path(checkpoint).expanduser() / "norm_stats.pkl"
    if not resolved.is_file():
        raise FileNotFoundError(
            f"LIBERO norm stats not found: {resolved}. Pass --norm-path or put "
            "norm_stats.pkl next to the Hy-VLA checkpoint."
        )
    return resolved.resolve()


def _stats_for_chunk(stats: np.ndarray, steps: int, dim: int) -> np.ndarray:
    """Return stats broadcastable to ``(steps, dim)``."""
    arr = np.asarray(stats, dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] < dim:
            raise ValueError(f"norm-stat width {arr.shape[0]} is smaller than action_dim={dim}")
        return np.broadcast_to(arr[:dim], (steps, dim))
    if arr.ndim != 2 or arr.shape[1] < dim:
        raise ValueError(f"expected action stats shaped [D] or [T,D], got {arr.shape}")
    if arr.shape[0] < steps:
        pad = np.repeat(arr[-1:], steps - arr.shape[0], axis=0)
        arr = np.concatenate((arr, pad), axis=0)
    return arr[:steps, :dim]


class HyVLALiberoPolicy:
    """Load a Hy-VLA checkpoint and produce executable LIBERO action chunks."""

    def __init__(
        self,
        checkpoint: str,
        *,
        norm_path: str | None = None,
        vlm_model_path: str | None = None,
        action_dim: int = 7,
        num_open_loop_steps: int | None = None,
        gripper_mode: str = "libero",
        device: str = "cuda",
        dtype: str = "bfloat16",
        img_history_size: int | None = None,
        img_history_interval: int = 1,
    ) -> None:
        if action_dim < 7:
            raise ValueError("LIBERO requires action_dim >= 7")
        if gripper_mode not in {"libero", "openvla", "raw"}:
            raise ValueError("gripper_mode must be libero|openvla|raw")

        self.checkpoint = checkpoint
        self.action_dim = int(action_dim)
        self.gripper_mode = gripper_mode
        self.device = torch.device(device)
        self.weight_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype]

        self.config = HyVLAConfig.from_pretrained(checkpoint)
        self.policy = HyVLA.from_pretrained(
            checkpoint,
            config=self.config,
            vlm_model_path=vlm_model_path,
        )
        self.policy.enable_video_encoder_if_needed()
        self.policy = self.policy.to(device=self.device, dtype=self.weight_dtype).eval()

        resolved_norm = resolve_norm_path(checkpoint, norm_path)
        self.norm = _load_norm_stats(resolved_norm)
        self.num_open_loop_steps = int(
            num_open_loop_steps or getattr(self.config, "n_action_steps", 1)
        )
        self.use_video_encoder = bool(getattr(self.config, "use_video_encoder", False))
        self.img_history_size = int(
            img_history_size
            or (6 if self.use_video_encoder else 1)
        )
        self.img_history_interval = int(img_history_interval)
        self._main_history: list[np.ndarray] = []
        self._wrist_history: list[np.ndarray] = []

    def reset(self) -> None:
        self.policy.reset()
        self._main_history.clear()
        self._wrist_history.clear()

    @staticmethod
    def _image_tensor(image: np.ndarray) -> torch.Tensor:
        arr = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
        return torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0)

    def _history_tensor(self, history: list[np.ndarray]) -> torch.Tensor:
        step = len(history) - 1
        indices: list[int] = []
        valid: list[bool] = []
        for slot in range(self.img_history_size):
            raw = step - (self.img_history_size - 1 - slot) * self.img_history_interval
            indices.append(max(raw, 0))
            valid.append(raw >= 0)
        frames = torch.stack([self._image_tensor(history[index]) for index in indices])
        for slot, is_valid in enumerate(valid):
            if not is_valid:
                frames[slot].zero_()
        return frames.unsqueeze(0)

    def _build_batch(
        self,
        instruction: str,
        main_image: np.ndarray,
        wrist_image: np.ndarray | None,
        state: np.ndarray,
    ) -> dict[str, Any]:
        state_arr = np.asarray(state, dtype=np.float32).reshape(-1)
        q_mean = self.norm["qpos_mean"].reshape(-1)
        q_std = self.norm["qpos_std"].reshape(-1)
        if q_mean.shape[0] > state_arr.shape[0]:
            raise ValueError(
                f"checkpoint norm state width {q_mean.shape[0]} exceeds LIBERO state "
                f"width {state_arr.shape[0]}"
            )
        state_arr = state_arr.copy()
        state_arr[: q_mean.shape[0]] = (
            state_arr[: q_mean.shape[0]] - q_mean
        ) / (q_std + 1e-8)

        main = np.asarray(main_image, dtype=np.uint8)
        wrist = main if wrist_image is None else np.asarray(wrist_image, dtype=np.uint8)
        self._main_history.append(main)
        self._wrist_history.append(wrist)

        image_keys = list(self.config.image_features)
        if not image_keys:
            raise ValueError("Hy-VLA checkpoint config has no image_features")
        views = (self._main_history, self._wrist_history)
        batch: dict[str, Any] = {
            "observation.state": torch.from_numpy(state_arr).unsqueeze(0),
            "task": [instruction],
        }
        for index, key in enumerate(image_keys):
            history = views[min(index, len(views) - 1)]
            if index >= len(views):
                history = [np.zeros_like(main) for _ in range(len(self._main_history))]
            if self.use_video_encoder:
                batch[key] = self._history_tensor(history)
            else:
                batch[key] = self._image_tensor(history[-1]).unsqueeze(0)

        for key, value in tuple(batch.items()):
            if isinstance(value, torch.Tensor):
                batch[key] = value.to(device=self.device, dtype=self.weight_dtype)
        return batch

    def _decode(self, normalized: np.ndarray) -> np.ndarray:
        if normalized.ndim != 2 or normalized.shape[1] < self.action_dim:
            raise ValueError(
                f"Hy-VLA returned {normalized.shape}; expected [T,D] with D >= {self.action_dim}"
            )
        steps = min(normalized.shape[0], self.num_open_loop_steps)
        pred = normalized[:steps, : self.action_dim]
        mean = _stats_for_chunk(self.norm["action_mean"], steps, self.action_dim)
        std = _stats_for_chunk(self.norm["action_std"], steps, self.action_dim)
        actions = pred * std + mean

        if self.gripper_mode == "openvla":
            # OpenVLA convention is [0, 1] with 0=close, 1=open. LIBERO uses
            # [-1, 1] with -1=open, +1=close.
            actions[:, -1] = np.where(actions[:, -1] > 0.5, -1.0, 1.0)
        elif self.gripper_mode == "libero":
            actions[:, -1] = np.where(actions[:, -1] >= 0.0, 1.0, -1.0)
        return actions.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        instruction: str,
        main_image: np.ndarray,
        wrist_image: np.ndarray | None,
        state: np.ndarray,
    ) -> np.ndarray:
        batch = self._build_batch(instruction, main_image, wrist_image, state)
        self.policy.reset()
        pred = self.policy.forward_evaluate(batch)["pred"][0].float().cpu().numpy()
        return self._decode(pred)


__all__ = ["HyVLALiberoPolicy", "resolve_norm_path"]
