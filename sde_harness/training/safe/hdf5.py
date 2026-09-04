"""Read the common subset of UMI-style and RobotWin recorder HDF5 files."""

from __future__ import annotations

import io
from typing import Any

import h5py
import numpy as np
from PIL import Image


CAMERA_PATHS = {
    "top_head": "cam_high",
    "hand_left": "cam_left_wrist",
    "hand_right": "cam_right_wrist",
}


def decode_text(value: Any) -> str:
    """Decode HDF5 string attributes without retaining numpy scalar wrappers."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def aligned_raw_index(stream: h5py.Group, step30: int) -> int:
    indices = stream["aligned_index30"]
    if step30 < 0 or step30 >= len(indices):
        raise IndexError(f"step30={step30} is outside aligned stream of length {len(indices)}")
    raw_index = int(indices[step30])
    if raw_index < 0 or raw_index >= len(stream["value"]):
        raise ValueError(f"stream has no valid raw sample for step30={step30}: {raw_index}")
    return raw_index


def decode_image(stream: h5py.Group, step30: int) -> np.ndarray:
    """Read one aligned JPEG stream frame as an RGB HWC uint8 array."""
    raw_index = aligned_raw_index(stream, step30)
    if "valid" in stream and not bool(stream["valid"][step30]):
        raise ValueError(f"image stream {stream.name} is invalid at step30={step30}")
    encoded = np.asarray(stream["value"][raw_index], dtype=np.uint8)
    if encoded.size == 0:
        raise ValueError(f"image stream {stream.name} is empty at step30={step30}")
    with Image.open(io.BytesIO(encoded.tobytes())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_qpos(h5_file: h5py.File, step30: int) -> np.ndarray:
    stream = h5_file["observations/qpos"]
    raw_index = aligned_raw_index(stream, step30)
    qpos = np.asarray(stream["value"][raw_index], dtype=np.float32).reshape(-1)
    if qpos.size < 16:
        raise ValueError(f"{h5_file.filename} qpos contains {qpos.size}, expected at least 16")
    qpos = qpos[:16].copy()
    # UMI-compatible files persist xyzw. RobotWin's online wrapper accepts
    # wxyz, so only this simulator source is converted at the boundary.
    if str(h5_file.attrs.get("source", "")) == "robotwin_harness":
        qpos[3:7] = qpos[[6, 3, 4, 5]]
        qpos[11:15] = qpos[[14, 11, 12, 13]]
    return qpos


def instruction_at(h5_file: h5py.File, step30: int) -> str:
    """Use per-step Harness instruction when present, otherwise root instruction."""
    step_instruction = h5_file.get("harness/instruction")
    if step_instruction is not None and 0 <= step30 < len(step_instruction):
        return decode_text(step_instruction[step30])
    if "instruction" not in h5_file.attrs:
        raise KeyError(f"{h5_file.filename} has no root instruction attribute")
    return decode_text(h5_file.attrs["instruction"])


def build_policy_batch(h5_file: h5py.File, step30: int) -> dict[str, Any]:
    """Build the exact raw-image batch contract used by RobotWin Hy-VLA.

    The policy wrapper performs resize/padding, normalization, tokenizer prompt
    formatting and optional video-history assembly.  Keeping this function at
    the raw HDF5 boundary prevents those model-specific operations from being
    duplicated in the data reader.
    """
    qpos = read_qpos(h5_file, step30)
    state = np.zeros((1, 32), dtype=np.float32)
    state[0, :16] = qpos
    images = {
        name: decode_image(h5_file[f"observations/images/{camera}"], step30)
        for name, camera in CAMERA_PATHS.items()
    }
    return {
        "observation.state": state,
        "task": [instruction_at(h5_file, step30)],
        "raw_images.top_head": images["top_head"],
        "raw_images.hand_left": images["hand_left"],
        "raw_images.hand_right": images["hand_right"],
    }


def terminal_risk_label(h5_file: h5py.File, step30: int) -> tuple[int, str] | None:
    """Return ``(risk, source)`` without conflating unknown labels with zero.

    Explicit safety-window annotations take priority.  Recorder files then
    fall back to their documented broadcast terminal task outcome, which is an
    *outcome risk* target, not a direct collision/constraint-risk label.
    """
    risk = h5_file.get("annotations/safety/risk_within_horizon")
    if risk is not None and step30 < len(risk["value"]) and bool(risk["valid"][step30]):
        value = int(risk["value"][step30])
        if value in (0, 1):
            return value, "safety_risk_within_horizon"

    correct = h5_file.get("annotations/task_correct")
    if correct is not None and step30 < len(correct["value"]) and bool(correct["valid"][step30]):
        value = int(correct["value"][step30])
        if value in (0, 1):
            return 1 - value, "terminal_task_failure"
    return None


__all__ = [
    "CAMERA_PATHS",
    "aligned_raw_index",
    "build_policy_batch",
    "decode_image",
    "decode_text",
    "instruction_at",
    "terminal_risk_label",
]
