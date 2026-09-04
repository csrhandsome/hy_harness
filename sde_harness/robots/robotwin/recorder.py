"""Versioned HDF5 episode recorder for RoboTwin Harness runs.

The core groups deliberately mirror the HDF5 produced by the UMI MCAP
converter.  A RoboTwin simulator only exposes end-effector poses, rather than
the robot joint angles available on the physical G2, so this recorder writes
the common ``qpos`` and image streams and does *not* invent joint-angle data.

The recorder is append-only while an episode is running and atomically
publishes the final ``.hdf5`` only from :meth:`finalize`.  A partial ``.tmp``
file is therefore never mistaken for trainable data after a crashed run.
"""

from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import imageio.v2 as imageio
import numpy as np


SCHEMA_NAME = "robotwin_harness_hdf5"
SCHEMA_VERSION = 1
CAMERAS = (
    ("cam_high", "raw_images.top_head", "observation.images.top_head"),
    (
        "cam_left_wrist",
        "raw_images.hand_left",
        "observation.images.hand_left",
    ),
    (
        "cam_right_wrist",
        "raw_images.hand_right",
        "observation.images.hand_right",
    ),
)


def _iso8601(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _as_rgb_uint8(value: Any) -> np.ndarray | None:
    """Normalize an HWC/CHW image to RGB ``uint8`` without guessing BGR."""
    if value is None:
        return None
    image = np.asarray(value)
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]
    if image.ndim == 3 and image.shape[-1] not in (1, 3, 4) and image.shape[0] in (
        1,
        3,
        4,
    ):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        return None
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        if image.size and float(np.nanmax(image)) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    return np.ascontiguousarray(image.astype(np.uint8, copy=False))


def _jpeg_bytes(image: np.ndarray, quality: int) -> np.ndarray:
    buffer = io.BytesIO()
    imageio.imwrite(buffer, image, format="jpeg", quality=quality)
    return np.frombuffer(buffer.getvalue(), dtype=np.uint8)


def _dual_arm_wxyz_to_xyzw(value: np.ndarray) -> np.ndarray:
    """Convert RobotWin dual-arm xyz+wxyz+gripper layout to the UMI qpos contract."""
    output = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if output.size != 16:
        raise ValueError(f"dual-arm qpos requires 16 values, got {output.size}")
    output[3:7] = output[[4, 5, 6, 3]]
    output[11:15] = output[[12, 13, 14, 11]]
    return output


class RobotWinHdf5Recorder:
    """Append RobotWin observations and executed EE actions to one episode file.

    Args:
        output_path: Final ``.hdf5`` target.  During recording its sibling
            ``.tmp`` file is used instead.
        instruction: Default task text; per-step overrides are retained under
            ``harness/instruction`` as well.
        task_name: RoboTwin task identifier, when the environment exposes one.

    ``record_step`` consumes the *pre-action* observation and the action that
    is about to be executed.  This makes a row a policy decision sample:
    ``(image_t, qpos_t, instruction_t) -> action_t``.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        instruction: str,
        task_name: str,
        test_num: int | str,
        fps: float = 30.0,
        jpeg_quality: int = 95,
    ) -> None:
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        if not 1 <= int(jpeg_quality) <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")

        self.output_path = Path(output_path).expanduser().resolve()
        if self.output_path.suffix.lower() not in {".h5", ".hdf5"}:
            raise ValueError("output_path must end in .h5 or .hdf5")
        self.temporary_path = self.output_path.with_suffix(
            self.output_path.suffix + ".tmp"
        )
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary_path.exists():
            self.temporary_path.unlink()

        self.fps = float(fps)
        self.jpeg_quality = int(jpeg_quality)
        self._started_at = time.time()
        self._closed = False
        self._frames = 0
        self._file = h5py.File(self.temporary_path, "w")
        self._create_schema(
            instruction=str(instruction), task_name=str(task_name), test_num=test_num
        )

    def _create_schema(self, *, instruction: str, task_name: str, test_num: int | str) -> None:
        f = self._file
        f.attrs["schema_name"] = SCHEMA_NAME
        f.attrs["schema_version"] = SCHEMA_VERSION
        f.attrs["source"] = "robotwin_harness"
        f.attrs["data_id"] = self.output_path.stem
        f.attrs["instruction"] = instruction
        f.attrs["name"] = task_name
        f.attrs["task_id"] = task_name
        f.attrs["test_num"] = str(test_num)
        f.attrs["fps_30"] = self.fps
        f.attrs["main_sensor_30"] = "camera/head"
        f.attrs["timebase"] = "virtual_30hz"
        f.attrs["t0_30"] = self._started_at
        f.attrs["n_frames_30"] = 0
        f.attrs["collect_info/started_at"] = _iso8601(self._started_at)
        f.attrs["collect_info/collector_uid"] = "robotwin_harness"
        f.attrs["available_modalities"] = json.dumps(
            ["qpos", "images", "executed_ee_action"], ensure_ascii=False
        )
        f.create_dataset("aligned_timestamp30", shape=(0,), maxshape=(None,), dtype=np.float64)

        observations = f.create_group("observations")
        actions = f.create_group("action")
        self._create_numeric_stream(observations, "qpos", width=16)
        self._create_numeric_stream(actions, "qpos", width=16)

        images = observations.create_group("images")
        for camera, _, _ in CAMERAS:
            stream = images.create_group(camera)
            stream.create_dataset(
                "value",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.vlen_dtype(np.dtype("uint8")),
            )
            stream.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
            stream.create_dataset(
                "aligned_index30", shape=(0,), maxshape=(None,), dtype=np.int64
            )
            stream.create_dataset("valid", shape=(0,), maxshape=(None,), dtype=np.bool_)
            stream.attrs["encoding"] = "jpeg"
            stream.attrs["color_space"] = "RGB"

        subtask = f.create_group("subtask")
        subtask.create_dataset(
            "aligned_timestamp30", shape=(0,), maxshape=(None,), dtype=np.int32
        )

        annotations = f.create_group("annotations")
        self._create_optional_label(annotations, "progress", np.float32, np.nan)
        self._create_optional_label(annotations, "task_correct", np.int8, -1)
        safety = annotations.create_group("safety")
        self._create_optional_label(safety, "risk_within_horizon", np.int8, -1)
        annotations["task_correct"].attrs[
            "definition"
        ] = "broadcast terminal RoboTwin success label; 1=success, 0=failure"

        outcome = annotations.create_group("outcome")
        outcome.create_dataset("success", data=np.array(-1, dtype=np.int8))
        outcome.create_dataset("status", data="unknown", dtype=h5py.string_dtype("utf-8"))
        outcome.create_dataset("terminal_timestamp", data=np.array(np.nan, dtype=np.float64))
        outcome.attrs[
            "definition"
        ] = "RoboTwin TASK_ENV.eval_success at recorder finalization"

        harness = f.create_group("harness")
        string_dtype = h5py.string_dtype("utf-8")
        harness.create_dataset("action_source", shape=(0,), maxshape=(None,), dtype=string_dtype)
        harness.create_dataset("instruction", shape=(0,), maxshape=(None,), dtype=string_dtype)
        harness.create_dataset("wall_timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        harness.create_dataset("pre_action_done", shape=(0,), maxshape=(None,), dtype=np.bool_)
        harness.create_dataset("pre_action_success", shape=(0,), maxshape=(None,), dtype=np.bool_)

    def _create_numeric_stream(self, parent: h5py.Group, name: str, *, width: int) -> None:
        stream = parent.create_group(name)
        stream.create_dataset(
            "value", shape=(0, width), maxshape=(None, width), dtype=np.float32
        )
        stream.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        stream.create_dataset(
            "aligned_index30", shape=(0,), maxshape=(None,), dtype=np.int64
        )
        stream.attrs["layout"] = "left xyz + quaternion xyzw + gripper, right xyz + quaternion xyzw + gripper"
        stream.attrs["quaternion_order"] = "xyzw"

    def _create_optional_label(
        self, parent: h5py.Group, name: str, dtype: Any, fill_value: Any
    ) -> None:
        group = parent.create_group(name)
        group.create_dataset(
            "value", shape=(0,), maxshape=(None,), dtype=dtype, fillvalue=fill_value
        )
        group.create_dataset("valid", shape=(0,), maxshape=(None,), dtype=np.bool_)

    @property
    def frames(self) -> int:
        return self._frames

    def record_step(
        self,
        *,
        observation: dict[str, Any],
        action: Any,
        action_source: str,
        pre_action_status: dict[str, Any] | None = None,
    ) -> None:
        """Append one pre-action observation / executed action pair."""
        if self._closed:
            raise RuntimeError("cannot record into a finalized episode")
        state = np.asarray(observation.get("observation.state"), dtype=np.float32)
        if state.ndim == 2:
            state = state[0]
        state = state.reshape(-1)
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if state.size < 16:
            raise ValueError(f"RobotWin observation.state requires 16 values, got {state.size}")
        if action_array.size != 16:
            raise ValueError(f"RobotWin EE action requires 16 values, got {action_array.size}")

        index = self._frames
        timestamp = self._started_at + index / self.fps
        f = self._file
        self._append_1d(f["aligned_timestamp30"], np.float64(timestamp))
        self._append_numeric_stream(
            f["observations/qpos"], _dual_arm_wxyz_to_xyzw(state[:16]), timestamp, index
        )
        self._append_numeric_stream(
            f["action/qpos"], _dual_arm_wxyz_to_xyzw(action_array), timestamp, index
        )

        for camera, raw_key, encoded_key in CAMERAS:
            image = _as_rgb_uint8(observation.get(raw_key, observation.get(encoded_key)))
            stream = f[f"observations/images/{camera}"]
            valid = image is not None
            encoded = _jpeg_bytes(image, self.jpeg_quality) if valid else np.empty(0, dtype=np.uint8)
            self._append_1d(stream["value"], encoded)
            self._append_1d(stream["timestamp"], np.float64(timestamp))
            self._append_1d(stream["aligned_index30"], np.int64(index))
            self._append_1d(stream["valid"], bool(valid))

        self._append_1d(f["subtask/aligned_timestamp30"], np.int32(-1))
        self._append_optional_unknown(f["annotations/progress"], np.nan)
        self._append_optional_unknown(f["annotations/task_correct"], -1)
        self._append_optional_unknown(f["annotations/safety/risk_within_horizon"], -1)

        task = observation.get("task") or [f.attrs["instruction"]]
        instruction = _text(task[0] if isinstance(task, (list, tuple)) else task)
        self._append_1d(f["harness/action_source"], str(action_source))
        self._append_1d(f["harness/instruction"], instruction)
        self._append_1d(f["harness/wall_timestamp"], np.float64(time.time()))
        status = dict(pre_action_status or {})
        self._append_1d(f["harness/pre_action_done"], bool(status.get("done", False)))
        self._append_1d(f["harness/pre_action_success"], bool(status.get("success", False)))
        self._frames += 1

    def finalize(
        self,
        *,
        success: bool,
        status: str,
        outcome: dict[str, Any] | None = None,
        failure_reason: str | None = None,
    ) -> Path:
        """Write terminal labels and atomically publish the completed file."""
        if self._closed:
            return self.output_path
        finished_at = time.time()
        f = self._file
        f.attrs["n_frames_30"] = self._frames
        f.attrs["collect_info/finished_at"] = _iso8601(finished_at)
        f.attrs["outcome_status"] = str(status)
        f.attrs["benchmark_success"] = bool(success)
        f.attrs["failure_reason"] = "" if failure_reason is None else str(failure_reason)
        f.attrs["outcome"] = json.dumps(outcome or {}, ensure_ascii=False, default=str)
        f["annotations/outcome/success"][()] = np.int8(bool(success))
        f["annotations/outcome/status"][()] = str(status)
        f["annotations/outcome/terminal_timestamp"][()] = np.float64(finished_at)

        if self._frames:
            correct = f["annotations/task_correct"]
            correct["value"][:] = np.int8(bool(success))
            correct["valid"][:] = True

        f.flush()
        f.close()
        self._closed = True
        os.replace(self.temporary_path, self.output_path)
        return self.output_path

    def abort(self, error: BaseException) -> Path:
        """Finalize a failed run while preserving samples recorded before an error."""
        return self.finalize(
            success=False,
            status="error",
            outcome={"exception_type": type(error).__name__},
            failure_reason=str(error),
        )

    @staticmethod
    def _append_1d(dataset: h5py.Dataset, value: Any) -> None:
        size = dataset.shape[0]
        dataset.resize((size + 1,))
        dataset[size] = value

    @staticmethod
    def _append_numeric_stream(
        stream: h5py.Group, value: np.ndarray, timestamp: float, aligned_index: int
    ) -> None:
        dataset = stream["value"]
        size = dataset.shape[0]
        dataset.resize((size + 1, dataset.shape[1]))
        dataset[size] = value
        RobotWinHdf5Recorder._append_1d(stream["timestamp"], np.float64(timestamp))
        RobotWinHdf5Recorder._append_1d(
            stream["aligned_index30"], np.int64(aligned_index)
        )

    @staticmethod
    def _append_optional_unknown(group: h5py.Group, value: Any) -> None:
        RobotWinHdf5Recorder._append_1d(group["value"], value)
        RobotWinHdf5Recorder._append_1d(group["valid"], False)


__all__ = ["CAMERAS", "SCHEMA_NAME", "SCHEMA_VERSION", "RobotWinHdf5Recorder"]
