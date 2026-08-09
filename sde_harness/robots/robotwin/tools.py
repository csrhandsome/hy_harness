"""RoboTwin toolkit backed by the official TASK_ENV interface.

RoboTwin owns the simulator process and calls a policy hook with the current
observation. The adapter in this module keeps that ownership model while
exposing a small HyHarness Toolkit:

* robotwin_observe returns the current state and camera images;
* robotwin_execute_ee sends an absolute dual-arm end-effector action;
* robotwin_vla_step asks the existing Hy-VLA policy wrapper for an action;
* robotwin_status reports the benchmark-side termination state.

The toolkit deliberately does not import RoboTwin itself. This makes it usable
with the official runner and keeps the repository importable when the external
RoboTwin checkout is not installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Protocol

import imageio.v2 as imageio
import numpy as np

from hy_harness.tools.base import BaseTool
from hy_harness.utils.logging import get_output_dir

from .execution import RobotTwinActionExecutor


class RobotTwinTaskEnv(Protocol):
    """Subset of RoboTwin TASK_ENV used by the toolkit."""

    def get_obs(self) -> dict[str, Any]: ...

    def get_instruction(self) -> str: ...

    def take_action(self, action: Any, action_type: str = "ee") -> Any: ...


def _jsonable(value: Any) -> Any:
    """Convert numpy/torch containers into JSON-friendly values."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _png_bytes(image: Any) -> bytes | None:
    """Encode an HWC RGB image for the planner's multimodal tool result."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[-1] not in (1, 3, 4) and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] not in (1, 3, 4):
            return None
    else:
        return None
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        finite_max = float(np.nanmax(arr)) if arr.size else 0.0
        if finite_max <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0.0, 255.0)
    arr = np.ascontiguousarray(arr.astype(np.uint8))
    import io

    buffer = io.BytesIO()
    imageio.imwrite(buffer, arr, format="png")
    return buffer.getvalue()


def _first_image(batch: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in batch and batch[key] is not None:
            return batch[key]
    return None


def _truthy(value: Any) -> bool:
    if callable(value):
        try:
            value = value()
        except TypeError:
            return False
    if isinstance(value, np.ndarray):
        return bool(value.any())
    return bool(value)


class RobotTwinEnvAdapter:
    """Normalize the official RoboTwin TASK_ENV surface for the toolkit."""

    def __init__(
        self,
        task_env: RobotTwinTaskEnv,
        *,
        observation: dict[str, Any] | None = None,
        observation_encoder: Callable[[dict[str, Any], str], dict[str, Any]]
        | None = None,
    ) -> None:
        self.task_env = task_env
        self._observation = observation
        self.observation_encoder = observation_encoder
        if self._observation is None:
            self.refresh()

    @property
    def observation(self) -> dict[str, Any]:
        if self._observation is None:
            self.refresh()
        if self._observation is None:
            raise RuntimeError("RoboTwin observation is unavailable")
        return self._observation

    def refresh(self) -> dict[str, Any]:
        getter = getattr(self.task_env, "get_obs", None)
        if not callable(getter):
            raise AttributeError("TASK_ENV must provide get_obs()")
        self._observation = getter()
        return self._observation

    def instruction(self) -> str:
        getter = getattr(self.task_env, "get_instruction", None)
        if callable(getter):
            return str(getter())
        return str(getattr(self.task_env, "instruction", ""))

    def encoded_observation(self) -> dict[str, Any]:
        observation = self.observation
        if self.observation_encoder is None:
            return observation
        return self.observation_encoder(observation, self.instruction())

    def take_action(self, action: Any, *, action_type: str = "ee") -> Any:
        if self.done():
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        result = self.task_env.take_action(arr, action_type=action_type)
        self.refresh()
        return result

    def done(self) -> bool:
        for name in ("eval_success", "episode_done", "terminated", "done"):
            if hasattr(self.task_env, name) and _truthy(getattr(self.task_env, name)):
                return True
        episode_end = getattr(self.task_env, "is_episode_end", None)
        if callable(episode_end) and _truthy(episode_end):
            return True
        count = getattr(self.task_env, "take_action_cnt", None)
        limit = getattr(self.task_env, "step_lim", None)
        if count is not None and limit is not None:
            return int(count) >= int(limit)
        return False

    def status(self) -> dict[str, Any]:
        count = getattr(self.task_env, "take_action_cnt", None)
        limit = getattr(self.task_env, "step_lim", None)
        success = getattr(self.task_env, "eval_success", False)
        return {
            "instruction": self.instruction(),
            "done": self.done(),
            "success": _truthy(success),
            "take_action_cnt": None if count is None else int(count),
            "step_lim": None if limit is None else int(limit),
        }


class RobotTwinTools(BaseTool):
    """Agent-facing tools for a live RoboTwin task environment."""

    def __init__(
        self,
        *,
        env: RobotTwinEnvAdapter,
        policy: Any | None = None,
        dashboard: Any = None,
    ) -> None:
        super().__init__(dashboard=dashboard)
        self.env = env
        self.policy = policy
        self._records: list[dict[str, Any]] = []
        self._executor = RobotTwinActionExecutor(
            env=env,
            policy=policy,
            record=self._record,
        )
        self._add_robotwin_tools()

    def _add_robotwin_tools(self) -> None:
        self.add_tool(
            "robotwin_observe",
            {
                "name": "robotwin_observe",
                "description": (
                    "Read the current RoboTwin observation. Returns the task "
                    "instruction, dual-arm state, termination status, and the "
                    "head/left-wrist/right-wrist RGB images. Call this before "
                    "choosing an action or after an action changes the scene."
                ),
                "input_schema": {"type": "object", "properties": {}},
            },
            self.observe,
        )
        self.add_tool(
            "robotwin_status",
            {
                "name": "robotwin_status",
                "description": "Read the current RoboTwin task status without taking an action.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.status,
        )
        self.add_tool(
            "robotwin_execute_ee",
            {
                "name": "robotwin_execute_ee",
                "description": (
                    "Execute an absolute dual-arm end-effector action in "
                    "RoboTwin. The action is 16 values: left xyz + quaternion "
                    "(wxyz) + gripper, followed by right xyz + quaternion "
                    "(wxyz) + gripper. Use only after inspecting the current "
                    "observation. RoboTwin receives action_type='ee'."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "array",
                            "description": "Absolute dual-arm EE action with 16 float values.",
                            "items": {"type": "number"},
                            "minItems": 16,
                            "maxItems": 16,
                        },
                        "repeat": {
                            "type": "integer",
                            "description": "Repeat the same action for 1-8 simulator steps.",
                        },
                    },
                    "required": ["action"],
                },
            },
            self.execute_ee,
        )
        self.add_tool(
            "robotwin_vla_step",
            {
                "name": "robotwin_vla_step",
                "description": (
                    "Run the loaded Hy-VLA policy for one or more closed-loop "
                    "steps. The policy sees the current RoboTwin images, "
                    "instruction, and dual-arm state, then its decoded EE "
                    "action is sent to the simulator. Prefer this tool for "
                    "physical manipulation; use robotwin_execute_ee only for "
                    "explicit corrective actions."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "integer",
                            "description": "Number of closed-loop VLA steps, 1-32.",
                        },
                    },
                },
            },
            self.vla_step,
        )

    def _payload(
        self,
        *,
        include_images: bool = True,
        last_action: Any | None = None,
    ) -> dict[str, Any]:
        batch = self.env.encoded_observation()
        payload: dict[str, Any] = {
            **self.env.status(),
            "state": _jsonable(batch.get("observation.state")),
        }
        if last_action is not None:
            payload["last_action"] = _jsonable(last_action)
        if include_images:
            head = _first_image(
                batch,
                "raw_images.top_head",
                "observation.images.top_head",
            )
            left = _first_image(
                batch,
                "raw_images.hand_left",
                "observation.images.hand_left",
            )
            right = _first_image(
                batch,
                "raw_images.hand_right",
                "observation.images.hand_right",
            )
            payload["_image_bytes"] = _png_bytes(head)
            payload["_image_cam_bytes"] = _png_bytes(left)
            payload["_image_wrist_bytes"] = _png_bytes(right)
        return payload

    def observe(self) -> dict[str, Any]:
        self.env.refresh()
        return self._payload(include_images=True)

    def status(self) -> dict[str, Any]:
        return self.env.status()

    def _record(self, action: Any, *, action_type: str, source: str) -> None:
        self._records.append(
            {
                "step": len(self._records),
                "source": source,
                "action_type": action_type,
                "action": _jsonable(action),
                **self.env.status(),
            }
        )

    def execute_ee(self, action: list[float], repeat: int = 1) -> dict[str, Any]:
        result = self._executor.execute_ee(action, repeat=repeat)
        return {
            "applied_steps": result.applied_steps,
            **self._payload(include_images=True, last_action=result.last_action),
        }

    def vla_step(self, steps: int = 1) -> dict[str, Any]:
        result = self._executor.vla_step(steps=steps)
        return {
            "applied_steps": result.applied_steps,
            **self._payload(include_images=True, last_action=result.last_action),
        }

    def write_recipe(self, recipe_tag: str) -> str:
        output_dir = get_output_dir() or Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"robotwin_recipe_{recipe_tag}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for record in self._records:
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return str(path)


__all__ = ["RobotTwinEnvAdapter", "RobotTwinTools"]
