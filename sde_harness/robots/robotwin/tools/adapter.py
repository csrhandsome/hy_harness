"""Small adapter around the official live RoboTwin ``TASK_ENV`` object."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

import numpy as np


class RobotTwinTaskEnv(Protocol):
    def get_obs(self) -> dict[str, Any]: ...
    def get_instruction(self) -> str: ...
    def take_action(self, action: Any, action_type: str = "ee") -> Any: ...


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
    """Normalize the subset of RoboTwin used by planner tools."""

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
            raise TypeError("TASK_ENV must provide get_obs()")
        self._observation = getter()
        return self._observation

    def instruction(self) -> str:
        getter = getattr(self.task_env, "get_instruction", None)
        if callable(getter):
            return str(getter())
        return str(getattr(self.task_env, "instruction", ""))

    def encoded_observation(self) -> dict[str, Any]:
        if self.observation_encoder is None:
            return self.observation
        return self.observation_encoder(self.observation, self.instruction())

    def ee_state(self) -> np.ndarray:
        """Return current ``left/right xyz + quat_wxyz + gripper`` state."""
        state = np.asarray(
            self.encoded_observation().get("observation.state"), dtype=np.float32
        )
        if state.ndim == 2:
            state = state[0]
        state = state.reshape(-1)
        if state.size < 16:
            raise ValueError(f"RoboTwin EE state needs 16 values, got {state.size}")
        return state[:16].copy()

    def take_action(self, action: Any, *, action_type: str = "ee") -> Any:
        if self.done():
            return None
        result = self.task_env.take_action(
            np.asarray(action, dtype=np.float32).reshape(-1), action_type=action_type
        )
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
        return count is not None and limit is not None and int(count) >= int(limit)

    def status(self) -> dict[str, Any]:
        count = getattr(self.task_env, "take_action_cnt", None)
        limit = getattr(self.task_env, "step_lim", None)
        return {
            "instruction": self.instruction(),
            "done": self.done(),
            "success": _truthy(getattr(self.task_env, "eval_success", False)),
            "take_action_cnt": None if count is None else int(count),
            "step_lim": None if limit is None else int(limit),
        }


__all__ = ["RobotTwinEnvAdapter", "RobotTwinTaskEnv"]
