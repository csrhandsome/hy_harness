"""Simulator-affecting RoboTwin action execution.

This module is intentionally independent of BaseTool and tool schemas, so the
same executor can be reused by a planner tool, a scripted rollout, or a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import numpy as np


class RobotTwinActionEnv(Protocol):
    """Environment operations needed by a simulator action executor."""

    def encoded_observation(self) -> dict[str, Any]: ...

    def take_action(self, action: Any, *, action_type: str = "ee") -> Any: ...

    def done(self) -> bool: ...


@dataclass(frozen=True)
class ExecutionResult:
    """The result of executing one or more simulator actions."""

    applied_steps: int
    last_action: np.ndarray | None


class RobotTwinActionExecutor:
    """Run direct EE actions and closed-loop Hy-VLA policy actions."""

    EE_ACTION_SIZE = 16
    MAX_DIRECT_REPEAT = 8
    MAX_VLA_STEPS = 32

    def __init__(
        self,
        *,
        env: RobotTwinActionEnv,
        policy: Any | None = None,
        record: Callable[..., None] | None = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self._record = record

    def execute_ee(self, action: Any, *, repeat: int = 1) -> ExecutionResult:
        """Execute a validated absolute dual-arm EE action."""
        arr = self._as_ee_action(action, source="RoboTwin EE action")
        count = max(1, min(int(repeat), self.MAX_DIRECT_REPEAT))
        applied = 0
        for _ in range(count):
            self.env.take_action(arr, action_type="ee")
            self._record_action(arr, source="robotwin_execute_ee")
            applied += 1
            if self.env.done():
                break
        return ExecutionResult(applied_steps=applied, last_action=arr)

    def vla_step(self, *, steps: int = 1) -> ExecutionResult:
        """Decode and execute one or more closed-loop Hy-VLA actions."""
        count = max(1, min(int(steps), self.MAX_VLA_STEPS))
        applied = 0
        last_action: np.ndarray | None = None
        for _ in range(count):
            if self.env.done():
                break
            last_action = self.predict_action()
            self.env.take_action(last_action, action_type="ee")
            self._record_action(last_action, source="robotwin_vla_step")
            applied += 1
        return ExecutionResult(applied_steps=applied, last_action=last_action)

    def predict_action(self) -> np.ndarray:
        """Get and validate one absolute dual-arm EE action from the policy."""
        if self.policy is None:
            raise RuntimeError(
                "robotwin_vla_step requires a Hy-VLA policy; "
                "disable direct Toolkit-only mode or provide policy"
            )
        batch = self.env.encoded_observation()
        getter = getattr(self.policy, "get_action", None)
        if callable(getter):
            action = getter(batch)
        elif callable(self.policy):
            action = self.policy(batch)
        else:
            raise TypeError("policy must provide get_action(batch) or be callable")
        return self._as_ee_action(
            action,
            source="Hy-VLA RoboTwin action",
            first_batch_item=True,
        )

    def _record_action(self, action: np.ndarray, *, source: str) -> None:
        if self._record is not None:
            self._record(action, action_type="ee", source=source)

    @classmethod
    def _as_ee_action(
        cls,
        action: Any,
        *,
        source: str,
        first_batch_item: bool = False,
    ) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32)
        if first_batch_item and arr.ndim == 2:
            arr = arr[0]
        arr = arr.reshape(-1)
        if arr.size != cls.EE_ACTION_SIZE:
            raise ValueError(
                f"{source} must contain {cls.EE_ACTION_SIZE} values, got {arr.size}"
            )
        return arr


__all__ = ["ExecutionResult", "RobotTwinActionExecutor"]
