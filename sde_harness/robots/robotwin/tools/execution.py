"""Simulator-affecting RoboTwin execution primitives."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class RobotTwinActionEnv(Protocol):
    def encoded_observation(self) -> dict[str, Any]: ...
    def ee_state(self) -> np.ndarray: ...
    def take_action(self, action: Any, *, action_type: str = "ee") -> Any: ...
    def done(self) -> bool: ...


@dataclass(frozen=True)
class ExecutionResult:
    applied_steps: int
    last_action: np.ndarray | None
    forward_count: int = 0
    generated_actions: int = 0
    discarded_actions: int = 0
    chunk_id: int | None = None


def _unit_quaternion(value: Any, *, label: str) -> np.ndarray:
    quat = np.asarray(value, dtype=np.float32).reshape(-1)
    if quat.size != 4:
        raise ValueError(f"{label} must contain 4 wxyz values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError(f"{label} cannot be the zero quaternion")
    return quat / norm


def _quat_angle(a: np.ndarray, b: np.ndarray) -> float:
    return float(2.0 * np.arccos(np.clip(abs(float(np.dot(a, b))), 0.0, 1.0)))


def _quat_step(a: np.ndarray, b: np.ndarray, max_angle: float) -> np.ndarray:
    """Shortest-path normalized interpolation for wxyz quaternions."""
    if float(np.dot(a, b)) < 0.0:
        b = -b
    angle = _quat_angle(a, b)
    if angle <= max_angle or angle < 1e-8:
        return b.copy()
    fraction = max_angle / angle
    mixed = (1.0 - fraction) * a + fraction * b
    return mixed / np.linalg.norm(mixed)


class RobotTwinActionExecutor:
    """Execute fresh VLA chunks and deterministic dual-arm EE primitives."""

    EE_ACTION_SIZE = 16
    MAX_DIRECT_REPEAT = 32
    MAX_VLA_EXECUTE_STEPS = 50
    MAX_TRANSLATION_METERS = 0.30

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
        self._chunk_id = 0

    def _invalidate_vla_cache(self) -> None:
        invalidator = getattr(self.policy, "invalidate_action_cache", None)
        if callable(invalidator):
            invalidator()

    def _sync_policy_observation(self) -> None:
        observer = getattr(self.policy, "observe", None)
        if callable(observer):
            observer(self.env.encoded_observation())

    def execute_ee(
        self, action: Any, *, repeat: int = 1, source: str = "robotwin_execute_ee"
    ) -> ExecutionResult:
        """Execute an absolute action after discarding stale VLA actions."""
        self._invalidate_vla_cache()
        arr = self._as_ee_action(action, source="RoboTwin EE action")
        return self._execute_actions(
            [arr] * self._bounded_repeat(repeat), source=source
        )

    def vla_chunk(
        self,
        *,
        execute_steps: int,
        instruction_override: str | None = None,
    ) -> ExecutionResult:
        """One fresh model forward followed by a prefix of that new chunk."""
        if self.policy is None:
            raise RuntimeError("robotwin_vla_chunk requires a Hy-VLA policy")
        getter = getattr(self.policy, "get_action_chunk", None)
        if not callable(getter):
            raise TypeError(
                "policy must implement get_action_chunk(batch, max_actions=...); "
                "cached get_action() would violate fresh-chunk semantics"
            )
        requested = max(1, min(int(execute_steps), self.MAX_VLA_EXECUTE_STEPS))
        batch = dict(self.env.encoded_observation())
        if instruction_override:
            batch["task"] = [str(instruction_override)]
        actions = np.asarray(getter(batch, max_actions=None), dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        if actions.ndim != 2 or actions.shape[1] != self.EE_ACTION_SIZE:
            raise ValueError(
                f"Hy-VLA chunk must have shape (T, 16), got {actions.shape}"
            )
        if not len(actions):
            raise ValueError("Hy-VLA returned an empty action chunk")

        self._chunk_id += 1
        executed = 0
        last: np.ndarray | None = None
        for index, action in enumerate(actions[:requested]):
            if self.env.done():
                break
            last = self._as_ee_action(action, source="Hy-VLA chunk action")
            self.env.take_action(last, action_type="ee")
            self._record_action(last, source="robotwin_vla_chunk")
            executed += 1
            # Preserve video-history cadence without an extra model forward.
            if index + 1 < min(requested, len(actions)) and not self.env.done():
                self._sync_policy_observation()
        return ExecutionResult(
            applied_steps=executed,
            last_action=last,
            forward_count=1,
            generated_actions=len(actions),
            discarded_actions=max(0, len(actions) - executed),
            chunk_id=self._chunk_id,
        )

    def move_arm(
        self,
        *,
        arm: str,
        xyz: Any | None = None,
        quaternion_wxyz: Any | None = None,
        gripper: float | None = None,
        max_steps: int = 12,
        position_step: float = 0.03,
        rotation_step: float = 0.20,
    ) -> ExecutionResult:
        arm = self._single_arm(arm)
        current = self.env.ee_state()
        target = current.copy()
        offset = 0 if arm == "left" else 8
        if xyz is not None:
            target[offset : offset + 3] = self._xyz(xyz, label=f"{arm} xyz")
        if quaternion_wxyz is not None:
            target[offset + 3 : offset + 7] = _unit_quaternion(
                quaternion_wxyz, label=f"{arm} quaternion_wxyz"
            )
        if gripper is not None:
            target[offset + 7] = float(gripper)
        if xyz is None and quaternion_wxyz is None and gripper is None:
            raise ValueError("xyz, quaternion_wxyz, or gripper is required")
        return self._move_to_target(
            target,
            active_arms=(arm,),
            max_steps=max_steps,
            position_step=position_step,
            rotation_step=rotation_step,
            source="robotwin_move_arm",
        )

    def translate_arm(
        self,
        *,
        arm: str,
        delta_xyz: Any,
        gripper: float | None = None,
        max_steps: int = 12,
        position_step: float = 0.03,
    ) -> ExecutionResult:
        arm = self._single_arm(arm)
        delta = self._xyz(delta_xyz, label="delta_xyz")
        distance = float(np.linalg.norm(delta))
        if distance > self.MAX_TRANSLATION_METERS:
            raise ValueError(
                f"relative translation {distance:.3f} m exceeds "
                f"{self.MAX_TRANSLATION_METERS:.2f} m safety bound"
            )
        offset = 0 if arm == "left" else 8
        return self.move_arm(
            arm=arm,
            xyz=self.env.ee_state()[offset : offset + 3] + delta,
            gripper=gripper,
            max_steps=max_steps,
            position_step=position_step,
        )

    def move_bimanual(
        self,
        *,
        left_xyz: Any | None = None,
        left_quaternion_wxyz: Any | None = None,
        left_gripper: float | None = None,
        right_xyz: Any | None = None,
        right_quaternion_wxyz: Any | None = None,
        right_gripper: float | None = None,
        max_steps: int = 12,
        position_step: float = 0.03,
        rotation_step: float = 0.20,
    ) -> ExecutionResult:
        target = self.env.ee_state()
        active: list[str] = []
        for arm, offset, xyz, quat, grip in (
            ("left", 0, left_xyz, left_quaternion_wxyz, left_gripper),
            ("right", 8, right_xyz, right_quaternion_wxyz, right_gripper),
        ):
            if xyz is not None or quat is not None or grip is not None:
                active.append(arm)
            if xyz is not None:
                target[offset : offset + 3] = self._xyz(xyz, label=f"{arm} xyz")
            if quat is not None:
                target[offset + 3 : offset + 7] = _unit_quaternion(
                    quat, label=f"{arm} quaternion_wxyz"
                )
            if grip is not None:
                target[offset + 7] = float(grip)
        if not active:
            raise ValueError("at least one left_* or right_* target is required")
        return self._move_to_target(
            target,
            active_arms=tuple(active),
            max_steps=max_steps,
            position_step=position_step,
            rotation_step=rotation_step,
            source="robotwin_move_bimanual",
        )

    def rotate_arm(
        self,
        *,
        arm: str,
        quaternion_wxyz: Any,
        gripper: float | None = None,
        max_steps: int = 12,
        rotation_step: float = 0.20,
    ) -> ExecutionResult:
        return self.move_arm(
            arm=arm,
            quaternion_wxyz=quaternion_wxyz,
            gripper=gripper,
            max_steps=max_steps,
            rotation_step=rotation_step,
        )

    def set_gripper(self, *, arm: str, value: float, steps: int = 4) -> ExecutionResult:
        arm = str(arm).strip().lower()
        if arm not in {"left", "right", "both"}:
            raise ValueError("arm must be 'left', 'right', or 'both'")
        current = self.env.ee_state()
        if arm in ("left", "both"):
            current[7] = float(value)
        if arm in ("right", "both"):
            current[15] = float(value)
        return self.execute_ee(current, repeat=steps, source="robotwin_set_gripper")

    def release(
        self, *, arm: str, open_value: float, steps: int = 4
    ) -> ExecutionResult:
        arm = str(arm).strip().lower()
        if arm not in {"left", "right", "both"}:
            raise ValueError("arm must be 'left', 'right', or 'both'")
        current = self.env.ee_state()
        if arm in ("left", "both"):
            current[7] = float(open_value)
        if arm in ("right", "both"):
            current[15] = float(open_value)
        return self.execute_ee(current, repeat=steps, source="robotwin_release")

    def hold(self, *, steps: int = 1) -> ExecutionResult:
        return self.execute_ee(
            self.env.ee_state(), repeat=steps, source="robotwin_hold"
        )

    def _move_to_target(
        self,
        target: np.ndarray,
        *,
        active_arms: tuple[str, ...],
        max_steps: int,
        position_step: float,
        rotation_step: float,
        source: str,
    ) -> ExecutionResult:
        self._invalidate_vla_cache()
        max_steps = self._bounded_repeat(max_steps)
        position_step = float(position_step)
        rotation_step = float(rotation_step)
        if not 0.0 < position_step <= 0.10:
            raise ValueError("position_step must be in (0, 0.10] meters")
        if not 0.0 < rotation_step <= 0.75:
            raise ValueError("rotation_step must be in (0, 0.75] radians")
        applied = 0
        last: np.ndarray | None = None
        for _ in range(max_steps):
            if self.env.done():
                break
            if applied:
                self._sync_policy_observation()
            current = self.env.ee_state()
            action = current.copy()
            complete = True
            for arm in active_arms:
                offset = 0 if arm == "left" else 8
                delta = target[offset : offset + 3] - current[offset : offset + 3]
                distance = float(np.linalg.norm(delta))
                if distance > 1e-4:
                    complete = False
                    action[offset : offset + 3] += delta * min(
                        1.0, position_step / distance
                    )
                current_q = _unit_quaternion(
                    current[offset + 3 : offset + 7], label=f"current {arm} quaternion"
                )
                target_q = _unit_quaternion(
                    target[offset + 3 : offset + 7], label=f"target {arm} quaternion"
                )
                if _quat_angle(current_q, target_q) > 1e-3:
                    complete = False
                    action[offset + 3 : offset + 7] = _quat_step(
                        current_q, target_q, rotation_step
                    )
                if abs(float(current[offset + 7] - target[offset + 7])) > 1e-4:
                    complete = False
                action[offset + 7] = target[offset + 7]
            if complete:
                break
            self.env.take_action(action, action_type="ee")
            self._record_action(action, source=source)
            applied += 1
            last = action
        return ExecutionResult(applied_steps=applied, last_action=last)

    def _execute_actions(self, actions: Any, *, source: str) -> ExecutionResult:
        actions = list(actions)
        applied = 0
        last: np.ndarray | None = None
        for index, action in enumerate(actions):
            if self.env.done():
                break
            last = self._as_ee_action(action, source=source)
            self.env.take_action(last, action_type="ee")
            self._record_action(last, source=source)
            applied += 1
            if index + 1 < len(actions) and not self.env.done():
                self._sync_policy_observation()
        return ExecutionResult(applied_steps=applied, last_action=last)

    def _record_action(self, action: np.ndarray, *, source: str) -> None:
        if self._record is not None:
            self._record(action, action_type="ee", source=source)

    @classmethod
    def _as_ee_action(cls, action: Any, *, source: str) -> np.ndarray:
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size != cls.EE_ACTION_SIZE:
            raise ValueError(
                f"{source} must contain {cls.EE_ACTION_SIZE} values, got {arr.size}"
            )
        return arr

    @staticmethod
    def _single_arm(value: str) -> str:
        arm = str(value).strip().lower()
        if arm not in {"left", "right"}:
            raise ValueError("arm must be 'left' or 'right'")
        return arm

    @staticmethod
    def _xyz(value: Any, *, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size != 3 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{label} must contain 3 finite values")
        return arr

    def _bounded_repeat(self, value: int) -> int:
        return max(1, min(int(value), self.MAX_DIRECT_REPEAT))


__all__ = ["ExecutionResult", "RobotTwinActionExecutor"]
