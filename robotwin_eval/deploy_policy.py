# coding=utf-8
# Copyright (C) 2026 Tencent.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RoboTwin entry point for Hy-VLA evaluation.

This is the single file imported by RoboTwin's ``script/eval_policy.py``
via ``importlib.import_module(<policy_name>)``. It exposes exactly four
public symbols (the contract RoboTwin expects):

* ``encode_obs(observation, instruction)`` -- dict packing helper used
  by ``eval``; exposed so user scripts can replay it offline.
* ``get_model(usr_args)`` -- factory returning a ``HyVLAPolicyWrapper``.
* ``eval(TASK_ENV, model, observation)`` -- per-step closed-loop hook.
* ``reset_model(model)`` -- per-episode hook.

Drop this directory under ``robotwin/policy/hy_vla/`` (or any other
RoboTwin policy slot) and point ``--config`` at the bundled
``deploy_policy.yml``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .policy_wrapper import HyVLAPolicyWrapper, build_policy


def _harness_policy(policy: HyVLAPolicyWrapper, usr_args: dict[str, Any]) -> Any:
    """Optionally wrap the normal policy with the embedded RPent Planner."""
    harness_config = usr_args.get("harness") or {}
    enabled = bool(harness_config.get("enabled", False)) or os.environ.get(
        "ROBOTWIN_HARNESS", ""
    ).lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return policy

    repo_root = Path(__file__).resolve().parents[1]
    harness_root = repo_root / "sde_harness"
    os.environ.setdefault("SDE_HARNESS_ROOT", str(harness_root))
    os.environ.setdefault("HY_VLA_ROOT", str(repo_root))
    if str(harness_root) not in sys.path:
        sys.path.insert(0, str(harness_root))

    from robots.robotwin.session import RobotTwinHarnessPolicy

    return RobotTwinHarnessPolicy(policy, harness_config)


# ---------------------------------------------------------------------------
# Observation packing
# ---------------------------------------------------------------------------
def _to_chw_float(img: np.ndarray) -> np.ndarray:
    """``(H, W, 3) uint8`` -> ``(1, 3, H, W) float32 in [0, 1]``."""
    return (img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32) / 255.0)


def _pad_state(state: np.ndarray, max_state_dim: int = 32) -> np.ndarray:
    """Pad a 16-d EE state to ``max_state_dim`` with zeros."""
    if state.shape[-1] == max_state_dim:
        return state
    shape = list(state.shape)
    cur = shape[-1]
    shape[-1] = max_state_dim
    out = np.zeros(shape, dtype=state.dtype)
    out[..., :cur] = state
    return out


def encode_obs(observation: dict[str, Any], instruction: str) -> dict[str, Any]:
    """Pack a RoboTwin ``observation`` dict + language instruction into the
    batch format expected by ``HyVLAPolicyWrapper.get_action``.

    The dual-arm 16-d EE state uses RoboTwin's native quaternion layout
    (wxyz). The wxyz->xyzw flip happens inside the wrapper.
    """
    eepose_16d = np.array([
        *observation["endpose"]["left_endpose"],   # 7-d (xyz + quat_wxyz)
        observation["endpose"]["left_gripper"],    # 1-d
        *observation["endpose"]["right_endpose"],  # 7-d
        observation["endpose"]["right_gripper"],   # 1-d
    ], dtype=np.float32)

    head = observation["observation"]["head_camera"]["rgb"]
    left = observation["observation"]["left_camera"]["rgb"]
    right = observation["observation"]["right_camera"]["rgb"]

    return {
        # Visual inputs (single-frame; the MEM path replaces these later
        # with a (1, K, C, H, W) stack built from raw_images.* below).
        "observation.images.top_head": _to_chw_float(head),
        "observation.images.hand_left": _to_chw_float(left),
        "observation.images.hand_right": _to_chw_float(right),
        # Padded EE state, ``(1, max_state_dim)``.
        "observation.state": _pad_state(eepose_16d[np.newaxis, :], max_state_dim=32),
    # Language instruction (HyVLA.prepare_language expects a list).
        "task": [instruction],
        # Raw uint8 frames -- the MEM video-encoder path needs the
        # un-normalized HWC layout to grow its per-camera history buffer.
        "raw_images.top_head": head,
        "raw_images.hand_left": left,
        "raw_images.hand_right": right,
    }


# ---------------------------------------------------------------------------
# RoboTwin hooks
# ---------------------------------------------------------------------------
def get_model(usr_args: dict[str, Any]) -> HyVLAPolicyWrapper:
    """Factory called once per evaluation run by RoboTwin."""
    return _harness_policy(build_policy(usr_args), usr_args)


def eval(TASK_ENV, model: HyVLAPolicyWrapper, observation: dict[str, Any]) -> None:  # noqa: A001
    """Per-step closed-loop hook.

    RoboTwin calls this in a tight loop; we pack the observation, query
    the wrapper for one 16-d action, and forward it back via
    ``TASK_ENV.take_action(..., action_type='ee')``.
    """
    if getattr(model, "is_harness", False):
        model.run(TASK_ENV, observation, encode_obs)
        return

    instruction = TASK_ENV.get_instruction()
    batch = encode_obs(observation, instruction)
    action = model.get_action(batch)
    TASK_ENV.take_action(action, action_type="ee")


def reset_model(model: HyVLAPolicyWrapper) -> str:
    """Per-episode reset hook."""
    return model.reset()


__all__ = ["encode_obs", "get_model", "eval", "reset_model"]
