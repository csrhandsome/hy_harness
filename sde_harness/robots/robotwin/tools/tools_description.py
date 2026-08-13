"""Central tool schemas for RoboTwin planner tools."""

from __future__ import annotations

from typing import Any


def build_tools_description(
    *, model_chunk_size: int = 50, default_execute_steps: int = 30
) -> list[dict[str, Any]]:
    chunk_size = max(1, int(model_chunk_size))
    default_steps = max(1, min(int(default_execute_steps), chunk_size))
    arm = {"type": "string", "enum": ["left", "right"]}
    xyz = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
    quat = {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}
    motion = {
        "max_steps": {"type": "integer", "minimum": 1, "maximum": 32},
        "position_step": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.10},
        "rotation_step": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.75},
    }
    return [
        {
            "name": "robotwin_observe",
            "description": (
                "Refresh and return the instruction, dual-arm state, benchmark status, and three RGB "
                "views. Motion tools already return post-action images; do not immediately observe "
                "again unless a later decision truly needs a fresh view."
            ),
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "robotwin_status",
            "description": "Read authoritative benchmark done/success and action-budget status without moving.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "robotwin_vla_chunk",
            "description": (
                f"Run exactly one fresh Hy-VLA forward from the current state, producing a "
                f"{chunk_size}-action model chunk, then execute only its requested prefix (default "
                f"{default_steps}). Old cached actions are never reused and the unused suffix is "
                "discarded. Use 1-5 steps near contact/final alignment and a longer prefix in stable "
                "motion. instruction_override conditions this forward on a concise current subgoal."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "execute_steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": chunk_size,
                        "default": default_steps,
                    },
                    "instruction_override": {
                        "type": "string",
                        "description": "Current-stage subgoal; omit to use the benchmark instruction.",
                    },
                },
            },
        },
        {
            "name": "robotwin_move_arm",
            "description": (
                "Interpolate one arm to an absolute world-frame EE target while holding the other arm. "
                "Supply xyz and/or quaternion_wxyz; gripper is preserved unless supplied. Invalidates "
                "the VLA cache."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": arm,
                    "xyz": xyz,
                    "quaternion_wxyz": quat,
                    "gripper": {"type": "number"},
                    **motion,
                },
                "required": ["arm"],
            },
        },
        {
            "name": "robotwin_translate_arm",
            "description": (
                "Translate one EE relative to its current world-frame position while preserving "
                "orientation and the other arm. delta_xyz is safety-limited to 0.30 m."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": arm,
                    "delta_xyz": xyz,
                    "gripper": {"type": "number"},
                    "max_steps": motion["max_steps"],
                    "position_step": motion["position_step"],
                },
                "required": ["arm", "delta_xyz"],
            },
        },
        {
            "name": "robotwin_move_bimanual",
            "description": (
                "Synchronously interpolate specified left/right absolute EE targets. Unspecified "
                "fields stay at current values. Use for coordinated transport or handover."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "left_xyz": xyz,
                    "left_quaternion_wxyz": quat,
                    "left_gripper": {"type": "number"},
                    "right_xyz": xyz,
                    "right_quaternion_wxyz": quat,
                    "right_gripper": {"type": "number"},
                    **motion,
                },
            },
        },
        {
            "name": "robotwin_rotate_arm",
            "description": "Rotate one EE to an absolute wxyz quaternion while holding position and the other arm.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": arm,
                    "quaternion_wxyz": quat,
                    "gripper": {"type": "number"},
                    "max_steps": motion["max_steps"],
                    "rotation_step": motion["rotation_step"],
                },
                "required": ["arm", "quaternion_wxyz"],
            },
        },
        {
            "name": "robotwin_set_gripper",
            "description": (
                "Set one or both grippers to an exact native target while holding both EE poses. Use "
                "values learned from state/memory; no +1/-1 open/close convention is assumed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "enum": ["left", "right", "both"]},
                    "value": {"type": "number"},
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 4,
                    },
                },
                "required": ["arm", "value"],
            },
        },
        {
            "name": "robotwin_release",
            "description": (
                "Open one or both grippers to an explicit validated RoboTwin native value while "
                "holding both EE poses. open_value is required because RoboTwin does not share "
                "LIBERO's fixed gripper sign convention."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "arm": {"type": "string", "enum": ["left", "right", "both"]},
                    "open_value": {"type": "number"},
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 4,
                    },
                },
                "required": ["arm", "open_value"],
            },
        },
        {
            "name": "robotwin_hold",
            "description": "Hold current dual-arm EE poses and gripper targets for a few simulator steps.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 1,
                    }
                },
            },
        },
        {
            "name": "robotwin_execute_ee",
            "description": (
                "Debug escape hatch: raw absolute 16-value dual-arm action [left xyz, quat_wxyz, "
                "gripper, right xyz, quat_wxyz, gripper]. Prefer named primitives."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 16,
                        "maxItems": 16,
                    },
                    "repeat": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 32,
                        "default": 1,
                    },
                },
                "required": ["action"],
            },
        },
    ]


__all__ = ["build_tools_description"]
