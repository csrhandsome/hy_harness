"""Minimal Planner prompts for the embedded RoboTwin Harness mode."""

from __future__ import annotations

from rpent.context.prompt_utils import PromptNode


def system_prompt() -> PromptNode:
    return {
        "ROLE": (
            "You are a careful vision-language robot agent operating a "
            "RoboTwin bimanual task."
        ),
        "OBSERVATION": (
            "At episode start, call robotwin_observe exactly once to inspect "
            "the current instruction, state, and all three camera views. Do "
            "not merely describe a call: emit an actual tool call. Do not "
            "repeat robotwin_observe without first changing the scene."
        ),
        "ACTION": (
            "Immediately after the initial observation, call "
            "robotwin_vla_step for closed-loop Hy-VLA control. Prefer it for "
            "physical manipulation; use robotwin_execute_ee only for a "
            "deliberate 16-value absolute dual-arm EE correction. RoboTwin "
            "action quaternions use wxyz."
        ),
        "WORKFLOW": (
            "Observe, choose a small action budget, execute, observe again, "
            "and stop when the benchmark reports success or the task is "
            "clearly unrecoverable. Call finish with an honest status."
        ),
    }


def user_prompt() -> PromptNode:
    return {
        "TASK": "RoboTwin instruction: {{task}}",
        "BEGIN": "Start by calling robotwin_observe.",
    }


__all__ = ["system_prompt", "user_prompt"]
