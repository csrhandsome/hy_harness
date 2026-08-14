"""Shared robot-loop control for planner backends.

CodeBuddy, Claude Code, and the pydantic-ai loop all have the same contract:
a natural-language completion is not a terminal state while the environment
reports ``success=false`` and ``done=false``. Finish is confirmed only by a
tool result that carries ``_finish``.
"""

from __future__ import annotations

import os
from typing import Any

QUIT_TOKENS = frozenset({"/quit", "/exit", "/q"})
DEFAULT_INVALID_TEXT_RETRIES = 3


def invalid_text_retries(*, env_keys: tuple[str, ...] = ()) -> int:
    """Read the bounded continuation retry limit from the environment."""
    keys = env_keys + (
        "HARNESS_INVALID_TEXT_RETRIES",
        "CODEBUDDY_INVALID_TEXT_RETRIES",
    )
    for key in keys:
        raw = os.environ.get(key)
        if raw not in (None, ""):
            return max(0, int(raw))
    return DEFAULT_INVALID_TEXT_RETRIES


def toolkit_status(toolkit: Any) -> dict[str, Any] | None:
    """Read a live environment status when the toolkit exposes one."""
    status = getattr(toolkit, "status", None)
    if not callable(status):
        return None
    try:
        value = status()
    except Exception:  # noqa: BLE001 - status is advisory for generic toolkits
        return None
    return value if isinstance(value, dict) else None


def status_requires_tool_call(status: dict[str, Any] | None) -> bool:
    """Return whether a text-only completion is invalid for this toolkit."""
    return (
        bool(status)
        and not bool(status.get("success"))
        and not bool(status.get("done"))
    )


def invalid_terminal_continuation(
    *,
    status: dict[str, Any],
    retry: int,
    limit: int,
) -> str:
    """User turn that forces the model back onto a real tool call."""
    remaining = ""
    count = status.get("take_action_cnt")
    step_limit = status.get("step_lim")
    if count is not None and step_limit is not None:
        remaining = f" Current simulator actions: {count}/{step_limit}."
    return (
        "INVALID TERMINATION: your previous response ended in prose without a "
        "valid RoboTwin terminal tool result. The authoritative state remains "
        f"success=false and done=false.{remaining} Do not explain, summarize, "
        "write an audit, emit literal <tool_call> text, or call finish. Use the "
        f"real tool-call channel to call an allowed RoboTwin tool now. Retry {retry}/{limit}."
    )


def next_user_line(input_queue) -> str | None:
    """Block for the next actionable user line from an interactive input queue.

    Returns the trimmed line, or ``None`` when the session should end (the queue
    yielded ``None`` or a quit token such as ``/quit``). Empty lines are skipped.
    """
    while True:
        line = input_queue.get()
        if line is None:
            return None
        line = line.strip()
        if line.lower() in QUIT_TOKENS:
            return None
        if line:
            return line
