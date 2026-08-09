"""Shared protocol for high-level reasoning backends."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from hy_harness.utils.config import (
    get_memory_dir,
    get_repo_root,
)

#: MCP namespace prefix for HyHarness tools (``mcp__<server>__<tool>``).
#: Toolkits expose plain tool names; planners add/strip this prefix.
MCP_TOOL_PREFIX = "mcp__hyharness__"


def add_mcp_prefix(name: str) -> str:
    """Return the namespaced MCP tool name for a bare tool name."""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """Return the bare tool name, dropping the MCP namespace if present."""
    return name.removeprefix(MCP_TOOL_PREFIX)


class PlannerResult:
    """Result returned by a planner invocation."""

    __slots__ = ("error", "finish_result", "messages", "stats")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
    ):
        """Initialize a serializable planner result."""
        self.finish_result = (
            finish_result  # {"status": "success"/"failure"/"stuck", "summary": "..."}
        )
        self.messages = messages or []  # serialisable conversation transcript
        self.stats = (
            stats or {}
        )  # {"total_input_tokens", "total_output_tokens", "turns_used", "tool_calls"}
        self.error = error  # str | None  — set when the planner raises


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def build_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    model: str | None = None,
    planner_timeout_s: int | None = None,
    dashboard: Any = None,
):
    """只保留了codebuddy的planer
    Build a planner for the given backend, resolving credentials from env vars."""

    if planner_type in {"codebuddy", "codebuddy_sdk"}:
        from hy_harness.planner.codebuddy import CodeBuddyPlanner

        cb_timeout_s = planner_timeout_s
        if cb_timeout_s is None:
            cb_timeout_s = int(
                os.environ.get(
                    "CODEBUDDY_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodeBuddyPlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=cb_timeout_s,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codebuddy_{recipe_tag}.txt",
            dashboard=dashboard,
        )
    raise ValueError(f"unknown planner_type: {planner_type}")
