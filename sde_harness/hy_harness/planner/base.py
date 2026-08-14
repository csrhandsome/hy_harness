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
    allowed_tools: str | None = None,
    dashboard: Any = None,
    base_url: str | None = None,
    max_tokens: int = 8192,
    no_images: bool = False,
    claude_code_max_budget_usd: float | None = None,
):
    """Build a planner for the given backend, resolving credentials from env vars."""
    planner_type = str(planner_type).strip().lower().replace("-", "_")

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
            allowed_tools=(
                "Bash Read Write Glob Grep" if allowed_tools is None else allowed_tools
            ),
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codebuddy_{recipe_tag}.txt",
            dashboard=dashboard,
        )

    if planner_type in {"api", "pydantic_ai", "pydanticai"}:
        return _build_api_planner(
            model=model,
            base_url=base_url,
            max_tokens=max_tokens,
            dashboard=dashboard,
            no_images=no_images,
            planner_timeout_s=planner_timeout_s,
        )

    if planner_type in {"claude_code", "claude", "claude_sdk"}:
        from hy_harness.planner.claude_code import ClaudeCodePlanner

        cc_timeout_s = planner_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(
                os.environ.get(
                    "CLAUDE_CODE_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))
        return ClaudeCodePlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            allowed_tools="" if allowed_tools is None else allowed_tools,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard=dashboard,
        )

    raise ValueError(f"unknown planner_type: {planner_type}")


def _build_api_planner(
    *,
    model: str | None,
    base_url: str | None,
    max_tokens: int,
    dashboard: Any,
    no_images: bool,
    planner_timeout_s: int | None,
):
    """Construct the pydantic-ai robot controller."""
    import inspect

    from pydantic_ai.models import infer_model
    from pydantic_ai.providers import infer_provider, infer_provider_class

    from hy_harness.planner.pydantic_loop import ApiAgentLoop
    from hy_harness.utils.config import load_local_env

    load_local_env()

    model_id = (model or os.environ.get("OPENAI_MODEL") or "").strip()
    if not model_id:
        raise ValueError(
            "the 'api' planner requires a model id; pass harness.model or "
            "ROBOTWIN_HARNESS_MODEL with a provider prefix "
            "(e.g. 'openai-chat:hy_a3b', 'anthropic:claude-opus-4-8')."
        )
    if ":" not in model_id:
        model_id = f"openai-chat:{model_id}"

    resolved_base = (base_url or "").strip() or None
    openai_base = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("CODEBUDDY_OPENAI_BASE_URL")
        or ""
    ).strip() or None

    def _provider_factory(provider_name: str):
        """Build the provider, optionally overriding base_url / api_key."""
        extra_base = resolved_base
        if extra_base is None and provider_name.startswith("openai"):
            extra_base = openai_base
        if extra_base is None:
            try:
                return infer_provider(provider_name)
            except Exception:
                pass
        provider_cls = infer_provider_class(provider_name)
        params = inspect.signature(provider_cls.__init__).parameters
        kwargs: dict[str, Any] = {}
        if extra_base and "base_url" in params:
            kwargs["base_url"] = extra_base
        if "api_key" in params:
            api_key = _api_key_for_provider(provider_name)
            if api_key is not None:
                kwargs["api_key"] = api_key
        return provider_cls(**kwargs) if kwargs else infer_provider(provider_name)

    api_model = infer_model(model_id, provider_factory=_provider_factory)
    timeout_s = planner_timeout_s
    if timeout_s is None:
        raw = os.environ.get("CELL_TIMEOUT_S")
        timeout_s = int(raw) if raw else None
    return ApiAgentLoop(
        model=api_model,
        max_tokens=max_tokens,
        dashboard=dashboard,
        no_images=no_images,
        timeout_s=timeout_s,
    )


def _api_key_for_provider(provider_name: str) -> str | None:
    if provider_name.startswith("anthropic"):
        return os.environ.get("ANTHROPIC_API_KEY")
    return (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CODEBUDDY_API_KEY")
        or "EMPTY"
    )
