"""Embedded HyHarness Planner session for the official RoboTwin runner."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from hy_harness.context.prompt_utils import format_prompt
from hy_harness.memory import is_recoverable_history_error
from hy_harness.planner.loop_control import status_requires_tool_call
from hy_harness.utils.logging import get_logger

from .prompt import PROMPT_PROFILES, system_prompt, user_prompt
from .tools import RobotTwinEnvAdapter, RobotTwinTools


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:80] or "task"


logger = get_logger("robotwin_session")


class RobotTwinHarnessPolicy:
    """Wrap a normal Hy-VLA policy with an optional high-level Planner.

    RoboTwin invokes eval once per simulator cycle. When enabled, the first
    invocation runs the complete HyHarness Planner loop against the live TASK_ENV
    object; later invocations are no-ops because the Planner has already
    consumed the episode actions.
    """

    is_harness = True

    def __init__(self, policy: Any, config: dict[str, Any] | None = None) -> None:
        self.policy = policy
        self.config = dict(config or {})
        self._has_run = False
        self.result: dict[str, Any] | None = None
        self.recipe_path: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.config.get("enabled", False)) or _truthy(
            os.environ.get("ROBOTWIN_HARNESS", "")
        )

    def reset(self) -> str:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()
        self._has_run = False
        self.result = None
        self.recipe_path = None
        return "RobotTwin Harness policy reset"

    def get_action(self, batch: dict[str, Any]) -> Any:
        """Keep the normal RoboTwin policy interface available."""
        return self.policy.get_action(batch)

    def run(
        self,
        task_env: Any,
        observation: dict[str, Any],
        observation_encoder: Callable[[dict[str, Any], str], dict[str, Any]],
    ) -> dict[str, Any] | None:
        if not self.enabled or self._has_run:
            return self.result

        from hy_harness.planner.base import build_planner
        from hy_harness.utils.config import get_hy_vla_root
        from hy_harness.utils.logging import init_output_dir
        from hy_harness.utils.resources import ensure_resources

        adapter = RobotTwinEnvAdapter(
            task_env,
            observation=observation,
            observation_encoder=observation_encoder,
        )
        task_name = str(
            getattr(task_env, "task_name", None)
            or getattr(task_env, "task", None)
            or "robotwin"
        )
        test_num = getattr(task_env, "test_num", 0)
        recipe_tag = f"{_safe_name(task_name)}_t{test_num}"

        output_value = self.config.get("output_dir")
        if output_value:
            output_dir = Path(str(output_value)).expanduser()
            if not output_dir.is_absolute():
                output_dir = get_hy_vla_root() / output_dir
        else:
            output_dir = (
                get_hy_vla_root()
                / "logs"
                / "robotwin_harness"
                / f"{recipe_tag}_{datetime.now().astimezone().strftime('%Y%m%d-%H%M%S')}"
            )
        output_dir = init_output_dir(output_dir)

        # RPent-style, reviewed read-only memory snapshot. Synchronization is
        # best-effort and falls back to the checked-in local starter index.
        resources_dir = ensure_resources("robotwin")
        memory_dir = resources_dir / "memory"
        memory_dir.mkdir(parents=True, exist_ok=True)
        reference_dir = resources_dir / "references"
        audit_path = output_dir / f"{recipe_tag}.json"
        tools = RobotTwinTools(
            env=adapter,
            policy=self.policy,
            output_dir=output_dir,
            audit_path=audit_path,
            read_roots=[memory_dir, reference_dir],
        )

        # Environment variables are explicit per-run launcher overrides. Read
        # them before YAML defaults so a smoke test can be tuned without
        # editing deploy_policy.yml.
        planner_type = str(
            os.environ.get("ROBOTWIN_HARNESS_PLANNER")
            or self.config.get("planner")
            or "codebuddy"
        )
        model = os.environ.get("ROBOTWIN_HARNESS_MODEL") or self.config.get("model")
        max_turns = int(
            os.environ.get("ROBOTWIN_HARNESS_MAX_TURNS")
            or self.config.get("max_turns", 100)
        )
        timeout_value = os.environ.get("ROBOTWIN_HARNESS_TIMEOUT_S")
        if timeout_value in (None, ""):
            timeout_value = self.config.get("planner_timeout_s")
        planner_timeout_s = None if timeout_value in (None, "") else int(timeout_value)
        base_url = os.environ.get("ROBOTWIN_HARNESS_BASE_URL") or self.config.get(
            "base_url"
        )
        max_tokens = int(
            os.environ.get("ROBOTWIN_HARNESS_MAX_TOKENS")
            or self.config.get("max_tokens", 8192)
        )
        no_images = _truthy(
            os.environ.get("ROBOTWIN_HARNESS_NO_IMAGES")
            if os.environ.get("ROBOTWIN_HARNESS_NO_IMAGES") not in (None, "")
            else self.config.get("no_images", False)
        )
        enable_thinking = _truthy(
            os.environ.get("ROBOTWIN_HARNESS_ENABLE_THINKING")
            if os.environ.get("ROBOTWIN_HARNESS_ENABLE_THINKING") not in (None, "")
            else self.config.get("enable_thinking", False)
        )
        budget_value = os.environ.get("ROBOTWIN_HARNESS_CLAUDE_BUDGET_USD")
        if budget_value in (None, ""):
            budget_value = self.config.get("claude_code_max_budget_usd")
        claude_code_max_budget_usd = (
            None if budget_value in (None, "") else float(budget_value)
        )
        prompt_profile = (
            str(
                os.environ.get("ROBOTWIN_HARNESS_PROMPT_PROFILE")
                or self.config.get("prompt_profile")
                or "primitive_first"
            )
            .strip()
            .lower()
        )
        if prompt_profile not in PROMPT_PROFILES:
            raise ValueError(
                f"unknown RoboTwin prompt_profile={prompt_profile!r}; "
                f"expected one of {PROMPT_PROFILES}"
            )
        planner = build_planner(
            planner_type,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
            env_name="robotwin",
            model=model,
            planner_timeout_s=planner_timeout_s,
            allowed_tools=str(self.config.get("allowed_tools", "")),
            base_url=None if base_url in (None, "") else str(base_url),
            max_tokens=max_tokens,
            no_images=no_images,
            enable_thinking=enable_thinking,
            claude_code_max_budget_usd=claude_code_max_budget_usd,
        )

        variables = {
            "task": adapter.instruction(),
            "task_name": task_name,
            "test_num": test_num,
            "output_dir": str(output_dir),
            "prompt_profile": prompt_profile,
            "memory_dir": str(memory_dir),
            "reference_dir": str(reference_dir),
            "audit_path": str(audit_path),
            "recipe_path": str(output_dir / f"recipe_{recipe_tag}.jsonl"),
            "model_chunk_size": int(getattr(self.policy, "action_chunk_size", 50)),
            "default_execute_steps": int(
                getattr(self.policy, "default_execute_steps", 30)
            ),
        }
        system = format_prompt(system_prompt(prompt_profile), variables=variables)
        user = format_prompt(user_prompt(), variables=variables)

        # Mark the episode consumed only after planner construction and solve.
        # This keeps a failed initialization retryable if RoboTwin invokes the
        # hook again, while a completed Planner run remains idempotent.
        planner_result = planner.solve(
            system_prompt=system,
            user_message=user,
            toolkit=tools,
            max_turns=max_turns,
        )
        if (
            planner_type in {"api", "pydantic_ai", "pydanticai"}
            and planner_result.error
            and is_recoverable_history_error(planner_result.error)
            and status_requires_tool_call(adapter.status())
        ):
            logger.warning(
                "planner interrupted by recoverable history error; "
                "starting a fresh pydantic session from live RoboTwin state"
            )
            planner_result = planner.solve(
                system_prompt=system,
                user_message=user,
                toolkit=tools,
                max_turns=max_turns,
            )
        self._has_run = True
        self.recipe_path = tools.write_recipe(recipe_tag)
        audit: dict[str, Any] = {}
        if audit_path.is_file():
            try:
                loaded = json.loads(audit_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    audit = loaded
            except (OSError, json.JSONDecodeError):
                pass
        audit.setdefault("task", adapter.instruction())
        audit.setdefault("task_name", task_name)
        audit.setdefault("test_num", test_num)
        audit.setdefault("prompt_profile", prompt_profile)
        audit.setdefault(
            "strategy",
            "automatic fallback audit; planner did not persist valid JSON",
        )
        audit.setdefault("outcome", planner_result.finish_result)
        audit.setdefault("failure_reason", planner_result.error)
        audit.update(
            benchmark_success=adapter.status()["success"],
            final_status=adapter.status(),
            memory_files_read=tools.artifact_summary()["files_read"],
            **tools.artifact_summary(),
        )
        audit_path.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        self.result = {
            "finish": planner_result.finish_result,
            "stats": planner_result.stats,
            "error": planner_result.error,
            "recipe": self.recipe_path,
            "audit": str(audit_path),
            "prompt_profile": prompt_profile,
            "output_dir": str(output_dir),
        }
        (output_dir / f"{recipe_tag}_result.json").write_text(
            json.dumps(self.result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.result


__all__ = ["RobotTwinHarnessPolicy"]
