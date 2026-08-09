"""Embedded HyHarness Planner session for the official RoboTwin runner."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .prompt import system_prompt, user_prompt
from hy_harness.context.prompt_utils import format_prompt

from .tools import RobotTwinEnvAdapter, RobotTwinTools


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")[:80] or "task"


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

        adapter = RobotTwinEnvAdapter(
            task_env,
            observation=observation,
            observation_encoder=observation_encoder,
        )
        tools = RobotTwinTools(env=adapter, policy=self.policy)

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
                / f"{recipe_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )
        output_dir = init_output_dir(output_dir)

        # Code-oriented planners may inspect environment-specific memory.
        # RoboTwin does not ship that directory with this repository, so make
        # the mount point available without requiring a fake memory file.
        (get_hy_vla_root() / "sde_harness" / "resources" / "robotwin" / "memory").mkdir(
            parents=True, exist_ok=True
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
        # codebuddy登场
        planner = build_planner(
            planner_type,
            output_dir=output_dir,
            recipe_tag=recipe_tag,
            env_name="robotwin",
            model=model,
            planner_timeout_s=planner_timeout_s,
        )

        variables = {
            "task": adapter.instruction(),
            "task_name": task_name,
            "test_num": test_num,
            "output_dir": str(output_dir),
        }
        system = format_prompt(system_prompt(), variables=variables)
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
        self._has_run = True
        self.recipe_path = tools.write_recipe(recipe_tag)
        self.result = {
            "finish": planner_result.finish_result,
            "stats": planner_result.stats,
            "error": planner_result.error,
            "recipe": self.recipe_path,
            "output_dir": str(output_dir),
        }
        (output_dir / f"{recipe_tag}_result.json").write_text(
            json.dumps(self.result, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.result


__all__ = ["RobotTwinHarnessPolicy"]
