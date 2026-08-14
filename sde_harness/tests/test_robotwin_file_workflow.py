from __future__ import annotations

import pytest
from test_robotwin_tools import FakePolicy, FakeTaskEnv, encode

from hy_harness.planner.base import PlannerResult, build_planner
from hy_harness.planner.claude_code import ClaudeCodePlanner
from hy_harness.planner.codebuddy import CodeBuddyPlanner
from robots.robotwin.session import RobotTwinHarnessPolicy
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools


def test_memory_reads_are_audited_and_path_scoped(tmp_path):
    memory = tmp_path / "memory"
    output = tmp_path / "output"
    memory.mkdir()
    output.mkdir()
    note = memory / "MEMORY.md"
    note.write_text("reviewed")
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(FakeTaskEnv(), observation_encoder=encode),
        policy=FakePolicy(),
        output_dir=output,
        read_roots=[memory],
    )
    read = toolkit.execute_tool("read_text_file", {"path": str(note)}).result
    assert read["content"] == "reviewed"
    assert toolkit.artifact_summary()["files_read"] == [str(note)]
    denied = toolkit.execute_tool(
        "read_text_file", {"path": str(tmp_path / "not-allowed.md")}
    ).result
    assert "error" in denied


def test_build_planner_preserves_explicit_empty_builtin_set(tmp_path):
    planner = build_planner(
        "codebuddy",
        output_dir=tmp_path,
        recipe_tag="task_t0",
        env_name="robotwin",
        allowed_tools="",
    )
    assert isinstance(planner, CodeBuddyPlanner)
    assert planner._allowed_tools == ""


def test_build_planner_rejects_unknown_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown planner_type"):
        build_planner(
            "not-a-planner",
            output_dir=tmp_path,
            recipe_tag="task_t0",
            env_name="robotwin",
        )


def test_build_planner_claude_alias(tmp_path):
    planner = build_planner(
        "claude",
        output_dir=tmp_path,
        recipe_tag="task_t0",
        env_name="robotwin",
        allowed_tools="",
    )
    assert isinstance(planner, ClaudeCodePlanner)


def test_session_retries_recoverable_pydantic_history_error_once(tmp_path, monkeypatch):
    class FakePlanner:
        def __init__(self) -> None:
            self.calls = 0

        def solve(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                return PlannerResult(
                    error=(
                        "ModelHTTPError: status_code: 400 Invalid JSON: "
                        "EOF while parsing a string"
                    )
                )
            return PlannerResult(
                finish_result={
                    "_finish": True,
                    "status": "failure",
                    "summary": "retried after history compact",
                }
            )

    fake = FakePlanner()
    monkeypatch.setattr(
        "hy_harness.planner.base.build_planner", lambda *_args, **_kwargs: fake
    )
    monkeypatch.setattr(
        "hy_harness.utils.resources.ensure_resources", lambda _env: tmp_path
    )
    env = FakeTaskEnv()
    policy = RobotTwinHarnessPolicy(
        FakePolicy(),
        {
            "enabled": True,
            "planner": "api",
            "model": "openai-chat:test",
            "output_dir": str(tmp_path / "out"),
        },
    )
    result = policy.run(env, env.observation, encode)
    assert fake.calls == 2
    assert result is not None
    assert result["finish"]["status"] == "failure"
    assert policy._has_run is True
