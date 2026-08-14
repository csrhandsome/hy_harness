from __future__ import annotations

import pytest

from hy_harness.planner.base import build_planner
from hy_harness.planner.claude_code import ClaudeCodePlanner
from hy_harness.planner.codebuddy import CodeBuddyPlanner
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools
from test_robotwin_tools import FakePolicy, FakeTaskEnv, encode


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
