from __future__ import annotations

import base64
from typing import Any

import pytest

from hy_harness.planner.base import build_planner
from hy_harness.planner.claude_code import (
    ClaudeCodePlanner,
    _Recorder,
    _tool_result_to_mcp,
)
from hy_harness.tools.base import BaseTool, ToolResult


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8Dw"
    "HwAFgAI/ScLttAAAAABJRU5ErkJggg=="
)


class _ObserveToolkit(BaseTool):
    def __init__(self) -> None:
        super().__init__()
        self.add_tool(
            "robotwin_observe",
            {
                "name": "robotwin_observe",
                "description": "Return the current RoboTwin observation.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.observe,
        )

    @staticmethod
    def observe() -> dict[str, Any]:
        return {
            "instruction": "pick up the bottle",
            "_image_bytes": _PNG,
        }


class _FakeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeSdk:
    ClaudeAgentOptions = _FakeAgentOptions

    @staticmethod
    def tool(name: str, description: str, input_schema: dict[str, Any]):
        def decorate(function):
            return function

        return decorate

    @staticmethod
    def create_sdk_mcp_server(**kwargs: Any) -> dict[str, Any]:
        return kwargs


def _serialized_tool_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "tool_use_id": "call_finish",
            "content": [
                {
                    "type": "text",
                    "text": json_dumps_nested(result),
                }
            ],
        }
    ]


def json_dumps_nested(result: dict[str, Any]) -> str:
    import json

    return json.dumps([{"type": "text", "text": json.dumps(result)}])


def test_build_planner_claude_code_preserves_empty_builtins(tmp_path) -> None:
    planner = build_planner(
        "claude_code",
        output_dir=tmp_path,
        recipe_tag="task_t0",
        env_name="robotwin",
        allowed_tools="",
        model="sonnet",
    )
    assert isinstance(planner, ClaudeCodePlanner)
    assert planner._allowed_tools == ""
    assert planner._model == "sonnet"


def test_empty_builtin_allowlist_disables_claude_builtins(tmp_path) -> None:
    planner = ClaudeCodePlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        allowed_tools="",
    )
    options = planner._build_options(
        _FakeSdk,
        toolkit=_ObserveToolkit(),
        max_turns=3,
    )
    assert options.kwargs["tools"] == []
    assert all(
        name.startswith("mcp__hyharness__") for name in options.kwargs["allowed_tools"]
    )
    assert "mcp__hyharness__robotwin_observe" in options.kwargs["allowed_tools"]


def test_mcp_bridge_keeps_native_images_and_marks_errors() -> None:
    result = ToolResult(
        name="robotwin_observe",
        result={
            "instruction": "pick up the bottle",
            "_image_bytes": _PNG,
            "error": "simulator unavailable",
        },
    )
    payload = _tool_result_to_mcp(result)
    assert payload["isError"] is True
    assert payload["is_error"] is True
    types = [block["type"] for block in payload["content"]]
    assert "text" in types
    assert "image" in types
    image = next(block for block in payload["content"] if block["type"] == "image")
    assert image["data"] == base64.b64encode(_PNG).decode("ascii")
    assert image["mimeType"] == "image/png"


def test_recorder_does_not_finish_on_rejected_finish_tool_result() -> None:
    recorder = _Recorder(max_turns=5)
    recorder.pending_finish["call_finish"] = {
        "status": "success",
        "summary": "hallucinated success",
    }
    recorder._tool_result_content(
        _serialized_tool_result(
            {
                "error": "benchmark success is false; continue or finish as failure/stuck",
                "success": False,
            }
        ),
        tool_use_id="call_finish",
    )
    assert recorder.finish_result is None


def test_recorder_finishes_only_when_tool_result_confirms_finish() -> None:
    recorder = _Recorder(max_turns=5)
    recorder.pending_finish["call_finish"] = {
        "status": "success",
        "summary": "benchmark confirmed",
    }
    recorder._tool_result_content(
        _serialized_tool_result(
            {
                "_finish": True,
                "status": "success",
                "summary": "benchmark confirmed",
                "success": True,
            }
        ),
        tool_use_id="call_finish",
    )
    assert recorder.finish_result is not None
    assert recorder.finish_result["_finish"] is True
    assert recorder.finish_result["status"] == "success"
