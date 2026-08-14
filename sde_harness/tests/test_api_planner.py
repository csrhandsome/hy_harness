from __future__ import annotations

import asyncio
import base64
from typing import Any

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")

from pydantic_ai import BinaryContent, ToolReturn
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from hy_harness.planner.pydantic_loop import (
    ApiAgentLoop,
    _RunState,
    _build_tools,
    _content_blocks_to_pydantic,
    _make_tool_function,
    _prune_history,
)
from hy_harness.planner.base import build_planner
from hy_harness.tools.base import BaseTool, ToolResult


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8Dw"
    "HwAFgAI/ScLttAAAAABJRU5ErkJggg=="
)


class _RobotToolkit(BaseTool):
    def __init__(self, *, success: bool = False, done: bool = False) -> None:
        super().__init__()
        self.calls: list[str] = []
        self._success = success
        self._done = done
        self.add_tool(
            "robotwin_observe",
            {
                "name": "robotwin_observe",
                "description": "Return the current RoboTwin observation.",
                "input_schema": {"type": "object", "properties": {}},
            },
            self.observe,
        )
        self.add_tool(
            "finish",
            {
                "name": "finish",
                "description": "End the planner loop.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string"},
                        "summary": {"type": "string"},
                    },
                    "required": ["status", "summary"],
                },
            },
            self.finish,
        )

    def status(self) -> dict[str, Any]:
        return {
            "success": self._success,
            "done": self._done,
            "take_action_cnt": len(self.calls),
            "step_lim": 400,
        }

    def observe(self) -> dict[str, Any]:
        self.calls.append("observe")
        return {
            "instruction": "pick up the bottle",
            "success": self._success,
            "done": self._done,
            "_image_bytes": _PNG,
        }

    def finish(self, status: str, summary: str) -> dict[str, Any]:
        self.calls.append(f"finish:{status}")
        if str(status).strip().lower() == "success" and not self._success:
            return {
                "error": "benchmark success is false; continue or finish as failure/stuck",
                "success": False,
            }
        self._done = True
        return {
            "_finish": True,
            "status": status,
            "summary": summary,
            "success": self._success,
        }


def _n_model_responses(messages: list[ModelMessage]) -> int:
    return sum(1 for message in messages if isinstance(message, ModelResponse))


def test_build_planner_api_requires_model(tmp_path) -> None:
    with pytest.raises(ValueError, match="api"):
        build_planner(
            "api",
            output_dir=tmp_path,
            recipe_tag="task_t0",
            env_name="robotwin",
            model=None,
        )


def test_build_planner_api_accepts_bare_model_id(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "EMPTY")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1")
    planner = build_planner(
        "api",
        output_dir=tmp_path,
        recipe_tag="task_t0",
        env_name="robotwin",
        model="hy_a3b",
        base_url="http://127.0.0.1:8080/v1",
    )
    assert isinstance(planner, ApiAgentLoop)


def test_content_blocks_to_pydantic_decodes_images() -> None:
    result = ToolResult(
        name="robotwin_observe",
        result={"instruction": "pick up the bottle", "_image_bytes": _PNG},
    )
    text, images = _content_blocks_to_pydantic(result.content_blocks)
    assert "pick up the bottle" in text
    assert len(images) == 1
    assert images[0].data == _PNG
    assert images[0].media_type == "image/png"


def test_tool_wrapper_confirms_finish_only_from_tool_result() -> None:
    toolkit = _RobotToolkit()
    state = _RunState()
    finish = _make_tool_function(toolkit, "finish", no_images=True, run_state=state)
    rejected = asyncio.run(finish(status="success", summary="hallucinated"))
    assert state.finish_result is None
    assert "error" in rejected

    state.begin_model_turn()
    confirmed = asyncio.run(finish(status="stuck", summary="cannot recover"))
    assert state.finish_result is not None
    assert state.finish_result["_finish"] is True
    assert state.finish_result["status"] == "stuck"
    assert "_finish" in confirmed


def test_tool_wrapper_returns_images_as_binary_content() -> None:
    toolkit = _RobotToolkit()
    state = _RunState()
    observe = _make_tool_function(
        toolkit, "robotwin_observe", no_images=False, run_state=state
    )
    returned = asyncio.run(observe())
    assert isinstance(returned, ToolReturn)
    assert any(isinstance(item, BinaryContent) for item in returned.content)


def test_prune_history_keeps_latest_observe_and_stubs_old_tool_returns() -> None:
    img = BinaryContent(data=_PNG, media_type="image/png")
    messages: list[ModelMessage] = []
    for index in range(6):
        messages.append(
            ModelRequest(
                parts=[
                    ToolReturnPart(
                        tool_name="robotwin_observe",
                        content=f"observation-{index}",
                        tool_call_id=f"call-{index}",
                    ),
                    UserPromptPart(content=["state", img, img, img]),
                ]
            )
        )
    pruned = _prune_history(messages)
    image_messages = 0
    stubbed = 0
    kept_returns = 0
    for message in pruned:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                images = [
                    item for item in part.content if isinstance(item, BinaryContent)
                ]
                if images:
                    image_messages += 1
                    assert len(images) == 3
            if isinstance(part, ToolReturnPart):
                if (
                    part.content
                    == "[earlier tool result omitted to bound request size]"
                ):
                    stubbed += 1
                else:
                    kept_returns += 1
    assert image_messages == 4
    assert kept_returns == 4
    assert stubbed == 2


def test_build_tools_hides_legacy_finish_and_serializes_environment_calls() -> None:
    toolkit = _RobotToolkit()
    tools = _build_tools(toolkit, no_images=True, run_state=_RunState())
    assert {tool.name for tool in tools} == {
        "read_text_file",
        "write_text_file",
        "list_dir",
        "robotwin_observe",
    }
    assert all(tool.sequential for tool in tools)


def test_prose_is_not_a_terminal_output_and_consumes_the_request_budget() -> None:
    toolkit = _RobotToolkit()

    async def model_function(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(parts=[TextPart("the task failed, giving up")])

    result = ApiAgentLoop(
        model=FunctionModel(model_function),
        invalid_text_retries=2,
    ).solve(
        system_prompt="Call tools until the environment is done.",
        user_message="Start.",
        toolkit=toolkit,
        max_turns=8,
    )
    assert result.finish_result is None
    assert result.error is not None
    assert "exhausted max_turns" in result.error
    assert result.stats["turns_used"] == 8
    assert result.stats["tool_calls"] == 0
    assert toolkit.calls == []


def test_structured_terminal_is_accepted_only_when_environment_is_done() -> None:
    toolkit = _RobotToolkit(done=True)

    async def model_function(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "robotwin_terminal",
                    {"status": "failure", "summary": "simulator budget exhausted"},
                )
            ]
        )

    result = ApiAgentLoop(
        model=FunctionModel(model_function),
        invalid_text_retries=3,
    ).solve(
        system_prompt="Use tools. Do not stop in prose.",
        user_message="Start.",
        toolkit=toolkit,
        max_turns=8,
    )
    assert result.error is None
    assert result.finish_result is not None
    assert result.finish_result["_finish"] is True
    assert result.finish_result["status"] == "failure"
    assert result.stats["structured_terminal_attempts"] == 1
    assert result.stats["structured_terminal_rejections"] == 0


def test_structured_terminal_is_rejected_while_environment_remains_active() -> None:
    toolkit = _RobotToolkit()

    async def model_function(
        messages: list[ModelMessage], _info: AgentInfo
    ) -> ModelResponse:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    "robotwin_terminal",
                    {"status": "failure", "summary": "gave up"},
                )
            ]
        )

    result = ApiAgentLoop(
        model=FunctionModel(model_function),
        invalid_text_retries=1,
    ).solve(
        system_prompt="Finish only after benchmark success.",
        user_message="Start.",
        toolkit=toolkit,
        max_turns=6,
    )
    assert result.finish_result is None
    assert result.error is not None
    assert "Exceeded maximum output retries" in result.error
    assert toolkit.calls == []
    assert result.stats["structured_terminal_attempts"] >= 2
    assert (
        result.stats["structured_terminal_rejections"]
        == result.stats["structured_terminal_attempts"]
    )
