from __future__ import annotations

import asyncio
import base64

import pytest

pydantic_ai = pytest.importorskip("pydantic_ai")
pytest.importorskip("pydantic_ai_harness")

from pydantic_ai import BinaryContent
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    TieredCompaction,
)

from hy_harness.memory import (
    HistoryMemory,
    is_recoverable_history_error,
    prune_history_images,
)
from hy_harness.memory.compaction import file_read_key

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8Dw"
    "HwAFgAI/ScLttAAAAABJRU5ErkJggg=="
)


def _tool_pair(
    name: str,
    index: int,
    *,
    args: dict | None = None,
    result: str | None = None,
) -> list[ModelMessage]:
    call_id = f"call-{index}"
    return [
        ModelResponse(
            parts=[
                ToolCallPart(
                    name,
                    args or {"step": index},
                    tool_call_id=call_id,
                )
            ]
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name=name,
                    content=result or f"{name}-{index}-" + ("x" * 200),
                    tool_call_id=call_id,
                )
            ]
        ),
    ]


def test_history_memory_uses_official_paired_compaction() -> None:
    memory = HistoryMemory()
    kinds = {type(item).__name__ for item in memory.capabilities()}
    assert kinds >= {
        "ClampOversizedMessages",
        "DeduplicateFileReads",
        "TieredCompaction",
        "ProcessHistory",
    }
    tiered = next(
        item for item in memory.capabilities() if isinstance(item, TieredCompaction)
    )
    assert isinstance(tiered.tiers[0], ClearToolResults)
    assert tiered.tiers[0].clear_tool_inputs is True
    assert tiered.tiers[0].keep_pairs == 2
    assert isinstance(tiered.tiers[1], SlidingWindowCompaction)
    clamp = next(
        item
        for item in memory.capabilities()
        if isinstance(item, ClampOversizedMessages)
    )
    assert clamp.clamp_tool_call_args is True
    dedupe = next(
        item for item in memory.capabilities() if isinstance(item, DeduplicateFileReads)
    )
    assert dedupe.file_key is file_read_key


def test_prune_history_images_keeps_latest_observe() -> None:
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
    pruned = prune_history_images(messages)
    image_messages = 0
    for message in pruned:
        for part in message.parts:
            if isinstance(part, UserPromptPart) and isinstance(part.content, list):
                images = [
                    item for item in part.content if isinstance(item, BinaryContent)
                ]
                if images:
                    image_messages += 1
                    assert len(images) == 3
    assert image_messages == 4


def test_compact_for_retry_clears_old_tool_pairs_and_inputs() -> None:
    messages: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="hang the mug")])
    ]
    for index in range(8):
        messages.extend(
            _tool_pair(
                "robotwin_execute_ee",
                index,
                args={"action": [0.9] * 16, "repeat": 1},
                result="[0.9065, 0.9983] " + ("payload " * 40),
            )
        )

    async def unused(_messages, _info):
        raise AssertionError("compaction retry must not call the model")

    compacted = asyncio.run(
        HistoryMemory().compact_for_retry(messages, model=FunctionModel(unused))
    )
    assert len(compacted) < len(messages)
    tool_returns = [
        part
        for message in compacted
        for part in getattr(message, "parts", ())
        if isinstance(part, ToolReturnPart)
    ]
    tool_calls = [
        part
        for message in compacted
        for part in getattr(message, "parts", ())
        if isinstance(part, ToolCallPart)
    ]
    assert tool_returns
    assert tool_calls
    assert any(str(part.content) == "[tool result cleared]" for part in tool_returns)
    assert any(part.args == "{}" or part.args_as_dict() == {} for part in tool_calls)


def test_is_recoverable_history_error_matches_truncated_tool_json() -> None:
    error = ModelHTTPError(
        status_code=400,
        model_name="hy_a3b",
        body='Invalid JSON: EOF while parsing a string input_value=\'[{"name": "write_t\'',
    )
    assert is_recoverable_history_error(error)
    assert is_recoverable_history_error(
        "ModelHTTPError: status_code: 400 Invalid JSON: EOF while parsing a string"
    )
    assert not is_recoverable_history_error(
        ModelHTTPError(
            status_code=400, model_name="hy_a3b", body="image input rejected"
        )
    )
    assert not is_recoverable_history_error("exhausted max_turns=48")
