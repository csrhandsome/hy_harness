from __future__ import annotations

import base64
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from hy_harness.planner.codebuddy import (
    CodeBuddyPlanner,
    _Recorder,
    _tool_result_to_mcp,
)
from hy_harness.tools.base import BaseTool, ToolResult


SERVING_ROOT = Path(__file__).resolve().parents[2] / "serving"
sys.path.insert(0, str(SERVING_ROOT / "src"))

from vla_serving.runtimes.vllm.tool_choice_proxy import ToolChoiceProxyHandler


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
    CodeBuddyAgentOptions = _FakeAgentOptions

    @staticmethod
    def tool(name: str, description: str, input_schema: dict[str, Any]):
        def decorate(function):
            return function

        return decorate

    @staticmethod
    def create_sdk_mcp_server(**kwargs: Any) -> dict[str, Any]:
        return kwargs


def _serialized_tool_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Match the nested text shape emitted by CodeBuddy's raw stream."""
    return [
        {
            "tool_use_id": "call_finish",
            "content": [
                {
                    "type": "text",
                    "text": json.dumps([{"type": "text", "text": json.dumps(result)}]),
                }
            ],
        }
    ]


def test_mcp_bridge_preserves_image_blocks_and_marks_errors() -> None:
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
    assert payload["content"][0]["type"] == "text"
    assert len(payload["content"]) == 1
    envelope = json.loads(payload["content"][0]["text"])["_hyharness_multimodal"]
    assert envelope == {
        "text": '{\n  "instruction": "pick up the bottle",\n  "error": "simulator unavailable"\n}',
        "images": [
            {
                "data": base64.b64encode(_PNG).decode("ascii"),
                "mimeType": "image/png",
            }
        ],
    }


def test_empty_builtin_allowlist_disables_codebuddy_builtins(tmp_path: Path) -> None:
    planner = CodeBuddyPlanner(
        output_dir=tmp_path,
        repo_root=tmp_path,
        allowed_tools="",
    )

    options = planner._build_agent_options(
        _FakeSdk,
        tools=_ObserveToolkit(),
        max_turns=3,
    )

    assert options.kwargs["tools"] == []
    assert all(
        name.startswith("mcp__hyharness__") for name in options.kwargs["allowed_tools"]
    )
    assert "mcp__hyharness__robotwin_observe" in options.kwargs["allowed_tools"]


def test_recorder_does_not_finish_on_rejected_finish_tool_result() -> None:
    recorder = _Recorder(max_turns=5)
    recorder.pending_finish["call_finish"] = {
        "status": "success",
        "summary": "hallucinated success",
    }

    rendered = recorder._tool_result_content(
        _serialized_tool_result(
            {
                "error": "benchmark success is false; continue or finish as failure/stuck",
                "success": False,
            }
        ),
        tool_use_id="call_finish",
        is_error=False,
    )

    assert recorder.finish_result is None
    assert '"is_error": true' in rendered


def test_recorder_finishes_only_when_tool_result_confirms_finish() -> None:
    recorder = _Recorder(max_turns=5)
    recorder.pending_finish["call_finish"] = {
        "status": "success",
        "summary": "requested success",
    }
    confirmed = {
        "_finish": True,
        "status": "success",
        "summary": "benchmark confirmed",
        "success": True,
    }

    recorder._tool_result_content(
        _serialized_tool_result(confirmed),
        tool_use_id="call_finish",
        is_error=False,
    )

    assert recorder.finish_result == confirmed


class _OpenAIHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.requests.append(json.loads(self.rfile.read(length)))
        request_number = len(self.requests)

        if request_number == 1:
            chunks = [
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "hy_a3b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_observe",
                                        "type": "function",
                                        "function": {
                                            "name": "mcp__hyharness__robotwin_observe",
                                            "arguments": "{}",
                                        },
                                    }
                                ],
                            },
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "hy_a3b",
                    "choices": [
                        {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                    ],
                },
            ]
        else:
            chunks = [
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "hy_a3b",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "done"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "test",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": "hy_a3b",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            ]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


class _RejectedFinishToolkit(BaseTool):
    def __init__(self) -> None:
        super().__init__()
        self.finish_calls = 0
        self.add_tool(
            "finish",
            {
                "name": "finish",
                "description": "End the episode.",
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
            "success": False,
            "done": False,
            "take_action_cnt": 12,
            "step_lim": 400,
        }

    def finish(self, status: str, summary: str) -> dict[str, Any]:
        self.finish_calls += 1
        if self.finish_calls == 1:
            return {
                "error": "benchmark success is false; continue or finish as failure/stuck",
                "success": False,
            }
        return {
            "_finish": True,
            "status": "success",
            "summary": "benchmark confirmed",
            "success": True,
        }


class _RejectedFinishHandler(BaseHTTPRequestHandler):
    requests: list[dict[str, Any]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.requests.append(json.loads(self.rfile.read(length)))
        request_number = len(self.requests)

        if request_number in (1, 3):
            call_id = (
                "call_finish_initial"
                if request_number == 1
                else "call_finish_confirmed"
            )
            delta = {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "mcp__hyharness__finish",
                            "arguments": json.dumps(
                                {
                                    "status": "success",
                                    "summary": (
                                        "not done"
                                        if request_number == 1
                                        else "confirmed after continuation"
                                    ),
                                }
                            ),
                        },
                    }
                ],
            }
            finish_reason = "tool_calls"
        else:
            delta = {"role": "assistant", "content": "continue after rejection"}
            finish_reason = "stop"

        chunks = [
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "hy_a3b",
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "hy_a3b",
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            },
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, fmt: str, *args: Any) -> None:
        pass


@pytest.mark.skipif(
    os.environ.get("RUN_CODEBUDDY_TRANSPORT_E2E") != "1",
    reason="set RUN_CODEBUDDY_TRANSPORT_E2E=1 with codebuddy-agent-sdk installed",
)
def test_codebuddy_headless_forwards_observation_image_as_multimodal_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("codebuddy_agent_sdk")
    _OpenAIHandler.requests = []
    backend = ThreadingHTTPServer(("127.0.0.1", 0), _OpenAIHandler)
    backend_thread = threading.Thread(target=backend.serve_forever, daemon=True)
    backend_thread.start()
    previous_backend = (
        ToolChoiceProxyHandler.backend_host,
        ToolChoiceProxyHandler.backend_port,
    )
    ToolChoiceProxyHandler.backend_host = "127.0.0.1"
    ToolChoiceProxyHandler.backend_port = backend.server_port
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), ToolChoiceProxyHandler)
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()
    monkeypatch.setenv(
        "CODEBUDDY_OPENAI_BASE_URL",
        f"http://127.0.0.1:{proxy.server_port}/v1",
    )

    try:
        result = CodeBuddyPlanner(
            output_dir=tmp_path,
            repo_root=tmp_path,
            model="hy_a3b",
            allowed_tools="",
            timeout_s=30,
            output_path=tmp_path / "codebuddy.out",
        ).solve(
            system_prompt="Call robotwin_observe.",
            user_message="Start.",
            toolkit=_ObserveToolkit(),
            max_turns=2,
        )
    finally:
        proxy.shutdown()
        proxy_thread.join()
        ToolChoiceProxyHandler.backend_host, ToolChoiceProxyHandler.backend_port = (
            previous_backend
        )
        backend.shutdown()
        backend_thread.join()

    assert result.error is None
    assert len(_OpenAIHandler.requests) == 2
    observation_message = next(
        message
        for message in _OpenAIHandler.requests[1]["messages"]
        if message.get("role") == "user"
        and isinstance(message.get("content"), list)
        and any(
            block.get("type") == "image_url"
            for block in message["content"]
            if isinstance(block, dict)
        )
    )
    image = next(
        block
        for block in observation_message["content"]
        if isinstance(block, dict) and block.get("type") == "image_url"
    )
    assert image["image_url"]["url"] == (
        "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii")
    )


@pytest.mark.skipif(
    os.environ.get("RUN_CODEBUDDY_TRANSPORT_E2E") != "1",
    reason="set RUN_CODEBUDDY_TRANSPORT_E2E=1 with codebuddy-agent-sdk installed",
)
def test_codebuddy_does_not_interrupt_after_rejected_finish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pytest.importorskip("codebuddy_agent_sdk")
    _RejectedFinishHandler.requests = []
    toolkit = _RejectedFinishToolkit()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RejectedFinishHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "CODEBUDDY_OPENAI_BASE_URL",
        f"http://127.0.0.1:{server.server_port}/v1",
    )
    monkeypatch.setenv("CODEBUDDY_INVALID_TEXT_RETRIES", "1")

    try:
        result = CodeBuddyPlanner(
            output_dir=tmp_path,
            repo_root=tmp_path,
            model="hy_a3b",
            allowed_tools="",
            timeout_s=30,
            output_path=tmp_path / "codebuddy.out",
        ).solve(
            system_prompt="Do not finish without benchmark success.",
            user_message="Start.",
            toolkit=toolkit,
            max_turns=4,
        )
    finally:
        server.shutdown()
        thread.join()

    assert result.error is None
    assert result.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "benchmark confirmed",
        "success": True,
    }
    assert result.stats["invalid_terminal_retries"] == 1
    assert toolkit.finish_calls == 2
    assert len(_RejectedFinishHandler.requests) >= 3
    assert "INVALID TERMINATION" in json.dumps(_RejectedFinishHandler.requests[2])
