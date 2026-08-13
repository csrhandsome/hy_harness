from __future__ import annotations

import json

from serving.src.vla_serving.runtimes.vllm.tool_choice_proxy import (
    inject_required_tool_choice,
)


def _tool(name: str) -> dict:
    return {"type": "function", "function": {"name": f"mcp__hyharness__{name}"}}


def test_robotwin_proxy_keeps_initial_choice_profile_neutral():
    payload = {
        "messages": [{"role": "user", "content": "start"}],
        "tools": [
            _tool("robotwin_observe"),
            _tool("robotwin_vla_chunk"),
            _tool("robotwin_move_arm"),
        ],
    }
    body, rewritten = inject_required_tool_choice(json.dumps(payload).encode())
    decoded = json.loads(body)
    assert rewritten
    assert decoded["tool_choice"] == "required"
    assert len(decoded["tools"]) == 3


def test_robotwin_proxy_keeps_full_tools_after_observe():
    payload = {
        "messages": [
            {"role": "user", "content": "start"},
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "mcp__hyharness__robotwin_observe"}}
                ],
            },
        ],
        "tools": [
            _tool("robotwin_observe"),
            _tool("robotwin_vla_chunk"),
            _tool("robotwin_move_arm"),
        ],
    }
    body, rewritten = inject_required_tool_choice(json.dumps(payload).encode())
    decoded = json.loads(body)
    assert rewritten
    assert len(decoded["tools"]) == 3
    assert "complete offered tool set" in decoded["messages"][0]["content"]
    assert "ONLY valid choices" not in decoded["messages"][0]["content"]
