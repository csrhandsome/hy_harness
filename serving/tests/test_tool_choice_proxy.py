from __future__ import annotations

import unittest
from pathlib import Path
import sys

SERVING_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVING_ROOT / "src"))

from vla_serving.runtimes.vllm.tool_choice_proxy import inject_required_tool_choice


def _body(
    *,
    tool_name: str | None = None,
    tool_names: list[str] | None = None,
    choice=None,
    history: list[dict] | None = None,
) -> bytes:
    names = tool_names or ([tool_name] if tool_name else [])
    payload = {
        "model": "hy_a3b",
        "messages": history or [{"role": "user", "content": "test"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": "test",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
            for name in names
        ],
    }
    if choice is not None:
        payload["tool_choice"] = choice
    import json

    return json.dumps(payload).encode("utf-8")


class ToolChoiceProxyTests(unittest.TestCase):
    def test_bare_pydantic_robotwin_tools_get_required_and_rehydrated(self) -> None:
        import json

        body, changed = inject_required_tool_choice(
            _body(
                tool_names=["robotwin_observe", "robotwin_vla_chunk"],
                history=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "observe",
                                "type": "function",
                                "function": {"name": "robotwin_observe"},
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "observe",
                        "content": json.dumps(
                            {
                                "content": [
                                    {"type": "text", "text": '{"success": false}'},
                                    {"type": "image", "image": "YWJj"},
                                ]
                            }
                        ),
                    },
                ],
            )
        )

        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(payload["tool_choice"], "required")
        observation = payload["messages"][-1]
        self.assertEqual(observation["role"], "user")
        self.assertTrue(
            any(item.get("type") == "image_url" for item in observation["content"])
        )

    def test_invalid_termination_retry_is_forced_to_fresh_vla_chunk(self) -> None:
        import json

        offered = [
            "mcp__hyharness__robotwin_observe",
            "mcp__hyharness__robotwin_vla_chunk",
            "mcp__hyharness__robotwin_move_arm",
            "mcp__hyharness__finish",
        ]
        body, changed = inject_required_tool_choice(
            _body(
                tool_names=offered,
                choice="required",
                history=[
                    {
                        "role": "user",
                        "content": (
                            "INVALID TERMINATION: do not summarize; call a "
                            "RoboTwin tool now."
                        ),
                    }
                ],
            )
        )

        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["mcp__hyharness__robotwin_vla_chunk"],
        )

    def test_rehydrates_serialized_harness_tool_images_as_user_content(self) -> None:
        import json

        body, changed = inject_required_tool_choice(
            _body(
                tool_name="mcp__hyharness__robotwin_observe",
                choice="required",
                history=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_observe",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call_observe",
                        "content": json.dumps(
                            {
                                "content": [
                                    {"type": "text", "text": '{"success": false}'},
                                    {
                                        "type": "image",
                                        "image": "YWJj",
                                        "mimeType": "image/png",
                                    },
                                ]
                            }
                        ),
                    },
                ],
            )
        )

        self.assertTrue(changed)
        payload = json.loads(body)
        message = payload["messages"][-1]
        self.assertEqual(message["role"], "user")
        self.assertNotIn("tool_call_id", message)
        self.assertEqual(
            message["content"],
            [
                {
                    "type": "text",
                    "text": "Tool result from robotwin_observe:\n",
                },
                {"type": "text", "text": '{"success": false}'},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,YWJj"},
                },
            ],
        )

    def test_rehydration_keeps_only_four_latest_tool_results(self) -> None:
        import json

        def tool_result(call_id: str, image: str) -> dict:
            return {
                "role": "tool",
                "tool_call_id": call_id,
                "content": json.dumps(
                    {
                        "content": [
                            {"type": "text", "text": f"state {call_id}"},
                            {"type": "image", "image": image},
                        ]
                    }
                ),
            }

        body, changed = inject_required_tool_choice(
            _body(
                tool_name="mcp__hyharness__robotwin_observe",
                choice="required",
                history=[
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "old",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    tool_result("old", "b2xk"),
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "new",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    tool_result("new", "bmV3"),
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "newer",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    tool_result("newer", "bmV3ZXI="),
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "latest",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    tool_result("latest", "bGF0ZXN0"),
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "very_latest",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_observe"
                                },
                            }
                        ],
                    },
                    tool_result("very_latest", "dmVyeV9sYXRlc3Q="),
                ],
            )
        )

        self.assertTrue(changed)
        payload = json.loads(body)
        rehydrated = [
            message
            for message in payload["messages"]
            if message.get("role") == "user"
            and isinstance(message.get("content"), list)
            and message["content"][0].get("text", "").startswith("Tool result from ")
        ]
        self.assertEqual(len(rehydrated), 4)
        self.assertFalse(
            any(
                "state old" in item.get("text", "") for item in rehydrated[0]["content"]
            )
        )
        self.assertTrue(
            all(
                any(item.get("type") == "image_url" for item in message["content"])
                for message in rehydrated
            )
        )

    def test_only_latest_user_turn_triggers_forced_recovery_tool(self) -> None:
        import json

        offered = [
            "mcp__hyharness__robotwin_observe",
            "mcp__hyharness__robotwin_vla_chunk",
            "mcp__hyharness__robotwin_move_arm",
        ]
        body, changed = inject_required_tool_choice(
            _body(
                tool_names=offered,
                choice="required",
                history=[
                    {
                        "role": "user",
                        "content": "INVALID TERMINATION: call a tool now.",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call",
                                "type": "function",
                                "function": {
                                    "name": "mcp__hyharness__robotwin_vla_chunk"
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": "call",
                        "content": json.dumps(
                            {"content": [{"type": "text", "text": "new state"}]}
                        ),
                    },
                ],
            )
        )

        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            offered,
        )

    def test_injects_required_for_harness_mcp_tool(self) -> None:
        import json

        body, changed = inject_required_tool_choice(
            _body(tool_name="mcp__rpent__robotwin_observe")
        )
        self.assertTrue(changed)
        self.assertEqual(json.loads(body)["tool_choice"], "required")

    def test_preserves_explicit_non_auto_choice(self) -> None:
        import json

        original = _body(
            tool_name="mcp__rpent__robotwin_observe",
            choice={"type": "function", "function": {"name": "mcp__rpent__finish"}},
        )
        body, changed = inject_required_tool_choice(original)
        self.assertFalse(changed)
        self.assertEqual(
            json.loads(body)["tool_choice"]["function"]["name"], "mcp__rpent__finish"
        )

    def test_leaves_non_harness_tool_request_unchanged(self) -> None:
        original = _body(tool_name="unrelated_tool")
        body, changed = inject_required_tool_choice(original)
        self.assertFalse(changed)
        self.assertEqual(body, original)

    def test_leaves_invalid_json_unchanged(self) -> None:
        body, changed = inject_required_tool_choice(b"not-json")
        self.assertFalse(changed)
        self.assertEqual(body, b"not-json")

    def test_robotwin_request_keeps_required_and_full_tool_list(self) -> None:
        import json

        offered = [
            "mcp__rpent__robotwin_observe",
            "mcp__rpent__robotwin_vla_chunk",
            "mcp__rpent__robotwin_move_arm",
            "mcp__rpent__finish",
        ]
        body, changed = inject_required_tool_choice(
            _body(
                tool_names=offered,
            ),
        )
        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(payload["tool_choice"], "required")
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            offered,
        )

    def test_robotwin_post_bootstrap_keeps_required_with_full_tool_list(self) -> None:
        """After the bootstrap pair the loop must stay under "required".

        Handing control back to "auto" stalled the episode: this model only
        emits a parseable tool call under "required", so under "auto" it
        produced prose (a literal "<tool_call>{...}" string) that the SDK never
        executed. The model still chooses *which* tool -- the full list is
        forwarded -- it just cannot answer without calling one.
        """
        import json

        history = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "mcp__rpent__robotwin_observe"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "mcp__rpent__robotwin_vla_chunk"},
                    },
                ],
            },
            {"role": "tool", "content": "state"},
        ]
        offered = [
            "mcp__rpent__robotwin_observe",
            "mcp__rpent__robotwin_vla_chunk",
            "mcp__rpent__robotwin_execute_ee",
            "mcp__rpent__finish",
        ]
        body, changed = inject_required_tool_choice(
            _body(tool_names=offered, history=history)
        )
        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(payload["tool_choice"], "required")
        # No narrowing: the model still picks among every offered tool.
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]], offered
        )

    def test_robotwin_stops_steering_once_finish_was_called(self) -> None:
        """``finish`` is a genuine model choice, so honour it and stop."""
        history = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "mcp__rpent__robotwin_observe"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "mcp__rpent__robotwin_vla_chunk"},
                    },
                    {"type": "function", "function": {"name": "mcp__rpent__finish"}},
                ],
            },
            {"role": "tool", "content": "done"},
        ]
        original = _body(
            tool_names=[
                "mcp__rpent__robotwin_observe",
                "mcp__rpent__robotwin_vla_chunk",
                "mcp__rpent__finish",
            ],
            history=history,
        )
        body, changed = inject_required_tool_choice(original)
        self.assertFalse(changed)
        self.assertEqual(body, original)

    def test_mcp_namespace_is_recognized_without_tool_restriction(self) -> None:
        """MCP namespace changes must not disable the required-tool policy.

        Renaming the MCP server (``mcp__rpent__`` -> ``mcp__hyharness__``) once
        silently disabled this proxy: requests stopped looking like Harness
        traffic, ``tool_choice`` stayed "auto", and the model emitted
        ``<tool_call>`` as prose that the SDK never executed.
        """
        import json

        for prefix in ("mcp__rpent__", "mcp__hyharness__", "mcp__anything__"):
            with self.subTest(prefix=prefix):
                observe = f"{prefix}robotwin_observe"
                vla_chunk = f"{prefix}robotwin_vla_chunk"

                body, changed = inject_required_tool_choice(
                    _body(tool_names=[observe, vla_chunk])
                )
                self.assertTrue(changed)
                payload = json.loads(body)
                self.assertEqual(payload["tool_choice"], "required")
                self.assertEqual(
                    [tool["function"]["name"] for tool in payload["tools"]],
                    [observe, vla_chunk],
                )

                history = [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {"type": "function", "function": {"name": observe}}
                        ],
                    },
                    {"role": "tool", "content": "state"},
                ]
                body, changed = inject_required_tool_choice(
                    _body(tool_names=[observe, vla_chunk], history=history)
                )
                self.assertTrue(changed)
                payload = json.loads(body)
                self.assertEqual(payload["tool_choice"], "required")
                self.assertEqual(
                    [tool["function"]["name"] for tool in payload["tools"]],
                    [observe, vla_chunk],
                )

    def test_appends_act_now_rule_only_after_an_observation(self) -> None:
        """The act-now rule must land at the very END of the prompt.

        Measured on Hy-Embodied-VLM: the identical rule was obeyed 0/4 times at
        the top of the system prompt and 4/4 at the end, so placement is part
        of the contract, not a cosmetic detail.
        """
        import json

        offered = [
            "mcp__rpent__robotwin_observe",
            "mcp__rpent__robotwin_vla_chunk",
            "mcp__rpent__finish",
        ]

        def call_msg(name: str) -> dict:
            return {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": name}}],
            }

        bootstrapped = [
            {"role": "user", "content": "RoboTwin instruction: beat the block."},
            call_msg("mcp__rpent__robotwin_observe"),
            {"role": "tool", "content": "state"},
            call_msg("mcp__rpent__robotwin_vla_chunk"),
            {"role": "tool", "content": "stepped"},
        ]

        # Last call was a VLA chunk -> acting already, no reminder needed.
        body, changed = inject_required_tool_choice(
            _body(tool_names=offered, history=list(bootstrapped))
        )
        self.assertTrue(changed)
        self.assertNotIn("CRITICAL RULE", body.decode())

        # Last call was observe -> the reminder must be appended, at the end.
        history = bootstrapped + [
            call_msg("mcp__rpent__robotwin_observe"),
            {"role": "tool", "content": "unchanged"},
        ]
        body, changed = inject_required_tool_choice(
            _body(tool_names=offered, history=history)
        )
        self.assertTrue(changed)
        payload = json.loads(body)
        self.assertEqual(payload["tool_choice"], "required")
        # Tool list stays intact: the model still chooses how to act.
        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]], offered
        )
        prompts = [
            message["content"]
            for message in payload["messages"]
            if message.get("role") in ("user", "system")
            and isinstance(message.get("content"), str)
        ]
        self.assertTrue(prompts[-1].rstrip().endswith("complete offered tool set."))


if __name__ == "__main__":
    unittest.main()
