"""Small local proxy that makes Harness MCP tool use deterministic.

CodeBuddy's OpenAI-compatible client sends the MCP tool schemas, but omits
``tool_choice``. Hy-Embodied-VLM's vLLM ``hy_v3`` tool parser reliably emits a
structured call for the Harness only when the request says
``tool_choice="required"``. For RoboTwin it does not force a particular tool; the prompt profile remains
free to select memory, perception, VLA chunks, or deterministic primitives. Subsequent turns keep ``tool_choice="required"`` but
offer the full tool list, so the model chooses its own next action while still
being obliged to emit a parseable call; ``finish`` ends the steering. It
otherwise transparently forwards HTTP traffic to the private vLLM backend.

The implementation uses only the standard library so the serving environment
does not need another runtime dependency. Streaming Server-Sent Events are
copied as they arrive instead of being buffered.
"""

from __future__ import annotations

import argparse
import http.client
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

#: MCP tool names arrive namespaced as ``mcp__<server>__<tool>``. Matching on
#: the bare tool name keeps this proxy working when the server namespace is
#: renamed -- a mismatch here silently disables the whole tool-choice
#: bootstrap, because requests then look like generic OpenAI traffic.
_MCP_PREFIX = "mcp__"
_OBSERVE_TOOL = "robotwin_observe"
_RECOVERY_TOOL = "robotwin_vla_chunk"
_FINISH_TOOL = "finish"
_INVALID_TERMINATION_MARKER = "INVALID TERMINATION:"
_MULTIMODAL_ENVELOPE_KEY = "_hyharness_multimodal"

#: Appended verbatim to the end of the prompt when the model has just observed.
#: Kept as prose (not a tool-list restriction) so the model still chooses
#: between acting, correcting and finishing.
_ACT_NOW_REMINDER = (
    "\n\nCRITICAL RULE: You have ALREADY observed and the observation is in "
    "your context. Do not call robotwin_observe again until a genuinely fresh "
    "view is needed. Choose the appropriate motion, memory, audit, status, or "
    "finish tool from the complete offered tool set."
)


def _is_mcp_tool(name: str) -> bool:
    """Return whether an OpenAI function name is MCP-namespaced."""
    return name.startswith(_MCP_PREFIX) and name.count("__") >= 2


def _bare_tool_name(name: str) -> str:
    """Return the tool name without its ``mcp__<server>__`` namespace."""
    if not _is_mcp_tool(name):
        return name
    return name.split("__", 2)[2]


def _has_harness_tool(payload: dict[str, Any]) -> bool:
    """Return whether an OpenAI request includes a Harness MCP tool."""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and _is_mcp_tool(str(function.get("name", ""))):
            return True
    return False


def _tool_name(tool: Any) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function")
    return str(function.get("name", "")) if isinstance(function, dict) else ""


def _restrict_to_tool(payload: dict[str, Any], name: str) -> bool:
    """Keep one named OpenAI function in the request's tool list."""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    selected = [tool for tool in tools if _tool_name(tool) == name]
    if not selected:
        return False
    payload["tools"] = selected
    return True


def _contains_invalid_termination_marker(content: Any) -> bool:
    if isinstance(content, str):
        return _INVALID_TERMINATION_MARKER in content
    if isinstance(content, list):
        return any(
            isinstance(item, dict)
            and _INVALID_TERMINATION_MARKER in str(item.get("text", ""))
            for item in content
        )
    return False


def _is_invalid_termination_retry(payload: dict[str, Any]) -> bool:
    """Return whether the runtime is retrying after a text-only completion."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    # The continuation marker remains in CodeBuddy's replayed history. Only
    # force the recovery VLA tool when that marker is the *latest* user turn;
    # later tool-result messages must restore the full RoboTwin tool set.
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        return _contains_invalid_termination_marker(message.get("content"))
    return False


def _harness_tool_calls_by_id(payload: dict[str, Any]) -> dict[str, str]:
    """Map CodeBuddy tool-call IDs to their MCP tool names."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return {}

    calls: dict[str, str] = {}
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            call_id = call.get("id")
            name = _tool_name(call)
            if isinstance(call_id, str) and _is_mcp_tool(name):
                calls[call_id] = _bare_tool_name(name)
    return calls


def _decode_serialized_tool_content(content: Any) -> list[dict[str, Any]]:
    """Return model-visible text/image blocks from CodeBuddy's tool string.

    CodeBuddy serializes an MCP ``CallToolResult`` into the OpenAI ``tool``
    message as a JSON string such as ``{"content": [{"type": "image",
    "image": "..."}]}``. Hy-Embodied's chat template intentionally renders
    only ``user`` messages, so leaving that string in a ``tool`` message makes
    both observation state and images invisible to the model.
    """
    decoded: Any = content
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return [{"type": "text", "text": content}]

    if isinstance(decoded, dict):
        decoded = decoded.get("content")
    if not isinstance(decoded, list):
        return [{"type": "text", "text": str(content)}]

    blocks: list[dict[str, Any]] = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if item_type == "text" and isinstance(item.get("text"), str):
            try:
                envelope = json.loads(item["text"])
            except json.JSONDecodeError:
                envelope = None
            multimodal = (
                envelope.get(_MULTIMODAL_ENVELOPE_KEY)
                if isinstance(envelope, dict)
                else None
            )
            if isinstance(multimodal, dict):
                text = multimodal.get("text")
                if isinstance(text, str) and text:
                    blocks.append({"type": "text", "text": text})
                for image in multimodal.get("images", []):
                    if not isinstance(image, dict):
                        continue
                    data = image.get("data")
                    if not isinstance(data, str) or not data:
                        continue
                    mime_type = image.get("mimeType") or "image/png"
                    if not isinstance(mime_type, str):
                        mime_type = "image/png"
                    url = (
                        data
                        if data.startswith("data:")
                        else f"data:{mime_type};base64,{data}"
                    )
                    blocks.append({"type": "image_url", "image_url": {"url": url}})
                continue
            blocks.append({"type": "text", "text": item["text"]})
            continue
        if item_type != "image":
            continue

        data = item.get("image") or item.get("data")
        if not isinstance(data, str) or not data:
            continue
        mime_type = item.get("mimeType") or item.get("media_type") or "image/png"
        if not isinstance(mime_type, str):
            mime_type = "image/png"
        url = data if data.startswith("data:") else f"data:{mime_type};base64,{data}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _rehydrate_harness_tool_results(payload: dict[str, Any]) -> bool:
    """Make serialized Harness tool results visible to Hy-Embodied-VLM.

    The OpenAI API's ``role=tool`` content is text-only in CodeBuddy's
    transport. Replace only Harness MCP tool-result messages with a following
    ``role=user`` multimodal message. The Hy-Embodied chat template renders
    those user blocks (including ``image_url``) into vision tokens.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False
    tool_names = _harness_tool_calls_by_id(payload)
    if not tool_names:
        return False

    changed = False
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "tool":
            continue
        tool_call_id = message.get("tool_call_id")
        if not isinstance(tool_call_id, str):
            continue
        tool_name = tool_names.get(tool_call_id)
        if tool_name is None:
            continue

        blocks = _decode_serialized_tool_content(message.get("content", ""))
        if not blocks:
            continue
        message["role"] = "user"
        message["content"] = [
            {"type": "text", "text": f"Tool result from {tool_name}:\n"},
            *blocks,
        ]
        message.pop("tool_call_id", None)
        changed = True
    return changed


def _prune_stale_harness_tool_results(payload: dict[str, Any]) -> bool:
    """Keep only recent rehydrated tool results in the model prompt.

    Every RoboTwin action returns three RGB views. Replaying all historical
    tool results into a continuation request would quickly exceed the model
    context window. The chat template ignores assistant/tool history, so keep
    only the four newest user-visible tool results: they contain the current
    state and images needed for the next decision.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return False

    result_indexes: list[int] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list) or not content:
            continue
        first = content[0]
        if not (
            isinstance(first, dict)
            and isinstance(first.get("text"), str)
            and first["text"].startswith("Tool result from ")
        ):
            continue
        result_indexes.append(index)

    stale = set(result_indexes[:-4])
    if not stale:
        return False
    messages[:] = [
        message for index, message in enumerate(messages) if index not in stale
    ]
    return True


def _message_tool_history(payload: dict[str, Any]) -> set[str]:
    """Return tool names already called in an OpenAI conversation history."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return set()
    names: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        names.update(name for call in calls if (name := _tool_name(call)))
    return names


def _last_tool_called(payload: dict[str, Any]) -> str | None:
    """Return the bare name of the most recent tool call, if any."""
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in reversed(calls):
            if name := _tool_name(call):
                return _bare_tool_name(name)
    return None


def _append_act_now_reminder(payload: dict[str, Any]) -> bool:
    """Append the act-next rule to the end of the last user-visible message.

    Placement matters: measured on Hy-Embodied-VLM, the identical rule obeyed
    0/4 times when placed at the top of the system prompt and 4/4 when placed
    at the very end, immediately before generation. The rule is also
    state-dependent ("you have already observed"), which is why it lives here
    rather than as static text in the harness prompt.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if message.get("role") not in ("user", "system"):
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + _ACT_NOW_REMINDER
            return True
        if isinstance(content, list):
            content.append({"type": "text", "text": _ACT_NOW_REMINDER})
            return True
    return False


def inject_required_tool_choice(
    body: bytes,
) -> tuple[bytes, bool]:
    """Rewrite Harness results and inject a bounded required tool choice.

    Returns the body to forward and whether it was rewritten. Non-JSON bodies
    and generic OpenAI clients are left untouched.
    """
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, False
    if not isinstance(payload, dict) or not _has_harness_tool(payload):
        return body, False

    # Resolve the namespaced names actually offered by this client, keyed by
    # bare tool name, so a renamed MCP server still matches.
    available = {
        _bare_tool_name(name): name
        for tool in payload.get("tools", [])
        if (name := _tool_name(tool))
    }
    rehydrated_tool_results = _rehydrate_harness_tool_results(payload)
    pruned_stale_results = _prune_stale_harness_tool_results(payload)
    if _is_invalid_termination_retry(payload):
        recovery_tool = available.get(_RECOVERY_TOOL)
        if recovery_tool and _restrict_to_tool(payload, recovery_tool):
            # ``required`` alone is not reliably honored by hy_v3 after a
            # prose completion. Constrain this recovery turn to a fresh VLA
            # action so the agent must re-enter the physical control loop.
            payload["tool_choice"] = "required"
            return json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"), True

    choice = payload.get("tool_choice")
    if choice not in (None, "", "auto"):
        if not rehydrated_tool_results and not pruned_stale_results:
            return body, False
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        ), True

    history = {_bare_tool_name(name) for name in _message_tool_history(payload)}

    # Keep every RoboTwin turn profile-neutral: ``required`` guarantees a parseable
    # structured call, while the prompt decides whether memory, perception, a VLA
    # chunk, a deterministic primitive, or finish is appropriate.
    #
    # Handing the loop back to "auto" does not work: this model only emits a
    # parseable HYV3 tool call under "required" (see the module docstring).
    # Under "auto" it degenerates into prose -- often a literal "<tool_call>
    # {...}" string that the SDK never executes, so the simulator can remain frozen
    # until max_turns or timeout.
    #
    # "required" alone is not enough either: it compels *a* call but not a
    # useful one, and the model then re-observes indefinitely (measured: 98 of
    # 101 calls in one episode were robotwin_observe, advancing the sim twice).
    # So when the previous call was an observation, append the act-now rule to
    # the end of the prompt. ``finish`` stays a genuine model choice: it is
    # offered every turn, and once called the harness stops steering.
    if _OBSERVE_TOOL in available:
        if _FINISH_TOOL in history:
            # The model already chose to finish; stop steering it.
            if not rehydrated_tool_results and not pruned_stale_results:
                return body, False
            return json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"), True
        payload["tool_choice"] = "required"
        if _last_tool_called(payload) == _OBSERVE_TOOL:
            _append_act_now_reminder(payload)
    else:
        # Non-RoboTwin Harnesses retain the conservative previous behavior:
        # make only their initial MCP request choose a tool, then leave later
        # requests in automatic mode.
        if history:
            if not rehydrated_tool_results and not pruned_stale_results:
                return body, False
            return json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8"), True
        payload["tool_choice"] = "required"

    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    ), True


class ToolChoiceProxyHandler(BaseHTTPRequestHandler):
    """Forward requests to vLLM, rewriting only Harness tool-call requests."""

    protocol_version = "HTTP/1.1"
    backend_host = "127.0.0.1"
    backend_port = 8081

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[tool-choice-proxy] {self.address_string()} {fmt % args}", flush=True)

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_OPTIONS(self) -> None:
        self._forward()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0"))
        return self.rfile.read(length) if length else b""

    def _forward(self) -> None:
        body = self._read_body()
        rewritten = False
        if self.command == "POST" and body:
            body, rewritten = inject_required_tool_choice(body)
        if rewritten:
            print(
                "[tool-choice-proxy] injected required Harness bootstrap tool choice",
                flush=True,
            )

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower()
            not in {
                "host",
                "content-length",
                "connection",
                "proxy-connection",
                "accept-encoding",
            }
        }
        if body:
            headers["Content-Length"] = str(len(body))

        connection = http.client.HTTPConnection(
            self.backend_host,
            self.backend_port,
            timeout=600,
        )
        try:
            connection.request(
                self.command, self.path, body=body or None, headers=headers
            )
            response = connection.getresponse()
            self.send_response(response.status, response.reason)
            for key, value in response.getheaders():
                if key.lower() in {
                    "connection",
                    "keep-alive",
                    "proxy-authenticate",
                    "proxy-authorization",
                    "te",
                    "trailers",
                    "transfer-encoding",
                    "upgrade",
                }:
                    continue
                self.send_header(key, value)
            # http.client de-chunks upstream bodies. Closing the downstream
            # response cleanly terminates streaming SSE for the client.
            self.send_header("Connection", "close")
            self.end_headers()
            while chunk := response.read(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
            self.close_connection = True
        except (ConnectionError, OSError, http.client.HTTPException) as exc:
            self.send_error(502, f"vLLM backend unavailable: {exc}")
        finally:
            connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forward local vLLM requests and force Harness MCP tool calls."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend-url", default="http://127.0.0.1:8081")
    args = parser.parse_args(argv)

    parsed = urlparse(args.backend_url)
    if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
        parser.error("--backend-url must be an http://host:port URL")

    ToolChoiceProxyHandler.backend_host = parsed.hostname
    ToolChoiceProxyHandler.backend_port = parsed.port
    server = ThreadingHTTPServer((args.host, args.port), ToolChoiceProxyHandler)
    print(
        "[tool-choice-proxy] "
        f"listening on http://{args.host}:{args.port} -> {args.backend_url}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
