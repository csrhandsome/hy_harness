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
_FINISH_TOOL = "finish"

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
    """Inject a bounded tool-choice bootstrap for Harness MCP requests.

    Returns the body to forward and whether it was rewritten. Non-JSON bodies
    and generic OpenAI clients are left untouched.
    """
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return body, False
    if not isinstance(payload, dict) or not _has_harness_tool(payload):
        return body, False

    choice = payload.get("tool_choice")
    if choice not in (None, "", "auto"):
        return body, False

    # Resolve the namespaced names actually offered by this client, keyed by
    # bare tool name, so a renamed MCP server still matches.
    available = {
        _bare_tool_name(name): name
        for tool in payload.get("tools", [])
        if (name := _tool_name(tool))
    }
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
            return body, False
        payload["tool_choice"] = "required"
        if _last_tool_called(payload) == _OBSERVE_TOOL:
            _append_act_now_reminder(payload)
    else:
        # Non-RoboTwin Harnesses retain the conservative previous behavior:
        # make only their initial MCP request choose a tool, then leave later
        # requests in automatic mode.
        if history:
            return body, False
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
