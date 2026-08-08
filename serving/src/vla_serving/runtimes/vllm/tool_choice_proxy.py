"""Small local proxy that makes Harness MCP tool use deterministic.

CodeBuddy's OpenAI-compatible client sends the MCP tool schemas, but omits
``tool_choice``. Hy-Embodied-VLM's vLLM ``hy_v3`` tool parser reliably emits a
structured call for the Harness only when the request says
``tool_choice="required"``. For RoboTwin it uses a small deterministic
bootstrap: first force ``robotwin_observe``, then force one
``robotwin_vla_step``. Later turns retain the normal ``auto`` policy so the
model can decide whether to continue acting or call ``finish`` rather than
being forced to observe forever. It otherwise transparently forwards HTTP
traffic to the private vLLM backend.

The implementation uses only the standard library so the serving environment
does not need another runtime dependency. Streaming Server-Sent Events are
copied as they arrive instead of being buffered.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


def _has_rpent_tool(payload: dict[str, Any]) -> bool:
    """Return whether an OpenAI request includes a Harness MCP tool."""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function")
        if isinstance(function, dict) and str(function.get("name", "")).startswith(
            "mcp__rpent__"
        ):
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


def _restrict_to_tool(payload: dict[str, Any], name: str) -> bool:
    """Keep exactly one named OpenAI function in the request's tool list."""
    tools = payload.get("tools")
    if not isinstance(tools, list):
        return False
    selected = [tool for tool in tools if _tool_name(tool) == name]
    if not selected:
        return False
    payload["tools"] = selected
    return True


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
    if not isinstance(payload, dict) or not _has_rpent_tool(payload):
        return body, False

    choice = payload.get("tool_choice")
    if choice not in (None, "", "auto"):
        return body, False

    available = {_tool_name(tool) for tool in payload.get("tools", [])}
    observe = "mcp__rpent__robotwin_observe"
    vla_step = "mcp__rpent__robotwin_vla_step"
    history = _message_tool_history(payload)

    # RoboTwin needs an initial perception/action pair. Without this, the
    # model's auto mode either returns no call or repeatedly describes an
    # observation in prose. Once the pair is complete, restore auto so finish
    # remains a genuine model choice.
    if observe in available:
        if observe not in history and _restrict_to_tool(payload, observe):
            payload["tool_choice"] = "required"
        elif vla_step not in history and _restrict_to_tool(payload, vla_step):
            payload["tool_choice"] = "required"
        else:
            return body, False
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

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._forward()

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._forward()

    def do_OPTIONS(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
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
