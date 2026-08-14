from __future__ import annotations

import asyncio
import json
import tempfile
import threading
from dataclasses import asdict, is_dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import codebuddy_agent_sdk as sdk


class Handler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.requests.append(json.loads(self.rfile.read(length)))
        chunks = [
            {"id": "schema-probe", "object": "chat.completion.chunk", "created": 0,
             "model": "hy_a3b", "choices": [{"index": 0, "delta": {
                 "role": "assistant", "content": "I am stopping with ordinary prose."
             }, "finish_reason": None}]},
            {"id": "schema-probe", "object": "chat.completion.chunk", "created": 0,
             "model": "hy_a3b", "choices": [{"index": 0, "delta": {},
                                                "finish_reason": "stop"}]},
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for chunk in chunks:
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, fmt: str, *args: object) -> None:
        pass


async def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = Path(tempfile.mkdtemp(prefix="codebuddy-schema-probe-"))
    config_dir = root / ".codebuddy"
    config_dir.mkdir()
    endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
    (config_dir / "models.json").write_text(json.dumps({
        "models": [{"id": "hy_a3b", "name": "hy_a3b", "vendor": "OpenAI",
                    "apiKey": "${CODEBUDDY_API_KEY}", "url": endpoint,
                    "supportsToolCall": True, "supportsImages": True}],
        "availableModels": ["hy_a3b"],
    }))
    schema = {"type": "object", "properties": {"action": {
        "type": "string", "enum": ["observe", "vla_chunk"]}},
        "required": ["action"], "additionalProperties": False}
    options = sdk.CodeBuddyAgentOptions(
        cwd=root, model="hy_a3b", max_turns=2, tools=["StructuredOutput"],
        permission_mode="bypassPermissions", setting_sources=["project"],
        extra_args={"json-schema": json.dumps(schema, separators=(",", ":"))},
        env={"CODEBUDDY_API_KEY": "EMPTY", "NO_PROXY": "127.0.0.1,localhost",
             "no_proxy": "127.0.0.1,localhost"},
    )
    messages = []
    try:
        async with sdk.CodeBuddySDKClient(options=options) as client:
            await client.query("Choose the next action. Do not answer in prose.")
            async for message in client.receive_response():
                messages.append(asdict(message) if is_dataclass(message) else repr(message))
    finally:
        server.shutdown()
        thread.join()
    print(json.dumps({"messages": messages}, ensure_ascii=False, default=str))
    for payload in Handler.requests:
        print(json.dumps({
            "tool_choice": payload.get("tool_choice"),
            "response_format": payload.get("response_format"),
            "tools": [x.get("function", {}).get("name") for x in payload.get("tools", [])],
        }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
