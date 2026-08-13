"""CodeBuddy Agent SDK planner.

SDK-first planner backend for the Harness.
``solve()`` prepares output files, binds the in-process toolkit via
``create_sdk_mcp_server``, drives the SDK query, and assembles a
``PlannerResult``. Requires the ``codebuddy-agent-sdk`` package; the default
local vLLM route uses ``CODEBUDDY_API_KEY=EMPTY`` and needs no remote login.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hy_harness.planner.base import (
    PlannerResult,
    add_mcp_prefix,
    strip_mcp_prefix,
)
from hy_harness.tools.base import BaseTool
from hy_harness.utils.config import get_repo_root, load_local_env
from hy_harness.utils.logging import get_logger, init_output_dir

logger = get_logger("codebuddy")

DEFAULT_MODEL = "hy_a3b"
DEFAULT_INVALID_TEXT_RETRIES = 3

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodeBuddyPlanner:
    """Planner backed by the CodeBuddy Agent SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        model: str | None = None,
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 1200,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        dashboard: Any = None,
    ):
        """Initialize the CodeBuddy Agent SDK backend."""
        load_local_env()
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model or os.environ.get("CODEBUDDY_MODEL", DEFAULT_MODEL)
        self._allowed_tools = allowed_tools
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._dashboard = dashboard
        self._api_key = os.environ.get("CODEBUDDY_API_KEY")
        self._invalid_text_retries = max(
            0,
            int(
                os.environ.get(
                    "CODEBUDDY_INVALID_TEXT_RETRIES",
                    str(DEFAULT_INVALID_TEXT_RETRIES),
                )
            ),
        )

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: BaseTool,
        max_turns: int,
        input_queue=None,
    ) -> PlannerResult:
        """Run a CodeBuddy Agent SDK session for the given prompt."""
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        return asyncio.run(
            self._solve_async(
                prompt,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        )

    # -- internal lifecycle -------------------------------------------------

    async def _solve_async(
        self,
        prompt: str,
        *,
        toolkit: BaseTool,
        max_turns: int,
        input_queue=None,
    ) -> PlannerResult:
        import codebuddy_agent_sdk as sdk

        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codebuddy_sdk_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        recorder = _Recorder(max_turns=max_turns, dashboard=self._dashboard)

        init_output_dir(self._output_dir)
        options = self._build_agent_options(sdk, tools=toolkit, max_turns=max_turns)

        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking CodeBuddy Agent SDK model %s (timeout=%ds)",
            self._model,
            self._timeout_s,
        )

        started = time.time()
        error: str | None = None
        rendered_chunks: list[str] = []
        with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:

            def _emit(message: Any) -> None:
                _write_jsonl(raw_f, _message_to_json(message))
                if rendered := recorder.observe(message):
                    rendered_chunks.append(rendered)
                    out_f.write(rendered)
                    out_f.flush()
                    logger.info(rendered.rstrip())

            def _emit_user(line: str) -> None:
                rendered = f"\n[user] {line}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                logger.info(rendered.strip())

            try:
                if input_queue is None:
                    await self._run_one_shot(
                        sdk,
                        prompt,
                        options,
                        recorder,
                        toolkit=toolkit,
                        timeout_s=self._timeout_s,
                        emit=_emit,
                    )
                else:
                    # TODO:这里没有被启动过。但是也许是一个改良点？
                    await self._run_interactive(
                        sdk,
                        prompt,
                        options,
                        recorder,
                        input_queue,
                        emit=_emit,
                        emit_user=_emit_user,
                    )
            except asyncio.TimeoutError:
                error = f"CodeBuddy Agent SDK timed out after {self._timeout_s}s"
                rendered = f"\n[codebuddy-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                rendered = f"\n[codebuddy-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())

        elapsed = time.time() - started
        text = "".join(rendered_chunks) or output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("CodeBuddy Agent SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codebuddy_sdk", "content": text}],
            stats={
                "backend": "codebuddy_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                **recorder.stats(),
            },
            error=error,
        )

    async def _run_one_shot(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        *,
        toolkit: BaseTool,
        timeout_s: int,
        emit,
    ) -> None:
        """Run one SDK session without cancelling its async generator directly.

        ``sdk.query()`` owns an AnyIO task group. Cancelling the task that is
        iterating that async generator can make older SDK releases close the
        group from an async-generator-finalizer task, producing:
        ``Attempted to exit cancel scope in a different task``. Use the
        stateful client API instead so timeout handling can interrupt and drain
        the same session before its context manager disconnects it.
        """
        async with sdk.CodeBuddySDKClient(options=options) as client:
            await client.query(prompt)
            deadline = asyncio.get_running_loop().time() + timeout_s
            retries = 0

            async def consume_response() -> None:
                async for message in client.receive_response():
                    emit(message)
                    if recorder.finish_result is not None:
                        logger.info("FINISH called: %s", recorder.finish_result)
                        with contextlib.suppress(Exception):
                            await client.interrupt()
                        return

            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise asyncio.TimeoutError

                consumer = asyncio.create_task(consume_response())
                done, _ = await asyncio.wait({consumer}, timeout=remaining)
                if consumer not in done:
                    # Do not use asyncio.wait_for() here: it cancels the task
                    # that is directly iterating the SDK stream. Instead, ask
                    # the CLI to stop and give that same stream a short chance
                    # to finish normally.
                    with contextlib.suppress(Exception):
                        await client.interrupt()
                    done, _ = await asyncio.wait({consumer}, timeout=10)
                    if consumer in done:
                        consumer.result()
                    if not consumer.done():
                        consumer.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await consumer
                    raise asyncio.TimeoutError

                # Propagate an SDK/CLI failure before the context manager
                # disconnects the session.
                consumer.result()
                if recorder.finish_result is not None:
                    return

                status = _toolkit_status(toolkit)
                if not _status_requires_tool_call(status):
                    return
                if retries >= self._invalid_text_retries:
                    recorder.error = (
                        "Planner returned non-terminal text "
                        f"{retries + 1} times while RoboTwin remained active"
                    )
                    return

                retries += 1
                recorder.invalid_terminal_retries += 1
                continuation = _invalid_terminal_continuation(
                    status=status,
                    retry=retries,
                    limit=self._invalid_text_retries,
                )
                logger.warning(
                    "planner emitted non-terminal text while RoboTwin remains active; "
                    "sending continuation retry %d/%d",
                    retries,
                    self._invalid_text_retries,
                )
                await client.query(continuation)

    async def _run_interactive(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        input_queue,
        *,
        emit,
        emit_user,
    ) -> None:
        """Drive a stateful ``CodeBuddySDKClient`` with live steering."""
        async with sdk.CodeBuddySDKClient(options=options) as client:
            stop = asyncio.Event()

            async def pump_input() -> None:
                while not stop.is_set():
                    nxt = await asyncio.to_thread(next_user_line, input_queue)
                    if nxt is None:
                        stop.set()
                        with contextlib.suppress(Exception):
                            await client.interrupt()
                        return
                    emit_user(nxt)
                    recorder.suppress_next_result_error = True
                    with contextlib.suppress(Exception):
                        await client.interrupt()
                    with contextlib.suppress(Exception):
                        await client.query(nxt)

            async def consume() -> None:
                async for message in client.receive_messages():
                    emit(message)
                    if recorder.finish_result is not None:
                        logger.info("FINISH called: %s", recorder.finish_result)
                        return

            await client.query(prompt)
            pump = asyncio.create_task(pump_input())
            consumer = asyncio.create_task(consume())
            stop_wait = asyncio.create_task(stop.wait())
            try:
                await asyncio.wait(
                    {consumer, stop_wait}, return_when=asyncio.FIRST_COMPLETED
                )
            finally:
                input_queue.put(None)
                for task in (consumer, stop_wait, pump):
                    task.cancel()
                for task in (consumer, stop_wait, pump):
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task

    # -- options + tool bridge ---------------------------------------------

    def _build_agent_options(self, sdk: Any, *, tools: BaseTool, max_turns: int) -> Any:
        allowed = [
            part for part in self._allowed_tools.replace(",", " ").split() if part
        ]
        builtins = [name for name in allowed if "__" not in name]
        allowed.extend(
            add_mcp_prefix(str(description["name"]))
            for description in tools.get_tools_description()
        )

        # The bundled headless CLI and the local OpenAI-compatible endpoint
        # exchange arbitrary Unicode. Keep the subprocess on UTF-8 even when
        # a batch launcher inherited ``LANG=C`` / an ASCII-compatible locale.
        env: dict[str, str] = {
            "PYTHONUTF8": "1",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
        }
        if self._api_key:
            env["CODEBUDDY_API_KEY"] = self._api_key
        for key in (
            "CODEBUDDY_INTERNET_ENVIRONMENT",
            "CODEBUDDY_AUTH_TOKEN",
            "CODEBUDDY_CODE_PATH",
            "CODEBUDDY_BASE_URL",
            "CODEBUDDY_CUSTOM_HEADERS",
        ):
            if val := os.environ.get(key):
                env[key] = val

        # OpenAI-compatible endpoint (e.g. local vLLM): inject project models.json
        # and load project settings. Otherwise keep isolation (no filesystem settings).
        setting_sources: list[str] = []
        openai_base = os.environ.get(
            "CODEBUDDY_OPENAI_BASE_URL", "http://127.0.0.1:8080/v1"
        ).strip()
        if openai_base:
            chat_url = _normalize_openai_chat_url(openai_base)
            # The SDK launches CodeBuddy as a child process and otherwise
            # inherits shell proxy variables.  A corporate HTTP proxy often
            # cannot route a loopback/private vLLM endpoint, so extend
            # NO_PROXY for the configured OpenAI-compatible service.
            _add_openai_endpoint_to_no_proxy(env, chat_url)
            if "CODEBUDDY_API_KEY" not in env:
                env["CODEBUDDY_API_KEY"] = self._api_key or "EMPTY"
            _write_project_openai_model(
                repo_root=Path(self._repo_root),
                model_id=self._model,
                chat_url=chat_url,
            )
            setting_sources = ["project"]
            logger.info(
                "CodeBuddy OpenAI-compatible endpoint: model=%s url=%s",
                self._model,
                chat_url,
            )

        return sdk.CodeBuddyAgentOptions(
            cwd=self._repo_root,
            model=self._model,
            max_turns=max_turns,
            # The SDK treats ``None`` as "enable every built-in tool", while
            # an empty list disables them. RoboTwin passes an empty built-in
            # allowlist so the planner cannot escape the scoped MCP toolkit.
            tools=builtins,
            allowed_tools=list(dict.fromkeys(allowed)),
            permission_mode="bypassPermissions",
            mcp_servers={
                "hyharness": _build_hyharness_server(sdk, toolkit=tools),
            },
            # Empty = ignore user/project .codebuddy; ["project"] only when
            # CODEBUDDY_OPENAI_BASE_URL injects a harness-managed models.json.
            setting_sources=setting_sources,
            env=env,
            stderr=lambda line: logger.debug("[codebuddy-sdk] %s", line.rstrip()),
        )


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume SDK messages, emit text + accumulate stats."""

    max_turns: int
    dashboard: Any = None
    turns: int = 0
    _seen_assistant_ids: set[str] = field(default_factory=set)
    tool_calls: int = 0
    tool_names: dict[str, str] = field(default_factory=dict)
    pending_finish: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_input_tokens": 0,
            "total_cache_read_input_tokens": 0,
        }
    )
    total_cost_usd: float | None = None
    finish_result: dict[str, Any] | None = None
    error: str | None = None
    suppress_next_result_error: bool = False
    invalid_terminal_retries: int = 0

    def stats(self) -> dict[str, int | float | None]:
        return {
            "turns_used": self.turns,
            "tool_calls": self.tool_calls,
            "invalid_terminal_retries": self.invalid_terminal_retries,
            "total_cost_usd": self.total_cost_usd,
            **self.usage,
        }

    def observe(self, message: Any) -> str:
        kind = _kind(message)
        if kind == "SystemMessage":
            rendered = self._system(message)
        elif kind == "AssistantMessage":
            rendered = self._assistant(message)
        elif kind == "UserMessage":
            rendered = self._user(message)
        elif kind == "ResultMessage":
            rendered = self._result(message)
        else:
            rendered = ""
        if self.dashboard is not None:
            self.dashboard.on_usage(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        return rendered

    def _system(self, message: Any) -> str:
        subtype = _get(message, "subtype", "")
        if subtype == "thinking_tokens":
            return ""
        data = _get(message, "data", {})
        session = data.get("session_id") if isinstance(data, dict) else ""
        return f"[cb-system] subtype={subtype} session={session}\n"

    def _assistant(self, message: Any) -> str:
        self._add_usage(_get(message, "usage"))
        lines: list[str] = []
        if _get(message, "parent_tool_use_id") is None:
            assistant_id = _get(message, "message_id") or _get(message, "uuid")
            if not assistant_id or assistant_id not in self._seen_assistant_ids:
                if assistant_id:
                    self._seen_assistant_ids.add(str(assistant_id))
                self.turns += 1
                lines.append(f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n")
        for block in _get(message, "content", []) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(_get(block, "text", "")).strip()
                if text:
                    lines.append(f"[codebuddy] {text}\n")
                    if self.dashboard is not None:
                        self.dashboard.on_event({"type": "text", "text": text})
            elif block_kind == "ToolUseBlock":
                tool_id = str(_get(block, "id", ""))
                name = strip_mcp_prefix(str(_get(block, "name", "tool")))
                self.tool_names[tool_id] = name
                tool_input = _get(block, "input", {}) or {}
                if name == "finish" and isinstance(tool_input, dict):
                    self.pending_finish[tool_id] = dict(tool_input)
                lines.append(f"[tool->] {name}: {_short_json(tool_input, limit=500)}\n")
                if self.dashboard is not None:
                    self.dashboard.on_event(
                        {"type": "tool_call", "tool": name, "args": tool_input}
                    )
            elif block_kind == "ToolResultBlock":
                lines.append(self._tool_result(block))
        if assistant_error := _get(message, "error"):
            lines.append(f"[cb-assistant-error] {assistant_error}\n")
        return "".join(lines)

    def _user(self, message: Any) -> str:
        tool_use_id = _get(message, "parent_tool_use_id")
        content = _get(message, "content", "")
        if tool_use_id:
            return self._tool_result_content(content, tool_use_id=str(tool_use_id))
        if isinstance(content, list):
            return "".join(
                self._tool_result(block)
                for block in content
                if _kind(block) == "ToolResultBlock"
            )
        return ""

    def _tool_result(self, block: Any, *, tool_use_id: str | None = None) -> str:
        return self._tool_result_content(
            _get(block, "content", ""),
            tool_use_id=tool_use_id or str(_get(block, "tool_use_id", "")),
            is_error=_get(block, "is_error"),
        )

    def _tool_result_content(
        self,
        content: Any,
        *,
        tool_use_id: str,
        is_error: Any = None,
    ) -> str:
        self.tool_calls += 1
        name = self.tool_names.get(tool_use_id, "tool_result")
        result_payload = _tool_result_payload(content)
        is_error = bool(is_error) or bool(
            isinstance(result_payload, dict) and result_payload.get("error")
        )
        summary: dict[str, Any] = {"size": _payload_size(content)}
        image_count = _content_image_count(content)
        if image_count:
            summary["images"] = image_count
        if is_error:
            summary["is_error"] = bool(is_error)
        pending = self.pending_finish.pop(tool_use_id, None)
        if (
            pending is not None
            and not is_error
            and isinstance(result_payload, dict)
            and result_payload.get("_finish") is True
            and self.finish_result is None
        ):
            # A finish request is only terminal when the *tool result*
            # explicitly confirms it. CodeBuddy can serialize rejected MCP
            # results as text with ``is_error=false``; trusting the original
            # tool request would incorrectly interrupt the planner.
            self.finish_result = result_payload
        if self.dashboard is not None:
            self.dashboard.on_event(
                {
                    "type": "tool_result",
                    "tool": name,
                    "result": {**summary, "is_error": bool(is_error)},
                }
            )
        return f"[tool<-] {name}: {json.dumps(summary, ensure_ascii=False)}\n"

    def _result(self, message: Any) -> str:
        if usage := _get(message, "usage"):
            self._set_usage(usage)
        if cost := _get(message, "total_cost_usd"):
            self.total_cost_usd = float(cost)
        suppress = self.suppress_next_result_error
        self.suppress_next_result_error = False
        if _get(message, "is_error", False) and not suppress:
            self.error = (
                f"CodeBuddy Agent SDK result {_get(message, 'subtype', 'error')}"
            )

        parts = ["[cb-result]", str(_get(message, "subtype", ""))]
        if duration_ms := _get(message, "duration_ms"):
            parts.append(f"duration={duration_ms / 1000:.1f}s")
        if self.total_cost_usd is not None:
            parts.append(f"cost=${self.total_cost_usd:.4f}")
        if result := str(_get(message, "result", "") or ""):
            parts.append(f"result_size={len(result)}")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cache_create={self.usage['total_cache_creation_input_tokens']} "
            f"cache_read={self.usage['total_cache_read_input_tokens']} "
            f"out={self.usage['total_output_tokens']} tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    def _add_usage(self, usage: Any) -> None:
        usage = _usage_dict(usage)
        if not usage:
            return
        self.usage["total_input_tokens"] += int(usage.get("input_tokens") or 0)
        self.usage["total_output_tokens"] += int(usage.get("output_tokens") or 0)
        self.usage["total_cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens") or 0
        )
        self.usage["total_cache_read_input_tokens"] += int(
            usage.get("cache_read_input_tokens") or 0
        )

    def _set_usage(self, usage: Any) -> None:
        usage = _usage_dict(usage)
        if not usage:
            return
        self.usage = {
            "total_input_tokens": int(usage.get("input_tokens") or 0),
            "total_output_tokens": int(usage.get("output_tokens") or 0),
            "total_cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
            "total_cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens") or 0
            ),
        }


# ---------------------------------------------------------------------------
# Tool bridge (HyHarness registry -> SDK MCP server)
# ---------------------------------------------------------------------------


def _build_hyharness_server(sdk: Any, *, toolkit: BaseTool) -> Any:
    sdk_tools = []
    for spec in toolkit.get_tools_description():
        name = str(spec["name"])
        description = str(spec.get("description", ""))
        input_schema = spec.get("input_schema", {"type": "object"})

        async def run_tool(
            args: dict[str, Any],
            *,
            tool_name: str = name,
        ) -> dict[str, Any]:
            return _tool_result_to_mcp(toolkit.execute_tool(tool_name, args or {}))

        run_tool.__name__ = f"hyharness_{name}"
        sdk_tools.append(sdk.tool(name, description, input_schema)(run_tool))

    return sdk.create_sdk_mcp_server(name="hyharness", version="0.1.0", tools=sdk_tools)


def _tool_result_to_mcp(tr: Any) -> dict[str, Any]:
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        return {"content": [{"type": "text", "text": str(tr)}]}

    text_blocks: list[str] = []
    images: list[dict[str, str]] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            text_blocks.append(str(_get(block, "text", "")))
        elif block_type == "image":
            src = _get(block, "source", {})
            images.append(
                {
                    "data": _get(src, "data", ""),
                    "mimeType": _get(src, "media_type", "image/png"),
                }
            )

    if images:
        # CodeBuddy's chat-completions transport accepts text-only tool
        # results. Preserve the images in a private JSON envelope; the local
        # proxy decodes it before forwarding the turn to vLLM as image_url
        # user content.
        content = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "_hyharness_multimodal": {
                            "text": "\n".join(text_blocks),
                            "images": images,
                        }
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ]
    else:
        content = [{"type": "text", "text": text} for text in text_blocks]

    response: dict[str, Any] = {"content": content}
    result_dict = getattr(tr, "result", None)
    if isinstance(result_dict, dict) and result_dict.get("error"):
        # MCP's CallToolResult schema uses camelCase. ``is_error`` is ignored
        # by CodeBuddy's MCP client and makes failed file/tool calls look like
        # normal observations to the planner.
        response["isError"] = True
    return response


# ---------------------------------------------------------------------------
# OpenAI-compatible endpoint (local vLLM / custom chat API)
# ---------------------------------------------------------------------------


def _normalize_openai_chat_url(base: str) -> str:
    """Return a full ``.../chat/completions`` URL for CodeBuddy models.json."""
    url = base.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    return f"{url}/v1/chat/completions"


def _add_openai_endpoint_to_no_proxy(env: dict[str, str], chat_url: str) -> None:
    """Ensure the configured OpenAI-compatible host bypasses inherited proxies."""
    from urllib.parse import urlparse

    hostname = urlparse(chat_url).hostname
    if not hostname:
        return

    existing = (
        env.get("NO_PROXY")
        or env.get("no_proxy")
        or os.environ.get("NO_PROXY")
        or os.environ.get("no_proxy", "")
    )
    entries = [part.strip() for part in existing.split(",") if part.strip()]
    if hostname not in entries:
        entries.append(hostname)
    # Some HTTP stacks only inspect lowercase while others inspect uppercase.
    value = ",".join(entries)
    env["NO_PROXY"] = value
    env["no_proxy"] = value


def _write_project_openai_model(
    *,
    repo_root: Path,
    model_id: str,
    chat_url: str,
) -> Path:
    """Write harness-managed ``.codebuddy/models.json`` under ``repo_root``."""
    codebuddy_dir = repo_root / ".codebuddy"
    codebuddy_dir.mkdir(parents=True, exist_ok=True)
    path = codebuddy_dir / "models.json"
    payload = {
        "models": [
            {
                "id": model_id,
                "name": model_id,
                "vendor": "OpenAI",
                "apiKey": "${CODEBUDDY_API_KEY}",
                "url": chat_url,
                "supportsToolCall": True,
                "supportsImages": True,
            }
        ],
        "availableModels": [model_id],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _usage_dict(usage: Any) -> dict[str, Any]:
    """Return ``usage`` as a dict, whether the SDK sends a dataclass or raw dict."""
    if isinstance(usage, dict):
        return usage
    if dataclasses.is_dataclass(usage):
        return dataclasses.asdict(usage)
    return {}


def _tool_result_payload(content: Any) -> dict[str, Any] | None:
    """Recover the result dict nested in CodeBuddy's tool-result text.

    The SDK can wrap an MCP result as nested lists/dicts and JSON-encoded text,
    e.g. ``[{"type": "text", "text": "[{\\"type\\": ...}]"}]``.
    """
    pending: list[Any] = [content]
    seen_strings: set[str] = set()
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            if "_finish" in value or "error" in value:
                return value
            pending.extend(
                value.get(key) for key in ("content", "text") if key in value
            )
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif dataclasses.is_dataclass(value):
            pending.append(dataclasses.asdict(value))
        elif isinstance(value, str) and value not in seen_strings:
            seen_strings.add(value)
            try:
                pending.append(json.loads(value))
            except json.JSONDecodeError:
                continue
        else:
            nested = [
                _get(value, key)
                for key in ("content", "text")
                if _get(value, key) is not None
            ]
            pending.extend(nested)
    return None


def _toolkit_status(toolkit: BaseTool) -> dict[str, Any] | None:
    """Read a live environment status when the toolkit exposes one."""
    status = getattr(toolkit, "status", None)
    if not callable(status):
        return None
    try:
        value = status()
    except Exception:  # noqa: BLE001 - status is advisory for generic toolkits
        return None
    return value if isinstance(value, dict) else None


def _status_requires_tool_call(status: dict[str, Any] | None) -> bool:
    """Return whether a text-only completion is invalid for this toolkit."""
    return (
        bool(status)
        and not bool(status.get("success"))
        and not bool(status.get("done"))
    )


def _invalid_terminal_continuation(
    *,
    status: dict[str, Any],
    retry: int,
    limit: int,
) -> str:
    remaining = ""
    count = status.get("take_action_cnt")
    step_limit = status.get("step_lim")
    if count is not None and step_limit is not None:
        remaining = f" Current simulator actions: {count}/{step_limit}."
    return (
        "INVALID TERMINATION: your previous response ended in prose without a "
        "valid RoboTwin terminal tool result. The authoritative state remains "
        f"success=false and done=false.{remaining} Do not explain, summarize, "
        "write an audit, emit literal <tool_call> text, or call finish. Use the "
        f"real tool-call channel to call an allowed RoboTwin tool now. Retry {retry}/{limit}."
    )


def _message_to_json(message: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(message):
        data = dataclasses.asdict(message)
    elif hasattr(message, "__dict__"):
        data = vars(message)
    else:
        data = {"value": repr(message)}
    return {"type": _kind(message), **_jsonable(data)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _content_image_count(value: Any) -> int:
    if isinstance(value, list):
        count = 0
        for item in value:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
        return count
    text = str(value)
    return text.count("'type': 'image'") + text.count('"type": "image"')
