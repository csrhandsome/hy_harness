"""Cursor SDK planner (local Agent + CustomTool bridge).

Exposes the shared :class:`~rpent.tools.toolkit.Toolkit` to a local Cursor
agent via ``LocalAgentOptions.custom_tools`` (no separate MCP subprocess).
Requires ``CURSOR_API_KEY`` and the ``cursor-sdk`` package.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.cli.tui import next_user_line
from rpent.planner.base import PlannerResult, strip_mcp_prefix
from rpent.tools.toolkit import ToolResult, Toolkit
from rpent.utils.config import get_repo_root, load_local_env
from rpent.utils.logging import get_logger

logger = get_logger("cursor")

DEFAULT_MODEL = "composer-2.5"


def _parse_model_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Parse ``id`` or ``id,key=val,...`` / ``id+key=val,...`` into (id, params)."""
    raw = spec.strip()
    if not raw:
        return DEFAULT_MODEL, {}
    # Split id from optional params on the first ',' or '+' that introduces key=value.
    model_id = raw
    param_blob = ""
    for sep in (",", "+"):
        if sep in raw:
            head, tail = raw.split(sep, 1)
            if "=" in tail:
                model_id, param_blob = head.strip(), tail
                break
    params: dict[str, str] = {}
    if param_blob:
        for piece in param_blob.replace("+", ",").split(","):
            piece = piece.strip()
            if not piece or "=" not in piece:
                continue
            k, _, v = piece.partition("=")
            k, v = k.strip(), v.strip()
            if k:
                params[k] = v
    return model_id or DEFAULT_MODEL, params


def _default_speed_params(model_id: str) -> dict[str, str]:
    """Prefer non-thinking / fast variants when the model supports them."""
    mid = model_id.strip().lower()
    if mid in {"auto", "default"}:
        return {}
    if mid.startswith("composer-"):
        return {"fast": "true"}
    if mid.startswith("claude-"):
        return {"thinking": "false"}
    if mid.startswith("gpt-") or mid.startswith("codex") or mid == "glm-5.2":
        # Prefer no/low reasoning when the model exposes it.
        if mid in {"gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.5", "gpt-5.4", "gpt-5.6-sol",
                   "gpt-5.6-terra", "gpt-5.6-luna"}:
            return {"reasoning": "none"}
        if mid == "glm-5.2":
            return {"reasoning": "high"}  # lowest available for this id
        return {"reasoning": "low", "fast": "true"}
    if mid.startswith("grok-"):
        return {"effort": "low", "fast": "true"}
    if mid.startswith("gemini-") and "flash" in mid:
        return {"effort": "minimal"} if mid == "gemini-3.6-flash" else {}
    return {}


def resolve_cursor_model(model: str | None) -> Any:
    """Build a Cursor ``model`` argument (str or ``ModelSelection``).

    Specs:
      - ``composer-2.5`` → Composer with ``fast=true`` (default)
      - ``composer-2.5,fast=false`` → override
      - ``auto`` / ``default`` → server pick (may think; slower)
      - ``CURSOR_MODEL_PARAMS=thinking=false,fast=true`` merges in
    """
    if model is None or str(model).strip() == "":
        model = os.environ.get("CURSOR_MODEL", DEFAULT_MODEL)
    model_id, explicit = _parse_model_spec(str(model))
    # Map common alias used in CLI/docs to the SDK id.
    if model_id == "auto":
        model_id = "default"

    params = _default_speed_params(model_id)
    env_blob = os.environ.get("CURSOR_MODEL_PARAMS", "").strip()
    if env_blob:
        _, env_params = _parse_model_spec(f"_,{env_blob}")
        params.update(env_params)
    params.update(explicit)

    if not params:
        return model_id if model_id != "default" else "auto"

    try:
        from cursor_sdk import ModelParameterValue, ModelSelection
    except ImportError:
        return model_id

    return ModelSelection(
        id=model_id,
        params=[ModelParameterValue(id=k, value=v) for k, v in params.items()],
    )


class CursorSdkPlanner:
    """Planner backed by the Cursor Python SDK (local runtime)."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        model: str | None = None,
        timeout_s: int = 1200,
        api_key: str | None = None,
        output_path: str | Path | None = None,
        dashboard: Any = None,
    ):
        """Initialize the Cursor SDK backend."""
        load_local_env()
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = resolve_cursor_model(model)
        self._timeout_s = timeout_s
        self._api_key = api_key or os.environ.get("CURSOR_API_KEY")
        self._output_path = Path(output_path) if output_path else None
        self._dashboard = dashboard

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        """Run a local Cursor agent session for the given prompt."""
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        if not self._api_key:
            return PlannerResult(
                error="CURSOR_API_KEY is not set",
                stats={"backend": "cursor_sdk"},
            )

        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="cursor_sdk_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)

        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        recorder = _Recorder(max_turns=max_turns, dashboard=self._dashboard)
        state: dict[str, Any] = {}

        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Cursor SDK model %s (timeout=%ds, cwd=%s)",
            self._model,
            self._timeout_s,
            self._repo_root,
        )

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                toolkit,
                output_path,
                raw_stream_path,
                recorder,
                state,
                input_queue,
            ),
            name="cursor-sdk",
            daemon=True,
        )
        worker.start()
        try:
            worker.join(timeout=self._timeout_s)
            error: str | None = None
            if worker.is_alive():
                error = f"Cursor SDK timed out after {self._timeout_s}s"
                _cancel_run(state)
                rendered = f"\n[cursor-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                worker.join(timeout=15)
            elif "error" in state:
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[cursor-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            _close_agent(state)

        elapsed = time.time() - started
        text = state.get("text", "") or output_path.read_text(errors="replace")
        error = error or recorder.error or state.get("run_error")

        logger.info("Cursor SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "cursor_sdk", "content": text}],
            stats={
                "backend": "cursor_sdk",
                "model": self._model,
                "agent_id": state.get("agent_id"),
                "run_id": state.get("run_id"),
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _run_session(
        self,
        prompt: str,
        toolkit: Toolkit,
        output_path: Path,
        raw_stream_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        input_queue: queue.Queue[str | None] | None,
    ) -> None:
        try:
            from cursor_sdk import (
                Agent,
                AgentOptions,
                CursorAgentError,
                CustomTool,
                LocalAgentOptions,
            )
        except ImportError as exc:
            state["error"] = ImportError(
                'cursor-sdk is not installed; run: uv pip install "cursor-sdk"'
            )
            state["error"].__cause__ = exc
            return

        custom_tools = _build_custom_tools(toolkit, recorder, CustomTool)
        chunks: list[str] = []
        try:
            with Agent.create(
                AgentOptions(
                    api_key=self._api_key,
                    model=self._model,
                    local=LocalAgentOptions(
                        cwd=self._repo_root,
                        # Inline tools only — ignore ambient .cursor MCP/settings.
                        setting_sources=[],
                        custom_tools=custom_tools,
                    ),
                )
            ) as agent:
                state["agent"] = agent
                state["agent_id"] = getattr(agent, "agent_id", None)
                logger.info("cursor agent_id=%s", state["agent_id"])

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    write_lock = threading.Lock()

                    def _emit(rendered: str) -> None:
                        if not rendered:
                            return
                        with write_lock:
                            chunks.append(rendered)
                            out_f.write(rendered)
                            out_f.flush()
                        logger.info(rendered.rstrip())

                    def _emit_user(line: str) -> None:
                        rendered = f"\n[user] {line}\n"
                        _emit(rendered)

                    run = agent.send(prompt)
                    state["run"] = run
                    state["run_id"] = getattr(run, "id", None)
                    logger.info("cursor run_id=%s", state["run_id"])

                    stop_steer: threading.Event | None = None
                    if input_queue is not None:
                        stop_steer = threading.Event()

                        def _steer() -> None:
                            while not stop_steer.is_set():
                                nxt = next_user_line(input_queue)
                                if stop_steer.is_set():
                                    return
                                if nxt is None:
                                    _cancel_run(state)
                                    return
                                _emit_user(nxt)
                                try:
                                    follow = agent.send(nxt)
                                    state["run"] = follow
                                    for message in follow.messages():
                                        _write_jsonl(raw_f, _message_to_json(message))
                                        _emit(recorder.observe(message))
                                        if recorder.finish_result is not None:
                                            _cancel_run({"run": follow})
                                            return
                                    follow.wait()
                                except Exception as e:
                                    _emit(f"\n[cursor-planner] steer failed: {e}\n")
                                    return

                        threading.Thread(
                            target=_steer, name="cursor-steer", daemon=True
                        ).start()

                    try:
                        for message in run.messages():
                            _write_jsonl(raw_f, _message_to_json(message))
                            _emit(recorder.observe(message))
                            if recorder.finish_result is not None:
                                logger.info(
                                    "FINISH called: %s", recorder.finish_result
                                )
                                _cancel_run(state)
                                break
                        result = run.wait()
                        status = getattr(result, "status", None)
                        if status == "error" and recorder.finish_result is None:
                            state["run_error"] = f"run failed: {getattr(result, 'id', '')}"
                        usage = getattr(result, "usage", None) or getattr(
                            run, "usage", None
                        )
                        if usage is not None:
                            recorder.set_usage(usage)
                    finally:
                        if stop_steer is not None:
                            stop_steer.set()
                            if input_queue is not None:
                                input_queue.put(None)

            state["text"] = "".join(chunks)
        except CursorAgentError as e:
            state["error"] = e
        except Exception as e:
            state["error"] = e


# ---------------------------------------------------------------------------
# CustomTool bridge
# ---------------------------------------------------------------------------


def _build_custom_tools(
    toolkit: Toolkit,
    recorder: "_Recorder",
    custom_tool_cls: Any,
) -> dict[str, Any]:
    """Map toolkit specs → Cursor ``CustomTool`` handlers."""
    tools: dict[str, Any] = {}
    for spec in toolkit.get_tools_spec():
        name = str(spec["name"])

        def _make_execute(tool_name: str):
            def execute(args: Any, context: Any) -> Any:
                del context  # tool_call_id available but unused
                payload = dict(args or {})
                tr = toolkit.execute_tool(tool_name, payload)
                recorder.note_tool_result(tool_name, tr, payload)
                return _tool_result_payload(tr)

            return execute

        tools[name] = custom_tool_cls(
            description=str(spec.get("description", "")),
            input_schema=spec.get("input_schema", {"type": "object"}),
            execute=_make_execute(name),
        )
    return tools


def _tool_result_payload(tr: ToolResult) -> Any:
    """Convert a :class:`ToolResult` into a Cursor CustomTool return value."""
    content: list[dict[str, Any]] = []
    for block in tr.content_blocks:
        btype = block.get("type")
        if btype == "text":
            content.append({"type": "text", "text": block.get("text", "")})
        elif btype == "image":
            src = block.get("source", {})
            content.append(
                {
                    "type": "image",
                    "data": src.get("data", ""),
                    "mimeType": src.get("media_type", "image/png"),
                }
            )
    if not content:
        return json.dumps(tr.result, ensure_ascii=False, default=str)
    return {"content": content}


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Consume Cursor SDK messages; emit text + accumulate stats / finish."""

    max_turns: int
    dashboard: Any = None
    turns: int = 0
    tool_calls: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    finish_result: dict[str, Any] | None = None
    error: str | None = None

    def stats(self) -> dict[str, int]:
        return {"turns_used": self.turns, "tool_calls": self.tool_calls, **self.usage}

    def note_tool_result(
        self, name: str, tr: ToolResult, args: dict[str, Any]
    ) -> None:
        """Record a CustomTool execution (authoritative for ``finish``)."""
        self.tool_calls += 1
        if self.dashboard is not None:
            self.dashboard.on_event(
                {"type": "tool_call", "tool": name, "args": args}
            )
            self.dashboard.on_event(
                {"type": "tool_result", "tool": name, "result": tr.result}
            )
        if tr.is_finish and self.finish_result is None:
            self.finish_result = dict(tr.result)

    def set_usage(self, usage: Any) -> None:
        self.usage = {
            "total_input_tokens": _int_attr(usage, "input_tokens"),
            "total_cached_input_tokens": _int_attr(
                usage, "cache_read_tokens", "cached_input_tokens"
            ),
            "total_output_tokens": _int_attr(usage, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                usage, "reasoning_tokens", "reasoning_output_tokens"
            ),
        }
        if self.dashboard is not None:
            self.dashboard.on_usage(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )

    def observe(self, message: Any) -> str:
        mtype = str(_get(message, "type", ""))

        if mtype == "assistant":
            text = _extract_text(_get(_get(message, "message"), "content"))
            if not text:
                return ""
            self.turns += 1
            if self.dashboard is not None:
                self.dashboard.on_event({"type": "text", "text": text})
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[cursor] {text}\n"
            )

        if mtype == "thinking":
            text = str(_get(message, "text", "")).strip()
            if text and self.dashboard is not None:
                self.dashboard.on_event({"type": "thinking", "text": text})
            return f"[cursor-thinking] {text}\n" if text else ""

        if mtype == "tool_call":
            name = strip_mcp_prefix(str(_get(message, "name", "tool")))
            status = str(_get(message, "status", ""))
            if status == "running":
                args = _get(message, "args") or {}
                return (
                    f"[tool->] {name}: "
                    f"{json.dumps(_jsonable(args), ensure_ascii=False)[:500]}\n"
                )
            if status in {"completed", "error"}:
                # finish is captured in note_tool_result (CustomTool path).
                result = _get(message, "result")
                return (
                    f"[tool<-] {name}: "
                    f"{_short_json(_jsonable(result), limit=500)}\n"
                )
            return ""

        if mtype == "usage":
            usage = _get(message, "usage")
            if usage is not None:
                self.set_usage(usage)
            return ""

        if mtype == "status":
            status = str(_get(message, "status", ""))
            msg = str(_get(message, "message", "")).strip()
            line = f"[cursor-status] {status}"
            if msg:
                line += f" {msg}"
            return line + "\n"

        if mtype == "system":
            return f"[cursor-system] {_get(message, 'subtype', '')}\n"

        return ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _cancel_run(state: dict[str, Any]) -> None:
    run = state.get("run")
    if run is None:
        return
    try:
        if hasattr(run, "supports") and not run.supports("cancel"):
            return
        run.cancel()
    except Exception:
        pass


def _close_agent(state: dict[str, Any]) -> None:
    agent = state.get("agent")
    if agent is None:
        return
    try:
        close = getattr(agent, "close", None)
        if callable(close):
            close()
    except Exception:
        pass


def _write_jsonl(f: Any, obj: Any) -> None:
    f.write(json.dumps(obj, ensure_ascii=False, default=str) + "\n")
    f.flush()


def _message_to_json(message: Any) -> Any:
    if isinstance(message, dict):
        return message
    if hasattr(message, "__dict__"):
        return _jsonable(message)
    return {"repr": repr(message)}


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            k: _jsonable(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return str(value)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _int_attr(obj: Any, *keys: str) -> int:
    for key in keys:
        val = _get(obj, key)
        if val is None:
            continue
        try:
            return int(val)
        except (TypeError, ValueError):
            continue
    return 0


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            else:
                text = _get(block, "text")
                if text:
                    parts.append(str(text))
        return "".join(parts).strip()
    text = _get(content, "text")
    return str(text).strip() if text else ""


def _short_json(value: Any, *, limit: int = 500) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text
