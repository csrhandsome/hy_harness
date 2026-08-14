"""Provider-independent tool-use loop built on pydantic-ai.

This is a robot controller, not a chat wrapper. The environment status
owns episode lifetime:

- prose / JSON / a natural-language "I'm done" is not a legal terminal
  state while ``success=false`` and ``done=false``
- ``finish`` is confirmed only when the toolkit result carries ``_finish``
- rejected finish and invalid structured output retry in-loop
- camera images in history are pruned so the resent request stays bounded
"""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import json
import queue
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import (
    Agent,
    BinaryContent,
    ModelRetry,
    ModelSettings,
    Tool,
    ToolOutput,
    ToolReturn,
)
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from hy_harness.planner.base import PlannerResult
from hy_harness.planner.loop_control import (
    QUIT_TOKENS,
    invalid_terminal_continuation,
    invalid_text_retries as env_invalid_text_retries,
    status_requires_tool_call,
    toolkit_status,
)
from hy_harness.tools.base import BaseTool
from hy_harness.utils.logging import get_logger

logger = get_logger("api_loop")

_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350

#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always keep every image from at least this many most-recent camera
#: messages (one RoboTwin observe is three RGB frames).
_MIN_IMAGE_MESSAGES = 1

#: Drop images from older observations beyond this many camera messages.
_MAX_IMAGE_MESSAGES = 4

#: Keep the last N tool-return texts; older ones are stubbed.
_MAX_RECENT_TOOL_RETURNS = 4


class _TerminalDecision(BaseModel):
    """The only valid structured terminal output for the robot controller."""

    status: Literal["success", "failure"]
    summary: str


class ApiAgentLoop:
    """Planner that runs a structured tool-calling loop via pydantic-ai."""

    def __init__(
        self,
        model: Model,
        max_tokens: int = 8192,
        dashboard: Any = None,
        no_images: bool = False,
        timeout_s: int | None = None,
        invalid_text_retries: int | None = None,
    ):
        """Store the pydantic-ai model and robot-loop limits."""
        self._model = model
        self._max_tokens = max_tokens
        self._dashboard = dashboard
        self._no_images = no_images
        self._timeout_s = timeout_s
        self._invalid_text_retries = (
            invalid_text_retries
            if invalid_text_retries is not None
            else env_invalid_text_retries()
        )

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: BaseTool,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        """Run the tool-calling loop until finish, budget, or timeout."""
        return asyncio.run(
            self._solve_with_timeout(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        )

    async def _solve_with_timeout(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: BaseTool,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        try:
            if self._timeout_s:
                return await asyncio.wait_for(
                    self._solve(
                        system_prompt=system_prompt,
                        user_message=user_message,
                        toolkit=toolkit,
                        max_turns=max_turns,
                        input_queue=input_queue,
                    ),
                    timeout=self._timeout_s,
                )
            return await self._solve(
                system_prompt=system_prompt,
                user_message=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
            )
        except asyncio.TimeoutError:
            error = f"pydantic-ai planner timed out after {self._timeout_s}s"
            logger.error(error)
            return PlannerResult(
                stats={"backend": "pydantic_ai", "timeout_s": self._timeout_s},
                error=error,
            )

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: BaseTool,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        run_state = _RunState()
        controller_instruction = (
            "\n\nPYDANTIC CONTROLLER PROTOCOL: ordinary text is disabled. "
            "Use one real RoboTwin environment tool at a time, then wait for "
            "its returned state before choosing another motion. The ordinary "
            "`finish` tool is not available in this backend. Only when "
            "`robotwin_status`/a returned action result proves success=true "
            "or done=true may you invoke the structured "
            "`robotwin_terminal` output tool. If success=false and done=false, "
            "you must make another real RoboTwin tool call."
        )
        agent = Agent(
            self._model,
            instructions=(system_prompt or "") + controller_instruction,
            tools=_build_tools(toolkit, no_images=self._no_images, run_state=run_state),
            # ToolOutput makes a free-text assistant response invalid. This is
            # the PydanticAI structured-output contract we need for a robot
            # controller: terminal intent is machine-readable and validated
            # against the live environment before the Agent may end.
            output_type=ToolOutput(
                _TerminalDecision,
                name="robotwin_terminal",
                description=(
                    "Use only after the live RoboTwin status is terminal. "
                    "Use status='success' only when benchmark success=true; "
                    "use status='failure' only when done=true and success=false."
                ),
                max_retries=max(3, self._invalid_text_retries),
                sequential=True,
            ),
            model_settings=_build_model_settings(self._model, self._max_tokens),
            capabilities=_build_capabilities(self._model),
            # The default PydanticAI output/tool retry budget is one. Local
            # HyV3 occasionally emits an incomplete or malformed tool call
            # after a rejected finish; retry inside the same structured loop
            # rather than abandoning the planner after a single bad turn.
            retries=max(3, max_turns),
        )
        _register_terminal_validator(agent, toolkit=toolkit, run_state=run_state)

        interactive = input_queue is not None
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        n_tool_calls = 0
        turns = 0
        last_error: str | None = None
        usage: RunUsage | None = None
        quit_requested = False
        retries = 0

        def _inject_pending(run: Any) -> bool:
            """Drain queued user lines into the live run; True => end session."""
            while True:
                try:
                    line = input_queue.get_nowait()  # type: ignore[union-attr]
                except queue.Empty:
                    return False
                if line is None:
                    return True
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return True
                if not line:
                    continue
                run.enqueue(line, priority="asap")
                messages.append({"role": "user", "content": line})
                logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))

        async def _await_next() -> str | None:
            logger.info("awaiting input — type a message to continue, /quit to end")
            while True:
                line = await asyncio.to_thread(input_queue.get)  # type: ignore[union-attr]
                if line is None:
                    return None
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return None
                if line:
                    logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))
                    return line

        seed: str | None = user_message
        history: list[ModelMessage] | None = None
        try:
            while True:
                run_turns = 0
                needs_continuation = False
                async with agent.iter(
                    seed,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    async for node in run:
                        if interactive and _inject_pending(run):
                            quit_requested = True
                            break
                        if Agent.is_call_tools_node(node):
                            turns += 1
                            run_turns += 1
                            run_state.begin_model_turn()
                            response = node.model_response
                            response_message = _serialize_response(response)
                            messages.append(response_message)
                            _log_response(response, run.usage, run_turns, max_turns)
                            _emit_dashboard_response(
                                self._dashboard,
                                response_message,
                                run.usage,
                                n_tool_calls,
                            )

                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    if isinstance(event, FunctionToolCallEvent):
                                        n_tool_calls += 1
                                        if self._dashboard is not None:
                                            self._dashboard.on_event(
                                                {
                                                    "type": "tool_call",
                                                    "tool": event.part.tool_name,
                                                    "args": event.part.args_as_dict(),
                                                }
                                            )
                                    elif isinstance(event, FunctionToolResultEvent):
                                        message = _serialize_tool_result(event)
                                        messages.append(message)
                                        _log_tool_result(message)
                                        if self._dashboard is not None:
                                            self._dashboard.on_event(
                                                {
                                                    "type": "tool_result",
                                                    "tool": message.get("name")
                                                    or "tool_result",
                                                    "result": {
                                                        "is_error": bool(
                                                            getattr(
                                                                event.part,
                                                                "is_error",
                                                                False,
                                                            )
                                                        ),
                                                        "size": len(
                                                            str(
                                                                message.get("content")
                                                                or ""
                                                            )
                                                        ),
                                                    },
                                                }
                                            )
                                    if self._dashboard is not None:
                                        self._dashboard.on_usage(
                                            inp=int(run.usage.input_tokens or 0),
                                            out=int(run.usage.output_tokens or 0),
                                            tool_calls=n_tool_calls,
                                        )

                            if run_state.finish_result is not None:
                                logger.info(
                                    "FINISH confirmed: %s", run_state.finish_result
                                )
                                break
                            if turns >= max_turns:
                                if run_state.finish_result is None:
                                    last_error = (
                                        "pydantic-ai planner exhausted "
                                        f"max_turns={max_turns} without a "
                                        "validated RoboTwin terminal state"
                                    )
                                    logger.warning(last_error)
                                else:
                                    logger.info(
                                        "reached max_turns=%d. Stopping.", max_turns
                                    )
                                break
                        elif Agent.is_end_node(node):
                            if run_state.finish_result is not None:
                                break
                            status = toolkit_status(toolkit)
                            if not interactive and status_requires_tool_call(status):
                                needs_continuation = True
                                logger.warning(
                                    "model ended turn without a tool call while "
                                    "RoboTwin remains active"
                                )
                            elif interactive:
                                logger.info(
                                    "model ended turn without a tool call "
                                    "— awaiting your input."
                                )
                            else:
                                logger.info(
                                    "model ended turn without a tool call. Stopping."
                                )
                            break

                    usage = run.usage
                    history = run.all_messages()

                if run_state.finish_result is not None or quit_requested:
                    break
                if turns >= max_turns:
                    if last_error is None:
                        last_error = (
                            "pydantic-ai planner exhausted "
                            f"max_turns={max_turns} without a validated "
                            "RoboTwin terminal state"
                        )
                    break
                if needs_continuation:
                    status = toolkit_status(toolkit) or {}
                    if retries >= self._invalid_text_retries:
                        last_error = (
                            "Planner returned non-terminal text "
                            f"{retries + 1} times while RoboTwin remained active"
                        )
                        logger.error(last_error)
                        break
                    retries += 1
                    run_state.invalid_terminal_retries = retries
                    seed = invalid_terminal_continuation(
                        status=status,
                        retry=retries,
                        limit=self._invalid_text_retries,
                    )
                    messages.append({"role": "user", "content": seed})
                    logger.warning(
                        "sending continuation retry %d/%d",
                        retries,
                        self._invalid_text_retries,
                    )
                    continue
                if not interactive:
                    break
                nxt = await _await_next()
                if nxt is None:
                    break
                seed = nxt
                messages.append({"role": "user", "content": seed})
        except UsageLimitExceeded as e:
            last_error = f"pydantic-ai usage limit reached: {e}"
            logger.info(last_error)
        except Exception as e:  # noqa: BLE001 - surfaced via PlannerResult.error
            last_error = f"{type(e).__name__}: {e}"
            if _is_image_rejection(e) and not self._no_images:
                last_error += (
                    "\n\nThe model rejected image input — it is likely a "
                    "text-only model (no vision support). Re-run with "
                    "no_images=true: the planner will then keep every visual "
                    "observation as text instead of sending image bytes."
                )
            logger.error("agent run failed: %s", last_error)

        stats = _build_stats(usage, turns, n_tool_calls)
        stats["backend"] = "pydantic_ai"
        stats["invalid_terminal_retries"] = run_state.invalid_terminal_retries
        stats["structured_terminal_attempts"] = run_state.terminal_attempts
        stats["structured_terminal_rejections"] = run_state.terminal_rejections
        return PlannerResult(
            finish_result=run_state.finish_result,
            messages=messages,
            stats=stats,
            error=last_error,
        )


@dataclasses.dataclass
class _RunState:
    """Mutable per-solve bookkeeping shared with toolkit wrappers."""

    finish_result: dict[str, Any] | None = None
    invalid_terminal_retries: int = 0
    terminal_attempts: int = 0
    terminal_rejections: int = 0
    environment_tool_this_turn: str | None = None

    def begin_model_turn(self) -> None:
        """Allow one environment operation in the next model response."""
        self.environment_tool_this_turn = None


def _register_terminal_validator(
    agent: Agent,
    *,
    toolkit: BaseTool,
    run_state: _RunState,
) -> None:
    """Accept a structured terminal output only at an environment terminal.

    ``ToolOutput`` removes free-text exits, but its schema alone cannot know
    whether a model's claimed completion matches the simulator. The validator
    makes the simulator authoritative and returns a Pydantic ``ModelRetry``
    when the model tries to end a live episode.
    """

    @agent.output_validator
    async def _validate_terminal(decision: _TerminalDecision) -> _TerminalDecision:
        run_state.terminal_attempts += 1
        status = toolkit_status(toolkit) or {}
        success = bool(status.get("success"))
        done = bool(status.get("done"))
        if success:
            if decision.status != "success":
                run_state.terminal_rejections += 1
                raise ModelRetry(
                    "RoboTwin reports success=true. Invoke robotwin_terminal "
                    "with status='success'."
                )
            run_state.finish_result = {
                "_finish": True,
                "status": "success",
                "summary": decision.summary,
                **status,
            }
            return decision
        if done:
            if decision.status != "failure":
                run_state.terminal_rejections += 1
                raise ModelRetry(
                    "RoboTwin reports done=true and success=false. Invoke "
                    "robotwin_terminal with status='failure'."
                )
            run_state.finish_result = {
                "_finish": True,
                "status": "failure",
                "summary": decision.summary,
                **status,
            }
            return decision

        run_state.terminal_rejections += 1
        count = status.get("take_action_cnt")
        limit = status.get("step_lim")
        budget = (
            f" Current simulator actions: {count}/{limit}."
            if count is not None and limit is not None
            else ""
        )
        raise ModelRetry(
            "INVALID TERMINATION: RoboTwin is still active "
            "(success=false, done=false). Do not terminate or write prose."
            f"{budget} Call one real RoboTwin environment tool now."
        )


def _build_capabilities(model: Model) -> list[Any]:
    # pydantic-ai dispatches synchronous ProcessHistory processors through an
    # AnyIO worker thread. In the RoboTwin Python 3.10 runtime that path can
    # stall an Agent run before its first model request. Keep the pruning logic
    # synchronous and deterministic, but expose it through an async wrapper so
    # pydantic-ai runs it directly in the event loop.
    capabilities: list[Any] = [ProcessHistory(processor=_async_prune_history)]
    if _is_anthropic_model(model):
        capabilities.insert(0, Thinking(effort="high"))
    return capabilities


def _is_anthropic_model(model: Model) -> bool:
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
    except ImportError:
        return False
    return isinstance(model, AnthropicModel)


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    if _is_anthropic_model(model):
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            max_tokens=max_tokens,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    return ModelSettings(max_tokens=max_tokens)


def _prune_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images and stub old tool returns to bound request size."""
    pruned = _prune_history_images(messages)
    return _stub_old_tool_returns(pruned)


async def _async_prune_history(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Async adapter for pydantic-ai's ProcessHistory capability."""
    return _prune_history(messages)


def _prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images so the resent request body stays bounded.

    Images are grouped by message so one RoboTwin observe (three RGB frames)
    is kept or dropped together. The newest camera message is always kept.
    """
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            content = getattr(part, "content", None)
            if isinstance(content, list):
                for ii, item in enumerate(content):
                    if isinstance(item, BinaryContent) and str(
                        getattr(item, "media_type", "")
                    ).startswith("image/"):
                        located.append((mi, pi, ii, len(item.data)))
            elif isinstance(content, BinaryContent) and str(
                getattr(content, "media_type", "")
            ).startswith("image/"):
                located.append((mi, pi, -1, len(content.data)))

    if not located:
        return messages

    bytes_by_message: dict[int, int] = {}
    for mi, _pi, _ii, nbytes in located:
        bytes_by_message[mi] = bytes_by_message.get(mi, 0) + nbytes
    image_messages = sorted(bytes_by_message)

    keep_messages: set[int] = set()
    total = 0
    for rank, mi in enumerate(reversed(image_messages)):
        nbytes = bytes_by_message[mi]
        if rank < _MIN_IMAGE_MESSAGES or (
            len(keep_messages) < _MAX_IMAGE_MESSAGES
            and total + nbytes <= _MAX_HISTORY_IMAGE_BYTES
        ):
            keep_messages.add(mi)
            total += nbytes

    keep_items = {(mi, pi, ii) for mi, pi, ii, _ in located if mi in keep_messages}
    if len(keep_items) == len(located):
        return messages

    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    drop_whole_part: set[tuple[int, int]] = set()
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) in keep_items:
            continue
        if ii < 0:
            drop_whole_part.add((mi, pi))
        else:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    return _replace_dropped_images(
        messages,
        drop_items_by_part=drop_items_by_part,
        drop_whole_part=drop_whole_part,
    )


def _replace_dropped_images(
    messages: list[ModelMessage],
    *,
    drop_items_by_part: dict[tuple[int, int], set[int]],
    drop_whole_part: set[tuple[int, int]],
) -> list[ModelMessage]:
    placeholder = "[earlier camera image omitted to bound request size]"
    new_messages = list(messages)
    touched = set(drop_items_by_part) | drop_whole_part
    for mi, pi in touched:
        message = new_messages[mi]
        part = message.parts[pi]
        if (mi, pi) in drop_whole_part:
            new_part = dataclasses.replace(part, content=placeholder)
        else:
            drop_items = drop_items_by_part[(mi, pi)]
            new_content = [
                placeholder if ci in drop_items else item
                for ci, item in enumerate(part.content)
            ]
            new_part = dataclasses.replace(part, content=new_content)
        new_parts = list(message.parts)
        new_parts[pi] = new_part
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)
    return new_messages


def _stub_old_tool_returns(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Replace older tool-return bodies with a short placeholder."""
    located: list[tuple[int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if isinstance(part, ToolReturnPart):
                located.append((mi, pi))
    if len(located) <= _MAX_RECENT_TOOL_RETURNS:
        return messages

    drop = set(located[:-_MAX_RECENT_TOOL_RETURNS])
    placeholder = "[earlier tool result omitted to bound request size]"
    new_messages = list(messages)
    for mi, pi in drop:
        message = new_messages[mi]
        part = message.parts[pi]
        if getattr(part, "content", None) == placeholder:
            continue
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=placeholder)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)
    return new_messages


def _is_image_rejection(e: Exception) -> bool:
    """True when the provider returned a 4xx complaining about image input."""
    if not isinstance(e, ModelHTTPError):
        return False
    if not 400 <= e.status_code < 500:
        return False
    return "image" in str(e).lower()


def _build_tools(
    toolkit: BaseTool,
    *,
    no_images: bool,
    run_state: _RunState,
) -> list[Tool]:
    """Wrap the HyHarness toolkit as pydantic-ai function tools."""
    tools: list[Tool] = []
    for spec in toolkit.get_tools_description():
        name = spec["name"]
        # ``robotwin_terminal`` is the sole terminal interface for this
        # backend. Keeping the legacy free-form ``finish`` function would
        # give the model two competing exit paths and reintroduce an
        # unstructured status/summary decision.
        if name == "finish":
            continue
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(
                    toolkit, name, no_images=no_images, run_state=run_state
                ),
                name=name,
                description=spec.get("description", ""),
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                takes_ctx=False,
                # The model must see each tool result before selecting the
                # next operation. This both serializes real simulator
                # actions and prevents a single response from committing to
                # observe/action chains based on stale perception.
                sequential=True,
            )
        )
    return tools


def _make_tool_function(
    toolkit: BaseTool,
    name: str,
    *,
    no_images: bool,
    run_state: _RunState,
):
    """Return a callable that dispatches one tool call to the toolkit."""

    async def _call(**kwargs: Any) -> Any:
        if run_state.environment_tool_this_turn is not None:
            raise ModelRetry(
                "Only one RoboTwin tool is allowed per assistant turn. "
                f"{run_state.environment_tool_this_turn!r} already ran and "
                "returned fresh state; choose the next tool in a new turn."
            )
        run_state.environment_tool_this_turn = name
        result = toolkit.execute_tool(name, kwargs)
        if result.is_finish and run_state.finish_result is None:
            # Only the tool result may terminate the loop. A rejected
            # finish returns error without ``_finish`` and is ignored here.
            run_state.finish_result = dict(result.result)
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images and not no_images:
            return ToolReturn(return_value=text, content=images)
        return text

    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[BinaryContent]]:
    """Split Anthropic-shaped content blocks into text and image content."""
    text_parts: list[str] = []
    images: list[BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                images.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
    text = "\n\n".join(part for part in text_parts if part) or "{}"
    return text, images


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        if isinstance(part, TextPart):
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            if part.content:
                content.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            content.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args_as_dict(),
                }
            )
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps({"size": _payload_size(content)}, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _emit_dashboard_response(
    dashboard: Any,
    response_message: dict[str, Any],
    usage: RunUsage,
    n_tool_calls: int,
) -> None:
    if dashboard is None:
        return
    for block in response_message["content"]:
        if block["type"] == "text":
            dashboard.on_event({"type": "text", "text": block["text"]})
        elif block["type"] == "thinking":
            dashboard.on_event({"type": "thinking", "text": block["thinking"]})
    dashboard.on_usage(
        inp=int(usage.input_tokens or 0),
        out=int(usage.output_tokens or 0),
        tool_calls=n_tool_calls,
    )


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(part.args_as_dict(), default=str)
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        return text
    return text[:limit] + "...(+%d)" % (len(text) - limit)


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (bytes, bytearray)):
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))
