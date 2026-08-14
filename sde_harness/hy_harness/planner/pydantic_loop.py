"""Provider-independent tool-use loop built on pydantic-ai.

This is a robot controller, not a chat wrapper. The environment status
owns episode lifetime:

- prose / JSON / a natural-language "I'm done" is not a legal terminal
  state while ``success=false`` and ``done=false``
- ``finish`` is confirmed only when the toolkit result carries ``_finish``
- rejected finish and invalid structured output retry in-loop
- history is compacted by official pydantic-ai-harness strategies so
  long tool-call / tool-return sequences stay paired and bounded
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
from pydantic_ai.capabilities import Thinking
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from hy_harness.memory import (
    HISTORY_COMPACT_CONTINUATION,
    HistoryMemory,
    build_history_memory,
    is_recoverable_history_error,
)
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
_HIDDEN_TOOLS = frozenset({"finish", "write_text_file", "robotwin_execute_ee"})
_HISTORY_ERROR_RETRIES = 2
_FILE_LOOKUP_TOOLS = frozenset({"read_text_file", "list_dir"})
_PASSIVE_TOOLS = _FILE_LOOKUP_TOOLS | {"robotwin_observe", "robotwin_status"}
_MOTION_TOOLS = frozenset(
    {
        "robotwin_vla_chunk",
        "robotwin_move_arm",
        "robotwin_translate_arm",
        "robotwin_move_bimanual",
        "robotwin_rotate_arm",
        "robotwin_set_gripper",
        "robotwin_release",
        "robotwin_hold",
    }
)

# The RoboTwin prompt asks for the index plus at most two relevant leaves.
# Allow one extra directory listing, then force the model back to perception
# and physical control instead of letting it burn an episode guessing paths.
_MAX_FILE_LOOKUP_CALLS = 4

# Every action result already includes a fresh state, status and three camera
# views. Re-observing without an intervening external change just inflates the
# multimodal transcript and was the main trigger for local-vLLM 32k overflows.
_MAX_OBSERVE_CALLS = 1

# A controller may consult memory and status before it moves, but repeated
# passive requests add tool-call transcript without changing the environment.
# After this many consecutive passive requests it must execute a real motion.
_MAX_CONSECUTIVE_PASSIVE_TOOLS = 6


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
        enable_thinking: bool = False,
        history_memory: HistoryMemory | None = None,
    ):
        """Store the pydantic-ai model and robot-loop limits."""
        self._model = model
        self._max_tokens = max_tokens
        self._dashboard = dashboard
        self._no_images = no_images
        self._timeout_s = timeout_s
        self._enable_thinking = enable_thinking
        self._history_memory = history_memory or build_history_memory()
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
            "`finish` and `write_text_file` tools are not available in this "
            "backend; the runner writes the audit and recipe when the "
            "session ends. Only when `robotwin_status`/a returned action "
            "result proves success=true or done=true may you invoke the "
            "structured `robotwin_terminal` output tool. If success=false "
            "and done=false, you must make another real RoboTwin tool call."
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
            model_settings=_build_model_settings(
                self._model,
                self._max_tokens,
                enable_thinking=self._enable_thinking,
            ),
            capabilities=_build_capabilities(self._model, self._history_memory),
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
        history_error_retries = 0
        try:
            while True:
                run_turns = 0
                needs_continuation = False
                run: Any | None = None
                try:
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
                                                                    message.get(
                                                                        "content"
                                                                    )
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
                                if not interactive and status_requires_tool_call(
                                    status
                                ):
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
                except ModelHTTPError as e:
                    # A request can fail while ``agent.iter`` is advancing to
                    # its next model node. In that case the assignment after
                    # the context manager is never reached, even though the
                    # live run already contains valid user/tool history that
                    # can be compacted. Recover it before deciding whether a
                    # bounded retry is possible.
                    if run is not None:
                        partial_history = run.all_messages()
                        if partial_history:
                            history = partial_history
                    if (
                        history
                        and history_error_retries < _HISTORY_ERROR_RETRIES
                        and is_recoverable_history_error(e)
                        and status_requires_tool_call(toolkit_status(toolkit))
                    ):
                        history_error_retries += 1
                        history = await self._history_memory.compact_for_retry(
                            history, model=self._model
                        )
                        seed = HISTORY_COMPACT_CONTINUATION
                        messages.append({"role": "user", "content": seed})
                        logger.warning(
                            "recoverable history 400; compacted tool pairs "
                            "and retrying %d/%d",
                            history_error_retries,
                            _HISTORY_ERROR_RETRIES,
                        )
                        continue
                    raise

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
        stats["history_error_retries"] = history_error_retries
        stats["file_lookup_calls"] = run_state.file_lookup_calls
        stats["file_lookup_rejections"] = run_state.file_lookup_rejections
        stats["passive_tool_calls"] = run_state.passive_tool_calls
        stats["passive_tool_rejections"] = run_state.passive_tool_rejections
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
    file_lookup_calls: int = 0
    file_lookup_rejections: int = 0
    observe_calls: int = 0
    observe_rejections: int = 0
    passive_tool_calls: int = 0
    passive_tool_rejections: int = 0

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


def _build_capabilities(model: Model, history_memory: HistoryMemory) -> list[Any]:
    capabilities: list[Any] = list(history_memory.capabilities())
    if _is_anthropic_model(model):
        capabilities.insert(0, Thinking(effort="high"))
    return capabilities


def _is_anthropic_model(model: Model) -> bool:
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
    except ImportError:
        return False
    return isinstance(model, AnthropicModel)


def _build_model_settings(
    model: Model,
    max_tokens: int,
    *,
    enable_thinking: bool,
) -> ModelSettings:
    """Build bounded, sequential model settings for a robot control turn.

    The Hy-Embodied vLLM deployment recognizes
    ``chat_template_kwargs.enable_thinking``. Leaving it on caused each
    controller decision to spend thousands of tokens narrating an action it
    could instead express as one function call. Disable it by default for
    RoboTwin smoke/evaluation; the planner still receives the observation and
    tool schema, and can use up to ``max_tokens`` for its actual call.
    """
    common: ModelSettings = {
        "max_tokens": max_tokens,
        # The environment is stateful. Request one function call at a time;
        # the wrapper also guards this in case a provider ignores the flag.
        "parallel_tool_calls": False,
    }
    if _is_anthropic_model(model):
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        return AnthropicModelSettings(
            **common,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    if _is_openai_chat_model(model):
        common["extra_body"] = {
            "chat_template_kwargs": {"enable_thinking": enable_thinking}
        }
    return common


def _is_openai_chat_model(model: Model) -> bool:
    """True for OpenAI-compatible chat models, including the local vLLM."""
    try:
        from pydantic_ai.models.openai import OpenAIChatModel
    except ImportError:
        return False
    return isinstance(model, OpenAIChatModel)


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
    """Wrap the HyHarness toolkit as pydantic-ai function tools.

    Tool visibility is evaluated before every model request. Once the initial
    memory/observation budget is consumed, we remove passive tools from the
    schema instead of repeatedly returning ``ModelRetry`` after the model
    chooses them. This keeps the backend-facing function schema compact and
    makes a physical control action the only available next decision.
    """
    specs = toolkit.get_tools_description()
    robotwin_vla_controller = any(
        spec.get("name") == "robotwin_vla_chunk" for spec in specs
    )
    tools: list[Tool] = []
    for spec in specs:
        name = spec["name"]
        # ``robotwin_terminal`` is the sole terminal interface for this
        # backend. ``finish`` would reintroduce an unstructured exit, and
        # ``write_text_file`` only inflates tool history; the runner writes
        # the audit and recipe when the session ends.
        if name in _HIDDEN_TOOLS:
            continue
        tool = Tool.from_schema(
            function=_make_tool_function(
                toolkit, name, no_images=no_images, run_state=run_state
            ),
            name=name,
            description=spec.get("description", ""),
            json_schema=spec.get("input_schema")
            or {"type": "object", "properties": {}},
            takes_ctx=False,
            # The model must see each tool result before selecting the next
            # operation. This serializes stateful simulator actions.
            sequential=True,
        )
        tool.prepare = _prepare_robotwin_tool(
            name,
            run_state,
            robotwin_vla_controller=robotwin_vla_controller,
        )
        tools.append(tool)
    return tools


def _prepare_robotwin_tool(
    name: str,
    run_state: _RunState,
    *,
    robotwin_vla_controller: bool,
):
    """Hide exhausted passive tools before the model sees the schema."""

    async def _prepare(_ctx: Any, tool_def: Any) -> Any:
        if robotwin_vla_controller:
            # The live RoboTwin action result always returns authoritative
            # status plus all three camera images. A closed-loop Hy-VLA
            # controller therefore needs exactly one bootstrap observation,
            # then fresh VLA chunks until the structured terminal validator
            # sees done/success. Exposing file I/O, status polling, raw EE,
            # and speculative primitives lets the local model create long
            # non-moving tool transcripts that its HYV3 parser later rejects.
            if run_state.observe_calls == 0:
                return tool_def if name == "robotwin_observe" else None
            return tool_def if name == "robotwin_vla_chunk" else None
        if name in _FILE_LOOKUP_TOOLS and (
            run_state.file_lookup_calls >= _MAX_FILE_LOOKUP_CALLS
        ):
            return None
        if name == "robotwin_observe" and (
            run_state.observe_calls >= _MAX_OBSERVE_CALLS
        ):
            return None
        # Once a physical action has occurred, repeated status polling has no
        # new visual information; each motion result already includes status.
        if name == "robotwin_status" and run_state.passive_tool_calls >= 2:
            return None
        return tool_def

    return _prepare


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
        if name in _FILE_LOOKUP_TOOLS:
            if run_state.file_lookup_calls >= _MAX_FILE_LOOKUP_CALLS:
                run_state.file_lookup_rejections += 1
                raise ModelRetry(
                    "The RoboTwin memory lookup budget is exhausted. Do not "
                    "guess more file paths or call list_dir again. Call "
                    "robotwin_observe if needed, then execute a real "
                    "RoboTwin motion tool."
                )
            run_state.file_lookup_calls += 1
        if name == "robotwin_observe":
            if run_state.observe_calls >= _MAX_OBSERVE_CALLS:
                run_state.observe_rejections += 1
                raise ModelRetry(
                    "The initial RoboTwin observation and every completed "
                    "motion result already provide fresh three-view imagery. "
                    "Do not call robotwin_observe again; choose a real "
                    "RoboTwin motion tool using the latest result."
                )
            run_state.observe_calls += 1
        if name in _PASSIVE_TOOLS:
            if run_state.passive_tool_calls >= _MAX_CONSECUTIVE_PASSIVE_TOOLS:
                run_state.passive_tool_rejections += 1
                raise ModelRetry(
                    "Too many consecutive memory/status/observation calls "
                    "without changing RoboTwin. The latest state is already "
                    "available; call a physical RoboTwin motion tool now."
                )
            run_state.passive_tool_calls += 1
        else:
            run_state.passive_tool_calls = 0
        run_state.environment_tool_this_turn = name
        result = toolkit.execute_tool(name, kwargs)
        if name in _MOTION_TOOLS and isinstance(result.result, dict):
            error = result.result.get("error")
            if error:
                # Do not append a large traceback from a malformed primitive
                # call to the model transcript. It does not change the live
                # environment and has no value as history; make the retry
                # short and force a valid next control decision instead.
                raise ModelRetry(
                    f"{name} was rejected: {error}. The robot did not move. "
                    "Do not repeat the malformed primitive; use valid "
                    "arguments or call robotwin_vla_chunk for the next "
                    "physical action."
                )
        if result.is_finish and run_state.finish_result is None:
            # Only the tool result may terminate the loop. A rejected
            # finish returns error without ``_finish`` and is ignored here.
            run_state.finish_result = dict(result.result)
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images and not no_images:
            # ``return_value`` remains the protocol-correct tool result.
            # ``content`` creates a following user-visible multimodal state
            # message for OpenAI-compatible models. Keep the text state and
            # all three RoboTwin camera images in the same bundle so a
            # backend-side history projection can retain one complete live
            # observation without separating cameras from their state.
            return ToolReturn(
                return_value=text,
                content=[
                    f"Latest live RoboTwin result from {name}:\n{text}",
                    *images,
                ],
            )
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
