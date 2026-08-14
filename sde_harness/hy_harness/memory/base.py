"""Planner-facing history memory built on official pydantic-ai compaction."""

from __future__ import annotations

from typing import Any

from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage
from pydantic_ai.models import Model
from pydantic_ai_harness.compaction import compact_now

from hy_harness.memory.compaction import (
    clamp_oversized_messages,
    deduplicate_file_reads,
    retry_history_compaction,
    robot_history_compaction,
)
from hy_harness.memory.images import prune_history_images_async

HISTORY_COMPACT_CONTINUATION = (
    "Conversation history was compacted after a transport/schema error. "
    "The live RoboTwin episode is still running (success=false, done=false). "
    "Do not terminate or write files. Call robotwin_status or robotwin_observe, "
    "then continue with one real RoboTwin motion tool."
)


def is_recoverable_history_error(error: object) -> bool:
    """True when a model-request 400 looks like truncated or invalid tool JSON."""
    if error is None:
        return False
    if isinstance(error, ModelHTTPError):
        if _is_image_rejection(error) or error.status_code != 400:
            return False
        text = f"{error} {error.body}".lower()
    else:
        text = str(error).lower()
        if "image" in text and ("reject" in text or "vision" in text):
            return False
        if "modelhttperror" not in text and "status_code: 400" not in text:
            return False
    return any(
        needle in text
        for needle in (
            "invalid json",
            "eof while parsing",
            "json",
            "tool",
            "function",
        )
    )


def _is_image_rejection(error: ModelHTTPError) -> bool:
    if not 400 <= error.status_code < 500:
        return False
    return "image" in str(error).lower()


class HistoryMemory:
    """Official compaction capabilities plus RoboTwin image bounding.

    Tool-call / tool-return pairing is handled entirely by
    ``pydantic-ai-harness`` strategies. Image pruning is the only
    robot-specific processor, because the SDK does not know about the
    three-view camera payload.
    """

    def __init__(self) -> None:
        self._clamp = clamp_oversized_messages()
        self._dedupe = deduplicate_file_reads()
        self._tiered = robot_history_compaction()
        self._retry = retry_history_compaction()

    def capabilities(self) -> list[Any]:
        """Capabilities attached to each pydantic-ai Agent run."""
        # pydantic-ai dispatches synchronous ProcessHistory processors through
        # an AnyIO worker thread. In the RoboTwin Python 3.10 runtime that path
        # can stall an Agent run before its first model request. Keep image
        # pruning synchronous and deterministic, but expose it through an async
        # wrapper so pydantic-ai runs it on the event loop.
        return [
            self._clamp,
            self._dedupe,
            self._tiered,
            ProcessHistory(processor=prune_history_images_async),
        ]

    async def compact_for_retry(
        self,
        messages: list[ModelMessage],
        *,
        model: Model,
    ) -> list[ModelMessage]:
        """Aggressively compact history after a recoverable 400."""
        compacted = await compact_now(self._retry, messages, model=model)
        return await prune_history_images_async(compacted)


def build_history_memory() -> HistoryMemory:
    """Construct the default robot-controller history memory."""
    return HistoryMemory()
