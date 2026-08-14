"""Official pydantic-ai-harness compaction strategies for the robot loop.

These capabilities persist paired tool-call / tool-return edits. That is the
property the previous placeholder-only ToolReturn stubbing lacked, and the
property local Hy-Embodied vLLM needs to keep OpenAI tool history well-formed.
"""

from __future__ import annotations

from pydantic_ai.messages import ToolCallPart
from pydantic_ai_harness.compaction import (
    ClampOversizedMessages,
    ClearToolResults,
    DeduplicateFileReads,
    SlidingWindowCompaction,
    TieredCompaction,
)

#: Keep the latest observation/action pair plus one more real tool pair.
KEEP_TOOL_PAIRS = 2

#: First user turn is preserved separately; this is the recent tail.
KEEP_MESSAGES = 8

#: Local vLLM's tool-history parser fails well before a 128k window.
TARGET_TOKENS = 24_000

#: Clamp runaway write_text_file / execute_ee argument copies in history.
MAX_PART_CHARS = 4_000

_FILE_TOOLS = frozenset({"read_text_file", "write_text_file", "list_dir"})


def file_read_key(call: ToolCallPart) -> str | None:
    """Identify scoped file reads so superseded copies can be blanked."""
    if call.tool_name not in _FILE_TOOLS:
        return None
    path = call.args_as_dict().get("path")
    return f"{call.tool_name}:{path}" if path else call.tool_name


def clamp_oversized_messages() -> ClampOversizedMessages:
    """Always-on clamp for a single oversized response or tool-call payload."""
    return ClampOversizedMessages(
        max_part_chars=MAX_PART_CHARS,
        keep_head_chars=800,
        keep_tail_chars=400,
        clamp_tool_call_args=True,
    )


def deduplicate_file_reads() -> DeduplicateFileReads:
    """Blank older reads of the same memory/reference path."""
    return DeduplicateFileReads(file_key=file_read_key)


def robot_history_compaction() -> TieredCompaction:
    """Cheap-to-cheaper paired history compression for the robot controller.

    ``ClearToolResults(..., clear_tool_inputs=True)`` blanks old tool results
    *and* the matching call arguments, so a long ``execute_ee`` float array
    cannot linger as a truncated JSON string. ``SlidingWindowCompaction``
    then drops older whole messages while keeping tool pairs intact.
    """
    return TieredCompaction(
        tiers=[
            ClearToolResults(
                max_tokens=1,
                keep_pairs=KEEP_TOOL_PAIRS,
                clear_tool_inputs=True,
            ),
            SlidingWindowCompaction(
                max_messages=1,
                keep_messages=KEEP_MESSAGES,
                preserve_first_user_message=True,
            ),
        ],
        target_tokens=TARGET_TOKENS,
    )


def retry_history_compaction() -> TieredCompaction:
    """Force every cheap tier after a recoverable 400 so the next request is short."""
    return TieredCompaction(
        tiers=[
            clamp_oversized_messages(),
            ClearToolResults(
                max_tokens=1,
                keep_pairs=1,
                clear_tool_inputs=True,
            ),
            SlidingWindowCompaction(
                max_messages=1,
                keep_messages=6,
                preserve_first_user_message=True,
            ),
        ],
        target_tokens=1,
    )
