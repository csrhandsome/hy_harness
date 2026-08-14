"""Bound camera images in planner history.

Official pydantic-ai-harness compaction does not understand RoboTwin's
three-view PNG observations, so this processor stays beside the SDK
strategies and runs on every model request.
"""

from __future__ import annotations

import dataclasses

from pydantic_ai import BinaryContent
from pydantic_ai.messages import ModelMessage

#: Cap on cumulative decoded image bytes kept in the resent request history.
MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: A RoboTwin camera observation is an inseparable **three-image bundle**:
#: head, left wrist and right wrist. Never retain one camera while dropping
#: its siblings; we only bound how many *complete observation bundles* appear
#: in resend history.
IMAGES_PER_ROBOTWIN_OBSERVATION = 3

#: Always keep every image from at least this many most-recent observation
#: bundles. ``1`` means all three images from the newest observation, not one
#: image.
MIN_IMAGE_OBSERVATIONS = 1

#: Drop image bundles from older observations beyond this many complete
#: observations. The local Hy-Embodied vLLM service has a 32k context; one
#: latest three-view bundle leaves room for system instructions, schemas and
#: the latest state text without breaking camera completeness.
MAX_IMAGE_OBSERVATIONS = 1

_IMAGE_PLACEHOLDER = "[earlier camera image omitted to bound request size]"


def prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
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
                    if _is_image(item):
                        located.append((mi, pi, ii, len(item.data)))
            elif _is_image(content):
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
        if rank < MIN_IMAGE_OBSERVATIONS or (
            len(keep_messages) < MAX_IMAGE_OBSERVATIONS
            and total + nbytes <= MAX_HISTORY_IMAGE_BYTES
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


async def prune_history_images_async(
    messages: list[ModelMessage],
) -> list[ModelMessage]:
    """Async adapter so pydantic-ai runs image pruning on the event loop."""
    return prune_history_images(messages)


def _is_image(item: object) -> bool:
    return isinstance(item, BinaryContent) and str(
        getattr(item, "media_type", "")
    ).startswith("image/")


def _replace_dropped_images(
    messages: list[ModelMessage],
    *,
    drop_items_by_part: dict[tuple[int, int], set[int]],
    drop_whole_part: set[tuple[int, int]],
) -> list[ModelMessage]:
    new_messages = list(messages)
    touched = set(drop_items_by_part) | drop_whole_part
    for mi, pi in touched:
        message = new_messages[mi]
        part = message.parts[pi]
        if (mi, pi) in drop_whole_part:
            new_part = dataclasses.replace(part, content=_IMAGE_PLACEHOLDER)
        else:
            drop_items = drop_items_by_part[(mi, pi)]
            new_content = [
                _IMAGE_PLACEHOLDER if ci in drop_items else item
                for ci, item in enumerate(part.content)
            ]
            new_part = dataclasses.replace(part, content=new_content)
        new_parts = list(message.parts)
        new_parts[pi] = new_part
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)
    return new_messages
