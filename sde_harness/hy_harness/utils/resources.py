"""Sync the env's text resources/ payload from its HuggingFace dataset.

``RLinf/HyHarness-memory`` is a small **text** dataset (MEMORY.md notes + seed-0
reference recipes), not model weights. Total size is on the order of hundreds
of KB.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

from hy_harness.utils.config import get_resources_dir
from hy_harness.utils.logging import get_logger

RESOURCES_HF_REPO = os.environ.get(
    "HYHARNESS_RESOURCES_HF_REPO", "RLinf/HyHarness-memory"
)

logger = get_logger("resources")


@cache
def ensure_resources(env_name: str) -> Path:
    """Sync an env text snapshot once per worker process.

    Set ``HF_HUB_OFFLINE=1`` to use the local copy only. Missing memory is
    non-fatal — the agent can still run without it.
    """
    resources_dir = get_resources_dir(env_name)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return resources_dir

    # Prefer hf-mirror (same default as vla-scripts/download_*.sh).
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=RESOURCES_HF_REPO,
            repo_type="dataset",
            local_dir=str(resources_dir.parent),
            allow_patterns=[f"{env_name}/**"],
        )
    except Exception as exc:  # noqa: BLE001 - offline fallback must be non-fatal
        logger.warning(
            "could not sync '%s' from '%s': %s; continuing with local files under %s",
            env_name,
            RESOURCES_HF_REPO,
            exc,
            resources_dir,
        )

    return resources_dir
