"""Static environment-extension descriptor for runner hooks."""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from hy_harness.dashboard.state import State
    from hy_harness.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """Derived per-run identifiers produced by :attr:`EnvSpec.parse_config`."""

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    dashboard_state: "State | None"
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """Environment-level (non-tool) extension points for HyHarness."""

    name: str
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]
    parse_config: Callable[[argparse.Namespace], RunConfig]
    init_runtime: Callable[
        [argparse.Namespace, Path],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]

