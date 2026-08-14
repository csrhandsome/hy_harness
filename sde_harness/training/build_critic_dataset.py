"""Build versioned Fast/Semantic Critic datasets from rollout checkpoints.

The eventual implementation will read Critic Hook event logs, construct a
single immutable manifest, and materialize level-specific labels without
leaking a post-recovery terminal outcome into the pre-recovery example.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, help="Critic event JSONL or directory")
    parser.add_argument("--output", required=True, help="Output dataset-manifest directory")
    parser.add_argument(
        "--target",
        choices=("fast", "semantic", "both"),
        default="both",
        help="Which level-specific views to materialize",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0, help="Episode-level split seed"
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Critic dataset construction is scaffolded; event-log and label contracts "
        f"must be defined before reading {args.events!r}."
    )


if __name__ == "__main__":
    main()

