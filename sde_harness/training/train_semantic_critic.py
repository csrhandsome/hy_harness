"""Offline training entry point for the frozen slow Semantic Critic."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Semantic-view dataset manifest")
    parser.add_argument("--output", required=True, help="Frozen checkpoint or adapter directory")
    parser.add_argument(
        "--architecture",
        default="vlm_judge",
        help="Implementation family, e.g. frozen_vlm or lora_vlm",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Semantic Critic training is scaffolded; it must preserve the three-way "
        "semantic-ok/drift/abstain contract before exporting to "
        f"{args.output!r}."
    )


if __name__ == "__main__":
    main()

