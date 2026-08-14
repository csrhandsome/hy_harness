"""Evaluate Fast Critic, Semantic Critic, and their routed Critic Stack."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True)
    parser.add_argument("--fast-checkpoint", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--stack-config", required=True)
    parser.add_argument("--output", required=True, help="JSON evaluation report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Critic Stack evaluation is scaffolded; the final report must separate "
        "Fast-risk calibration, semantic drift recall/abstention, Router VLM-call "
        f"rate, and end-to-end intervention cost for {args.test_manifest!r}."
    )


if __name__ == "__main__":
    main()

