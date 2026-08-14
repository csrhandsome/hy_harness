"""Offline training entry point for the frozen Fast Critic (Agent-SAFE)."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Fast-view dataset manifest")
    parser.add_argument("--output", required=True, help="Frozen checkpoint output directory")
    parser.add_argument(
        "--architecture",
        default="agent_safe",
        help="Implementation family, e.g. agent_safe_mlp or agent_safe_lstm",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Fast Critic training is scaffolded; it will train only from the Fast "
        f"view in {args.manifest!r} and export a frozen artifact to {args.output!r}."
    )


if __name__ == "__main__":
    main()

