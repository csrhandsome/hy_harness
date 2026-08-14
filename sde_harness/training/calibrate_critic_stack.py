"""Calibrate frozen Fast/Semantic Critics and deterministic routing knobs."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--fast-checkpoint", required=True)
    parser.add_argument("--semantic-checkpoint", required=True)
    parser.add_argument("--output", required=True, help="Versioned stack-config path")
    parser.add_argument(
        "--target-semantic-miss-rate",
        type=float,
        required=True,
        help="Maximum allowed semantic-drift miss rate during calibration",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Critic Stack calibration is scaffolded; it will write temperatures, "
        "thresholds, evidence gates, and VLM-call budgets—not model weights—to "
        f"{args.output!r}."
    )


if __name__ == "__main__":
    main()

