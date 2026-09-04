"""Build an immutable, episode-split SAFE manifest from prefix sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import h5py

from .hdf5 import terminal_risk_label


def _episodes(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.hdf5")):
        if not path.name.endswith(".prefix.hdf5"):
            yield path


def _prefix_path(source: Path, source_root: Path, prefix_root: Path) -> Path:
    return prefix_root / source.relative_to(source_root).with_suffix(".prefix.hdf5")


def _split(episode_id: str, *, seed: int, val_ratio: float, test_ratio: float) -> str:
    value = int.from_bytes(
        hashlib.sha256(f"{seed}:{episode_id}".encode("utf-8")).digest()[:8], "big"
    ) / 2**64
    if value < test_ratio:
        return "test"
    if value < test_ratio + val_ratio:
        return "val"
    return "train"


def build_manifest(
    *,
    hdf5_root: Path,
    prefix_root: Path,
    output: Path,
    split_seed: int,
    val_ratio: float,
    test_ratio: float,
) -> dict:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio + test_ratio must be in [0, 1)")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    split_counts: Counter[str] = Counter()
    label_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    samples = 0
    episodes = 0

    with temporary.open("w", encoding="utf-8") as handle:
        for source in _episodes(hdf5_root):
            prefix_path = _prefix_path(source, hdf5_root, prefix_root)
            if not prefix_path.is_file():
                continue
            episode_id = str(source.relative_to(hdf5_root).with_suffix(""))
            split = _split(
                episode_id,
                seed=split_seed,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
            )
            with h5py.File(source, "r") as episode, h5py.File(prefix_path, "r") as cache:
                if cache.attrs.get("schema_name", "") != "safe_prefix_cache":
                    raise ValueError(f"unsupported prefix cache: {prefix_path}")
                steps = cache["prefix/step30"]
                timestamps = cache["prefix/timestamp"]
                for cache_index, step in enumerate(steps):
                    step30 = int(step)
                    label = terminal_risk_label(episode, step30)
                    if label is None:
                        continue
                    risk, label_source = label
                    record = {
                        "schema_version": 1,
                        "sample_id": f"{episode_id}:{step30}",
                        "episode_id": episode_id,
                        "source_hdf5": str(source.resolve()),
                        "prefix_hdf5": str(prefix_path.resolve()),
                        "prefix_index": cache_index,
                        "step30": step30,
                        "timestamp": float(timestamps[cache_index]),
                        "risk_label": risk,
                        "label_source": label_source,
                        "split": split,
                    }
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                    samples += 1
                    split_counts[split] += 1
                    label_counts[risk] += 1
                    source_counts[label_source] += 1
            episodes += 1
    os.replace(temporary, output)
    metadata = {
        "schema_version": 1,
        "manifest": str(output.resolve()),
        "hdf5_root": str(hdf5_root.resolve()),
        "prefix_root": str(prefix_root.resolve()),
        "split_seed": split_seed,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "episodes_with_prefix": episodes,
        "samples": samples,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": {str(key): value for key, value in sorted(label_counts.items())},
        "label_sources": dict(sorted(source_counts.items())),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", required=True)
    parser.add_argument("--prefix-dir", required=True)
    parser.add_argument("--output", required=True, help="Output JSONL manifest")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = build_manifest(
        hdf5_root=Path(args.hdf5_dir).expanduser().resolve(),
        prefix_root=Path(args.prefix_dir).expanduser().resolve(),
        output=Path(args.output).expanduser().resolve(),
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
