"""Cache frozen Hy-VLA prefix representations from RobotWin-compatible HDF5.

The cache is a sidecar rather than a mutation of source HDF5.  Its identity is
bound to the frozen checkpoint and preprocessing code, while labels such as
progress or safety events may be updated later without recalculating VLA
features.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from .hdf5 import build_policy_batch


CACHE_SCHEMA_NAME = "safe_prefix_cache"
CACHE_SCHEMA_VERSION = 1


def _episodes(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.hdf5")):
        if not path.name.endswith(".prefix.hdf5"):
            yield path


def _cache_path(source: Path, source_root: Path, cache_root: Path) -> Path:
    relative = source.relative_to(source_root)
    return cache_root / relative.with_suffix(".prefix.hdf5")


class PrefixCacheWriter:
    """Incrementally write bfloat16 prefix tensors as portable uint16 bits."""

    def __init__(
        self,
        output: Path,
        *,
        source: Path,
        checkpoint: str,
        stride: int,
    ) -> None:
        self.output = output
        self.temporary = output.with_suffix(output.suffix + ".tmp")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if self.temporary.exists():
            self.temporary.unlink()
        self.file = h5py.File(self.temporary, "w")
        self.file.attrs["schema_name"] = CACHE_SCHEMA_NAME
        self.file.attrs["schema_version"] = CACHE_SCHEMA_VERSION
        self.file.attrs["source_hdf5"] = str(source.resolve())
        self.file.attrs["checkpoint"] = checkpoint
        self.file.attrs["sample_stride_30hz"] = int(stride)
        self.file.attrs["storage_dtype"] = "bfloat16_bits"
        prefix = self.file.create_group("prefix")
        prefix.create_dataset("step30", shape=(0,), maxshape=(None,), dtype=np.int32)
        prefix.create_dataset("timestamp", shape=(0,), maxshape=(None,), dtype=np.float64)
        self._value: h5py.Dataset | None = None
        self._mask: h5py.Dataset | None = None

    def append(self, *, step30: int, timestamp: float, value, pad_mask) -> None:
        """Append one ``(L,D)`` bfloat16 prefix and its ``(L,)`` valid mask."""
        import torch

        tensor = value.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        mask = pad_mask.detach().to(device="cpu", dtype=torch.bool).contiguous()
        if tensor.ndim != 2 or mask.ndim != 1 or tensor.shape[0] != mask.shape[0]:
            raise ValueError(
                f"prefix/value+mask must be (L,D)+(L,), got {tuple(tensor.shape)} and {tuple(mask.shape)}"
            )
        bits = tensor.view(torch.uint16).numpy()
        mask_np = mask.numpy()
        if self._value is None:
            prefix = self.file["prefix"]
            self._value = prefix.create_dataset(
                "value",
                shape=(0, *bits.shape),
                maxshape=(None, *bits.shape),
                chunks=(1, *bits.shape),
                dtype=np.uint16,
            )
            self._mask = prefix.create_dataset(
                "pad_mask",
                shape=(0, *mask_np.shape),
                maxshape=(None, *mask_np.shape),
                chunks=(1, *mask_np.shape),
                dtype=np.bool_,
            )
        assert self._value is not None and self._mask is not None
        if tuple(self._value.shape[1:]) != tuple(bits.shape):
            raise ValueError("prefix shape changed within one episode")
        row = self._value.shape[0]
        self._value.resize((row + 1, *self._value.shape[1:]))
        self._mask.resize((row + 1, *self._mask.shape[1:]))
        self._value[row] = bits
        self._mask[row] = mask_np
        for name, payload in (("step30", np.int32(step30)), ("timestamp", np.float64(timestamp))):
            dataset = self.file[f"prefix/{name}"]
            dataset.resize((row + 1,))
            dataset[row] = payload

    def close(self) -> Path:
        if self._value is None:
            self.file.close()
            self.temporary.unlink(missing_ok=True)
            raise ValueError("episode has no cacheable prefix samples")
        self.file.flush()
        self.file.close()
        os.replace(self.temporary, self.output)
        return self.output


def cache_episode(
    source: Path,
    output: Path,
    *,
    policy,
    checkpoint: str,
    stride: int,
) -> Path:
    if stride < 1:
        raise ValueError("stride must be >= 1")
    policy.reset()
    writer = PrefixCacheWriter(output, source=source, checkpoint=checkpoint, stride=stride)
    try:
        with h5py.File(source, "r") as h5_file:
            timeline = h5_file["aligned_timestamp30"]
            for step30, timestamp in enumerate(timeline):
                batch = build_policy_batch(h5_file, step30)
                # Every raw 30 Hz frame must enter the MEM history, even when
                # only every Nth frame is materialized as a SAFE example.
                policy.observe(batch)
                if step30 % stride:
                    continue
                prefix = policy.extract_prefix(batch, observe=False)
                writer.append(
                    step30=step30,
                    timestamp=float(timestamp),
                    value=prefix["value"],
                    pad_mask=prefix["pad_mask"],
                )
        return writer.close()
    except BaseException:
        writer.file.close()
        writer.temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5-dir", required=True, help="Root containing episode HDF5 files")
    parser.add_argument("--output", required=True, help="Sidecar prefix-cache root")
    parser.add_argument("--ckpt-path", required=True, help="Frozen Hy-VLA checkpoint")
    parser.add_argument("--norm-path", default=None, help="Optional norm_stats.pkl override")
    parser.add_argument("--vlm-model-path", default=None)
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth 30 Hz frame")
    parser.add_argument("--img-history-size", type=int, default=6)
    parser.add_argument("--img-history-interval", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    source_root = Path(args.hdf5_dir).expanduser().resolve()
    cache_root = Path(args.output).expanduser().resolve()
    if not source_root.is_dir():
        raise ValueError(f"not an HDF5 directory: {source_root}")

    # The wrapper owns the exact online preprocessing and VLM prefix forward.
    from robotwin_eval.policy_wrapper import build_policy

    policy = build_policy(
        {
            "ckpt_path": args.ckpt_path,
            "norm_path": args.norm_path,
            "vlm_model_path": args.vlm_model_path,
            "img_history_size": args.img_history_size,
            "img_history_interval": args.img_history_interval,
        }
    )
    completed: list[str] = []
    for source in _episodes(source_root):
        output = _cache_path(source, source_root, cache_root)
        if output.exists() and not args.overwrite:
            continue
        completed.append(str(cache_episode(
            source,
            output,
            policy=policy,
            checkpoint=str(args.ckpt_path),
            stride=args.stride,
        )))
    print(json.dumps({"cached": completed, "count": len(completed)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
