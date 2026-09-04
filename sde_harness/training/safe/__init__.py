"""Offline SAFE training pipeline for versioned RobotWin HDF5 episodes.

The VLA stays frozen.  The pipeline is intentionally split into three CLI
stages so labels can evolve independently of expensive prefix extraction:

``cache_prefix`` -> ``build_manifest`` -> ``train``.
"""

from .hdf5 import build_policy_batch, decode_text

__all__ = ["build_policy_batch", "decode_text"]
