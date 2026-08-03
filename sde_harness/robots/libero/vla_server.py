"""RPC server exposing a LIBERO-finetuned Hy-VLA policy to the Harness."""

from __future__ import annotations

import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from rpent.utils.config import get_checkpoint_path, get_hy_vla_root
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

root = str(get_hy_vla_root())
if root not in sys.path:
    sys.path.insert(0, root)

import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402

from libero_eval.policy_wrapper import HyVLALiberoPolicy  # noqa: E402

logger = get_logger("vla_server")


def _decode_image(block: dict[str, Any]) -> np.ndarray:
    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block is missing base64 data")
    image = np.asarray(imageio.imread(io.BytesIO(base64.b64decode(data))))
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {image.shape}")
    return image.astype(np.uint8, copy=False)


class VLAFacade(RpcFacade):
    def __init__(self, policy: HyVLALiberoPolicy):
        super().__init__()
        self.policy = policy

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method == "predict":
            return self.predict(*args, **kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        del mode
        if "main" not in images:
            raise ValueError("images.main is required")
        main = _decode_image(images["main"])
        wrist = _decode_image(images["wrist"]) if isinstance(images.get("wrist"), dict) else None
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim == 2 and state_array.shape[0] == 1:
            state_array = state_array[0]
        actions = self.policy.predict(instruction, main, wrist, state_array)
        batched = actions[None]
        return {"actions": batched.tolist(), "shape": list(batched.shape), "dtype": "float32"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Hy-VLA RPC server for sde_harness")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--model-path", "--checkpoint", dest="checkpoint", default=None)
    parser.add_argument("--norm-path", default=os.environ.get("NORM_PATH"))
    parser.add_argument("--vlm-model-path", default=os.environ.get("VLM_MODEL_PATH"))
    parser.add_argument("--action-dim", type=int, default=int(os.environ.get("ACTION_DIM", "7")))
    parser.add_argument("--num-open-loop-steps", type=int, default=None)
    parser.add_argument(
        "--gripper-mode",
        choices=["libero", "openvla", "raw"],
        default=os.environ.get("GRIPPER_MODE", "libero"),
    )
    parser.add_argument("--device", default=os.environ.get("DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "bfloat16"))
    # Accepted only for compatibility with the copied VLA-Adapter launcher.
    parser.add_argument("--unnorm-key", default=None)
    parser.add_argument("--use-pro-version", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    checkpoint = args.checkpoint or get_checkpoint_path()
    if not checkpoint:
        raise RuntimeError("set HY_VLA_CHECKPOINT or pass --model-path/--checkpoint")
    logger.info("loading Hy-VLA checkpoint=%s", checkpoint)
    policy = HyVLALiberoPolicy(
        checkpoint,
        norm_path=args.norm_path,
        vlm_model_path=args.vlm_model_path,
        action_dim=args.action_dim,
        num_open_loop_steps=args.num_open_loop_steps,
        gripper_mode=args.gripper_mode,
        device=args.device,
        dtype=args.dtype,
    )
    VLAFacade(policy).serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
