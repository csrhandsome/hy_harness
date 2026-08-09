"""RPC server wrapping a single-env LIBERO environment (no RLinf).

Uses the local ``third_party/LIBERO-PRO`` / ``LIBERO-plus`` / ``LIBERO`` trees
already wired by VLA-Adapter eval scripts, and exposes the same harness
protocol that ``robots.libero.env_client.LiberoEnvClient`` expects.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any

# MuJoCo env vars must be set BEFORE importing anything that touches MuJoCo.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from hy_harness.utils.config import (
    get_libero_config_dir,
    get_libero_root,
    get_libero_type,
    get_vla_adapter_root,
)
from hy_harness.utils.logging import get_logger
from hy_harness.utils.rpc import RpcFacade

logger = get_logger("env_server")


def _configure_libero_imports(libero_type: str) -> None:
    """Put the selected LIBERO package on ``sys.path`` and set config path."""
    adapter = get_vla_adapter_root()
    if str(adapter) not in sys.path:
        sys.path.insert(0, str(adapter))

    libero_root = get_libero_root(libero_type)
    if not libero_root.is_dir():
        raise FileNotFoundError(
            f"LIBERO root missing for type={libero_type!r}: {libero_root}"
        )
    root_s = str(libero_root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)

    cfg_dir = get_libero_config_dir(libero_type)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg_dir)


import numpy as np  # noqa: E402


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x,y,z,w) to axis-angle (same as VLA-Adapter eval)."""
    q = np.asarray(quat, dtype=np.float64).copy()
    if q[3] > 1.0:
        q[3] = 1.0
    elif q[3] < -1.0:
        q[3] = -1.0
    den = math.sqrt(1.0 - q[3] * q[3])
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def _flip180(img: np.ndarray) -> np.ndarray:
    """Rotate 180° to match VLA-Adapter / OpenVLA train preprocessing."""
    return np.ascontiguousarray(np.asarray(img)[::-1, ::-1])


def _to_numpy_tree(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return x


class NativeLiberoEnv:
    """Single-env LIBERO wrapper exposing the HyHarness harness obs protocol."""

    def __init__(
        self,
        *,
        suite_name: str,
        task_id: int,
        seed: int,
        max_episode_steps: int = 10000,
        resolution: int = 256,
    ):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        self.suite_name = suite_name
        self.task_id = int(task_id)
        self.seed = int(seed)
        self.max_episode_steps = int(max_episode_steps)
        self.resolution = int(resolution)

        bench = benchmark.get_benchmark_dict()[suite_name]()
        task = bench.get_task(self.task_id)
        self.task_descriptions = task.language
        init_states = bench.get_task_init_states(self.task_id)
        self._init_state = init_states[self.seed % len(init_states)]

        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        self._env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=resolution,
            camera_widths=resolution,
            camera_depths=True,
            horizon=max_episode_steps,
        )
        self._env.seed(self.seed)

        self._raw_obs: dict[str, Any] | None = None
        self._cached_full_image: np.ndarray | None = None
        self._steps = 0
        self._done = False

    # ---- obs helpers ----

    def _build_policy_obs(self, raw: dict[str, Any]) -> dict[str, Any]:
        main = _flip180(raw["agentview_image"])
        wrist = _flip180(raw["robot0_eye_in_hand_image"])
        state = np.concatenate(
            (
                np.asarray(raw["robot0_eef_pos"], dtype=np.float32),
                quat2axisangle(raw["robot0_eef_quat"]),
                np.asarray(raw["robot0_gripper_qpos"], dtype=np.float32),
            )
        ).astype(np.float32)
        self._cached_full_image = main
        return {
            "main_images": main,
            "wrist_images": wrist,
            "extra_view_images": None,
            "states": state,
            "task_descriptions": self.task_descriptions,
        }

    def _success(self) -> bool:
        try:
            return bool(self._env.check_success())
        except Exception:
            return False

    # ---- gym-like surface (single-env, no leading batch dim) ----

    def reset(self):
        self._env.reset()
        self._raw_obs = self._env.set_init_state(self._init_state)
        self._steps = 0
        self._done = False
        obs = self._build_policy_obs(self._raw_obs)
        return obs, {"task_description": self.task_descriptions}

    def step(self, action):
        assert not self._done, "step called after episode done"
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        raw, reward, done, info = self._env.step(action.tolist())
        self._raw_obs = raw
        self._steps += 1
        truncated = self._steps >= self.max_episode_steps
        # Native LIBERO can report a horizon stop through done. Do not turn
        # that time-limit event into task success.
        terminated = self._success() or (bool(done) and not truncated)
        terminated = bool(terminated)
        truncated = bool(truncated)
        self._done = terminated or truncated
        obs = self._build_policy_obs(raw)
        return obs, float(reward), terminated, truncated, _to_numpy_tree(info)

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                f"actions must be [chunk, action_dim]; got shape {actions.shape}"
            )
        obs_list = []
        rewards = []
        terms = []
        truncs = []
        last_info: dict[str, Any] = {}
        last_term = False
        last_trunc = False
        for a in actions:
            obs, rew, term, trunc, info = self.step(a)
            obs_list.append(obs)
            rewards.append(rew)
            terms.append(term)
            truncs.append(trunc)
            last_info = info
            last_term = bool(term)
            last_trunc = bool(trunc)
            if self._done:
                # Preserve the reason the episode ended when padding.
                while len(terms) < len(actions):
                    rewards.append(0.0)
                    terms.append(last_term)
                    truncs.append(last_trunc)
                    if return_all_frames:
                        obs_list.append(obs)
                break
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            np.asarray(rewards, dtype=np.float32),
            np.asarray(terms, dtype=bool),
            np.asarray(truncs, dtype=bool),
            last_info,
        )

    def raw_obs(self) -> dict:
        if self._raw_obs is None:
            raise RuntimeError("raw_obs called before reset")
        return _to_numpy_tree(self._raw_obs)

    def get_task_language(self) -> str:
        return self.task_descriptions

    def cached_image(self) -> np.ndarray | None:
        return self._cached_full_image

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        sim = self._env.sim
        # robosuite OffScreenRenderEnv exposes sim via ControlEnv
        rgb = sim.render(camera_name=camera_name, height=height, width=width)
        # MuJoCo returns flipped vertically relative to robosuite obs buffers;
        # match robosuite observation orientation (no extra 180 here — callers
        # that need Pi0/VLA frame flip themselves, as in tools.dump_state).
        rgb = np.asarray(rgb)[::-1]
        if not depth:
            return rgb.astype(np.uint8)
        depth_map = sim.render(
            camera_name=camera_name, height=height, width=width, depth=True
        )
        if isinstance(depth_map, tuple):
            # Some backends return (rgb, depth)
            depth_map = depth_map[1]
        depth_map = np.asarray(depth_map)[::-1]
        return rgb.astype(np.uint8), depth_map.astype(np.float32)

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        try:
            from robosuite.utils.camera_utils import (
                get_camera_extrinsic_matrix,
                get_camera_intrinsic_matrix,
            )
        except Exception as e:
            logger.warning("camera_utils unavailable: %s", e)
            return None

        sim = self._env.sim
        K = get_camera_intrinsic_matrix(sim, camera_name, height, width)
        # cam2world
        E = get_camera_extrinsic_matrix(sim, camera_name)
        extent = float(sim.model.stat.extent)
        near = float(sim.model.vis.map.znear) * extent
        far = float(sim.model.vis.map.zfar) * extent
        return {
            "camera_name": camera_name,
            "height": int(height),
            "width": int(width),
            "intrinsic_K": np.asarray(K, dtype=np.float64).tolist(),
            "extrinsic_cam2world": np.asarray(E, dtype=np.float64).tolist(),
            "depth_near": near,
            "depth_far": far,
        }

    def close(self) -> None:
        try:
            self._env.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Facade
# ---------------------------------------------------------------------------


class LiberoEnvFacade(RpcFacade):
    """Implements :class:`robots.libero.env_client.LiberoEnvClient`."""

    def __init__(self, env: NativeLiberoEnv, *, meta: dict):
        super().__init__()
        self._env = env
        self._done = False
        self._meta = dict(meta)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if method.startswith("env."):
            attr = method[len("env.") :]
            try:
                return getattr(self, attr)(*args, **kwargs)
            except Exception as e:
                logger.warning("run method %s failed: %s", method, e)
                raise e
        raise ValueError(f"unknown RPC method: {method!r}")

    def _record_done(self, *signals: Any) -> None:
        for s in signals:
            if np.asarray(s).any():
                self._done = True
                return

    def reset(self):
        obs, info = self._env.reset()
        self._done = False
        return obs, info

    def step(self, action):
        assert not self._done, "step called after episode done"
        obs, rew, term, trunc, info = self._env.step(action)
        self._record_done(term, trunc)
        return obs, rew, term, trunc, info

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        assert not self._done, "chunk_step called after episode done"
        ret = self._env.chunk_step(actions, return_all_frames=return_all_frames)
        _, _, term, trunc, _ = ret
        self._record_done(term, trunc)
        return ret

    def raw_obs(self) -> dict:
        return self._env.raw_obs()

    def get_env_meta(self) -> dict:
        return dict(self._meta)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        return self._env.render_camera(
            camera_name=camera_name,
            height=height,
            width=width,
            depth=depth,
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        return self._env.get_camera_meta(
            camera_name=camera_name, height=height, width=width
        )

    def get_task_language(self) -> str | None:
        return self._env.get_task_language()

    def cached_image(self) -> np.ndarray | None:
        return self._env.cached_image()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument(
        "--libero-type",
        default=None,
        choices=["standard", "pro", "plus"],
        help="LIBERO variant (defaults to LIBERO_TYPE env / pro)",
    )
    p.add_argument(
        "--cuda-device", type=int, default=None,
        help="GPU device to pin Native LIBERO EGL rendering to.",
    )
    args = p.parse_args()

    if args.cuda_device is not None:
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + --cuda-device=%s",
                prev, args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        from hy_harness.utils.egl import configure_egl_device
        configure_egl_device(args.cuda_device)
        import torch
        torch.cuda.set_device(args.cuda_device)

    libero_type = args.libero_type or get_libero_type()
    _configure_libero_imports(libero_type)
    logger.info("LIBERO_TYPE=%s root=%s", libero_type, get_libero_root(libero_type))

    raw_env = NativeLiberoEnv(
        suite_name=args.suite,
        task_id=args.task,
        seed=args.seed,
        max_episode_steps=args.max_episode_steps,
    )
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    facade.serve(transport=args.transport, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
