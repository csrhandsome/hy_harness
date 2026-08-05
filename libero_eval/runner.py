"""Shared evaluator for standard LIBERO, LIBERO-plus, and LIBERO-Pro."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import yaml

from libero_eval.remote_policy import RemoteLiberoPolicy

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = Path(__file__).resolve().parent
BASE_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90")
MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
logger = logging.getLogger("libero_eval")


def base_suite(name: str) -> str:
    for base in BASE_SUITES:
        if name == base or name.startswith(base + "_"):
            return base
    return name


def _variant_root(variant: str) -> Path:
    return {
        "standard": REPO_ROOT / "third_party" / "LIBERO",
        "plus": REPO_ROOT / "third_party" / "LIBERO-plus",
        "pro": REPO_ROOT / "third_party" / "LIBERO-PRO",
    }[variant]


def configure_libero(variant: str):
    """Select one vendored LIBERO tree before importing ``libero``."""
    root = _variant_root(variant)
    package_root = root / "libero" / "libero"
    if not package_root.is_dir():
        raise FileNotFoundError(f"{variant} LIBERO package missing: {package_root}")
    config_name = ".libero_config" if variant == "standard" else f".libero_{variant}_config"
    config_dir = EVAL_ROOT / "configs" / config_name
    config_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "benchmark_root": str(package_root),
        "bddl_files": str(package_root / "bddl_files"),
        "init_states": str(package_root / "init_files"),
        "datasets": str(root / "libero" / "datasets"),
        "assets": str(package_root / "assets"),
    }
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True), encoding="utf-8"
    )
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    sys.path.insert(0, str(root))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    return benchmark, OffScreenRenderEnv, get_libero_path, package_root


def resolve_pro_suite(requested: str, benchmark_dict: dict[str, Any], config_path: Path) -> str:
    if requested in benchmark_dict and requested != base_suite(requested):
        return requested
    base = base_suite(requested)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    enabled = [
        key
        for key in ("use_environment", "use_swap", "use_object", "use_language", "use_task")
        if config.get(key)
    ]
    if len(enabled) != 1:
        raise ValueError(f"{config_path}: enable exactly one Pro perturbation; enabled={enabled}")
    suffix = config.get("perturbation_mapping", {}).get(enabled[0], "temp")
    target = f"{base}_{suffix}"
    if target not in benchmark_dict:
        raise FileNotFoundError(
            f"prebuilt LIBERO-Pro suite {target!r} is missing. Run "
            "bash libero_eval/download.sh pro first."
        )
    return target


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1.0 - float(quat[3] ** 2)))
    if np.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * np.arccos(quat[3]) / denominator).astype(np.float32)


def policy_observation(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    main = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
        )
    )
    return main, wrist, state.astype(np.float32)


def make_env(task, OffScreenRenderEnv, get_libero_path, resolution: int):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl,
        camera_heights=resolution,
        camera_widths=resolution,
    )
    env.seed(0)
    return env


def save_video(frames: list[np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=30) as writer:
        for frame in frames:
            writer.append_data(frame)


def task_ids_for_plus(args, suite, package_root: Path) -> list[int]:
    classification_path = package_root / "benchmark" / "task_classification.json"
    classification: dict[str, dict[str, Any]] = {}
    if classification_path.is_file():
        data = json.loads(classification_path.read_text(encoding="utf-8"))
        classification = {item["name"]: item for item in data.get(args.task_suite_name, [])}
    selected: list[int] = []
    for task_id in range(args.start_task_id, suite.n_tasks):
        task = suite.get_task(task_id)
        category = classification.get(task.name, {}).get("category", "Unknown")
        if args.perturbation_category and category != args.perturbation_category:
            continue
        selected.append(task_id)
        if args.max_tasks and len(selected) >= args.max_tasks:
            break
    return selected


def run_episode(args, env, initial_state, instruction: str, policy: RemoteLiberoPolicy):
    env.reset()
    obs = env.set_init_state(initial_state) if initial_state is not None else env.get_observation()
    policy.reset()
    queue: deque[np.ndarray] = deque()
    frames: list[np.ndarray] = []
    max_steps = MAX_STEPS[base_suite(args.task_suite_name)]
    for step in range(max_steps + args.num_steps_wait):
        if step < args.num_steps_wait:
            obs, _, done, _ = env.step([0, 0, 0, 0, 0, 0, -1])
            if done:
                return True, frames
            continue
        main, wrist, state = policy_observation(obs)
        frames.append(main)
        if not queue:
            queue.extend(policy.predict(instruction, main, wrist, state))
        obs, _, done, _ = env.step(np.asarray(queue.popleft(), dtype=np.float32).tolist())
        if done:
            return True, frames
    return False, frames


def build_parser(variant: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Evaluate Hy-VLA on LIBERO ({variant}).")
    parser.add_argument("--policy-endpoint", default=os.environ.get("POLICY_ENDPOINT", "http://127.0.0.1:8001"))
    parser.add_argument("--policy-timeout-s", type=float, default=120.0)
    parser.add_argument("--checkpoint", "--pretrained_checkpoint", dest="checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--task-suite-name", "--task_suite_name", default="libero_spatial")
    parser.add_argument("--num-trials-per-task", "--num_trials_per_task", type=int, default=1 if variant == "plus" else 20)
    parser.add_argument("--num-steps-wait", "--num_steps_wait", type=int, default=10)
    parser.add_argument("--env-img-res", "--env_img_res", type=int, default=256)
    parser.add_argument("--max-tasks", "--max_tasks", type=int, default=0)
    parser.add_argument("--start-task-id", "--start_task_id", type=int, default=0)
    parser.add_argument("--perturbation-category", "--perturbation_category", default="")
    parser.add_argument("--evaluation-config-path", "--evaluation_config_path", default=str(EVAL_ROOT / "configs" / "libero_pro_evaluation_config.yaml"))
    parser.add_argument("--local-log-dir", default=str(REPO_ROOT / "eval_logs" / "libero"))
    parser.add_argument("--save-videos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--seed", type=int, default=7)
    return parser


def evaluate(variant: str, argv: list[str] | None = None) -> float:
    args, unknown = build_parser(variant).parse_known_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if unknown:
        logger.warning("ignoring legacy VLA-Adapter arguments: %s", " ".join(unknown))
    random.seed(args.seed)
    np.random.seed(args.seed)

    benchmark, OffScreenRenderEnv, get_libero_path, package_root = configure_libero(variant)
    benchmark_dict = benchmark.get_benchmark_dict()
    if variant == "pro":
        args.task_suite_name = resolve_pro_suite(
            args.task_suite_name, benchmark_dict, Path(args.evaluation_config_path)
        )
    if args.task_suite_name not in benchmark_dict:
        raise KeyError(f"suite {args.task_suite_name!r} is unavailable")

    policy = RemoteLiberoPolicy(args.policy_endpoint, timeout_s=args.policy_timeout_s)
    suite = benchmark_dict[args.task_suite_name]()
    if variant == "plus":
        task_ids = task_ids_for_plus(args, suite, package_root)
    else:
        task_ids = list(range(args.start_task_id, suite.n_tasks))
        if args.max_tasks:
            task_ids = task_ids[: args.max_tasks]
    if not task_ids:
        raise RuntimeError("no tasks selected")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(args.local_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{variant}-{args.task_suite_name}-{timestamp}.log"
    video_root = REPO_ROOT / "rollouts" / f"hy-vla-libero-{variant}"
    total = successes = 0

    with log_path.open("w", encoding="utf-8") as log:
        def emit(message: str) -> None:
            logger.info(message)
            log.write(message + "\n")
            log.flush()

        emit(f"variant={variant} suite={args.task_suite_name} tasks={len(task_ids)}")
        emit(f"policy_endpoint={args.policy_endpoint}")
        for task_id in task_ids:
            task = suite.get_task(task_id)
            env = make_env(task, OffScreenRenderEnv, get_libero_path, args.env_img_res)
            initial_states = suite.get_task_init_states(task_id)
            task_successes = 0
            try:
                for episode in range(args.num_trials_per_task):
                    initial = initial_states[episode % len(initial_states)]
                    success, frames = run_episode(args, env, initial, task.language, policy)
                    total += 1
                    task_successes += int(success)
                    successes += int(success)
                    emit(
                        f"task={task_id} episode={episode} success={success} "
                        f"overall={successes}/{total} ({successes / total:.2%})"
                    )
                    if args.save_videos:
                        safe_task = task.language.lower().replace(" ", "_")[:60]
                        save_video(
                            frames,
                            video_root / f"{timestamp}-t{task_id}-e{episode}-{success}-{safe_task}.mp4",
                        )
            finally:
                env.close()
            emit(
                f"task={task_id} rate={task_successes / args.num_trials_per_task:.2%} "
                f"description={task.language}"
            )
        rate = successes / total if total else 0.0
        emit(f"final_success_rate={rate:.6f} successes={successes} episodes={total}")
        emit(f"log={log_path}")
    policy.close()
    return rate


def main(variant: str | None = None, argv: list[str] | None = None) -> None:
    """Run an evaluator selected by the shell launcher or command line."""
    if variant is None:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--variant", choices=("standard", "plus", "pro"), default="standard")
        selection, argv = parser.parse_known_args(argv)
        variant = selection.variant
    evaluate(variant, argv)


__all__ = ["evaluate", "main"]


if __name__ == "__main__":
    main()
