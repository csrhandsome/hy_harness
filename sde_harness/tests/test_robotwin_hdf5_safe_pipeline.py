from __future__ import annotations

import io

import h5py
import numpy as np
import torch
from PIL import Image

from robots.robotwin.recorder import RobotWinHdf5Recorder
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools
from test_robotwin_tools import FakePolicy, FakeTaskEnv, encode
from training.safe.build_manifest import build_manifest
from training.safe.train import train


def _record_episode(path, *, success: bool) -> None:
    recorder = RobotWinHdf5Recorder(
        path,
        instruction="move the red block",
        task_name="robotwin_demo",
        test_num=0,
    )
    image = np.full((8, 9, 3), 55, dtype=np.uint8)
    state = np.array(
        [[0, 0, 0, 1, 0, 0, 0, 0.04, 1, 0, 0, 1, 0, 0, 0, 0.04]],
        dtype=np.float32,
    )
    recorder.record_step(
        observation={
            "observation.state": state,
            "task": ["move the red block"],
            "raw_images.top_head": image,
            "raw_images.hand_left": image,
            "raw_images.hand_right": image,
        },
        action=state[0, :16],
        action_source="robotwin_vla_chunk",
    )
    recorder.finalize(success=success, status="success" if success else "failure")


def _write_prefix_cache(path, source, *, value: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = torch.full((4, 6), value, dtype=torch.bfloat16)
    with h5py.File(path, "w") as cache:
        cache.attrs["schema_name"] = "safe_prefix_cache"
        cache.attrs["source_hdf5"] = str(source)
        cache.create_dataset("prefix/step30", data=np.array([0], dtype=np.int32))
        cache.create_dataset("prefix/timestamp", data=np.array([1.0], dtype=np.float64))
        cache.create_dataset(
            "prefix/value", data=prefix.view(torch.uint16).numpy()[None], dtype=np.uint16
        )
        cache.create_dataset("prefix/pad_mask", data=np.ones((1, 4), dtype=bool))


def test_recorder_pairs_pre_action_observation_with_executed_action(tmp_path):
    output = tmp_path / "episode.hdf5"
    recorder = RobotWinHdf5Recorder(
        output, instruction="move the red block", task_name="demo", test_num=0
    )
    env = FakeTaskEnv()
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
        output_dir=tmp_path,
        recorder=recorder,
    )
    toolkit.execute_tool("robotwin_vla_chunk", {"execute_steps": 2})
    recorder.finalize(success=False, status="failure")

    with h5py.File(output, "r") as data:
        assert data.attrs["schema_name"] == "robotwin_harness_hdf5"
        assert data["aligned_timestamp30"].shape == (2,)
        np.testing.assert_allclose(data["observations/qpos/value"][:, 0], [0.0, 11.0])
        np.testing.assert_allclose(data["action/qpos/value"][:, 0], [11.0, 12.0])
        np.testing.assert_allclose(data["observations/qpos/value"][0, 3:7], [0.0, 0.0, 0.0, 1.0])
        assert data["observations/qpos"].attrs["quaternion_order"] == "xyzw"
        assert data["harness/action_source"][:].tolist() == [
            b"robotwin_vla_chunk",
            b"robotwin_vla_chunk",
        ]
        assert int(data["annotations/outcome/success"][()]) == 0
        assert data["annotations/task_correct/value"][:].tolist() == [0, 0]
        encoded = data["observations/images/cam_high/value"][0]
        with Image.open(io.BytesIO(encoded.tobytes())) as image:
            assert image.convert("RGB").size == (8, 8)


def test_manifest_and_safe_train_consume_recorder_outputs(tmp_path):
    hdf5_root = tmp_path / "episodes"
    prefix_root = tmp_path / "prefixes"
    success = hdf5_root / "success.hdf5"
    failure = hdf5_root / "failure.hdf5"
    _record_episode(success, success=True)
    _record_episode(failure, success=False)
    _write_prefix_cache(prefix_root / "success.prefix.hdf5", success, value=0.25)
    _write_prefix_cache(prefix_root / "failure.prefix.hdf5", failure, value=-0.25)

    manifest = tmp_path / "safe.jsonl"
    metadata = build_manifest(
        hdf5_root=hdf5_root,
        prefix_root=prefix_root,
        output=manifest,
        split_seed=42,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    assert metadata["label_counts"] == {"0": 1, "1": 1}
    output = tmp_path / "model"
    report = train(
        manifest=manifest,
        output=output,
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        weight_decay=0.0,
        width=4,
        dropout=0.0,
        seed=0,
        device_name="cpu",
    )
    assert report["train_samples"] == 2
    assert (output / "safe_prefix_risk.pt").is_file()
    assert (output / "metrics.json").is_file()
