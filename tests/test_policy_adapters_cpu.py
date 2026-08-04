from __future__ import annotations

import unittest

import numpy as np

from hy_vla.policy_server.adapters import LiberoPolicyAdapter, RobotwinPolicyAdapter


class _FakeLiberoModel:
    def predict(self, instruction, main, wrist, state):
        assert instruction == "pick"
        assert main.shape == (6, 7, 3)
        assert wrist.shape == (6, 7, 3)
        assert state.shape == (8,)
        return np.ones((3, 7), dtype=np.float32)


class _FakeRobotwinModel:
    def get_action(self, batch):
        assert batch["observation.state"].shape == (1, 32)
        assert batch["observation.images.top_head"].shape == (1, 3, 6, 7)
        assert batch["raw_images.hand_left"].dtype == np.uint8
        return np.arange(16, dtype=np.float32)


class PolicyAdapterCpuTest(unittest.TestCase):
    def test_libero_wire_to_policy_shape(self):
        adapter = LiberoPolicyAdapter.__new__(LiberoPolicyAdapter)
        adapter.policy = _FakeLiberoModel()
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        result = adapter.dispatch(
            "libero.predict",
            {
                "instruction": "pick",
                "images": {"main": image, "wrist": image},
                "state": np.zeros(8, dtype=np.float32),
            },
        )
        self.assertEqual(result["actions"].shape, (1, 3, 7))

    def test_robotwin_wire_to_policy_shape(self):
        adapter = RobotwinPolicyAdapter.__new__(RobotwinPolicyAdapter)
        adapter.policy = _FakeRobotwinModel()
        image = np.zeros((6, 7, 3), dtype=np.uint8)
        result = adapter.dispatch(
            "robotwin.step",
            {
                "instruction": "move",
                "images": {"top": image, "left": image, "right": image},
                "state": np.zeros(16, dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(result["action"], np.arange(16, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()

