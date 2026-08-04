from __future__ import annotations

import threading
import unittest

import numpy as np

from hy_vla.policy_server.server import PolicyRpcService
from libero_eval.remote_policy import RemoteLiberoPolicy
from robotwin_eval.remote_policy import RemoteRobotwinPolicy
from vla_protocol import RpcServer


class _FakeAdapter:
    def __init__(self, benchmark: str):
        self.benchmark = benchmark
        self.reset_count = 0

    def metadata(self):
        return {"fake": True}

    def reset(self):
        self.reset_count += 1
        return "reset"

    def dispatch(self, method, kwargs):
        if method == "libero.predict":
            assert kwargs["images"]["main"].dtype == np.uint8
            return {
                "actions": np.arange(14, dtype=np.float32).reshape(1, 2, 7),
                "shape": [1, 2, 7],
                "dtype": "float32",
            }
        if method == "robotwin.step":
            assert kwargs["images"]["top"].shape == (4, 5, 3)
            assert kwargs["state"].shape == (16,)
            return {
                "action": np.arange(16, dtype=np.float32),
                "shape": [16],
                "dtype": "float32",
            }
        raise AssertionError(method)


class _RunningServer:
    def __init__(self, benchmark: str):
        self.adapter = _FakeAdapter(benchmark)
        service = PolicyRpcService(self.adapter)
        self.server = RpcServer(("127.0.0.1", 0), service.dispatch)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        return self.adapter, f"http://{host}:{port}"

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class RemotePolicyRpcTest(unittest.TestCase):
    def test_libero_client_round_trip(self):
        with _RunningServer("libero") as (adapter, endpoint):
            policy = RemoteLiberoPolicy(endpoint)
            policy.reset()
            actions = policy.predict(
                "pick up the block",
                np.zeros((4, 5, 3), dtype=np.uint8),
                np.ones((4, 5, 3), dtype=np.uint8),
                np.arange(8, dtype=np.float32),
            )
            np.testing.assert_array_equal(
                actions, np.arange(14, dtype=np.float32).reshape(2, 7)
            )
            self.assertGreaterEqual(adapter.reset_count, 1)
            policy.close()

    def test_robotwin_client_round_trip(self):
        with _RunningServer("robotwin") as (adapter, endpoint):
            policy = RemoteRobotwinPolicy(
                {"policy_endpoint": endpoint, "policy_timeout_s": 5}
            )
            policy.reset()
            image = np.zeros((4, 5, 3), dtype=np.uint8)
            batch = {
                "observation.state": np.arange(32, dtype=np.float32)[None],
                "task": ["move the object"],
                "raw_images.top_head": image,
                "raw_images.hand_left": image,
                "raw_images.hand_right": image,
            }
            np.testing.assert_array_equal(
                policy.get_action(batch), np.arange(16, dtype=np.float32)
            )
            self.assertGreaterEqual(adapter.reset_count, 1)
            policy.close()


if __name__ == "__main__":
    unittest.main()

