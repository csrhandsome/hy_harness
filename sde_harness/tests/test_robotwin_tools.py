from __future__ import annotations

import numpy as np

from robots.robotwin.execution import RobotTwinActionExecutor
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools


class FakeTaskEnv:
    def __init__(self) -> None:
        self.observation = {"step": 0}
        self.take_action_cnt = 0
        self.step_lim = 10
        self.eval_success = False

    def get_obs(self):
        return self.observation

    def get_instruction(self):
        return "move the red block"

    def take_action(self, action, action_type="ee"):
        assert action_type == "ee"
        assert np.asarray(action).shape == (16,)
        self.take_action_cnt += 1
        self.observation = {"step": self.take_action_cnt}
        if self.take_action_cnt >= 2:
            self.eval_success = True


def encode(observation, instruction):
    step = int(observation["step"])
    image = np.full((8, 8, 3), step, dtype=np.uint8)
    return {
        "observation.state": np.zeros((1, 32), dtype=np.float32),
        "task": [instruction],
        "raw_images.top_head": image,
        "raw_images.hand_left": image,
        "raw_images.hand_right": image,
    }


class FakePolicy:
    def get_action(self, batch):
        return np.zeros(16, dtype=np.float32)


def test_robotwin_toolkit_observe_and_action():
    env = FakeTaskEnv()
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
    )

    observed = toolkit.execute_tool("robotwin_observe", {}).result
    assert observed["instruction"] == "move the red block"
    assert observed["done"] is False
    assert observed["_image_bytes"]

    result = toolkit.execute_tool(
        "robotwin_execute_ee",
        {"action": [0.0] * 16, "repeat": 2},
    ).result
    assert result["applied_steps"] == 2
    assert result["success"] is True


def test_robotwin_toolkit_vla_step():
    env = FakeTaskEnv()
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
    )
    result = toolkit.execute_tool("robotwin_vla_step", {"steps": 1}).result
    assert result["applied_steps"] == 1
    assert env.take_action_cnt == 1


def test_robotwin_action_executor_runs_without_toolkit():
    env = FakeTaskEnv()
    executor = RobotTwinActionExecutor(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
    )

    direct_result = executor.execute_ee([0.0] * 16)
    policy_result = executor.vla_step()

    assert direct_result.applied_steps == 1
    assert policy_result.applied_steps == 1
    assert env.take_action_cnt == 2
    assert env.eval_success is True
