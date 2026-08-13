from __future__ import annotations

import json

import numpy as np
from robots.robotwin.execution import RobotTwinActionExecutor
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools


class FakeTaskEnv:
    def __init__(self, *, success_at: int = 100) -> None:
        self.observation = {"step": 0, "state": self._initial_state()}
        self.take_action_cnt = 0
        self.step_lim = 100
        self.eval_success = False
        self.success_at = success_at

    @staticmethod
    def _initial_state() -> np.ndarray:
        return np.array(
            [0, 0, 0, 1, 0, 0, 0, 0.04, 1, 0, 0, 1, 0, 0, 0, 0.04],
            dtype=np.float32,
        )

    def get_obs(self):
        return self.observation

    def get_instruction(self):
        return "move the red block"

    def take_action(self, action, action_type="ee"):
        assert action_type == "ee"
        action = np.asarray(action, dtype=np.float32)
        assert action.shape == (16,)
        self.take_action_cnt += 1
        self.observation = {
            "step": self.take_action_cnt,
            "state": action.copy(),
        }
        if self.take_action_cnt >= self.success_at:
            self.eval_success = True


def encode(observation, instruction):
    step = int(observation["step"])
    image = np.full((8, 8, 3), step, dtype=np.uint8)
    state = np.zeros(32, dtype=np.float32)
    state[:16] = observation["state"]
    return {
        "observation.state": state[None],
        "task": [instruction],
        "raw_images.top_head": image,
        "raw_images.hand_left": image,
        "raw_images.hand_right": image,
    }


class FakePolicy:
    action_chunk_size = 5
    default_execute_steps = 3

    def __init__(self) -> None:
        self.forward_count = 0
        self.invalidate_count = 0
        self.observed = 0

    def get_action_chunk(self, batch, *, max_actions=None):
        self.forward_count += 1
        actions = np.repeat(batch["observation.state"][:, :16], 5, axis=0)
        actions[:, 0] = np.arange(1, 6) + 10 * self.forward_count
        return actions

    def invalidate_action_cache(self):
        self.invalidate_count += 1

    def observe(self, batch):
        self.observed += 1


def make_toolkit(tmp_path=None, *, success_at=100):
    env = FakeTaskEnv(success_at=success_at)
    policy = FakePolicy()
    return (
        env,
        policy,
        RobotTwinTools(
            env=RobotTwinEnvAdapter(env, observation_encoder=encode),
            policy=policy,
            output_dir=tmp_path,
        ),
    )


def test_toolkit_registers_named_primitives_and_observes(tmp_path):
    _, _, toolkit = make_toolkit(tmp_path)
    names = {item["name"] for item in toolkit.get_tools_description()}
    assert {
        "robotwin_vla_chunk",
        "robotwin_move_arm",
        "robotwin_translate_arm",
        "robotwin_move_bimanual",
        "robotwin_rotate_arm",
        "robotwin_set_gripper",
        "robotwin_release",
        "robotwin_hold",
    } <= names
    observed = toolkit.execute_tool("robotwin_observe", {}).result
    assert observed["instruction"] == "move the red block"
    assert observed["_image_bytes"]


def test_vla_chunk_is_one_fresh_forward_and_discards_suffix(tmp_path):
    env, policy, toolkit = make_toolkit(tmp_path)
    result = toolkit.execute_tool("robotwin_vla_chunk", {"execute_steps": 3}).result
    assert result["applied_steps"] == 3
    assert result["forward_count"] == 1
    assert result["generated_actions"] == 5
    assert result["discarded_actions"] == 2
    assert policy.forward_count == 1
    assert env.take_action_cnt == 3
    assert env.observation["state"][0] == 13

    second = toolkit.execute_tool("robotwin_vla_chunk", {"execute_steps": 1}).result
    assert second["chunk_id"] == 2
    assert policy.forward_count == 2
    assert env.observation["state"][0] == 21


def test_direct_primitive_invalidates_vla_cache_and_holds_other_arm(tmp_path):
    env, policy, toolkit = make_toolkit(tmp_path)
    before = env.observation["state"].copy()
    result = toolkit.execute_tool(
        "robotwin_translate_arm",
        {"arm": "left", "delta_xyz": [0.06, 0.0, 0.0], "position_step": 0.03},
    ).result
    assert result["applied_steps"] == 2
    assert policy.invalidate_count == 1
    np.testing.assert_allclose(env.observation["state"][8:16], before[8:16])
    np.testing.assert_allclose(env.observation["state"][0], 0.06, atol=1e-6)


def test_write_is_output_scoped_and_recipe_is_automatic(tmp_path):
    _, _, toolkit = make_toolkit(tmp_path)
    audit = toolkit.execute_tool(
        "write_text_file", {"path": "audit.json", "content": json.dumps({"ok": True})}
    ).result
    assert audit["path"] == str(tmp_path / "audit.json")
    denied = toolkit.execute_tool(
        "write_text_file",
        {"path": str(tmp_path.parent / "outside.json"), "content": "{}"},
    ).result
    assert "error" in denied
    toolkit.execute_tool("robotwin_hold", {"steps": 1})
    recipe = toolkit.write_recipe("task_t0")
    assert recipe == str(tmp_path / "recipe_task_t0.jsonl")
    assert (
        json.loads((tmp_path / "recipe_task_t0.jsonl").read_text())["source"]
        == "robotwin_hold"
    )


def test_executor_stops_on_authoritative_success():
    env = FakeTaskEnv(success_at=2)
    executor = RobotTwinActionExecutor(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
    )
    result = executor.vla_chunk(execute_steps=5)
    assert result.applied_steps == 2
    assert result.discarded_actions == 3
    assert env.eval_success is True
