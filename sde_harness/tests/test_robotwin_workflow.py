from __future__ import annotations

import json

import numpy as np
from robots.robotwin.tools import RobotTwinEnvAdapter, RobotTwinTools
from test_robotwin_tools import FakePolicy, FakeTaskEnv, encode


def test_finish_requires_audit_and_rejects_false_success(tmp_path):
    env = FakeTaskEnv()
    audit_path = tmp_path / "task_t0.json"
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=FakePolicy(),
        output_dir=tmp_path,
        audit_path=audit_path,
    )
    missing = toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "cannot recover"}
    )
    assert not missing.is_finish
    assert "required audit" in missing.result["error"]

    audit_path.write_text(json.dumps({"benchmark_success": False}))
    false_success = toolkit.execute_tool(
        "finish", {"status": "success", "summary": "looks done"}
    )
    assert not false_success.is_finish
    assert false_success.result["success"] is False

    stuck = toolkit.execute_tool(
        "finish", {"status": "stuck", "summary": "cannot recover"}
    )
    assert stuck.is_finish
    assert stuck.result["status"] == "stuck"


def test_gripper_only_move_arm_executes_and_invalidates_cache(tmp_path):
    env = FakeTaskEnv()
    policy = FakePolicy()
    toolkit = RobotTwinTools(
        env=RobotTwinEnvAdapter(env, observation_encoder=encode),
        policy=policy,
        output_dir=tmp_path,
    )
    result = toolkit.execute_tool(
        "robotwin_move_arm", {"arm": "left", "gripper": 0.01}
    ).result
    assert result["applied_steps"] == 1
    assert policy.invalidate_count == 1
    assert np.isclose(env.observation["state"][7], 0.01)
