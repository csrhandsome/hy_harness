"""RoboTwin integrations for the HyHarness Harness.

The RoboTwin simulator is owned by RoboTwin's evaluation runner. This package
provides the agent-facing toolkit and an embedded Harness bridge for the
runner's TASK_ENV object.
"""

from .tools import RobotTwinEnvAdapter, RobotTwinTools

__all__ = ["RobotTwinEnvAdapter", "RobotTwinHarnessPolicy", "RobotTwinTools"]


def __getattr__(name: str):
    if name == "RobotTwinHarnessPolicy":
        # Keep low-level tools and the offline HDF5 recorder importable in
        # environments that intentionally do not install planner providers.
        from .session import RobotTwinHarnessPolicy

        return RobotTwinHarnessPolicy
    raise AttributeError(name)
