"""Gymnasium environment for the real Vega."""

import time
from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import RenderFrame

from prpl_dexmate.interfaces.interface import Interface
from prpl_dexmate.structs import VegaAction, VegaObservation

# One policy tick. Matches the streaming control period that dexcontrol's
# set_joint_pos interface is comfortable with at low rates.
POLICY_CONTROL_PERIOD = 0.1


class RealVegaEnv(gymnasium.Env[VegaObservation, VegaAction]):
    """Gymnasium environment for the real Vega.

    ``step`` issues one command to each commanded sub-interface and returns
    a fresh observation after ``control_period``. Trajectory tracking —
    including convergence tolerances — lives in the configured
    ``PlanExecutor``, not here.

    Reward is always 0 and terminated / truncated are always False: the
    real environment has no task semantics. Convergence is the executor's
    call.
    """

    def __init__(
        self,
        interface: Interface,
        control_period: float = POLICY_CONTROL_PERIOD,
    ) -> None:
        self._interface = interface
        self._control_period = control_period

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VegaObservation, dict[str, Any]]:
        super().reset(seed=seed)
        return self._interface.get_observation(), {}

    def step(
        self, action: VegaAction
    ) -> tuple[VegaObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        if action.right_arm_goal is not None:
            self._interface.right_arm_interface.execute_action(action.right_arm_goal)
        if action.left_arm_goal is not None:
            self._interface.left_arm_interface.execute_action(action.left_arm_goal)
        if action.head_goal is not None:
            self._interface.head_interface.execute_action(action.head_goal)
        for command, gripper in (
            (action.right_gripper, self._interface.right_gripper_interface),
            (action.left_gripper, self._interface.left_gripper_interface),
        ):
            if command == "open":
                gripper.open()
            elif command == "close":
                gripper.close()
        time.sleep(self._control_period)
        return self._interface.get_observation(), 0.0, False, False, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def close(self) -> None:
        """Tear down the underlying interface (robot connection)."""
        self._interface.close()
