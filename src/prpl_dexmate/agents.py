"""Planning agents for the Vega pipeline.

ScriptedJointDeltaAgent is a stand-in that exercises the full pipeline deterministically
until a VegaMotion3D env model lands in kinder-baselines and a bilevel planning agent
can take its place.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from prpl_utils.planning_agent import PlanningAgent
from relational_structs import ObjectCentricState

from prpl_dexmate.real_sim.perceivers.vega_motion3d import ROBOT_OBJECT
from prpl_dexmate.structs import NUM_ARM_JOINTS


class PlanExhausted(Exception):
    """Raised by an agent whose plan has been fully executed."""


class ScriptedJointDeltaAgent(
    PlanningAgent[ObjectCentricState, NDArray[np.floating], ObjectCentricState]
):
    """Plan a fixed out-and-back delta script on one right-arm joint.

    The plan moves ``joint_index`` by ``delta`` per step for ``num_steps``
    steps and then back, rolling the state estimate forward kinematically
    for each pair. One plan per episode: a second ``plan`` call raises
    PlanExhausted, which the pipeline treats as the rollout's natural end.
    """

    def __init__(
        self,
        seed: int,
        joint_index: int = 6,
        delta: float = 0.02,
        num_steps: int = 10,
    ) -> None:
        super().__init__(seed)
        assert 0 <= joint_index < NUM_ARM_JOINTS
        self._joint_index = joint_index
        self._delta = delta
        self._num_steps = num_steps
        self._planned = False

    def reset(self, obs: ObjectCentricState, info: dict[str, Any]) -> None:
        super().reset(obs, info)
        self._planned = False

    def plan(self) -> list[tuple[ObjectCentricState, NDArray[np.floating]]]:
        if self._planned:
            raise PlanExhausted("Scripted plan already executed")
        assert self._last_observation is not None
        self._planned = True
        deltas = [self._delta] * self._num_steps + [-self._delta] * self._num_steps
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]] = []
        state = self._last_observation
        feature = f"joint_{self._joint_index + 1}"
        robot = state.get_object_from_name(ROBOT_OBJECT.name)
        for delta in deltas:
            action = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
            action[self._joint_index] = delta
            trajectory.append((state, action))
            state = state.copy()
            state.set(robot, feature, state.get(robot, feature) + delta)
        return trajectory

    def _get_action(self) -> NDArray[np.floating]:
        raise NotImplementedError("Driven via plan(), not step()")
