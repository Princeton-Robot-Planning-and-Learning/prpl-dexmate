"""Tests for prpl_dexmate.real_sim.plan_executors.vega_motion3d."""

import numpy as np
from relational_structs import ObjectCentricState

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_motion3d import VegaMotion3DPerceiver
from prpl_dexmate.real_sim.plan_executors.vega_motion3d import (
    VegaMotion3DPlanExecutor,
)
from prpl_dexmate.structs import NUM_ARM_JOINTS


def _perceive_fake(interface: FakeInterface) -> ObjectCentricState:
    perceiver = VegaMotion3DPerceiver(ConstantTargetSource(0.5, -0.4, 0.8))
    return perceiver.reset(interface.get_observation(), {})


def test_executor_tracks_delta_trajectory_to_completion() -> None:
    """Against a snapping fake, each waypoint converges in one tick."""
    interface = FakeInterface()
    state = _perceive_fake(interface)
    delta = np.zeros(NUM_ARM_JOINTS, dtype=np.float32)
    delta[6] = 0.1
    executor = VegaMotion3DPlanExecutor()
    executor.set_trajectory([(state, delta)])
    assert not executor.done(state)
    real_action, sim_action = executor.step(state)
    assert real_action.right_arm_goal is not None
    assert np.allclose(sim_action, delta)
    interface.right_arm_interface.execute_action(real_action.right_arm_goal)
    new_state = _perceive_fake(interface)
    assert executor.done(new_state)


def test_executor_with_empty_trajectory_is_done() -> None:
    """An empty trajectory is trivially complete."""
    executor = VegaMotion3DPlanExecutor()
    executor.set_trajectory([])
    state = _perceive_fake(FakeInterface())
    assert executor.done(state)
