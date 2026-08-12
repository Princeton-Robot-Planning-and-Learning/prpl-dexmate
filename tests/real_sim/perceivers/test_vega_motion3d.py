"""Tests for prpl_dexmate.real_sim.perceivers.vega_motion3d."""

import numpy as np

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_motion3d import VegaMotion3DPerceiver
from prpl_dexmate.structs import NUM_ARM_JOINTS


def test_perceiver_builds_matching_state() -> None:
    """The perceived state carries the right-arm joints and the target."""
    perceiver = VegaMotion3DPerceiver(ConstantTargetSource(0.5, -0.4, 0.8))
    obs = FakeInterface().get_observation()
    state = perceiver.reset(obs, {})
    robot = state.get_object_from_name("robot")
    for i in range(NUM_ARM_JOINTS):
        assert state.get(robot, f"joint_{i + 1}") == obs.right_arm_conf[i]
    target = state.get_object_from_name("target")
    assert state.get(target, "x") == 0.5
    assert state.get(target, "y") == -0.4
    assert state.get(target, "z") == 0.8
    stepped = perceiver.step(obs, {})
    for obj, values in state.data.items():
        assert np.allclose(stepped.data[obj], values)
