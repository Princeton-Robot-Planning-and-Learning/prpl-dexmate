"""Tests for prpl_dexmate.sim_env."""

import numpy as np
from relational_structs import ObjectCentricState

from prpl_dexmate.sim_env import KinderSimEnv


def test_kinder_sim_env_devectorizes_states() -> None:
    """Reset and step return structured ObjectCentricStates."""
    env = KinderSimEnv("kinder/VegaMotion3D-v0")
    state, _ = env.reset(seed=0)
    assert isinstance(state, ObjectCentricState)
    robot = state.get_object_from_name("robot")
    assert state.get(robot, "joint_1") is not None
    action = np.zeros(7, dtype=np.float32)
    next_state, reward, terminated, truncated, _ = env.step(action)
    assert isinstance(next_state, ObjectCentricState)
    assert isinstance(float(reward), float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    env.close()
