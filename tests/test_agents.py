"""Tests for prpl_dexmate.agents."""

import numpy as np
import pytest
from relational_structs import ObjectCentricState

from prpl_dexmate.agents import PlanExhausted, ScriptedJointDeltaAgent
from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_motion3d import VegaMotion3DPerceiver


def _initial_state() -> ObjectCentricState:
    perceiver = VegaMotion3DPerceiver(ConstantTargetSource(0.5, -0.4, 0.8))
    return perceiver.reset(FakeInterface().get_observation(), {})


def test_scripted_agent_plans_out_and_back() -> None:
    """The plan applies +delta then -delta and returns to the start."""
    agent = ScriptedJointDeltaAgent(seed=0, joint_index=6, delta=0.02, num_steps=3)
    state = _initial_state()
    agent.reset(state, {})
    trajectory = agent.plan()
    assert len(trajectory) == 6
    deltas = [float(action[6]) for _, action in trajectory]
    assert deltas == pytest.approx([0.02] * 3 + [-0.02] * 3)
    robot = state.get_object_from_name("robot")
    assert trajectory[0][0].get(robot, "joint_7") == state.get(robot, "joint_7")


def test_scripted_agent_raises_when_exhausted() -> None:
    """A second plan call in the same episode raises PlanExhausted."""
    agent = ScriptedJointDeltaAgent(seed=0, num_steps=1)
    state = _initial_state()
    agent.reset(state, {})
    first = agent.plan()
    assert np.asarray(first[0][1]).shape == (7,)
    with pytest.raises(PlanExhausted):
        agent.plan()
    agent.reset(state, {})
    assert len(agent.plan()) == 2
