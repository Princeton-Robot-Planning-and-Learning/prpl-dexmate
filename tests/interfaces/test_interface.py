"""Tests for prpl_dexmate.interfaces.interface."""

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.structs import NUM_ARM_JOINTS, NUM_HEAD_JOINTS


def test_fake_interface_builds_observation() -> None:
    """The composite fake produces a well-formed observation."""
    interface = FakeInterface()
    obs = interface.get_observation()
    assert len(obs.right_arm_conf) == NUM_ARM_JOINTS
    assert len(obs.left_arm_conf) == NUM_ARM_JOINTS
    assert len(obs.head_conf) == NUM_HEAD_JOINTS
    interface.close()


def test_fake_interface_component_commands_are_independent() -> None:
    """Commanding one component leaves the others unchanged."""
    interface = FakeInterface()
    before = interface.get_observation()
    goal = [0.1] * NUM_ARM_JOINTS
    interface.right_arm_interface.execute_action(goal)
    after = interface.get_observation()
    assert after.right_arm_conf == goal
    assert after.left_arm_conf == before.left_arm_conf
    assert after.head_conf == before.head_conf
