"""Tests for prpl_dexmate.interfaces.joint_interface."""

from prpl_dexmate.interfaces.joint_interface import FakeJointInterface


def test_fake_joint_interface_snaps_to_goal() -> None:
    """The fake component reports the last commanded goal."""
    interface = FakeJointInterface([0.0, 1.0])
    assert interface.get_joint_state() == [0.0, 1.0]
    interface.execute_action([0.5, -0.5])
    assert interface.get_joint_state() == [0.5, -0.5]
    interface.close()
