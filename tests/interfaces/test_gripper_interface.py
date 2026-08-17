"""Tests for prpl_dexmate.interfaces.gripper_interface."""

import pytest

from prpl_dexmate.interfaces.gripper_interface import (
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
    FakeGripperInterface,
    RealGripperInterface,
)


def test_fake_gripper_snaps_between_ends() -> None:
    """The fake gripper toggles between the open and closed positions."""
    gripper = FakeGripperInterface()
    assert gripper.get_position() == GRIPPER_CLOSED_POS
    gripper.open()
    assert gripper.get_position() == GRIPPER_OPEN_POS
    gripper.close()
    assert gripper.get_position() == GRIPPER_CLOSED_POS


def test_real_gripper_without_hand_fails_loudly() -> None:
    """When dexcontrol detected no gripper, use raises instead of pretending."""
    gripper = RealGripperInterface(None, "right")
    with pytest.raises(RuntimeError, match="No right gripper detected"):
        gripper.open()
    assert gripper.get_position() == 0.0


def test_real_gripper_drives_hand_component() -> None:
    """The wrapper forwards to the dexcontrol-style hand component."""

    class _FakeHand:
        def __init__(self) -> None:
            self.pos = [0.1]

        def open_hand(self, wait_time: float = 0.0) -> None:
            """Snap to the open position."""
            del wait_time
            self.pos = [0.6]

        def close_hand(self, wait_time: float = 0.0) -> None:
            """Snap to the closed position."""
            del wait_time
            self.pos = [0.1]

        def get_joint_pos(self) -> list[float]:
            """Return the current position."""
            return self.pos

    hand = _FakeHand()
    gripper = RealGripperInterface(hand, "left")
    gripper.open()
    assert gripper.get_position() == pytest.approx(0.6)
    gripper.close()
    assert gripper.get_position() == pytest.approx(0.1)
