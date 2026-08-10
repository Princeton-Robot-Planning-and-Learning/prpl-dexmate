"""Tests for prpl_dexmate.motion."""

import numpy as np

from prpl_dexmate.motion import move_and_wait


class _FakeComponent:
    """Reports whatever position was last commanded."""

    def __init__(self) -> None:
        self._pos: np.ndarray = np.zeros(3)

    def move_to_joint_pos(self, target: np.ndarray) -> None:
        """Snap instantly to the commanded position."""
        self._pos = np.asarray(target, dtype=float).copy()

    def get_joint_pos(self) -> np.ndarray:
        """Return the current position."""
        return self._pos


class _StuckComponent(_FakeComponent):
    """Never moves regardless of commands."""

    def move_to_joint_pos(self, target: np.ndarray) -> None:
        """Ignore the command."""


def test_move_and_wait_reaches_target() -> None:
    """A component that reaches the target ends at the target."""
    component = _FakeComponent()
    target = np.array([0.1, -0.2, 0.3])
    move_and_wait(component, target)
    assert np.allclose(component.get_joint_pos(), target)


def test_move_and_wait_times_out_without_reaching_target() -> None:
    """A component that never moves leaves move_and_wait after the timeout."""
    component = _StuckComponent()
    target = np.array([0.1, -0.2, 0.3])
    move_and_wait(component, target, timeout=0.3)
    assert np.allclose(component.get_joint_pos(), np.zeros(3))
