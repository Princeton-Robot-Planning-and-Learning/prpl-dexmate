"""Tests for prpl_dexmate.motion."""

import numpy as np
import pytest

from prpl_dexmate.motion import (
    follow_joint_trajectory,
    min_jerk_trajectory,
    move_and_wait,
)


class _FakeComponent:
    """Reports whatever position was last commanded."""

    def __init__(self, pos: np.ndarray | None = None) -> None:
        self._pos: np.ndarray = (
            np.zeros(3) if pos is None else np.asarray(pos, dtype=float)
        )

    def move_to_joint_pos(self, target: np.ndarray) -> None:
        """Snap instantly to the commanded position."""
        self._pos = np.asarray(target, dtype=float).copy()

    def set_joint_pos(self, target: np.ndarray) -> None:
        """Snap instantly to the commanded position."""
        self._pos = np.asarray(target, dtype=float).copy()

    def get_joint_pos(self) -> np.ndarray:
        """Return the current position."""
        return self._pos


class _StuckComponent(_FakeComponent):
    """Never moves regardless of commands."""

    def move_to_joint_pos(self, target: np.ndarray) -> None:
        """Ignore the command."""

    def set_joint_pos(self, target: np.ndarray) -> None:
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


def test_follow_joint_trajectory_tracks_all_frames() -> None:
    """A snapping component follows the whole trajectory with zero error."""
    start = np.zeros(3)
    end = np.array([0.5, -0.5, 0.2])
    trajectory = min_jerk_trajectory(start, end, duration=0.05, hz=100.0)
    component = _FakeComponent(start)
    max_error = follow_joint_trajectory(component, trajectory, hz=1000.0)
    assert max_error == 0.0
    assert np.allclose(component.get_joint_pos(), end)


def test_follow_joint_trajectory_rejects_distant_start() -> None:
    """A trajectory starting far from the current pose is refused."""
    component = _FakeComponent()
    trajectory = np.full((10, 3), 1.0)
    with pytest.raises(ValueError, match="trajectory start"):
        follow_joint_trajectory(component, trajectory)


def test_follow_joint_trajectory_aborts_on_tracking_error() -> None:
    """A component that stops responding triggers an abort."""
    component = _StuckComponent()
    trajectory = np.linspace(np.zeros(3), np.ones(3), 50)
    with pytest.raises(RuntimeError, match="Tracking error"):
        follow_joint_trajectory(
            component, trajectory, hz=1000.0, max_tracking_error=0.1
        )


def test_min_jerk_trajectory_endpoints_and_shape() -> None:
    """The trajectory starts at start, ends at end, and moves monotonically."""
    start = np.array([0.0, 1.0])
    end = np.array([1.0, -1.0])
    trajectory = min_jerk_trajectory(start, end, duration=2.0, hz=50.0)
    assert trajectory.shape == (101, 2)
    assert np.allclose(trajectory[0], start)
    assert np.allclose(trajectory[-1], end)
    steps = np.diff(trajectory[:, 0])
    assert np.all(steps >= 0)
