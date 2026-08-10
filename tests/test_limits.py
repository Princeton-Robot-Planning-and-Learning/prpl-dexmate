"""Tests for prpl_dexmate.limits."""

import numpy as np
import pytest

from prpl_dexmate.limits import get_joint_limits, validate_trajectory


def test_get_joint_limits_shapes_and_ordering() -> None:
    """Arms have 7 limited joints, the head has 3, and bounds are ordered."""
    for component, dof in (("left_arm", 7), ("right_arm", 7), ("head", 3)):
        lower, upper, velocity = get_joint_limits(component)
        assert lower.shape == upper.shape == velocity.shape == (dof,)
        assert np.all(lower < upper)
        assert np.all(velocity > 0)


def test_get_joint_limits_matches_observed_wrist_limit() -> None:
    """The right wrist's URDF upper limit matches the observed clamp."""
    _, upper, _ = get_joint_limits("right_arm")
    assert upper[6] == pytest.approx(1.378, abs=1e-3)


def test_arms_are_mirrored() -> None:
    """Left and right arm ranges mirror each other."""
    l_lower, l_upper, _ = get_joint_limits("left_arm")
    r_lower, r_upper, _ = get_joint_limits("right_arm")
    for joint in (1, 6):
        assert l_lower[joint] == pytest.approx(-r_upper[joint])
        assert l_upper[joint] == pytest.approx(-r_lower[joint])


def test_validate_trajectory_accepts_interior_motion() -> None:
    """A slow trajectory well inside the limits passes."""
    trajectory = np.linspace(np.zeros(7), np.full(7, 0.1), 100)
    validate_trajectory(trajectory, "left_arm", hz=100.0)


def test_validate_trajectory_rejects_position_violation() -> None:
    """A frame beyond a joint's upper limit is rejected by name."""
    trajectory = np.zeros((5, 7))
    trajectory[3, 6] = 2.0
    with pytest.raises(ValueError, match="joint 7"):
        validate_trajectory(trajectory, "right_arm", hz=100.0)


def test_validate_trajectory_rejects_velocity_violation() -> None:
    """A step that implies excessive joint speed is rejected."""
    trajectory = np.zeros((3, 7))
    trajectory[1, 0] = 0.5
    with pytest.raises(ValueError, match="rad/s"):
        validate_trajectory(trajectory, "left_arm", hz=100.0)


def test_validate_trajectory_rejects_wrong_shape() -> None:
    """A trajectory with the wrong joint count is rejected."""
    with pytest.raises(ValueError, match="shape"):
        validate_trajectory(np.zeros((5, 3)), "left_arm", hz=100.0)
