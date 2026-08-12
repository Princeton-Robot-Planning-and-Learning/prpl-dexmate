"""Tests for prpl_dexmate.structs."""

import pytest

from prpl_dexmate.structs import VegaAction, VegaObservation


def test_observation_validates_lengths() -> None:
    """Wrong joint counts are rejected."""
    with pytest.raises(AssertionError):
        VegaObservation(
            right_arm_conf=[0.0] * 6, left_arm_conf=[0.0] * 7, head_conf=[0.0] * 3
        )


def test_action_allows_partial_commands() -> None:
    """Uncommanded components default to None."""
    action = VegaAction(right_arm_goal=[0.0] * 7)
    assert action.left_arm_goal is None
    assert action.head_goal is None
    with pytest.raises(AssertionError):
        VegaAction(head_goal=[0.0] * 7)
