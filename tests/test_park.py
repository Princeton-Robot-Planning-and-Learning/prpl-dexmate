"""Tests for prpl_dexmate.park."""

from typing import Iterator

import numpy as np
import pytest

from prpl_dexmate.park import ParkingBlocked, ParkingPlanner


@pytest.fixture(name="planner", scope="module")
def _planner_fixture() -> Iterator[ParkingPlanner]:
    planner = ParkingPlanner()
    yield planner
    planner.close()


def test_fold_to_home_moves_each_arm_once(planner: ParkingPlanner) -> None:
    """From the fold, going home is one move per arm, right first."""
    moves = planner.plan_parking_moves(planner.fold_right, planner.fold_left, "home")
    assert [m.component for m in moves] == ["right_arm", "left_arm"]
    assert np.allclose(moves[0].end, planner.home_right)
    assert np.allclose(moves[1].end, planner.home_left)


def test_home_to_fold_moves_each_arm_once(planner: ParkingPlanner) -> None:
    """From home, folding is one move per arm (home waypoint is skipped)."""
    moves = planner.plan_parking_moves(planner.home_right, planner.home_left, "fold")
    assert [m.component for m in moves] == ["right_arm", "left_arm"]
    assert np.allclose(moves[0].end, planner.fold_right)
    assert np.allclose(moves[1].end, planner.fold_left)


def test_arbitrary_start_routes_through_home(planner: ParkingPlanner) -> None:
    """An arm away from both named poses goes via home on its way to fold."""
    displaced = planner.home_right + np.array([0.2, 0.1, -0.1, 0.1, 0.0, 0.1, -0.2])
    moves = planner.plan_parking_moves(displaced, planner.fold_left, "fold")
    right_moves = [m for m in moves if m.component == "right_arm"]
    assert len(right_moves) == 2
    assert np.allclose(right_moves[0].end, planner.home_right)
    assert np.allclose(right_moves[1].end, planner.fold_right)
    # The left arm is already folded and must not detour through home.
    assert [m.component for m in moves] == ["right_arm", "right_arm"]


def test_already_at_goal_plans_nothing(planner: ParkingPlanner) -> None:
    """Arms at the goal produce an empty plan (no fold->home->fold detour)."""
    assert not planner.plan_parking_moves(planner.fold_right, planner.fold_left, "fold")
    assert not planner.plan_parking_moves(planner.home_right, planner.home_left, "home")


def test_durations_keep_peak_velocity_gentle(planner: ParkingPlanner) -> None:
    """Move durations scale with distance (~0.3 rad/s min-jerk peak)."""
    moves = planner.plan_parking_moves(planner.fold_right, planner.fold_left, "home")
    for move in moves:
        max_delta = float(np.max(np.abs(move.end - move.start)))
        peak_velocity = 1.875 * max_delta / move.seconds
        assert peak_velocity <= 0.31


def test_blocked_path_refuses(
    planner: ParkingPlanner, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A segment failing the collision check raises instead of planning."""
    monkeypatch.setattr(planner, "_path_clear", lambda *args: False)
    with pytest.raises(ParkingBlocked, match="not collision-free"):
        planner.plan_parking_moves(planner.fold_right, planner.fold_left, "home")
