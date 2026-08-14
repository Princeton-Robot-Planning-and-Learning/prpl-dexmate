"""Tests for prpl_dexmate.park."""

from typing import Iterator

import numpy as np
import pytest

from prpl_dexmate.park import (
    ParkingBlocked,
    ParkingPlanner,
    clip_conf_to_limits,
    vendor_allowed_collision_pairs,
)


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


def test_storage_round_trips_through_home(planner: ParkingPlanner) -> None:
    """Storage is reachable from home directly and from fold via home."""
    moves = planner.plan_parking_moves(planner.home_right, planner.home_left, "storage")
    assert [m.component for m in moves] == ["right_arm", "left_arm"]
    assert np.allclose(moves[0].end, planner.storage_right)
    from_fold = planner.plan_parking_moves(
        planner.fold_right, planner.fold_left, "storage"
    )
    assert [m.component for m in from_fold] == [
        "right_arm",
        "right_arm",
        "left_arm",
        "left_arm",
    ]
    assert np.allclose(from_fold[1].end, planner.storage_right)
    assert np.allclose(from_fold[3].end, planner.storage_left)


@pytest.fixture(name="gripper_planner", scope="module")
def _gripper_planner_fixture() -> Iterator[ParkingPlanner]:
    planner = ParkingPlanner(grippers_mounted=True)
    yield planner
    planner.close()


def test_grippers_mounted_refuses_fold(gripper_planner: ParkingPlanner) -> None:
    """With grippers mounted the shipping fold is not a legal goal."""
    with pytest.raises(ValueError, match="self-collides"):
        gripper_planner.plan_parking_moves(
            gripper_planner.home_right, gripper_planner.home_left, "fold"
        )


def test_grippers_mounted_home_storage_round_trip(
    gripper_planner: ParkingPlanner,
) -> None:
    """Home <-> storage plans cleanly with gripper geometry included."""
    there = gripper_planner.plan_parking_moves(
        gripper_planner.home_right, gripper_planner.home_left, "storage"
    )
    assert [m.component for m in there] == ["right_arm", "left_arm"]
    back = gripper_planner.plan_parking_moves(
        gripper_planner.storage_right, gripper_planner.storage_left, "home"
    )
    assert [m.component for m in back] == ["right_arm", "left_arm"]


def test_clip_conf_to_limits_recovers_backdriven_wrist() -> None:
    """The incident wrist values clip to just inside the limits.

    Gripper mounting left the right wrist at 1.522 rad against a 1.378 URDF limit (left
    mirrored); the overshoot feeds the recovery move's start-error allowance.
    """
    incident_right = np.array([-1.809, -0.637, 0.243, -2.029, -0.831, -0.117, 1.522])
    clipped, overshoot = clip_conf_to_limits(incident_right, "right_arm")
    assert clipped[6] == pytest.approx(1.368, abs=1e-3)
    assert overshoot == pytest.approx(0.154, abs=1e-3)
    assert np.allclose(clipped[:6], incident_right[:6])
    in_range = np.zeros(7)
    same, none = clip_conf_to_limits(in_range, "left_arm")
    assert np.allclose(same, in_range)
    assert none == 0.0


def test_vendor_srdf_pairs_parse() -> None:
    """The vendor SRDF yields the wrist pairs that bit us on hardware."""
    pairs = {frozenset(p) for p in vendor_allowed_collision_pairs()}
    assert frozenset(("L_arm_l5", "L_arm_l7")) in pairs
    assert frozenset(("R_arm_l5", "R_arm_l7")) in pairs
    assert len(pairs) > 50


def test_mounting_backdriven_wrists_can_repark(
    gripper_planner: ParkingPlanner,
) -> None:
    """Regression: the exact post-mounting poses from hardware plan cleanly.

    Gripper installation back-drove both wrist rolls ~0.7 rad; at those angles the
    coarse wrist-link hulls overlap in the model, and before the vendor SRDF pairs were
    honored the planner refused to move at all (2026-08-13 incident).
    """
    current_right = np.array([-1.809, -0.637, 0.243, -2.029, -0.831, -0.117, 1.522])
    current_left = np.array([1.817, 0.632, -0.236, -2.03, 0.759, -0.023, -1.506])
    moves = gripper_planner.plan_parking_moves(current_right, current_left, "home")
    assert [m.component for m in moves] == ["right_arm", "left_arm"]
    assert np.allclose(moves[0].end, gripper_planner.home_right)
    assert np.allclose(moves[1].end, gripper_planner.home_left)


def test_grippers_mounted_blocks_leaving_the_fold(
    gripper_planner: ParkingPlanner,
) -> None:
    """From the (illegal) fold with grippers, even moving out is blocked.

    The fold itself is in collision with gripper geometry, so the first path sample
    fails — the planner refuses rather than sweeping the grippers through each other.
    """
    with pytest.raises(ParkingBlocked):
        gripper_planner.plan_parking_moves(
            gripper_planner.fold_right, gripper_planner.fold_left, "home"
        )
