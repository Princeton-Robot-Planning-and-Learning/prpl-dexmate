"""Helpers for commanding robot motion through dexcontrol."""

import time
from typing import Any

import numpy as np


def move_and_wait(
    component: Any,
    target: np.ndarray,
    timeout: float = 5.0,
    tolerance: float = 0.02,
) -> None:
    """Command a joint-space move and poll until the target is reached.

    Polling covers both blocking and non-blocking move implementations.
    """
    component.move_to_joint_pos(target)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        error = np.max(np.abs(np.asarray(component.get_joint_pos()) - target))
        if error < tolerance:
            return
        time.sleep(0.1)
    print(f"Warning: target not reached within {timeout}s (error {error:.4f} rad)")


def follow_joint_trajectory(
    component: Any,
    trajectory: np.ndarray,
    hz: float = 100.0,
    max_start_error: float = 0.05,
    max_tracking_error: float = 0.5,
) -> float:
    """Stream a joint-space trajectory to a component at a fixed rate.

    The trajectory is an array of shape (num_frames, num_joints), one frame
    per control period. The component must already be at the first frame
    (use move_and_wait to get there); this is checked against
    max_start_error so a distant first frame cannot cause a violent jump.
    Aborts by raising RuntimeError if the tracking error ever exceeds
    max_tracking_error. Returns the maximum tracking error observed.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    current = np.asarray(component.get_joint_pos())
    start_error = np.max(np.abs(current - trajectory[0]))
    if start_error > max_start_error:
        raise ValueError(
            f"Component is {start_error:.4f} rad from the trajectory start; "
            f"move there first (max allowed: {max_start_error})"
        )
    period = 1.0 / hz
    start_time = time.monotonic()
    max_error = 0.0
    for i, frame in enumerate(trajectory):
        component.set_joint_pos(frame)
        error = np.max(np.abs(np.asarray(component.get_joint_pos()) - frame))
        max_error = max(max_error, error)
        if error > max_tracking_error:
            raise RuntimeError(
                f"Tracking error {error:.4f} rad exceeded "
                f"{max_tracking_error} at frame {i}; aborting"
            )
        time.sleep(max(0.0, start_time + (i + 1) * period - time.monotonic()))
    return max_error


def min_jerk_trajectory(
    start: np.ndarray,
    end: np.ndarray,
    duration: float,
    hz: float = 100.0,
) -> np.ndarray:
    """Generate a minimum-jerk joint trajectory between two poses.

    Returns an array of shape (num_frames, num_joints) suitable for
    follow_joint_trajectory, starting exactly at start and ending at end
    with zero velocity and acceleration at both ends.
    """
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    num_frames = max(2, int(round(duration * hz)) + 1)
    phase = np.linspace(0.0, 1.0, num_frames)
    s = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
    return start + s[:, None] * (end - start)


def waypoint_trajectory(
    waypoints: list[np.ndarray],
    segment_duration: float,
    hz: float = 100.0,
) -> np.ndarray:
    """Build one trajectory through a sequence of waypoints.

    Consecutive waypoints are joined by min-jerk segments of
    segment_duration seconds each, so the motion pauses momentarily
    (zero velocity) at every waypoint. Returns an array of shape
    (num_frames, num_joints) for follow_joint_trajectory.
    """
    if len(waypoints) < 2:
        raise ValueError("Need at least two waypoints")
    segments: list[np.ndarray] = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        segment = min_jerk_trajectory(a, b, segment_duration, hz)
        if segments:
            segment = segment[1:]
        segments.append(segment)
    return np.vstack(segments)
