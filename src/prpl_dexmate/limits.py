"""Joint limits for the Vega robot, read from Dexmate's published URDF."""

import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import dexmate_urdf
import numpy as np

_COMPONENT_JOINT_PREFIXES = {
    "left_arm": "L_arm_j",
    "right_arm": "R_arm_j",
    "head": "head_j",
}


@lru_cache(maxsize=None)
def get_joint_limits(component: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lower, upper, velocity) limit arrays for a vega_1u component.

    Values come from the vega_1u URDF in the dexmate-urdf package, in
    joint order (j1, j2, ...). component is "left_arm", "right_arm",
    or "head".
    """
    prefix = _COMPONENT_JOINT_PREFIXES[component]
    urdf_path = (
        Path(dexmate_urdf.__file__).parent / "robots/humanoid/vega_1u/vega_1u.urdf"
    )
    joints = []
    for joint in ET.parse(urdf_path).getroot().iter("joint"):
        name = joint.get("name", "")
        limit = joint.find("limit")
        if not name.startswith(prefix) or limit is None:
            continue
        index = int(name.removeprefix(prefix))
        joints.append(
            (
                index,
                float(limit.get("lower", "nan")),
                float(limit.get("upper", "nan")),
                float(limit.get("velocity", "nan")),
            )
        )
    joints.sort()
    _, lower, upper, velocity = (np.array(column) for column in zip(*joints))
    return lower, upper, velocity


def validate_trajectory(
    trajectory: np.ndarray,
    component: str,
    hz: float,
) -> None:
    """Check a joint trajectory against URDF position and velocity limits.

    Raises ValueError naming the first offending joint and frame. Position
    limits are applied as-is (no margin): the robot can legitimately rest
    within millimeters of a limit, e.g. joint 4 near its folded pose.
    """
    trajectory = np.asarray(trajectory, dtype=float)
    lower, upper, velocity = get_joint_limits(component)
    if trajectory.ndim != 2 or trajectory.shape[1] != lower.shape[0]:
        raise ValueError(
            f"Expected trajectory of shape (frames, {lower.shape[0]}), "
            f"got {trajectory.shape}"
        )
    too_low = trajectory < lower
    too_high = trajectory > upper
    if np.any(too_low) or np.any(too_high):
        frame, joint = np.argwhere(too_low | too_high)[0]
        raise ValueError(
            f"{component} joint {joint + 1} = {trajectory[frame, joint]:.4f} "
            f"at frame {frame} violates limits "
            f"[{lower[joint]:.4f}, {upper[joint]:.4f}]"
        )
    speeds = np.abs(np.diff(trajectory, axis=0)) * hz
    if np.any(speeds > velocity):
        frame, joint = np.argwhere(speeds > velocity)[0]
        raise ValueError(
            f"{component} joint {joint + 1} moves at {speeds[frame, joint]:.3f} "
            f"rad/s at frame {frame}, exceeding the "
            f"{velocity[joint]:.2f} rad/s limit"
        )
