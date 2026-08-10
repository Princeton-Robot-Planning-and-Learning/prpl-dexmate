"""Example: follow a coordinated multi-joint waypoint trajectory with one arm."""

import argparse

import numpy as np
from dexcontrol.robot import Robot

from prpl_dexmate.limits import validate_trajectory
from prpl_dexmate.motion import follow_joint_trajectory, waypoint_trajectory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("side", choices=["left", "right"], help="which arm to move")
parser.add_argument("--hz", type=float, default=100.0, help="control rate")
parser.add_argument(
    "--segment-duration", type=float, default=3.0, help="seconds per segment"
)
args = parser.parse_args()


def toward_center(angle: float) -> float:
    """Sign of a delta that moves a joint toward its zero position."""
    return -1.0 if angle > 0 else 1.0


robot = Robot()
arm = robot.left_arm if args.side == "left" else robot.right_arm

initial_pos = np.asarray(arm.get_joint_pos())
print(f"Initial {args.side} arm joint positions: {initial_pos}")

# All deltas go toward joint centers so no segment can run into a limit.
elbow_open = initial_pos.copy()
elbow_open[3] += 0.4 * toward_center(initial_pos[3])
wrists_turned = elbow_open.copy()
wrists_turned[5] += 0.3 * toward_center(initial_pos[5])
wrists_turned[6] += 0.4 * toward_center(initial_pos[6])

waypoints = [initial_pos, elbow_open, wrists_turned, elbow_open, initial_pos]
trajectory = waypoint_trajectory(waypoints, args.segment_duration, args.hz)
print(
    f"Following {trajectory.shape[0]}-frame trajectory through "
    f"{len(waypoints)} waypoints at {args.hz:.0f} Hz"
)
validate_trajectory(trajectory, f"{args.side}_arm", hz=args.hz)
max_error = follow_joint_trajectory(arm, trajectory, hz=args.hz)
print(f"Done. Max tracking error: {max_error:.4f} rad")
print(f"Final {args.side} arm joint positions: {arm.get_joint_pos()}")

robot.shutdown()
