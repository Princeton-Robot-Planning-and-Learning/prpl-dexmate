"""Example: stream a smooth sine-wave wrist trajectory to an arm."""

import argparse

import numpy as np
from dexcontrol.robot import Robot

from prpl_dexmate.motion import follow_joint_trajectory

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("side", choices=["left", "right"], help="which arm to move")
parser.add_argument("--hz", type=float, default=100.0, help="control rate")
parser.add_argument(
    "--amplitude", type=float, default=0.3, help="wrist swing amplitude (rad)"
)
parser.add_argument("--duration", type=float, default=8.0, help="seconds")
args = parser.parse_args()

robot = Robot()
arm = robot.left_arm if args.side == "left" else robot.right_arm

initial_pos = np.asarray(arm.get_joint_pos())
print(f"Initial {args.side} arm joint positions: {initial_pos}")

# One full sine period on the wrist joint, swinging toward joint center so
# the trajectory cannot run into the wrist's travel limit.
direction = -1.0 if initial_pos[-1] > 0 else 1.0
num_frames = int(args.duration * args.hz) + 1
phase = np.linspace(0.0, 2.0 * np.pi, num_frames)
trajectory = np.tile(initial_pos, (num_frames, 1))
trajectory[:, -1] += direction * args.amplitude * 0.5 * (1.0 - np.cos(phase))

print(
    f"Following {num_frames}-frame wrist sine "
    f"(amplitude {direction * args.amplitude:+.2f} rad) at {args.hz:.0f} Hz"
)
max_error = follow_joint_trajectory(arm, trajectory, hz=args.hz)
print(f"Done. Max tracking error: {max_error:.4f} rad")
print(f"Final {args.side} arm joint positions: {arm.get_joint_pos()}")

robot.shutdown()
