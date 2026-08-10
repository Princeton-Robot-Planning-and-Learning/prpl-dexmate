"""Example: rotate an arm's wrist joint by a small amount and back."""

import argparse

import numpy as np
from dexcontrol.robot import Robot

from prpl_dexmate.motion import move_and_wait

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("side", choices=["left", "right"], help="which arm to move")
args = parser.parse_args()

robot = Robot()
arm = robot.left_arm if args.side == "left" else robot.right_arm

initial_pos = np.asarray(arm.get_joint_pos())
print(f"Initial {args.side} arm joint positions: {initial_pos}")

target = initial_pos.copy()
target[-1] += 0.5
print(f"Rotating wrist joint by +0.5 rad to: {target}")
move_and_wait(arm, target, timeout=10.0)
print(f"Arm joint positions after move: {arm.get_joint_pos()}")

print(f"Returning {args.side} arm to initial position")
move_and_wait(arm, initial_pos, timeout=10.0)
print(f"Arm joint positions after return: {arm.get_joint_pos()}")

robot.shutdown()
