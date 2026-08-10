"""Example: rotate the left arm's wrist joint by a small amount and back."""

import numpy as np
from dexcontrol.robot import Robot

from prpl_dexmate.motion import move_and_wait

robot = Robot()

initial_pos = np.asarray(robot.left_arm.get_joint_pos())
print(f"Initial left arm joint positions: {initial_pos}")

target = initial_pos.copy()
target[-1] += 0.5
print(f"Rotating wrist joint by +0.5 rad to: {target}")
move_and_wait(robot.left_arm, target, timeout=10.0)
print(f"Left arm joint positions after move: {robot.left_arm.get_joint_pos()}")

print("Returning left arm to initial position")
move_and_wait(robot.left_arm, initial_pos, timeout=10.0)
print(f"Left arm joint positions after return: {robot.left_arm.get_joint_pos()}")

robot.shutdown()
