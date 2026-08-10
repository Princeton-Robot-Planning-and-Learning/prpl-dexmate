"""Example: tilt the head to a target pose and return to the initial pose."""

import numpy as np
from dexcontrol.robot import Robot

from prpl_dexmate.motion import move_and_wait

robot = Robot()

initial_pos = np.asarray(robot.head.get_joint_pos())
print(f"Initial head joint positions: {initial_pos}")

target_pos = np.array([-np.pi / 6, 0.0, 0.0])
print(f"Moving head to: {target_pos}")
move_and_wait(robot.head, target_pos)
print(f"Head joint positions after move: {robot.head.get_joint_pos()}")

print("Returning head to initial position")
move_and_wait(robot.head, initial_pos)
print(f"Head joint positions after return: {robot.head.get_joint_pos()}")

robot.shutdown()
