"""Example: tilt the head to a target pose and return to the initial pose."""

import numpy as np
from dexcontrol.robot import Robot

robot = Robot()

initial_pos = robot.head.get_joint_pos()
print(f"Initial head joint positions: {initial_pos}")

target_pos = np.array([-np.pi / 6, 0.0, 0.0])
print(f"Moving head to: {target_pos}")
robot.head.set_joint_pos(target_pos, wait_time=2.0)
print(f"Head joint positions after move: {robot.head.get_joint_pos()}")

print("Returning head to initial position")
robot.head.set_joint_pos(initial_pos, wait_time=2.0)
print(f"Head joint positions after return: {robot.head.get_joint_pos()}")

robot.shutdown()
