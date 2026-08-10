"""Example: tilt the head to a target pose and return to the initial pose."""

import time
from typing import Any

import numpy as np
from dexcontrol.robot import Robot


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
