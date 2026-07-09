"""Example: read the right arm joint current from the robot."""

from dexcontrol.robot import Robot

robot = Robot()

arm_current = robot.right_arm.get_joint_current()
print(f"Right arm joint current:{arm_current}")

robot.shutdown()
