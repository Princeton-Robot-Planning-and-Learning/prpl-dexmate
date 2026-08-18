"""The top-level interface composing both arms and the head.

Keep all real-world code behind this abstraction so that the rest of the package is
testable without the real robot.
"""

import abc

from dexcontrol.robot import Robot

from prpl_dexmate.interfaces.gripper_interface import (
    FakeGripperInterface,
    GripperInterface,
    RealGripperInterface,
)
from prpl_dexmate.interfaces.joint_interface import (
    FakeJointInterface,
    JointInterface,
    RealJointInterface,
)
from prpl_dexmate.structs import NUM_ARM_JOINTS, NUM_HEAD_JOINTS, VegaObservation

# Rest pose observed on the real robot with the arms folded; used as the
# fake interface's home so fake rollouts start from a realistic pose.
FOLDED_RIGHT_ARM_CONF = [-1.577, 0.019, -0.013, -3.061, -0.044, 0.291, 1.192]
FOLDED_LEFT_ARM_CONF = [1.595, 0.013, 0.010, -3.061, -0.048, -0.288, -1.122]
HOME_HEAD_CONF = [0.0] * NUM_HEAD_JOINTS


class Interface(abc.ABC):
    """A generic interface to the Vega, real or fake.

    The component sub-interfaces are exposed as attributes so that callers
    (e.g. ``RealVegaEnv.step``) can address each component directly when a
    composite action commands only a subset of them.
    """

    right_arm_interface: JointInterface
    left_arm_interface: JointInterface
    head_interface: JointInterface
    right_gripper_interface: GripperInterface
    left_gripper_interface: GripperInterface

    def get_observation(self) -> VegaObservation:
        """Build a full VegaObservation from the component getters."""
        return VegaObservation(
            right_arm_conf=self.right_arm_interface.get_joint_state(),
            left_arm_conf=self.left_arm_interface.get_joint_state(),
            head_conf=self.head_interface.get_joint_state(),
            right_gripper_pos=self.right_gripper_interface.get_position(),
            left_gripper_pos=self.left_gripper_interface.get_position(),
        )

    def close(self) -> None:
        """Tear down hardware connections in all sub-interfaces."""
        self.right_arm_interface.close()
        self.left_arm_interface.close()
        self.head_interface.close()


class FakeInterface(Interface):
    """A fake interface composing fake arm, head, and gripper interfaces."""

    def __init__(self) -> None:
        self.right_arm_interface = FakeJointInterface(FOLDED_RIGHT_ARM_CONF)
        self.left_arm_interface = FakeJointInterface(FOLDED_LEFT_ARM_CONF)
        self.head_interface = FakeJointInterface(HOME_HEAD_CONF)
        self.right_gripper_interface = FakeGripperInterface()
        self.left_gripper_interface = FakeGripperInterface()


class RealInterface(Interface):
    """The real interface to the Vega robot, backed by dexcontrol.

    Constructing this connects to the robot (it must be reachable over
    Zenoh); ``close`` shuts the connection down. Only import-time-safe on
    machines with dexcontrol installed; only constructible where the robot
    is reachable.
    """

    def __init__(self) -> None:
        self._robot = Robot()
        self.right_arm_interface = RealJointInterface(self._robot.right_arm)
        self.left_arm_interface = RealJointInterface(self._robot.left_arm)
        self.head_interface = RealJointInterface(self._robot.head)
        # dexcontrol creates the hand components only when grippers are
        # detected at the wrists; the wrappers fail loudly on use if not.
        self.right_gripper_interface = RealGripperInterface(
            getattr(self._robot, "right_hand", None), "right"
        )
        self.left_gripper_interface = RealGripperInterface(
            getattr(self._robot, "left_hand", None), "left"
        )

    def close(self) -> None:
        self._robot.shutdown()


assert len(FOLDED_RIGHT_ARM_CONF) == NUM_ARM_JOINTS
assert len(FOLDED_LEFT_ARM_CONF) == NUM_ARM_JOINTS
