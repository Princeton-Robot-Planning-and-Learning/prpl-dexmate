"""Perceiver mapping VegaObservation to a VegaPickPlace3D sim state.

The real robot cannot see the cube yet, so this is a staged-scene belief
tracker rather than perception proper: the cube's initial position and
the target patch come from configured sources (a taped spot on the
table), and from then on the cube is dead-reckoned through the grasp
cycle using the observation alone:

* An arm acquires the cube when its gripper reads closed and its end
  effector (forward kinematics on the observed joints) is within
  ``grasp_radius`` of the believed cube position.
* While held, the cube rides the holding arm's end effector.
* When the holder's gripper opens, the cube is set down at the resting
  height directly beneath the end effector.

Known belief-vs-reality gaps, accepted for the staged-scene stage: a
closed-empty gripper within range is indistinguishable from a real
grasp (grasp *verification* via the gripper's stall position is the
planned upgrade), and a cube knocked or dropped outside the model's
rules goes untracked. The environment's release-height discipline lives
in the skills, not here — this tracker believes whatever the gripper
does.
"""

from typing import Any

import numpy as np
from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2EnvTypeFeatures,
    Kinematic3Dv2GraspArmRobotType,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    ARM_SIDES,
    CUBE_NODE,
    TARGET_NODE,
    VegaPickPlace3DObjectCentricState,
)
from prpl_kinematics.robots import make_vega
from prpl_utils.real_sim import Perceiver
from relational_structs import Object
from relational_structs.utils import create_state_from_dict

from prpl_dexmate.real_sim.perceivers.target_source import TargetSource
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaObservation

# A gripper reading below this is treated as closed (closed reads ~0.1 on
# hardware, open ~0.6; a gripper stalled on the 6 cm cube sits between,
# still on the closed side of this line).
GRIPPER_CLOSED_THRESHOLD = 0.35

# Matches VegaPickPlace3DEnvConfig defaults: grasp acquisition radius and
# the cube's resting height on the table (table_height + cube_half_size).
DEFAULT_GRASP_RADIUS = 0.10
DEFAULT_CUBE_RESTING_Z = 0.58

_SIDE_PREFIX = {"left": "L", "right": "R"}


class VegaPickPlace3DPerceiver(
    Perceiver[VegaObservation, VegaPickPlace3DObjectCentricState]
):
    """Build a VegaPickPlace3D ObjectCentricState from a VegaObservation."""

    def __init__(
        self,
        cube_source: TargetSource,
        target_source: TargetSource,
        grasp_radius: float = DEFAULT_GRASP_RADIUS,
        cube_resting_z: float = DEFAULT_CUBE_RESTING_Z,
    ) -> None:
        self._cube_source = cube_source
        self._target_source = target_source
        self._grasp_radius = grasp_radius
        self._cube_resting_z = cube_resting_z
        robot = make_vega()
        self._tree = robot.tree
        self._home = dict(robot.home)
        self._ee_frames = {
            side: robot.manipulators[side].ee_frame for side in ARM_SIDES
        }
        self._cube = np.zeros(3)
        self._holder: str | None = None

    def reset(
        self, obs: VegaObservation, info: dict[str, Any]
    ) -> VegaPickPlace3DObjectCentricState:
        del info
        self._cube = np.array(self._cube_source.get_target(), dtype=float)
        self._holder = None
        return self._perceive(obs)

    def step(
        self, obs: VegaObservation, info: dict[str, Any]
    ) -> VegaPickPlace3DObjectCentricState:
        del info
        self._update_belief(obs)
        return self._perceive(obs)

    def _ee_position(self, obs: VegaObservation, side: str) -> np.ndarray:
        config = dict(self._home)
        prefix = _SIDE_PREFIX[side]
        joints = obs.left_arm_conf if side == "left" else obs.right_arm_conf
        for i in range(NUM_ARM_JOINTS):
            config[f"{prefix}_arm_j{i + 1}"] = [float(joints[i])]
        return np.asarray(
            self._tree.forward_kinematics(self._ee_frames[side], config).t
        )

    def _gripper_closed(self, obs: VegaObservation, side: str) -> bool:
        position = obs.left_gripper_pos if side == "left" else obs.right_gripper_pos
        return position < GRIPPER_CLOSED_THRESHOLD

    def _update_belief(self, obs: VegaObservation) -> None:
        if self._holder is not None:
            if self._gripper_closed(obs, self._holder):
                # Held cube rides the holding arm's end effector.
                self._cube = self._ee_position(obs, self._holder)
                return
            # Released: set down at resting height beneath the end effector.
            ee = self._ee_position(obs, self._holder)
            self._cube = np.array([ee[0], ee[1], self._cube_resting_z])
            self._holder = None
        for side in ARM_SIDES:
            if not self._gripper_closed(obs, side):
                continue
            ee = self._ee_position(obs, side)
            if float(np.linalg.norm(ee - self._cube)) < self._grasp_radius:
                self._holder = side
                self._cube = ee
                return

    def _perceive(self, obs: VegaObservation) -> VegaPickPlace3DObjectCentricState:
        state_dict: dict[Object, dict[str, float]] = {}
        for side in ARM_SIDES:
            joints = obs.left_arm_conf if side == "left" else obs.right_arm_conf
            feats = {f"joint_{i + 1}": float(v) for i, v in enumerate(joints)}
            feats["grasping"] = 1.0 if self._holder == side else 0.0
            state_dict[Object(f"{side}_arm", Kinematic3Dv2GraspArmRobotType)] = feats
        target = self._target_source.get_target()
        for node, position in ((CUBE_NODE, tuple(self._cube)), (TARGET_NODE, target)):
            state_dict[Object(node, Kinematic3Dv2PointType)] = {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        base = create_state_from_dict(
            state_dict,
            Kinematic3Dv2EnvTypeFeatures,
            state_cls=VegaPickPlace3DObjectCentricState,
        )
        assert isinstance(base, VegaPickPlace3DObjectCentricState)
        return base
