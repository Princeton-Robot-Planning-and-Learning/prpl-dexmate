"""Perceiver mapping VegaObservation to a VegaPickPlace3D sim state.

The real robot cannot see the cube yet, so this is a staged-scene belief
tracker rather than perception proper: the cube's initial position and
the target patch come from configured sources (a taped spot on the
table), and from then on the cube is dead-reckoned through the grasp
cycle using the observation alone:

* An arm acquires the cube when its gripper reads closed and its end
  effector (forward kinematics on the observed joints) is within
  ``grasp_radius`` of the believed cube position.
* While held, the cube rides the holding arm rigidly, keeping the
  cube-to-end-effector offset it had at acquisition (the env's kinematic
  grasp does the same, and plans place the cube — offset included — onto
  the target, so discarding the offset would bias every set-down).
* When the holder's gripper opens, the cube is set down at the resting
  height, directly beneath its carried position.

Grasp classification is object-size independent: a gripper is "open"
when its reading is near the open position (skills only command full
open or full close, so anything else is an attempted hold at whatever
stall position the object dictates), and a hold is *verified* by the
fingers NOT closing fully — a gripper that reaches the empty-closed
reading grasped air, whatever the object was. The fake gripper snaps
to exactly closed, so fake-mode pipelines construct this perceiver
with ``verify_grasps=False``.

Remaining belief-vs-reality gap, accepted for the staged-scene stage: a
cube knocked or dropped outside the model's rules goes untracked. The
environment's release-height discipline lives in the skills, not here —
this tracker believes whatever the gripper does.
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

from prpl_dexmate.interfaces.gripper_interface import GRIPPER_OPEN_POS
from prpl_dexmate.real_sim.perceivers.target_source import TargetSource
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaObservation

# A gripper reading above this is "open" (skills command full open, which
# reads 0.785 on hardware); anything below is an attempted hold at
# whatever stall position the grasped object dictates — object-size
# independent by construction.
GRIPPER_OPEN_THRESHOLD = GRIPPER_OPEN_POS - 0.15
# A gripper reading below this closed fully: the hand is empty (measured
# empty-close is ~0.003). Used to verify grasps without any per-object
# stall constant; objects thinner than this epsilon are out of scope.
GRIPPER_EMPTY_CLOSED_EPSILON = 0.05

# Matches the env config: grasp acquisition radius
# (VegaPickPlace3DEnvConfig default) and the cube's resting height on the
# real table (table_height 0.617 + cube_half_size 0.03, per the measured
# geometry in conf/env/vega_pickplace3d.yaml).
DEFAULT_GRASP_RADIUS = 0.10
DEFAULT_CUBE_RESTING_Z = 0.647

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
        verify_grasps: bool = True,
    ) -> None:
        self._cube_source = cube_source
        self._target_source = target_source
        self._grasp_radius = grasp_radius
        self._cube_resting_z = cube_resting_z
        self._verify_grasps = verify_grasps
        robot = make_vega()
        self._tree = robot.tree
        self._home = dict(robot.home)
        self._ee_frames = {
            side: robot.manipulators[side].ee_frame for side in ARM_SIDES
        }
        self._cube = np.zeros(3)
        self._holder: str | None = None
        self._grasp_offset = np.zeros(3)

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

    def _gripper_position(self, obs: VegaObservation, side: str) -> float:
        return obs.left_gripper_pos if side == "left" else obs.right_gripper_pos

    def _gripper_open(self, obs: VegaObservation, side: str) -> bool:
        return self._gripper_position(obs, side) > GRIPPER_OPEN_THRESHOLD

    def _gripper_empty(self, obs: VegaObservation, side: str) -> bool:
        """Fingers closed fully: whatever was commanded, nothing is held."""
        if not self._verify_grasps:
            return False
        return self._gripper_position(obs, side) < GRIPPER_EMPTY_CLOSED_EPSILON

    def _update_belief(self, obs: VegaObservation) -> None:
        if self._holder is not None:
            if self._gripper_empty(obs, self._holder):
                # Fingers closed fully: the hold was lost (or never real);
                # the cube stays wherever it was last believed to be.
                self._holder = None
            elif self._gripper_open(obs, self._holder):
                # Released: set down at resting height beneath the
                # carried position.
                carried = self._ee_position(obs, self._holder) + self._grasp_offset
                self._cube = np.array([carried[0], carried[1], self._cube_resting_z])
                self._holder = None
            else:
                # Held cube rides the holding arm rigidly.
                self._cube = self._ee_position(obs, self._holder) + self._grasp_offset
                return
        for side in ARM_SIDES:
            if self._gripper_open(obs, side) or self._gripper_empty(obs, side):
                continue
            ee = self._ee_position(obs, side)
            if float(np.linalg.norm(ee - self._cube)) < self._grasp_radius:
                self._holder = side
                self._grasp_offset = self._cube - ee
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
