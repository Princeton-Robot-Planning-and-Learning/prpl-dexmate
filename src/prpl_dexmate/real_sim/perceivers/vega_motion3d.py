"""Perceiver mapping VegaObservation to a VegaMotion3D sim state.

The kinder VegaMotion3D env models the right arm only (the left arm and head hold their
home values), so the state is built from the right arm's joints plus a target position
from the configured TargetSource.
"""

from typing import Any

from kinder.envs.kinematic3d_v2.object_types import (
    Kinematic3Dv2ArmRobotType,
    Kinematic3Dv2EnvTypeFeatures,
    Kinematic3Dv2PointType,
)
from kinder.envs.kinematic3d_v2.vega_motion3d import VegaMotion3DObjectCentricState
from prpl_utils.real_sim import Perceiver
from relational_structs import Object, ObjectCentricState
from relational_structs.utils import create_state_from_dict

from prpl_dexmate.real_sim.perceivers.target_source import TargetSource
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaObservation

# Object names must match the kinder env's own state so that env models
# and goal checks trained/written against sim transfer unchanged.
ROBOT_OBJECT = Object("robot", Kinematic3Dv2ArmRobotType)
TARGET_OBJECT = Object("target", Kinematic3Dv2PointType)


class VegaMotion3DPerceiver(Perceiver[VegaObservation, ObjectCentricState]):
    """Build a VegaMotion3D ObjectCentricState from a VegaObservation."""

    def __init__(self, target_source: TargetSource) -> None:
        self._target_source = target_source

    def reset(self, obs: VegaObservation, info: dict[str, Any]) -> ObjectCentricState:
        del info
        return self._perceive(obs)

    def step(self, obs: VegaObservation, info: dict[str, Any]) -> ObjectCentricState:
        del info
        return self._perceive(obs)

    def _perceive(self, obs: VegaObservation) -> ObjectCentricState:
        x, y, z = self._target_source.get_target()
        state_dict = {
            ROBOT_OBJECT: {
                f"joint_{i + 1}": obs.right_arm_conf[i] for i in range(NUM_ARM_JOINTS)
            },
            TARGET_OBJECT: {"x": x, "y": y, "z": z},
        }
        base = create_state_from_dict(state_dict, Kinematic3Dv2EnvTypeFeatures)
        # The kinder env models type-check for the env-specific state
        # subclass (it carries convenience accessors), so rebuild the
        # plain state as that class.
        return VegaMotion3DObjectCentricState(base.data, base.type_features)
