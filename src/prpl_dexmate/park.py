"""Plan safe arm-parking moves between the shipping fold and the model home.

Operational rules this module encodes (established during the 2026-08-13
bring-up):

* The model home is the resting pose while powered (collision-free with
  grippers, valid planning start; the remote pipeline's init move from it
  is null).
* The shipping fold is the gripper-less power-off pose: the arms rest on
  mechanical end-stops there, so nothing sags when motor power cuts.
  Whether the joints hold position unpowered elsewhere is unverified, so
  fold (or storage, below) before every power-off.
* Once grippers are mounted the shipping fold self-collides and is
  refused; the storage pose (the fold with the forearm-roll joints
  turned so the grippers face outward) inherits the end-stop property
  and replaces it.
* All parking motion routes through home, one arm at a time, and every
  straight-line segment is checked against the collision model (gripper
  links excluded, matching the gripper-less robot) from the arms'
  *observed* positions before anything moves.

``plan_parking_moves`` produces the move list; executing it is the job of
``scripts/park_arms.py``.
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import dexmate_urdf
import numpy as np
from kinder.envs.kinematic3d_v2.vega_motion3d import ObjectCentricVegaMotion3DEnv
from numpy.typing import NDArray

from prpl_dexmate.interfaces.interface import (
    FOLDED_LEFT_ARM_CONF,
    FOLDED_RIGHT_ARM_CONF,
)
from prpl_dexmate.limits import get_joint_limits


def clip_conf_to_limits(
    conf: NDArray[np.float64], component: str
) -> tuple[NDArray[np.float64], float]:
    """Clip a configuration into URDF limits; return it and the overshoot.

    Physical service can back-drive unpowered joints past their soft limits (gripper
    mounting pushed both wrist rolls ~0.15 rad out). A trajectory may not contain out-
    of-limit positions, so a recovery move must *start* at the clipped boundary; the
    returned overshoot is how far the real joint sits beyond it, which the caller adds
    to the directive's start-error allowance. The first setpoint then pulls the joint
    back inside its range, the same way a joint resettles off a mechanical end-stop.
    """
    lower, upper, _ = get_joint_limits(component)
    margin = 0.01  # Command just inside the limit, not exactly on it.
    clipped = np.clip(conf, lower + margin, upper - margin)
    overshoot = float(np.max(np.abs(conf - clipped)))
    return clipped, overshoot


def vendor_allowed_collision_pairs() -> list[tuple[str, str]]:
    """Link pairs DexMate's SRDF declares as expected-touching.

    The robot model's own allowed-pair list is discovered at the home
    pose, which misses pairs whose coarse collision hulls only overlap at
    other joint angles (e.g. the wrist links ``arm_l5``/``arm_l7`` at
    some rolls — encountered on hardware when gripper mounting
    back-drove the wrists). The vendor's SRDF is authoritative about
    which link pairs may legitimately touch across the whole range.
    """
    srdf = (
        Path(dexmate_urdf.__file__).parent
        / "robots/humanoid/vega_1u/vega_1u_gripper.srdf"
    )
    return [
        (str(entry.get("link1")), str(entry.get("link2")))
        for entry in ET.parse(srdf).getroot().iter("disable_collisions")
    ]


# The gripper-safe compact fold: the shipping fold with only the
# forearm-roll joints (j5) turned so the grippers face outward instead of
# into each other (11.7 cm modeled cross clearance). The elbows stay on
# their mechanical end-stops, so this remains a safe power-off pose. Once
# grippers are mounted this REPLACES the shipping fold, which
# self-collides with gripper geometry (see the parked-pose analysis,
# 2026-08-13, and prpl-mono#532 for the gripper model provenance).
_STORAGE_J5_RIGHT = -1.0
_STORAGE_J5_LEFT = 2.0

# A joint already within this distance of its target needs no move.
_SKIP_TOLERANCE = 0.05
# Move durations scale so the min-jerk peak velocity stays near 0.3 rad/s.
_SECONDS_PER_RADIAN = 6.5
_MIN_MOVE_SECONDS = 4.0


class ParkingBlocked(Exception):
    """A required straight-line segment is not collision-free in sim."""


@dataclass(frozen=True)
class ParkingMove:
    """One single-arm min-jerk move for the executor to run."""

    component: str
    start: NDArray[np.float64]
    end: NDArray[np.float64]

    @property
    def seconds(self) -> float:
        """A duration keeping the min-jerk peak velocity near 0.3 rad/s."""
        max_delta = float(np.max(np.abs(self.end - self.start)))
        return max(_MIN_MOVE_SECONDS, _SECONDS_PER_RADIAN * max_delta)


class ParkingPlanner:
    """Plan fold/home/storage parking moves, collision-checked in sim.

    Home confs come from the robot model at runtime (not hardcoded), so
    changes to the model home propagate here automatically.

    With ``grippers_mounted=True``, collision checks include the gripper
    links (before mounting they are phantoms and are excluded), and the
    shipping fold is refused as a goal — with grippers it is a real
    self-collision. The storage pose is the gripper-safe replacement.
    """

    def __init__(self, grippers_mounted: bool = False) -> None:
        self._env = ObjectCentricVegaMotion3DEnv()
        self._env.reset(seed=0)
        self._checker = self._env._collision_checker  # pylint: disable=protected-access
        self._checker.ignore(vendor_allowed_collision_pairs())
        self._base_config = dict(
            self._env._configuration
        )  # pylint: disable=protected-access
        self.grippers_mounted = grippers_mounted
        if grippers_mounted:
            self._ignored_nodes: list[str] = []
        else:
            self._ignored_nodes = [n for n in self._env.tree.nodes if "gripper" in n]
        home = self._env.robot.home
        self.home_right = np.array([home[f"R_arm_j{i}"][0] for i in range(1, 8)])
        self.home_left = np.array([home[f"L_arm_j{i}"][0] for i in range(1, 8)])
        self.fold_right = np.array(FOLDED_RIGHT_ARM_CONF)
        self.fold_left = np.array(FOLDED_LEFT_ARM_CONF)
        self.storage_right = self.fold_right.copy()
        self.storage_right[4] = _STORAGE_J5_RIGHT
        self.storage_left = self.fold_left.copy()
        self.storage_left[4] = _STORAGE_J5_LEFT

    def close(self) -> None:
        """Tear down the sim env."""
        self._env.close()

    def _clear(self, right: NDArray[np.float64], left: NDArray[np.float64]) -> bool:
        config = dict(self._base_config)
        for i in range(7):
            config[f"R_arm_j{i+1}"] = [float(right[i])]
            config[f"L_arm_j{i+1}"] = [float(left[i])]
        return not self._checker.in_collision(config, ignored_nodes=self._ignored_nodes)

    def _path_clear(self, start_r, start_l, end_r, end_l) -> bool:
        return all(
            self._clear(
                start_r + t * (end_r - start_r), start_l + t * (end_l - start_l)
            )
            for t in np.linspace(0.0, 1.0, 100)
        )

    def parking_deviation(
        self,
        right: NDArray[np.float64],
        left: NDArray[np.float64],
        goal: str,
    ) -> float:
        """The max per-joint distance of the observed confs from ``goal``.

        The final-verdict check for the parking script: a small value
        means the robot really is parked where it was asked to go. Added
        after an incident where two directives failed mid-route, the
        script marched on, and the arms were quietly left ~60% of the
        way to home for days (2026-08-14).
        """
        goals = {
            "home": (self.home_right, self.home_left),
            "fold": (self.fold_right, self.fold_left),
            "storage": (self.storage_right, self.storage_left),
        }
        goal_right, goal_left = goals[goal]
        return float(
            max(np.max(np.abs(right - goal_right)), np.max(np.abs(left - goal_left)))
        )

    def plan_parking_moves(
        self,
        current_right: NDArray[np.float64],
        current_left: NDArray[np.float64],
        goal: str,
    ) -> list[ParkingMove]:
        """Moves taking both arms from their current confs to ``goal``.

        ``goal`` is ``"home"``, ``"fold"``, or ``"storage"``. Each arm
        routes through home (skipping segments it is already at), right
        arm first. Raises ParkingBlocked if any segment is not
        collision-free in sim, and ValueError for the fold goal with
        grippers mounted (a real self-collision, not merely blocked).
        """
        assert goal in ("home", "fold", "storage")
        if goal == "fold" and self.grippers_mounted:
            raise ValueError(
                "The shipping fold self-collides with grippers mounted; "
                "park to 'storage' instead."
            )
        goals = {
            "home": (self.home_right, self.home_left),
            "fold": (self.fold_right, self.fold_left),
            "storage": (self.storage_right, self.storage_left),
        }
        goal_right, goal_left = goals[goal]
        moves: list[ParkingMove] = []
        # Track where each arm is as the move sequence progresses, so each
        # segment is checked with the other arm where it will actually be.
        position = {"right_arm": current_right, "left_arm": current_left}
        for component, home, goal_conf in (
            ("right_arm", self.home_right, goal_right),
            ("left_arm", self.home_left, goal_left),
        ):
            if float(np.max(np.abs(goal_conf - position[component]))) < _SKIP_TOLERANCE:
                continue  # Already at the goal; no detour through home.
            waypoints = [home] if goal == "home" else [home, goal_conf]
            for waypoint in waypoints:
                start = position[component]
                if float(np.max(np.abs(waypoint - start))) < _SKIP_TOLERANCE:
                    position[component] = waypoint
                    continue
                right = position["right_arm"]
                left = position["left_arm"]
                if component == "right_arm":
                    end_r, end_l = waypoint, left
                else:
                    end_r, end_l = right, waypoint
                if not self._path_clear(right, left, end_r, end_l):
                    raise ParkingBlocked(
                        f"{component} straight-line segment to "
                        f"{np.round(waypoint, 3).tolist()} is not collision-free "
                        "in sim; resolve manually before parking."
                    )
                moves.append(ParkingMove(component, start, waypoint))
                position[component] = waypoint
        return moves
