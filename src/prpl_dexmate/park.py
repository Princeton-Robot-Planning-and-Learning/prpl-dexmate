"""Plan safe arm-parking moves between the shipping fold and the model home.

Operational rules this module encodes (established during the 2026-08-13
bring-up):

* The model home is the resting pose while powered (collision-free with
  grippers, valid planning start; the remote pipeline's init move from it
  is null).
* The shipping fold is the power-off pose: the arms rest on mechanical
  end-stops there, so nothing sags when motor power cuts. Whether the
  joints hold position unpowered elsewhere is unverified, so fold before
  every power-off.
* All parking motion routes through home, one arm at a time, and every
  straight-line segment is checked against the collision model (gripper
  links excluded, matching the gripper-less robot) from the arms'
  *observed* positions before anything moves.

``plan_parking_moves`` produces the move list; executing it is the job of
``scripts/park_arms.py``.
"""

from dataclasses import dataclass

import numpy as np
from kinder.envs.kinematic3d_v2.vega_motion3d import ObjectCentricVegaMotion3DEnv
from numpy.typing import NDArray

from prpl_dexmate.interfaces.interface import (
    FOLDED_LEFT_ARM_CONF,
    FOLDED_RIGHT_ARM_CONF,
)

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
    """Plan fold/home parking moves, collision-checked against the model.

    Home confs come from the robot model at runtime (not hardcoded), so
    changes to the model home propagate here automatically.
    """

    def __init__(self) -> None:
        self._env = ObjectCentricVegaMotion3DEnv()
        self._env.reset(seed=0)
        self._checker = self._env._collision_checker  # pylint: disable=protected-access
        self._base_config = dict(
            self._env._configuration
        )  # pylint: disable=protected-access
        self._gripper_nodes = [n for n in self._env.tree.nodes if "gripper" in n]
        home = self._env.robot.home
        self.home_right = np.array([home[f"R_arm_j{i}"][0] for i in range(1, 8)])
        self.home_left = np.array([home[f"L_arm_j{i}"][0] for i in range(1, 8)])
        self.fold_right = np.array(FOLDED_RIGHT_ARM_CONF)
        self.fold_left = np.array(FOLDED_LEFT_ARM_CONF)

    def close(self) -> None:
        """Tear down the sim env."""
        self._env.close()

    def _clear(self, right: NDArray[np.float64], left: NDArray[np.float64]) -> bool:
        config = dict(self._base_config)
        for i in range(7):
            config[f"R_arm_j{i+1}"] = [float(right[i])]
            config[f"L_arm_j{i+1}"] = [float(left[i])]
        return not self._checker.in_collision(config, ignored_nodes=self._gripper_nodes)

    def _path_clear(self, start_r, start_l, end_r, end_l) -> bool:
        return all(
            self._clear(
                start_r + t * (end_r - start_r), start_l + t * (end_l - start_l)
            )
            for t in np.linspace(0.0, 1.0, 100)
        )

    def plan_parking_moves(
        self,
        current_right: NDArray[np.float64],
        current_left: NDArray[np.float64],
        goal: str,
    ) -> list[ParkingMove]:
        """Moves taking both arms from their current confs to ``goal``.

        ``goal`` is ``"home"`` or ``"fold"``. Each arm routes through home
        (skipping segments it is already at), right arm first. Raises
        ParkingBlocked if any segment is not collision-free in sim.
        """
        assert goal in ("home", "fold")
        goal_right = self.home_right if goal == "home" else self.fold_right
        goal_left = self.home_left if goal == "home" else self.fold_left
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
