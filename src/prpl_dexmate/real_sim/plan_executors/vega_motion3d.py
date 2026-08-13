"""Plan executor for VegaMotion3D right-arm joint trajectories.

A VegaMotion3D trajectory is a sequence of (state, action) pairs whose
kinder action is a 7-vector of right-arm joint deltas. Each pair's
intended absolute joint target (state joints + delta) is precomputed at
``set_trajectory`` time; per tick, the cursor advances past any waypoints
already within ``advance_radius`` of the perceived joints, and the
current waypoint is emitted as a VegaAction. Done when the final waypoint
is within ``tolerance``.
"""

import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor
from relational_structs import ObjectCentricState

from prpl_dexmate.motion import waypoint_trajectory
from prpl_dexmate.real_sim.perceivers.vega_motion3d import ROBOT_OBJECT
from prpl_dexmate.remote.protocol import TrajectoryDirective
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaAction


def _state_joints(state: ObjectCentricState) -> NDArray[np.float64]:
    robot = state.get_object_from_name(ROBOT_OBJECT.name)
    return np.array([state.get(robot, f"joint_{i + 1}") for i in range(NUM_ARM_JOINTS)])


class VegaMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], VegaAction, ObjectCentricState]
):
    """Track a planned right-arm joint trajectory waypoint-by-waypoint."""

    def __init__(self, advance_radius: float = 0.05, tolerance: float = 0.05) -> None:
        self._advance_radius = advance_radius
        self._tolerance = tolerance
        self._waypoints: list[NDArray[np.float64]] = []
        self._sim_actions: list[NDArray[np.floating]] = []
        self._cursor = 0

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        self._waypoints = [
            _state_joints(state) + np.asarray(action, dtype=float)
            for state, action in trajectory
        ]
        self._sim_actions = [action for _, action in trajectory]
        self._cursor = 0

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[VegaAction, NDArray[np.floating]]:
        current = _state_joints(sim_state)
        # Advance past waypoints we are already close to, but always keep
        # the final waypoint as the last target.
        while (
            self._cursor < len(self._waypoints) - 1
            and np.max(np.abs(self._waypoints[self._cursor] - current))
            < self._advance_radius
        ):
            self._cursor += 1
        waypoint = self._waypoints[self._cursor]
        real_action = VegaAction(right_arm_goal=[float(v) for v in waypoint])
        return real_action, self._sim_actions[self._cursor]

    def done(self, sim_state: ObjectCentricState) -> bool:
        if not self._waypoints:
            return True
        current = _state_joints(sim_state)
        return (
            self._cursor >= len(self._waypoints) - 1
            and np.max(np.abs(self._waypoints[-1] - current)) < self._tolerance
        )


class RemoteVegaMotion3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], TrajectoryDirective, ObjectCentricState]
):
    """Send a planned right-arm trajectory as one directive per skill.

    The remote counterpart of VegaMotion3DPlanExecutor: rather than
    emitting one waypoint per policy tick, the whole plan is
    time-parameterized into a dense min-jerk trajectory (starting at the
    perceived current joints and pausing momentarily at each planner
    waypoint) and returned as a single TrajectoryDirective. The executor
    is done after that one step; closed-loop tracking happens on the
    robot, not across the network.
    """

    def __init__(self, segment_duration: float = 1.0, hz: float = 50.0) -> None:
        self._segment_duration = segment_duration
        self._hz = hz
        self._waypoints: list[NDArray[np.float64]] = []
        self._sim_actions: list[NDArray[np.floating]] = []
        self._issued = False

    def set_trajectory(
        self,
        trajectory: list[tuple[ObjectCentricState, NDArray[np.floating]]],
    ) -> None:
        self._waypoints = [
            _state_joints(state) + np.asarray(action, dtype=float)
            for state, action in trajectory
        ]
        self._sim_actions = [action for _, action in trajectory]
        self._issued = False

    def step(
        self, sim_state: ObjectCentricState
    ) -> tuple[TrajectoryDirective, NDArray[np.floating]]:
        current = _state_joints(sim_state)
        dense = waypoint_trajectory(
            [current] + self._waypoints, self._segment_duration, self._hz
        )
        directive = TrajectoryDirective.from_array("right_arm", dense, hz=self._hz)
        self._issued = True
        return directive, self._sim_actions[-1]

    def done(self, sim_state: ObjectCentricState) -> bool:
        return self._issued or not self._waypoints
