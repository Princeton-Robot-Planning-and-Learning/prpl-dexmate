"""Plan executors for VegaPickPlace3D bimanual pick-and-place plans.

A VegaPickPlace3D plan step is a 16-vector: joint deltas for the left
then the right arm, then a grasp command per arm (positive = hold). The
env applies these simultaneously, but our directives are
single-component, so ``decompose_plan`` rewrites a plan as an ordered
event sequence: contiguous single-arm motion runs, broken wherever a
grasp command changes sign, with a gripper open/close event at each
flip. The kinder skills move at most one arm per step and re-assert
every grasp command until their final toggle, so a skill trajectory
decomposes into a handful of events; a plan that moves both arms in the
same step is rejected loudly rather than silently serialized.

``VegaPickPlace3DPlanExecutor`` tracks the events tick-by-tick against a
streaming env (fake or real); ``RemoteVegaPickPlace3DPlanExecutor``
emits one directive per event for the skill-server pipeline, with each
arm run time-parameterized as a dense min-jerk trajectory from the
currently perceived joints (same scheme as the motion3d remote
executor).
"""

from dataclasses import dataclass, field

import numpy as np
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    ARM_SIDES,
    VegaPickPlace3DObjectCentricState,
)
from numpy.typing import NDArray
from prpl_utils.real_sim import PlanExecutor

from prpl_dexmate.motion import waypoint_trajectory
from prpl_dexmate.remote.protocol import GripperDirective, TrajectoryDirective
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaAction

_ARM_SLICE = {
    side: slice(i * NUM_ARM_JOINTS, (i + 1) * NUM_ARM_JOINTS)
    for i, side in enumerate(ARM_SIDES)
}
_GRASP_INDEX = {side: 2 * NUM_ARM_JOINTS + i for i, side in enumerate(ARM_SIDES)}
# Deltas below this are "the arm is not moving" (idle arms get exact
# zeros from the skills; this only guards float noise).
_MOTION_EPS = 1e-6

_Trajectory = list[tuple[VegaPickPlace3DObjectCentricState, NDArray[np.floating]]]


@dataclass
class _ArmMove:
    """A contiguous run of single-arm waypoints."""

    side: str
    waypoints: list[NDArray[np.float64]] = field(default_factory=list)
    sim_action: NDArray[np.floating] | None = None


@dataclass
class _GripperMove:
    """A single gripper open/close at a grasp-command flip."""

    side: str
    action: str
    sim_action: NDArray[np.floating] | None = None


def decompose_plan(trajectory: _Trajectory) -> list[_ArmMove | _GripperMove]:
    """Rewrite a bimanual plan as an ordered single-component event list.

    Each step's intended absolute waypoint (state joints + delta) extends
    the current arm run; a grasp-command sign change emits a gripper
    event after that step's motion (arrive, then actuate) and breaks the
    run. The initial grasp commands are implied by the first state's
    grasping flags, so a plan that merely re-asserts an existing hold
    emits no gripper event for it.
    """
    events: list[_ArmMove | _GripperMove] = []
    run: _ArmMove | None = None
    first_state = trajectory[0][0]
    previous_command = {
        side: 1.0 if first_state.grasping(side) else -1.0 for side in ARM_SIDES
    }
    for state, action in trajectory:
        action = np.asarray(action, dtype=float)
        moving = [
            side
            for side in ARM_SIDES
            if float(np.max(np.abs(action[_ARM_SLICE[side]]))) > _MOTION_EPS
        ]
        if len(moving) > 1:
            raise ValueError(
                "Simultaneous bimanual motion cannot be decomposed into "
                "single-component directives"
            )
        if moving:
            side = moving[0]
            waypoint = (
                np.asarray(state.arm_joint_positions(side), dtype=float)
                + action[_ARM_SLICE[side]]
            )
            if run is None or run.side != side:
                if run is not None:
                    events.append(run)
                run = _ArmMove(side=side)
            run.waypoints.append(waypoint)
            run.sim_action = action
        for side in ARM_SIDES:
            command = float(action[_GRASP_INDEX[side]])
            if (command > 0) != (previous_command[side] > 0):
                if run is not None:
                    events.append(run)
                    run = None
                events.append(
                    _GripperMove(
                        side=side,
                        action="close" if command > 0 else "open",
                        sim_action=action,
                    )
                )
            previous_command[side] = command
    if run is not None:
        events.append(run)
    return events


def _state_joints(
    state: VegaPickPlace3DObjectCentricState, side: str
) -> NDArray[np.float64]:
    return np.asarray(state.arm_joint_positions(side), dtype=float)


class VegaPickPlace3DPlanExecutor(
    PlanExecutor[NDArray[np.floating], VegaAction, VegaPickPlace3DObjectCentricState]
):
    """Track a decomposed pick-and-place plan waypoint-by-waypoint.

    Arm runs are followed like the motion3d executor (advance past
    waypoints already within ``advance_radius``, an event completes when
    its final waypoint is within ``tolerance``); gripper events are
    issued once and complete on the next tick, when the observation
    reflects the actuated gripper.
    """

    def __init__(self, advance_radius: float = 0.05, tolerance: float = 0.05) -> None:
        self._advance_radius = advance_radius
        self._tolerance = tolerance
        self._events: list[_ArmMove | _GripperMove] = []
        self._event_index = 0
        self._waypoint_index = 0
        self._gripper_issued = False

    def set_trajectory(self, trajectory: _Trajectory) -> None:
        self._events = decompose_plan(trajectory) if trajectory else []
        self._event_index = 0
        self._waypoint_index = 0
        self._gripper_issued = False

    def _advance(self, state: VegaPickPlace3DObjectCentricState) -> None:
        while self._event_index < len(self._events):
            event = self._events[self._event_index]
            if isinstance(event, _GripperMove):
                if not self._gripper_issued:
                    return
            else:
                current = _state_joints(state, event.side)
                while (
                    self._waypoint_index < len(event.waypoints) - 1
                    and float(
                        np.max(np.abs(event.waypoints[self._waypoint_index] - current))
                    )
                    < self._advance_radius
                ):
                    self._waypoint_index += 1
                last = event.waypoints[-1]
                if (
                    self._waypoint_index < len(event.waypoints) - 1
                    or float(np.max(np.abs(last - current))) >= self._tolerance
                ):
                    return
            self._event_index += 1
            self._waypoint_index = 0
            self._gripper_issued = False

    def step(
        self, sim_state: VegaPickPlace3DObjectCentricState
    ) -> tuple[VegaAction, NDArray[np.floating]]:
        self._advance(sim_state)
        event = self._events[self._event_index]
        assert event.sim_action is not None
        if isinstance(event, _GripperMove):
            self._gripper_issued = True
            gripper_action = (
                VegaAction(right_gripper=event.action)
                if event.side == "right"
                else VegaAction(left_gripper=event.action)
            )
            return gripper_action, event.sim_action
        waypoint = event.waypoints[self._waypoint_index]
        goal = [float(v) for v in waypoint]
        arm_action = (
            VegaAction(right_arm_goal=goal)
            if event.side == "right"
            else VegaAction(left_arm_goal=goal)
        )
        return arm_action, event.sim_action

    def done(self, sim_state: VegaPickPlace3DObjectCentricState) -> bool:
        if not self._events:
            return True
        self._advance(sim_state)
        return self._event_index >= len(self._events)


class RemoteVegaPickPlace3DPlanExecutor(
    PlanExecutor[
        NDArray[np.floating],
        TrajectoryDirective | GripperDirective,
        VegaPickPlace3DObjectCentricState,
    ]
):
    """Send a decomposed pick-and-place plan as one directive per event.

    Arm runs become dense min-jerk TrajectoryDirectives starting at the
    perceived current joints and pausing momentarily at each planner
    waypoint; gripper events become GripperDirectives. Closed-loop
    tracking happens on the robot; here each ``step`` just emits the next
    event's directive.
    """

    def __init__(self, segment_duration: float = 1.0, hz: float = 50.0) -> None:
        self._segment_duration = segment_duration
        self._hz = hz
        self._events: list[_ArmMove | _GripperMove] = []
        self._event_index = 0

    def set_trajectory(self, trajectory: _Trajectory) -> None:
        self._events = decompose_plan(trajectory) if trajectory else []
        self._event_index = 0

    def step(
        self, sim_state: VegaPickPlace3DObjectCentricState
    ) -> tuple[TrajectoryDirective | GripperDirective, NDArray[np.floating]]:
        event = self._events[self._event_index]
        self._event_index += 1
        assert event.sim_action is not None
        if isinstance(event, _GripperMove):
            return GripperDirective(side=event.side, action=event.action), (
                event.sim_action
            )
        current = _state_joints(sim_state, event.side)
        dense = waypoint_trajectory(
            [current] + event.waypoints, self._segment_duration, self._hz
        )
        directive = TrajectoryDirective.from_array(
            f"{event.side}_arm", dense, hz=self._hz
        )
        return directive, event.sim_action

    def done(self, sim_state: VegaPickPlace3DObjectCentricState) -> bool:
        return self._event_index >= len(self._events)
