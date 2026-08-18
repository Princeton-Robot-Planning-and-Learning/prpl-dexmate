"""Tests for prpl_dexmate.real_sim.plan_executors.vega_pickplace3d."""

import numpy as np
import pytest
from kinder.envs.kinematic3d_v2.vega_pickplace3d import (
    VegaPickPlace3DObjectCentricState,
)
from numpy.typing import NDArray

from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_pickplace3d import (
    VegaPickPlace3DPerceiver,
)
from prpl_dexmate.real_sim.plan_executors.vega_pickplace3d import (
    RemoteVegaPickPlace3DPlanExecutor,
    VegaPickPlace3DPlanExecutor,
    _ArmMove,
    _GripperMove,
    decompose_plan,
)
from prpl_dexmate.remote.protocol import GripperDirective, TrajectoryDirective
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaObservation

_Plan = list[tuple[VegaPickPlace3DObjectCentricState, NDArray[np.floating]]]

HOME_RIGHT = (-1.809, -0.636, 0.244, -2.04, -0.841, -0.129, 0.833)
HOME_LEFT = (1.809, 0.636, -0.244, -2.04, 0.841, 0.129, -0.833)
RIGHT_SLICE = slice(NUM_ARM_JOINTS, 2 * NUM_ARM_JOINTS)


def _home_state() -> VegaPickPlace3DObjectCentricState:
    """A VegaPickPlace3D state at home via the perceiver (nothing held)."""
    perceiver = VegaPickPlace3DPerceiver(
        cube_source=ConstantTargetSource(0.46, -0.16, 0.647),
        target_source=ConstantTargetSource(0.46, -0.01, 0.6195),
    )
    obs = VegaObservation(
        right_arm_conf=list(HOME_RIGHT),
        left_arm_conf=list(HOME_LEFT),
        head_conf=[0.0, 0.0, 0.0],
    )
    return perceiver.reset(obs, {})


def _action(
    right_delta: NDArray[np.floating] | None = None,
    left_delta: NDArray[np.floating] | None = None,
    grasp_left: float = -1.0,
    grasp_right: float = -1.0,
) -> NDArray[np.float64]:
    action = np.zeros(2 * NUM_ARM_JOINTS + 2)
    if left_delta is not None:
        action[:NUM_ARM_JOINTS] = left_delta
    if right_delta is not None:
        action[RIGHT_SLICE] = right_delta
    action[2 * NUM_ARM_JOINTS] = grasp_left
    action[2 * NUM_ARM_JOINTS + 1] = grasp_right
    return action


def _right_reach_plan(
    num_steps: int = 3, delta: float = 0.02, close_at_end: bool = True
) -> tuple[_Plan, VegaPickPlace3DObjectCentricState]:
    """A plan moving right joint 7 by delta per step, closing at the end."""
    state = _home_state()
    robot = state.arm("right")
    trajectory = []
    for i in range(num_steps):
        d = np.zeros(NUM_ARM_JOINTS)
        d[6] = delta
        last = close_at_end and i == num_steps - 1
        trajectory.append(
            (state, _action(right_delta=d, grasp_right=1.0 if last else -1.0))
        )
        state = state.copy()
        state.set(robot, "joint_7", state.get(robot, "joint_7") + delta)
    return trajectory, state


def test_decompose_single_arm_run_then_close() -> None:
    """A right-arm reach ending in a grasp becomes one run plus one close."""
    trajectory, final_state = _right_reach_plan()
    events = decompose_plan(trajectory)
    assert len(events) == 2
    run, grip = events[0], events[1]
    assert isinstance(run, _ArmMove) and isinstance(grip, _GripperMove)
    assert run.side == "right" and len(run.waypoints) == 3
    assert np.allclose(run.waypoints[-1], final_state.arm_joint_positions("right"))
    assert grip.side == "right" and grip.action == "close"


def test_decompose_release_flip_emits_open() -> None:
    """Starting from a held cube, a negative grasp command emits an open."""
    state = _home_state()
    state.set(state.arm("right"), "grasping", 1.0)
    trajectory = [(state, _action(grasp_right=-1.0))]
    events = decompose_plan(trajectory)
    assert len(events) == 1
    assert isinstance(events[0], _GripperMove)
    assert events[0].action == "open"


def test_decompose_reasserted_commands_emit_nothing() -> None:
    """Commands that just re-assert the current grasp state are not events."""
    state = _home_state()
    trajectory = [(state, _action()), (state, _action())]
    assert not decompose_plan(trajectory)


def test_decompose_rejects_bimanual_motion() -> None:
    """A step moving both arms at once fails loudly."""
    state = _home_state()
    d = np.full(NUM_ARM_JOINTS, 0.01)
    with pytest.raises(ValueError, match="bimanual"):
        decompose_plan([(state, _action(right_delta=d, left_delta=d))])


def test_local_executor_tracks_waypoints_then_actuates_gripper() -> None:
    """The streaming executor emits arm goals, then the gripper command."""
    trajectory, final_state = _right_reach_plan()
    executor = VegaPickPlace3DPlanExecutor()
    executor.set_trajectory(trajectory)
    home = _home_state()
    assert not executor.done(home)
    action, _ = executor.step(home)
    assert action.right_arm_goal is not None and action.right_gripper is None
    # Once the arm reads at the final waypoint, the gripper event is next.
    action, _ = executor.step(final_state)
    assert action.right_arm_goal is None and action.right_gripper == "close"
    assert executor.done(final_state)


def test_remote_executor_emits_directive_per_event() -> None:
    """The remote executor emits a dense trajectory, then a gripper directive."""
    trajectory, final_state = _right_reach_plan()
    executor = RemoteVegaPickPlace3DPlanExecutor(segment_duration=0.1, hz=20.0)
    executor.set_trajectory(trajectory)
    home = _home_state()
    directive, _ = executor.step(home)
    assert isinstance(directive, TrajectoryDirective)
    assert directive.component == "right_arm"
    dense = directive.as_array()
    assert np.allclose(dense[0], home.arm_joint_positions("right"))
    assert np.allclose(dense[-1], final_state.arm_joint_positions("right"))
    assert not executor.done(home)
    directive, _ = executor.step(home)
    assert directive == GripperDirective(side="right", action="close")
    assert executor.done(home)
