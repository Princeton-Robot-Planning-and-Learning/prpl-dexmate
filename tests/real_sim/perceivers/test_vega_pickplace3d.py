"""Tests for prpl_dexmate.real_sim.perceivers.vega_pickplace3d."""

import numpy as np

from prpl_dexmate.interfaces.gripper_interface import (
    GRIPPER_CLOSED_POS,
    GRIPPER_OPEN_POS,
)
from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_motion3d import (
    VegaMotion3DPerceiver,
)
from prpl_dexmate.real_sim.perceivers.vega_pickplace3d import (
    VegaPickPlace3DPerceiver,
)
from prpl_dexmate.structs import NUM_HEAD_JOINTS, VegaObservation

HOME_RIGHT = (-1.809, -0.636, 0.244, -2.04, -0.841, -0.129, 0.833)
HOME_LEFT = (1.809, 0.636, -0.244, -2.04, 0.841, 0.129, -0.833)

# A plausible reading for fingers stalled on the ~6 cm cube: between the
# empty-closed epsilon and the open threshold.
STALLED_ON_CUBE = 0.3


def _obs(
    right=HOME_RIGHT,
    left=HOME_LEFT,
    right_gripper=GRIPPER_CLOSED_POS,
    left_gripper=GRIPPER_CLOSED_POS,
) -> VegaObservation:
    return VegaObservation(
        right_arm_conf=list(right),
        left_arm_conf=list(left),
        head_conf=[0.0] * NUM_HEAD_JOINTS,
        right_gripper_pos=right_gripper,
        left_gripper_pos=left_gripper,
    )


def _make_perceiver(cube_xyz, target_xyz=(0.6, 0.5, 0.58)) -> VegaPickPlace3DPerceiver:
    return VegaPickPlace3DPerceiver(
        cube_source=ConstantTargetSource(*cube_xyz),
        target_source=ConstantTargetSource(*target_xyz),
    )


def test_reset_stages_the_scene() -> None:
    """The cube and target come from the sources; nothing is grasped."""
    perceiver = _make_perceiver((0.55, -0.3, 0.58))
    state = perceiver.reset(_obs(), {})
    cube = state.get_object_from_name("cube")
    assert np.allclose(
        [state.get(cube, "x"), state.get(cube, "y"), state.get(cube, "z")],
        [0.55, -0.3, 0.58],
    )
    assert not state.grasping("left")
    assert not state.grasping("right")
    assert np.allclose(state.arm_joint_positions("right"), HOME_RIGHT)


def test_closed_gripper_far_from_cube_does_not_grasp() -> None:
    """Parked with closed-empty grippers at home, nothing acquires the cube."""
    perceiver = _make_perceiver((0.55, -0.3, 0.58))
    perceiver.reset(_obs(), {})
    state = perceiver.step(_obs(), {})
    assert state.holder is None


def test_fully_closed_fingers_near_cube_is_a_failed_grasp() -> None:
    """Closing all the way to empty near the cube must NOT read as holding."""
    perceiver = _make_perceiver((0.55, -0.3, 0.58))
    perceiver.reset(_obs(), {})
    ee_home = perceiver._ee_position(  # pylint: disable=protected-access
        _obs(), "right"
    )
    perceiver = _make_perceiver(tuple(ee_home))
    perceiver.reset(_obs(right_gripper=GRIPPER_OPEN_POS), {})
    # The fingers sail to the empty-closed reading: grasped air.
    state = perceiver.step(_obs(right_gripper=GRIPPER_CLOSED_POS), {})
    assert state.holder is None
    # With verification off (fake mode), the same reading counts as a hold.
    lenient = VegaPickPlace3DPerceiver(
        cube_source=ConstantTargetSource(*tuple(ee_home)),
        target_source=ConstantTargetSource(0.6, 0.5, 0.58),
        verify_grasps=False,
    )
    lenient.reset(_obs(right_gripper=GRIPPER_OPEN_POS), {})
    state = lenient.step(_obs(right_gripper=GRIPPER_CLOSED_POS), {})
    assert state.holder == "right"


def test_pick_carry_place_cycle() -> None:
    """The belief tracks a full pick, carry, and set-down by the right arm."""
    perceiver = _make_perceiver((0.55, -0.3, 0.58))
    perceiver.reset(_obs(), {})
    # Find where the right ee actually is at home, and stage the cube there
    # so a grasp is in range (the test does not need IK, just consistency).
    ee_home = perceiver._ee_position(  # pylint: disable=protected-access
        _obs(), "right"
    )
    perceiver = _make_perceiver(tuple(ee_home))
    perceiver.reset(_obs(right_gripper=GRIPPER_OPEN_POS), {})

    # Fingers stall on the cube: right arm acquires it (verified hold).
    state = perceiver.step(_obs(right_gripper=STALLED_ON_CUBE), {})
    assert state.holder == "right"

    # Move the arm (wrist twist changes the ee pose): the cube rides along.
    moved = list(HOME_RIGHT)
    moved[6] += 0.5
    state = perceiver.step(_obs(right=moved, right_gripper=STALLED_ON_CUBE), {})
    cube = state.get_object_from_name("cube")
    ee_moved = perceiver._ee_position(  # pylint: disable=protected-access
        _obs(right=moved), "right"
    )
    assert np.allclose(
        [state.get(cube, "x"), state.get(cube, "y"), state.get(cube, "z")], ee_moved
    )
    assert state.holder == "right"

    # Open the gripper: cube set down at resting height beneath the ee.
    state = perceiver.step(_obs(right=moved, right_gripper=GRIPPER_OPEN_POS), {})
    assert state.holder is None
    cube = state.get_object_from_name("cube")
    assert np.isclose(state.get(cube, "x"), ee_moved[0], atol=1e-5)
    assert np.isclose(state.get(cube, "z"), 0.58, atol=1e-5)

    # Re-closing (empty) away from the cube does not re-acquire it.
    state = perceiver.step(_obs(right_gripper=GRIPPER_CLOSED_POS), {})
    assert state.holder is None


def test_ee_fk_agrees_with_motion3d_perceiver_family() -> None:
    """Sanity: the FK-backed ee position is finite and plausible at home."""
    perceiver = _make_perceiver((0.5, 0.0, 0.58))
    ee = perceiver._ee_position(_obs(), "right")  # pylint: disable=protected-access
    assert np.all(np.isfinite(ee))
    assert 0.2 < ee[0] < 1.2 and ee[2] > 0.3


def test_motion3d_perceiver_unaffected() -> None:
    """The motion3d perceiver still builds its state from the same obs type."""
    perceiver = VegaMotion3DPerceiver(ConstantTargetSource(0.5, -0.4, 0.8))
    state = perceiver.reset(_obs(), {})
    assert state.get_object_from_name("robot") is not None
