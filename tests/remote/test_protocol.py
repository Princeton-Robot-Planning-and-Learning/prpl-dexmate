"""Tests for prpl_dexmate.remote.protocol."""

import numpy as np
import pytest

from prpl_dexmate.remote.protocol import (
    PROTOCOL_VERSION,
    DirectiveResult,
    DirectiveStatus,
    GripperDirective,
    PolicyRolloutDirective,
    TrajectoryDirective,
    deserialize_message,
    serialize_message,
)
from prpl_dexmate.structs import NUM_ARM_JOINTS, NUM_HEAD_JOINTS, VegaObservation


def test_trajectory_directive_round_trip() -> None:
    """A trajectory directive survives JSON serialization."""
    directive = TrajectoryDirective(
        component="right_arm",
        trajectory=[[0.1] * NUM_ARM_JOINTS, [0.2] * NUM_ARM_JOINTS],
        hz=50.0,
    )
    assert deserialize_message(serialize_message(directive)) == directive


def test_trajectory_directive_validates() -> None:
    """Bad components, joint counts, and rates are rejected."""
    frame = [0.0] * NUM_ARM_JOINTS
    with pytest.raises(AssertionError):
        TrajectoryDirective(component="torso", trajectory=[frame])
    with pytest.raises(AssertionError):
        TrajectoryDirective(component="head", trajectory=[frame])
    with pytest.raises(AssertionError):
        TrajectoryDirective(component="left_arm", trajectory=[])
    with pytest.raises(AssertionError):
        TrajectoryDirective(component="left_arm", trajectory=[frame], hz=0.0)


def test_trajectory_array_round_trip() -> None:
    """from_array and as_array invert each other."""
    trajectory = np.linspace(np.zeros(NUM_HEAD_JOINTS), np.ones(NUM_HEAD_JOINTS), num=5)
    directive = TrajectoryDirective.from_array("head", trajectory, hz=20.0)
    assert directive.hz == 20.0
    assert np.allclose(directive.as_array(), trajectory)


def test_gripper_directive_round_trip_and_validation() -> None:
    """Gripper directives serialize and reject bad sides/actions."""
    directive = GripperDirective(side="left", action="open")
    assert deserialize_message(serialize_message(directive)) == directive
    with pytest.raises(AssertionError):
        GripperDirective(side="torso", action="open")
    with pytest.raises(AssertionError):
        GripperDirective(side="left", action="wave")


def test_policy_rollout_directive_round_trip() -> None:
    """A policy rollout directive survives JSON serialization."""
    directive = PolicyRolloutDirective(
        policy="pick", policy_args={"object": "mug"}, timeout=10.0
    )
    assert deserialize_message(serialize_message(directive)) == directive


def test_directive_result_round_trip() -> None:
    """Results with every status survive JSON serialization."""
    for status in DirectiveStatus:
        result = DirectiveResult(
            status=status, message="context", max_tracking_error=0.01, duration=2.5
        )
        assert deserialize_message(serialize_message(result)) == result


def test_observation_round_trip() -> None:
    """A VegaObservation survives JSON serialization."""
    observation = VegaObservation(
        right_arm_conf=[0.1] * NUM_ARM_JOINTS,
        left_arm_conf=[-0.1] * NUM_ARM_JOINTS,
        head_conf=[0.0] * NUM_HEAD_JOINTS,
    )
    assert deserialize_message(serialize_message(observation)) == observation


def test_directive_status_terminal() -> None:
    """RUNNING is the only non-terminal status."""
    assert not DirectiveStatus.RUNNING.is_terminal
    assert DirectiveStatus.SUCCEEDED.is_terminal
    assert DirectiveStatus.STOPPED.is_terminal


def test_deserialize_rejects_unknown_type() -> None:
    """An unknown or missing type tag raises ValueError."""
    with pytest.raises(ValueError):
        deserialize_message('{"type": "not_a_message"}')
    with pytest.raises(ValueError):
        deserialize_message('{"component": "head"}')


def test_deserialize_rejects_unknown_fields() -> None:
    """Fields outside the schema raise TypeError (version-skew guard)."""
    directive = TrajectoryDirective(
        component="head", trajectory=[[0.0] * NUM_HEAD_JOINTS]
    )
    tampered = serialize_message(directive).replace(
        '"hz": 100.0', '"hz": 100.0, "new_field": 1'
    )
    with pytest.raises(TypeError):
        deserialize_message(tampered)


def test_protocol_version_format() -> None:
    """The protocol version is a short deterministic hex digest."""
    assert len(PROTOCOL_VERSION) == 12
    int(PROTOCOL_VERSION, 16)
