"""Tests for prpl_dexmate.remote.server, exercised through the client."""

import time
from typing import Iterator

import numpy as np
import pytest

from prpl_dexmate.interfaces.interface import HOME_HEAD_CONF, FakeInterface
from prpl_dexmate.motion import min_jerk_trajectory
from prpl_dexmate.remote.client import SkillClient, SkillServerError
from prpl_dexmate.remote.protocol import (
    DirectiveStatus,
    PolicyRolloutDirective,
    TrajectoryDirective,
)
from prpl_dexmate.remote.server import SkillServer
from prpl_dexmate.structs import NUM_HEAD_JOINTS


@pytest.fixture(name="server")
def _server_fixture() -> Iterator[SkillServer]:
    interface = FakeInterface()
    server = SkillServer(interface, host="127.0.0.1", port=0, watchdog_timeout=0.5)
    server.start()
    yield server
    server.close()
    interface.close()


@pytest.fixture(name="client")
def _client_fixture(server: SkillServer) -> Iterator[SkillClient]:
    client = SkillClient("127.0.0.1", server.port)
    yield client
    client.close()


def _slow_head_directive(num_seconds: float) -> TrajectoryDirective:
    """A directive that holds the head at home for num_seconds."""
    frames = [list(HOME_HEAD_CONF)] * int(num_seconds * 10)
    return TrajectoryDirective(component="head", trajectory=frames, hz=10.0)


def test_trajectory_directive_succeeds(
    server: SkillServer, client: SkillClient
) -> None:
    """A head trajectory executes to completion with telemetry."""
    target = [0.1] * NUM_HEAD_JOINTS
    trajectory = min_jerk_trajectory(
        np.array(HOME_HEAD_CONF), np.array(target), duration=0.2, hz=50.0
    )
    directive = TrajectoryDirective.from_array("head", trajectory, hz=50.0)
    result = client.execute_directive(directive, poll_period=0.05)
    assert result.status == DirectiveStatus.SUCCEEDED
    assert result.max_tracking_error is not None
    assert result.duration is not None
    interface = server._interface  # pylint: disable=protected-access
    assert np.allclose(interface.head_interface.get_joint_state(), target)


def test_start_error_fails(client: SkillClient) -> None:
    """A trajectory starting far from the current conf fails, not moves."""
    directive = TrajectoryDirective(
        component="head", trajectory=[[1.0] * NUM_HEAD_JOINTS], hz=10.0
    )
    result = client.execute_directive(directive, poll_period=0.05)
    assert result.status == DirectiveStatus.FAILED
    assert "trajectory start" in result.message


def test_one_directive_at_a_time(client: SkillClient) -> None:
    """A second start is rejected while a directive is running."""
    directive_id = client.start_directive(_slow_head_directive(5.0))
    with pytest.raises(SkillServerError, match="already running"):
        client.start_directive(_slow_head_directive(5.0))
    client.stop()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        result = client.get_result(directive_id)
        if result.status.is_terminal:
            break
        time.sleep(0.05)
    assert result.status == DirectiveStatus.STOPPED
    assert "stop requested" in result.message
    # After the stop, a new directive is accepted again.
    client.start_directive(_slow_head_directive(0.2))


def test_watchdog_stops_on_silence(client: SkillClient) -> None:
    """With no polling past the watchdog timeout, execution stops."""
    directive_id = client.start_directive(_slow_head_directive(5.0))
    time.sleep(1.5)  # Watchdog timeout is 0.5s in the fixture.
    result = client.get_result(directive_id)
    assert result.status == DirectiveStatus.STOPPED
    assert "watchdog" in result.message


def test_limit_violating_directives_rejected(client: SkillClient) -> None:
    """Directives breaching URDF position or velocity limits never start."""
    # Head joint 3's URDF upper limit is 1.483 rad.
    over_limit = [[0.0, 0.0, 2.0]]
    with pytest.raises(SkillServerError, match="violates limits"):
        client.start_directive(
            TrajectoryDirective(component="head", trajectory=over_limit, hz=10.0)
        )
    # 0.5 rad in one frame at 20 Hz = 10 rad/s, over the 3.2 rad/s limit.
    too_fast = [list(HOME_HEAD_CONF), [0.5, 0.0, 0.0]]
    with pytest.raises(SkillServerError, match="rad/s"):
        client.start_directive(
            TrajectoryDirective(component="head", trajectory=too_fast, hz=20.0)
        )


def test_observe_returns_joint_state(client: SkillClient) -> None:
    """The observe op reports the fake interface's home configuration."""
    observation = client.get_observation()
    assert np.allclose(observation.head_conf, HOME_HEAD_CONF)


def test_policy_rollout_not_implemented(client: SkillClient) -> None:
    """Policy rollout directives are rejected for now."""
    with pytest.raises(SkillServerError, match="not implemented"):
        client.start_directive(PolicyRolloutDirective(policy="pick"))


def test_unknown_requests_rejected(server: SkillServer, client: SkillClient) -> None:
    """Unknown ops, directive ids, and malformed directives are errors."""
    assert not server.handle_request({"op": "nope"})["ok"]
    assert not server.handle_request({"op": "start"})["ok"]
    with pytest.raises(SkillServerError, match="Unknown directive_id"):
        client.get_result(999)
