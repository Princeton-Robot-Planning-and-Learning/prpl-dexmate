"""Tests for prpl_dexmate.remote_env."""

from typing import Iterator

import numpy as np
import pytest

from prpl_dexmate.interfaces.interface import HOME_HEAD_CONF, FakeInterface
from prpl_dexmate.remote.protocol import DirectiveStatus, TrajectoryDirective
from prpl_dexmate.remote.server import SkillServer
from prpl_dexmate.remote_env import RemoteVegaEnv
from prpl_dexmate.structs import NUM_HEAD_JOINTS


@pytest.fixture(name="server")
def _server_fixture() -> Iterator[SkillServer]:
    interface = FakeInterface()
    server = SkillServer(interface, host="127.0.0.1", port=0)
    server.start()
    yield server
    server.close()
    interface.close()


def test_step_executes_directive_and_observes(server: SkillServer) -> None:
    """One env step runs a whole directive and returns the new joints."""
    env = RemoteVegaEnv("127.0.0.1", server.port, poll_period=0.05)
    obs, _ = env.reset()
    assert np.allclose(obs.head_conf, HOME_HEAD_CONF)
    target = [0.1] * NUM_HEAD_JOINTS
    directive = TrajectoryDirective(
        component="head", trajectory=[list(HOME_HEAD_CONF), target], hz=20.0
    )
    obs, reward, terminated, truncated, info = env.step(directive)
    assert np.allclose(obs.head_conf, target)
    assert float(reward) == 0.0
    assert not terminated
    assert not truncated
    assert info["directive_result"].status == DirectiveStatus.SUCCEEDED
    env.close()


def test_failed_directive_is_surfaced_not_raised(server: SkillServer) -> None:
    """A failing directive comes back in info, and the rollout can go on."""
    env = RemoteVegaEnv("127.0.0.1", server.port, poll_period=0.05)
    far_away = [[1.0] * NUM_HEAD_JOINTS]
    directive = TrajectoryDirective(component="head", trajectory=far_away, hz=20.0)
    obs, _, _, _, info = env.step(directive)
    assert info["directive_result"].status == DirectiveStatus.FAILED
    assert np.allclose(obs.head_conf, HOME_HEAD_CONF)
    env.close()
