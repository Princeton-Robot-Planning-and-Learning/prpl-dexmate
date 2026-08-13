"""Tests for prpl_dexmate.remote_env."""

from typing import Iterator

import numpy as np
import pytest

from prpl_dexmate.interfaces.interface import HOME_HEAD_CONF, FakeInterface
from prpl_dexmate.remote.protocol import DirectiveStatus, TrajectoryDirective
from prpl_dexmate.remote.server import SkillServer
from prpl_dexmate.remote_env import (
    DirectiveRejected,
    RemoteVegaEnv,
    describe_directive,
)
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


def test_confirm_gate_approves_and_executes(server: SkillServer) -> None:
    """With approval, the gated directive executes and shows its summary."""
    prompts: list[str] = []

    def fake_prompt(message: str) -> str:
        prompts.append(message)
        return "y"

    env = RemoteVegaEnv(
        "127.0.0.1", server.port, poll_period=0.05, confirm=True, prompt_fn=fake_prompt
    )
    target = [0.1] * NUM_HEAD_JOINTS
    directive = TrajectoryDirective(
        component="head", trajectory=[list(HOME_HEAD_CONF), target], hz=20.0
    )
    obs, _, _, _, _ = env.step(directive)
    assert np.allclose(obs.head_conf, target)
    assert len(prompts) == 1
    assert "head" in prompts[0]
    assert "peak velocity" in prompts[0]
    env.close()


def test_confirm_gate_rejection_sends_nothing(server: SkillServer) -> None:
    """Rejection raises before anything reaches the robot."""
    env = RemoteVegaEnv(
        "127.0.0.1",
        server.port,
        poll_period=0.05,
        confirm=True,
        prompt_fn=lambda _: "n",
    )
    directive = TrajectoryDirective(
        component="head",
        trajectory=[list(HOME_HEAD_CONF), [0.1] * NUM_HEAD_JOINTS],
        hz=20.0,
    )
    with pytest.raises(DirectiveRejected):
        env.step(directive)
    interface = server._interface  # pylint: disable=protected-access
    assert np.allclose(interface.head_interface.get_joint_state(), HOME_HEAD_CONF)
    env.close()


def test_describe_directive_reports_trajectory_facts() -> None:
    """The summary includes duration, excursion, and peak velocity."""
    directive = TrajectoryDirective(
        component="head",
        trajectory=[[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]],
        hz=10.0,
    )
    summary = describe_directive(directive)
    assert "head" in summary
    assert "3 frames" in summary
    assert "0.3s" in summary
    assert "1.000 rad/s" in summary  # 0.1 rad per frame at 10 Hz.


def test_failed_directive_is_surfaced_not_raised(server: SkillServer) -> None:
    """A failing directive comes back in info, and the rollout can go on."""
    env = RemoteVegaEnv("127.0.0.1", server.port, poll_period=0.05)
    far_away = [[1.0] * NUM_HEAD_JOINTS]
    directive = TrajectoryDirective(component="head", trajectory=far_away, hz=20.0)
    obs, _, _, _, info = env.step(directive)
    assert info["directive_result"].status == DirectiveStatus.FAILED
    assert np.allclose(obs.head_conf, HOME_HEAD_CONF)
    env.close()
