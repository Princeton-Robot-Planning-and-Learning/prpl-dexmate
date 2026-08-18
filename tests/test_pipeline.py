"""Tests for prpl_dexmate.pipeline.

Configs are composed with hydra.compose, mirroring how the scripts/run_pipeline.py entry
point builds them via @hydra.main.
"""

from pathlib import Path
from typing import Iterator

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.pipeline import run_pipeline
from prpl_dexmate.remote.server import SkillServer


def _compose(
    mode: str, agent: str = "bilevel_planning", extra_overrides: list[str] | None = None
) -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name="config",
            overrides=[f"mode={mode}", "env=vega_motion3d", f"agent={agent}"]
            + (extra_overrides or []),
        )


def test_fake_mode_rollout_completes() -> None:
    """A fake-mode rollout executes the scripted plan and finishes."""
    summary = run_pipeline(_compose("fake", agent="scripted"))
    assert summary.mode == "fake"
    assert summary.env_name == "vega_motion3d"
    assert summary.steps >= 1
    assert summary.finish_reason.startswith("plan_exhausted")


def test_sim_mode_rollout_completes() -> None:
    """A sim-mode rollout drives the kinder env end-to-end."""
    summary = run_pipeline(_compose("sim", agent="scripted"))
    assert summary.mode == "sim"
    assert summary.steps >= 1
    assert summary.finish_reason.startswith(("plan_exhausted", "terminated"))
    assert summary.video_path is None


def test_sim_mode_rollout_records_video(tmp_path: Path) -> None:
    """With record.video=true and a log dir, sim produces video.mp4."""
    cfg = _compose("sim", agent="scripted")
    cfg.record.video = True
    summary = run_pipeline(cfg, log_dir=tmp_path)
    assert summary.video_path is not None
    assert summary.video_path.exists()
    assert summary.video_path.stat().st_size > 0


def test_fake_mode_records_no_video(tmp_path: Path) -> None:
    """The fake env renders nothing, so no video is produced."""
    cfg = _compose("fake", agent="scripted")
    cfg.record.video = True
    summary = run_pipeline(cfg, log_dir=tmp_path)
    assert summary.video_path is None


@pytest.fixture(name="skill_server")
def _skill_server_fixture() -> Iterator[SkillServer]:
    interface = FakeInterface()
    server = SkillServer(interface, host="127.0.0.1", port=0)
    server.start()
    yield server
    server.close()
    interface.close()


def _remote_overrides(skill_server: SkillServer, confirm: bool) -> list[str]:
    return [
        "env.pipelines.remote.real_env.host=127.0.0.1",
        f"env.pipelines.remote.real_env.port={skill_server.port}",
        "env.pipelines.remote.real_env.poll_period=0.05",
        f"env.pipelines.remote.real_env.confirm={str(confirm).lower()}",
        "env.pipelines.remote.plan_executor.segment_duration=0.05",
        "env.pipelines.remote.plan_executor.hz=20.0",
        # Must stay slow enough that the fold-to-init move (~1.3 rad on
        # the biggest joint) passes the server's velocity-limit check.
        "env.init_move_seconds=1.5",
    ]


def test_remote_mode_rollout_completes(skill_server: SkillServer) -> None:
    """A remote-mode rollout runs one directive per skill against the server (the open-
    loop joint-plan end-to-end demo, minus hardware)."""
    cfg = _compose(
        "remote",
        agent="scripted",
        extra_overrides=_remote_overrides(skill_server, confirm=False),
    )
    summary = run_pipeline(cfg)
    assert summary.mode == "remote"
    assert summary.steps >= 1
    assert summary.finish_reason.startswith("plan_exhausted")


def test_remote_confirm_gate_approval_completes(
    skill_server: SkillServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With the gate on and every directive approved, the rollout completes."""
    monkeypatch.setattr("builtins.input", lambda _: "y")
    cfg = _compose(
        "remote",
        agent="scripted",
        extra_overrides=_remote_overrides(skill_server, confirm=True),
    )
    summary = run_pipeline(cfg)
    assert summary.steps >= 1
    assert summary.finish_reason.startswith("plan_exhausted")


def test_remote_confirm_gate_rejection_stops_cleanly(
    skill_server: SkillServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rejecting the first directive (the init move) ends with no steps."""
    monkeypatch.setattr("builtins.input", lambda _: "n")
    cfg = _compose(
        "remote",
        agent="scripted",
        extra_overrides=_remote_overrides(skill_server, confirm=True),
    )
    summary = run_pipeline(cfg)
    assert summary.steps == 0
    assert summary.finish_reason.startswith("directive_rejected")


def test_fake_mode_with_bilevel_planning_executes_plan() -> None:
    """The bilevel planner plans from the init pose and the executor tracks it."""
    summary = run_pipeline(_compose("fake"))
    assert summary.steps >= 1
    assert summary.finish_reason.startswith("plan_exhausted")


def test_sim_mode_with_bilevel_planning_reaches_goal() -> None:
    """In sim the planner reaches the target and the env terminates."""
    summary = run_pipeline(_compose("sim"))
    assert summary.finish_reason == "terminated"


def test_pickplace_sim_with_bilevel_planning_reaches_goal() -> None:
    """The planner solves a VegaPickPlace3D episode end to end in sim.

    With the real-table geometry, seed 4 stages cube and target both on the robot's
    right (mirroring the first planned hardware run: a right-arm-only pick-and-place);
    planning takes tens of seconds under the env's budgets.
    """
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(
            config_name="config",
            overrides=["mode=sim", "env=vega_pickplace3d", "seed=4"],
        )
    summary = run_pipeline(cfg)
    assert summary.env_name == "vega_pickplace3d"
    assert summary.finish_reason == "terminated"
