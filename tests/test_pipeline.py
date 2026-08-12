"""Tests for prpl_dexmate.pipeline.

Configs are composed with hydra.compose, mirroring how the scripts/run_pipeline.py entry
point builds them via @hydra.main.
"""

from pathlib import Path

from hydra import compose, initialize
from omegaconf import DictConfig

from prpl_dexmate.pipeline import run_pipeline


def _compose(mode: str) -> DictConfig:
    with initialize(version_base=None, config_path="../conf"):
        return compose(
            config_name="config", overrides=[f"mode={mode}", "env=vega_motion3d"]
        )


def test_fake_mode_rollout_completes() -> None:
    """A fake-mode rollout executes the scripted plan and finishes."""
    summary = run_pipeline(_compose("fake"))
    assert summary.mode == "fake"
    assert summary.env_name == "vega_motion3d"
    assert summary.steps >= 1
    assert summary.finish_reason.startswith("plan_exhausted")


def test_sim_mode_rollout_completes() -> None:
    """A sim-mode rollout drives the kinder env end-to-end."""
    summary = run_pipeline(_compose("sim"))
    assert summary.mode == "sim"
    assert summary.steps >= 1
    assert summary.finish_reason.startswith(("plan_exhausted", "terminated"))
    assert summary.video_path is None


def test_sim_mode_rollout_records_video(tmp_path: Path) -> None:
    """With record.video=true and a log dir, sim produces video.mp4."""
    cfg = _compose("sim")
    cfg.record.video = True
    summary = run_pipeline(cfg, log_dir=tmp_path)
    assert summary.video_path is not None
    assert summary.video_path.exists()
    assert summary.video_path.stat().st_size > 0


def test_fake_mode_records_no_video(tmp_path: Path) -> None:
    """The fake env renders nothing, so no video is produced."""
    cfg = _compose("fake")
    cfg.record.video = True
    summary = run_pipeline(cfg, log_dir=tmp_path)
    assert summary.video_path is None
