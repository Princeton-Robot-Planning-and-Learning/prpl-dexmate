"""Tests for prpl_dexmate.pipeline.

Configs are composed with hydra.compose, mirroring how the scripts/run_pipeline.py entry
point builds them via @hydra.main.
"""

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
