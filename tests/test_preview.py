"""Tests for prpl_dexmate.preview."""

from pathlib import Path

import numpy as np

from prpl_dexmate.motion import min_jerk_trajectory
from prpl_dexmate.preview import DirectivePreviewer
from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource
from prpl_dexmate.real_sim.perceivers.vega_motion3d import VegaMotion3DPerceiver
from prpl_dexmate.remote.protocol import TrajectoryDirective
from prpl_dexmate.sim_env import KinderSimEnv
from prpl_dexmate.structs import NUM_HEAD_JOINTS

INIT_CONF = np.array([-1.809, -0.636, 0.244, -2.04, -0.841, -0.129, 0.833])


def _make_previewer() -> DirectivePreviewer:
    return DirectivePreviewer(
        shadow_env=KinderSimEnv("kinder/VegaMotion3D-v0", allow_state_access=True),
        perceiver=VegaMotion3DPerceiver(ConstantTargetSource(0.5, -0.4, 0.8)),
        fps=10,
    )


def test_right_arm_directive_renders_video(tmp_path: Path) -> None:
    """A right-arm directive produces a non-empty mp4; paths increment."""
    previewer = _make_previewer()
    trajectory = min_jerk_trajectory(INIT_CONF, INIT_CONF + 0.1, duration=0.5, hz=20.0)
    directive = TrajectoryDirective.from_array("right_arm", trajectory, hz=20.0)
    path = previewer.render_directive(directive, tmp_path)
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0
    second = previewer.render_directive(directive, tmp_path)
    assert second is not None
    assert second != path
    previewer.close()


def test_other_components_fall_back_to_text_only(tmp_path: Path) -> None:
    """Head directives are not renderable (the sim models the right arm)."""
    previewer = _make_previewer()
    directive = TrajectoryDirective(
        component="head", trajectory=[[0.0] * NUM_HEAD_JOINTS], hz=10.0
    )
    assert previewer.render_directive(directive, tmp_path) is None
    previewer.close()
