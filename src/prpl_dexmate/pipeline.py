"""Glue that wires a Hydra config into a Runner rollout.

The ``scripts/run_pipeline.py`` entry point delegates to ``run_pipeline``
here; tests compose configs with ``hydra.compose`` and call
``run_pipeline`` directly, without going through the Hydra ``@main``
decorator.
"""

from dataclasses import dataclass
from pathlib import Path

import hydra
import numpy as np
from hydra.core.hydra_config import HydraConfig
from kinder_bilevel_planning.agent import AgentFailure
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner

from prpl_dexmate.agents import PlanExhausted
from prpl_dexmate.motion import min_jerk_trajectory
from prpl_dexmate.recording import RecordingRunner, VideoRecorder
from prpl_dexmate.remote.protocol import TrajectoryDirective
from prpl_dexmate.remote_env import DirectiveRejected
from prpl_dexmate.structs import VegaAction

# Frame rate for the remote-mode init move directive.
INIT_MOVE_HZ = 50.0


@dataclass(frozen=True)
class RolloutSummary:
    """Result of one rollout — handy for assertions in tests."""

    env_name: str
    mode: str
    seed: int
    steps: int
    finish_reason: str
    total_reward: float
    video_path: Path | None = None


def run_pipeline(cfg: DictConfig, log_dir: Path | str | None = None) -> RolloutSummary:
    """Build the pipeline from ``cfg``, run a rollout, return a summary.

    Mode and env are picked from ``cfg.mode`` and
    ``cfg.env.pipelines[mode]`` respectively; there is no env-specific
    branching here. The agent is likewise instantiated from
    ``cfg.agent`` so that swapping in a different planner is a config
    change, not a code change.

    ``log_dir`` is where ``video.mp4`` is written when
    ``cfg.record.video`` is true. When unset, falls back to the Hydra
    runtime output dir (set by ``@hydra.main``); when neither is
    available (tests composing configs directly), recording is skipped.
    """
    pipeline = cfg.env.pipelines[cfg.mode]
    # `_convert_="all"` returns plain Python objects rather than OmegaConf
    # wrappers, so nested dataclass-typed constructor arguments behave.
    real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
    perceiver = hydra.utils.instantiate(pipeline.perceiver, _convert_="all")
    plan_executor = hydra.utils.instantiate(pipeline.plan_executor, _convert_="all")
    agent = hydra.utils.instantiate(cfg.agent, seed=cfg.seed, _convert_="all")

    record_cfg = cfg.get("record")
    resolved_log_dir = _resolve_log_dir(log_dir)
    # Point the confirm gate's preview videos at the rollout log dir when
    # one exists (RemoteVegaEnv defaults to the system temp dir otherwise).
    if resolved_log_dir is not None and hasattr(real_env, "preview_dir"):
        real_env.preview_dir = Path(resolved_log_dir)
    recorder: VideoRecorder | None = None
    runner_kwargs = {
        "real_env": real_env,
        "perceiver": perceiver,
        "agent": agent,
        "plan_executor": plan_executor,
    }
    runner: Runner  # type: ignore[type-arg]
    if (
        record_cfg is not None
        and bool(record_cfg.get("video"))
        and resolved_log_dir is not None
    ):
        recorder = VideoRecorder()
        runner = RecordingRunner(recorder, **runner_kwargs)
    else:
        runner = Runner(**runner_kwargs)

    # try/finally so `real_env.close()` always runs — without it a real-mode
    # rollout can leave the robot connection open for the next session.
    try:
        # Move to the env's init pose before perceiving/planning: the real
        # robot's parked pose is not a valid planning start. In fake mode
        # one snapped setpoint suffices; in remote mode the init move is a
        # directive like any other, a min-jerk trajectory from the current
        # pose over env.init_move_seconds. The streaming real mode still
        # needs a gentle move_and_wait; not yet implemented.
        init_conf = cfg.env.get("init_right_arm_conf")
        if init_conf is not None and cfg.mode == "fake":
            real_env.step(VegaAction(right_arm_goal=list(init_conf)))
        elif init_conf is not None and cfg.mode == "remote":
            obs, _ = real_env.reset()
            trajectory = min_jerk_trajectory(
                np.array(obs.right_arm_conf),
                np.array(list(init_conf)),
                duration=float(cfg.env.get("init_move_seconds", 5.0)),
                hz=INIT_MOVE_HZ,
            )
            try:
                real_env.step(
                    TrajectoryDirective.from_array(
                        "right_arm", trajectory, hz=INIT_MOVE_HZ
                    )
                )
            except DirectiveRejected as e:
                return RolloutSummary(
                    env_name=cfg.env.env_name,
                    mode=cfg.mode,
                    seed=cfg.seed,
                    steps=0,
                    finish_reason=f"directive_rejected: {e}",
                    total_reward=0.0,
                )
        runner.reset(seed=cfg.seed)
        total_reward = 0.0
        steps = 0
        finish_reason = "max_steps_reached"
        for _ in range(cfg.max_eval_steps):
            try:
                _, reward, terminated, truncated, _ = runner.step()
            except (AgentFailure, PlanExhausted) as e:
                # A finite plan running out is the natural rollout end for
                # envs whose real mode has no goal detection.
                finish_reason = f"plan_exhausted: {e}"
                break
            except DirectiveRejected as e:
                # The operator declined a directive at the confirm gate;
                # nothing was sent to the robot. End the rollout cleanly.
                finish_reason = f"directive_rejected: {e}"
                break
            steps += 1
            total_reward += float(reward)
            if terminated:
                finish_reason = "terminated"
                break
            if truncated:
                finish_reason = "truncated"
                break
        video_path: Path | None = None
        if recorder is not None and resolved_log_dir is not None:
            fps = int(record_cfg.get("fps", 10)) if record_cfg is not None else 10
            video_path = recorder.save(Path(resolved_log_dir) / "video.mp4", fps)
        return RolloutSummary(
            env_name=cfg.env.env_name,
            mode=cfg.mode,
            seed=cfg.seed,
            steps=steps,
            finish_reason=finish_reason,
            total_reward=total_reward,
            video_path=video_path,
        )
    finally:
        real_env.close()


def _resolve_log_dir(explicit: Path | str | None) -> Path | None:
    """Use the explicit path if given; else the Hydra runtime dir; else None."""
    if explicit is not None:
        return Path(explicit)
    try:
        return Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        # HydraConfig.get() raises ValueError when no Hydra context is
        # active (tests composing configs without @hydra.main).
        return None
