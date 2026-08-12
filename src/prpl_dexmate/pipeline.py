"""Glue that wires a Hydra config into a Runner rollout.

The ``scripts/run_pipeline.py`` entry point delegates to ``run_pipeline``
here; tests compose configs with ``hydra.compose`` and call
``run_pipeline`` directly, without going through the Hydra ``@main``
decorator.
"""

from dataclasses import dataclass

import hydra
from omegaconf import DictConfig
from prpl_utils.real_sim import Runner

from prpl_dexmate.agents import PlanExhausted


@dataclass(frozen=True)
class RolloutSummary:
    """Result of one rollout — handy for assertions in tests."""

    env_name: str
    mode: str
    seed: int
    steps: int
    finish_reason: str
    total_reward: float


def run_pipeline(cfg: DictConfig) -> RolloutSummary:
    """Build the pipeline from ``cfg``, run a rollout, return a summary.

    Mode and env are picked from ``cfg.mode`` and
    ``cfg.env.pipelines[mode]`` respectively; there is no env-specific
    branching here. The agent is likewise instantiated from
    ``cfg.agent`` so that swapping in a different planner is a config
    change, not a code change.
    """
    pipeline = cfg.env.pipelines[cfg.mode]
    # `_convert_="all"` returns plain Python objects rather than OmegaConf
    # wrappers, so nested dataclass-typed constructor arguments behave.
    real_env = hydra.utils.instantiate(pipeline.real_env, _convert_="all")
    perceiver = hydra.utils.instantiate(pipeline.perceiver, _convert_="all")
    plan_executor = hydra.utils.instantiate(pipeline.plan_executor, _convert_="all")
    agent = hydra.utils.instantiate(cfg.agent, seed=cfg.seed, _convert_="all")

    runner = Runner(
        real_env=real_env,
        perceiver=perceiver,
        agent=agent,
        plan_executor=plan_executor,
    )

    # try/finally so `real_env.close()` always runs — without it a real-mode
    # rollout can leave the robot connection open for the next session.
    try:
        runner.reset(seed=cfg.seed)
        total_reward = 0.0
        steps = 0
        finish_reason = "max_steps_reached"
        for _ in range(cfg.max_eval_steps):
            try:
                _, reward, terminated, truncated, _ = runner.step()
            except PlanExhausted as e:
                finish_reason = f"plan_exhausted: {e}"
                break
            steps += 1
            total_reward += float(reward)
            if terminated:
                finish_reason = "terminated"
                break
            if truncated:
                finish_reason = "truncated"
                break
        return RolloutSummary(
            env_name=cfg.env.env_name,
            mode=cfg.mode,
            seed=cfg.seed,
            steps=steps,
            finish_reason=finish_reason,
            total_reward=total_reward,
        )
    finally:
        real_env.close()
