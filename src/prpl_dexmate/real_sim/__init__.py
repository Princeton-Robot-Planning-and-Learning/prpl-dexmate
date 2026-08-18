"""Real-to-sim-to-real components for the Vega.

``build_planner_env_models`` is a tiny adapter for using
``kinder_bilevel_planning.BilevelPlanningAgent`` inside this pipeline;
see its docstring.
"""

from dataclasses import replace
from typing import Any

import gymnasium
import kinder
from bilevel_planning.structs import SesameModels
from kinder_bilevel_planning.env_models import create_bilevel_planning_models


def build_planner_env_models(
    env_name: str, env_id: str, env_config: Any | None = None
) -> SesameModels:
    """Build kinder-bilevel-planning env models for our perceiver pipeline.

    Spins up a one-shot reference kinder env purely to source the
    observation and action spaces the bilevel-planning factory expects —
    the env is closed before this function returns.

    ``env_config`` is the kinder env's config object (e.g. a
    ``VegaPickPlace3DEnvConfig`` with the measured real-table geometry).
    It is forwarded to both the reference env and the factory's internal
    sim, so the planner's models assume the same world the pipeline env
    builds. ``None`` keeps the env's defaults.

    The factory's ``observation_to_state`` callback devectorizes raw
    vectorized observations, but in this pipeline the perceiver layer has
    already produced an ``ObjectCentricState`` by the time the agent sees
    it, so it is swapped for an identity. Everything else (types,
    predicates, skills, transition_fn, state_abstractor, goal_deriver) is
    untouched.
    """
    kinder.register_all_environments()
    make_kwargs = {} if env_config is None else {"config": env_config}
    ref_env = gymnasium.make(env_id, **make_kwargs)
    try:
        base = create_bilevel_planning_models(
            env_name, ref_env.observation_space, ref_env.action_space, **make_kwargs
        )
    finally:
        ref_env.close()
    return replace(base, observation_to_state=lambda x: x)


__all__ = ["build_planner_env_models"]
