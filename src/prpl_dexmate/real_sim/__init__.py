"""Real-to-sim-to-real components for the Vega.

``build_planner_env_models`` is a tiny adapter for using
``kinder_bilevel_planning.BilevelPlanningAgent`` inside this pipeline;
see its docstring.
"""

from dataclasses import replace

import gymnasium
import kinder
from bilevel_planning.structs import SesameModels
from kinder_bilevel_planning.env_models import create_bilevel_planning_models


def build_planner_env_models(env_name: str, env_id: str) -> SesameModels:
    """Build kinder-bilevel-planning env models for our perceiver pipeline.

    Spins up a one-shot reference kinder env purely to source the
    observation and action spaces the bilevel-planning factory expects —
    the env is closed before this function returns.

    The factory's ``observation_to_state`` callback devectorizes raw
    vectorized observations, but in this pipeline the perceiver layer has
    already produced an ``ObjectCentricState`` by the time the agent sees
    it, so it is swapped for an identity. Everything else (types,
    predicates, skills, transition_fn, state_abstractor, goal_deriver) is
    untouched.
    """
    kinder.register_all_environments()
    ref_env = gymnasium.make(env_id)
    try:
        base = create_bilevel_planning_models(
            env_name, ref_env.observation_space, ref_env.action_space
        )
    finally:
        ref_env.close()
    return replace(base, observation_to_state=lambda x: x)


__all__ = ["build_planner_env_models"]
