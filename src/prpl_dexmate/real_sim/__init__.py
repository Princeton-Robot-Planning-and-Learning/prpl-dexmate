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
from relational_structs.utils import all_ground_operators


def build_planner_env_models(
    env_name: str,
    env_id: str,
    env_config: Any | None = None,
    exclude_object_names: list[str] | None = None,
    grasp_approach_max_tilt: float | None = None,
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

    ``grasp_approach_max_tilt`` bounds how far (radians) the end effector
    may point off straight down in sampled pick and place configurations,
    yaw free (kinder-baselines#135); None leaves grasp orientations
    unconstrained. Only forwarded when set, since not every env's factory
    accepts it.

    ``exclude_object_names`` removes objects from the planner's operator
    grounding without touching the state: an operator can never bind to
    an excluded object, but the object still appears in every state. Used
    to keep pick-and-place plans right-arm-only
    (``exclude_object_names=["left_arm"]``) while the hardware runs are
    single-arm.

    The factory's ``observation_to_state`` callback devectorizes raw
    vectorized observations, but in this pipeline the perceiver layer has
    already produced an ``ObjectCentricState`` by the time the agent sees
    it, so it is swapped for an identity. Everything else (types,
    predicates, skills, transition_fn, state_abstractor, goal_deriver) is
    untouched.
    """
    kinder.register_all_environments()
    make_kwargs = {} if env_config is None else {"config": env_config}
    factory_kwargs = dict(make_kwargs)
    if grasp_approach_max_tilt is not None:
        factory_kwargs["grasp_approach_max_tilt"] = grasp_approach_max_tilt
    ref_env = gymnasium.make(env_id, **make_kwargs)
    try:
        base = create_bilevel_planning_models(
            env_name, ref_env.observation_space, ref_env.action_space, **factory_kwargs
        )
        ground_operators = None
        if exclude_object_names:
            obs0, _ = ref_env.reset(seed=0)
            state0 = ref_env.observation_space.devectorize(  # type: ignore[attr-defined]
                obs0
            )
            excluded = set(exclude_object_names)
            missing = excluded - set(state0.get_object_names())
            assert not missing, f"Objects to exclude not in the env: {missing}"
            objects = {
                state0.get_object_from_name(name)
                for name in state0.get_object_names()
                if name not in excluded
            }
            ground_operators = all_ground_operators(base.operators, objects)
    finally:
        ref_env.close()
    return replace(
        base,
        observation_to_state=lambda x: x,
        ground_operators=ground_operators,
    )


__all__ = ["build_planner_env_models"]
