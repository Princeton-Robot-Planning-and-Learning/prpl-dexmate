"""Tests for prpl_dexmate.real_sim.plan_executors.passthrough."""

from prpl_dexmate.real_sim.plan_executors.passthrough import PassThroughPlanExecutor


def test_passthrough_executor_emits_actions_in_order() -> None:
    """Each planned action is emitted once; done after the last."""
    executor: PassThroughPlanExecutor[int] = PassThroughPlanExecutor()
    executor.set_trajectory([("s0", 10), ("s1", 11)])
    assert not executor.done(None)
    assert executor.step(None) == (10, 10)
    assert executor.step(None) == (11, 11)
    assert executor.done(None)
