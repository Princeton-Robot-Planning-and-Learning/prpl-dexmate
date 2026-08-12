"""Tests for prpl_dexmate.real_env."""

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.real_env import RealVegaEnv
from prpl_dexmate.structs import NUM_ARM_JOINTS, VegaAction


def test_real_env_step_commands_and_observes() -> None:
    """A step commands the interfaces and returns the fresh observation."""
    env = RealVegaEnv(FakeInterface(), control_period=0.0)
    obs, info = env.reset(seed=0)
    assert not info
    goal = [v + 0.1 for v in obs.right_arm_conf]
    action = VegaAction(right_arm_goal=goal)
    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert next_obs.right_arm_conf == goal
    assert next_obs.left_arm_conf == obs.left_arm_conf
    assert reward == 0.0
    assert not terminated
    assert not truncated
    env.close()


def test_real_env_none_goals_hold_position() -> None:
    """An all-None action leaves every component unchanged."""
    env = RealVegaEnv(FakeInterface(), control_period=0.0)
    obs, _ = env.reset(seed=0)
    next_obs, _, _, _, _ = env.step(VegaAction())
    assert next_obs.right_arm_conf == obs.right_arm_conf
    assert len(next_obs.right_arm_conf) == NUM_ARM_JOINTS
    env.close()
