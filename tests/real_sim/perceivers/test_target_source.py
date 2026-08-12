"""Tests for prpl_dexmate.real_sim.perceivers.target_source."""

from prpl_dexmate.real_sim.perceivers.target_source import ConstantTargetSource


def test_constant_target_source_returns_configured_target() -> None:
    """The source echoes its constructor arguments."""
    source = ConstantTargetSource(0.5, -0.4, 0.8)
    assert source.get_target() == (0.5, -0.4, 0.8)
