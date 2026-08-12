"""Tests for prpl_dexmate.real_sim.perceivers.passthrough."""

from prpl_dexmate.real_sim.perceivers.passthrough import PassThroughPerceiver


def test_passthrough_perceiver_is_identity() -> None:
    """Reset and step return the observation unchanged."""
    perceiver: PassThroughPerceiver[str] = PassThroughPerceiver()
    assert perceiver.reset("obs", {}) == "obs"
    assert perceiver.step("obs2", {"k": 1}) == "obs2"
