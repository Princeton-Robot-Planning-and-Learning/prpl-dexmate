"""Sources for the target position perceived in VegaMotion3D.

The sim env samples its own target; outside sim the target has to come
from somewhere else. ConstantTargetSource is the fake/test source; a
camera-based source can implement the same interface later.
"""

import abc


class TargetSource(abc.ABC):
    """Where the perceiver gets the target position from."""

    @abc.abstractmethod
    def get_target(self) -> tuple[float, float, float]:
        """Return the target's (x, y, z) in the env frame."""


class ConstantTargetSource(TargetSource):
    """A fixed target position, configured up front."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self._target = (x, y, z)

    def get_target(self) -> tuple[float, float, float]:
        return self._target
