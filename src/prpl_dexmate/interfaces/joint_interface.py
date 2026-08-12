"""Interfaces to a joint-position-controlled robot component.

Both Vega arms and the head are joint-position components; the only
difference is the joint count, so one interface family covers all three.
"""

import abc
from typing import Any

import numpy as np


class JointInterface(abc.ABC):
    """A generic interface to one joint-position component, real or fake."""

    @abc.abstractmethod
    def get_joint_state(self) -> list[float]:
        """Get the component's joint positions."""

    @abc.abstractmethod
    def execute_action(self, goal: list[float]) -> None:
        """Command an absolute joint position goal (non-blocking)."""

    def close(self) -> None:
        """Tear down any hardware connection."""


class FakeJointInterface(JointInterface):
    """A fake joint component that snaps instantly to commanded goals.

    Snapping means a tracking executor converges on its first tick, which
    keeps fake-mode rollouts and tests fast.
    """

    def __init__(self, home_conf: list[float]) -> None:
        self._conf = list(home_conf)

    def get_joint_state(self) -> list[float]:
        return list(self._conf)

    def execute_action(self, goal: list[float]) -> None:
        self._conf = list(goal)


class RealJointInterface(JointInterface):
    """A joint component backed by a dexcontrol component handle.

    The handle is e.g. ``robot.right_arm`` or ``robot.head`` from a
    connected ``dexcontrol.robot.Robot``. The Robot's lifecycle is owned
    by the composite interface, not here.
    """

    def __init__(self, component: Any) -> None:
        self._component = component

    def get_joint_state(self) -> list[float]:
        return [float(v) for v in np.asarray(self._component.get_joint_pos())]

    def execute_action(self, goal: list[float]) -> None:
        self._component.set_joint_pos(np.asarray(goal, dtype=float))
