"""Interfaces to a DexGripper parallel-jaw end effector, real or fake.

Each Vega wrist carries one Dexgripper S with a single actuated joint.
The interface is deliberately semantic (open / close) rather than
positional for now: the grippers' useful states are the two ends of
travel, and dexcontrol's DexGripper class drives to its own predefined
pose pool for each.
"""

import abc

import numpy as np

# Approximate joint readings at the two ends of travel. CLOSED matches
# the reading measured on hardware after mounting (~0.097); OPEN is the
# model's pose-pool value and should be refined from a hardware reading
# during the next robot session.
GRIPPER_CLOSED_POS = 0.1
GRIPPER_OPEN_POS = 0.6

# How long the real gripper is given to complete an open/close command.
_GRIPPER_MOVE_SECONDS = 3.0


class GripperInterface(abc.ABC):
    """A generic interface to one gripper, real or fake."""

    @abc.abstractmethod
    def open(self) -> None:
        """Open the gripper (blocks until the motion has had time to finish)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Close the gripper (blocks until the motion has had time to finish)."""

    @abc.abstractmethod
    def get_position(self) -> float:
        """The gripper's joint position."""

    def teardown(self) -> None:
        """Tear down any hardware connection."""


class FakeGripperInterface(GripperInterface):
    """A fake gripper that snaps instantly between open and closed."""

    def __init__(self, position: float = GRIPPER_CLOSED_POS) -> None:
        self._position = position

    def open(self) -> None:
        self._position = GRIPPER_OPEN_POS

    def close(self) -> None:
        self._position = GRIPPER_CLOSED_POS

    def get_position(self) -> float:
        return self._position


class RealGripperInterface(GripperInterface):
    """A gripper backed by a dexcontrol DexGripper hand component.

    The component is ``robot.right_hand`` / ``robot.left_hand`` from a
    connected Robot; dexcontrol creates those only when a gripper is
    detected at the wrist, so ``hand`` may be None — operations then
    fail loudly rather than pretending.
    """

    def __init__(self, hand: object | None, side: str) -> None:
        self._hand = hand
        self._side = side

    def _require_hand(self) -> object:
        if self._hand is None:
            raise RuntimeError(
                f"No {self._side} gripper detected by dexcontrol; "
                "check mounting/cabling and the hand-type banner at startup."
            )
        return self._hand

    def open(self) -> None:
        self._require_hand().open_hand(  # type: ignore[attr-defined]
            wait_time=_GRIPPER_MOVE_SECONDS
        )

    def close(self) -> None:
        self._require_hand().close_hand(  # type: ignore[attr-defined]
            wait_time=_GRIPPER_MOVE_SECONDS
        )

    def get_position(self) -> float:
        if self._hand is None:
            return 0.0
        joint_pos = self._hand.get_joint_pos()  # type: ignore[attr-defined]
        return float(np.asarray(joint_pos).reshape(-1)[0])
