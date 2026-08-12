"""Data structures."""

from dataclasses import dataclass

NUM_ARM_JOINTS = 7
NUM_HEAD_JOINTS = 3


@dataclass(frozen=True)
class VegaObservation:
    """Raw joint observations from the Vega environment."""

    right_arm_conf: list[float]
    left_arm_conf: list[float]
    head_conf: list[float]

    def __post_init__(self) -> None:
        assert len(self.right_arm_conf) == NUM_ARM_JOINTS
        assert len(self.left_arm_conf) == NUM_ARM_JOINTS
        assert len(self.head_conf) == NUM_HEAD_JOINTS


@dataclass(frozen=True)
class VegaAction:
    """Absolute joint position commands for the real Vega environment.

    A component whose goal is None holds its current position.
    """

    right_arm_goal: list[float] | None = None
    left_arm_goal: list[float] | None = None
    head_goal: list[float] | None = None

    def __post_init__(self) -> None:
        assert self.right_arm_goal is None or len(self.right_arm_goal) == NUM_ARM_JOINTS
        assert self.left_arm_goal is None or len(self.left_arm_goal) == NUM_ARM_JOINTS
        assert self.head_goal is None or len(self.head_goal) == NUM_HEAD_JOINTS
