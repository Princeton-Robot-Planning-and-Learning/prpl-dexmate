"""Helpers for commanding robot motion through dexcontrol."""

import time
from typing import Any

import numpy as np


def move_and_wait(
    component: Any,
    target: np.ndarray,
    timeout: float = 5.0,
    tolerance: float = 0.02,
) -> None:
    """Command a joint-space move and poll until the target is reached.

    Polling covers both blocking and non-blocking move implementations.
    """
    component.move_to_joint_pos(target)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        error = np.max(np.abs(np.asarray(component.get_joint_pos()) - target))
        if error < tolerance:
            return
        time.sleep(0.1)
    print(f"Warning: target not reached within {timeout}s (error {error:.4f} rad)")
