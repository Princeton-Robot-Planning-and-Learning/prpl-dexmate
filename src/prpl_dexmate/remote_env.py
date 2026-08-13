"""Gymnasium environment backed by a remote skill server.

The counterpart of ``RealVegaEnv`` for the remote architecture: instead
of streaming one joint command per policy tick, each ``step`` sends a
whole directive across the RPC boundary and blocks until the skill
server reports a terminal result. The result (status plus telemetry) is
surfaced in the info dict under ``"directive_result"``; a non-success is
not an exception, because the outer loop's answer to a failed skill is
to perceive and replan from wherever the robot stopped.
"""

from typing import Any, SupportsFloat

import gymnasium
from gymnasium.core import RenderFrame

from prpl_dexmate.remote.client import SkillClient
from prpl_dexmate.remote.protocol import DirectiveStatus, TrajectoryDirective
from prpl_dexmate.remote.server import DEFAULT_PORT
from prpl_dexmate.structs import VegaObservation


class RemoteVegaEnv(gymnasium.Env[VegaObservation, TrajectoryDirective]):
    """One env step = one directive executed to completion on the robot.

    Constructing this connects to the skill server (which performs the
    protocol version handshake); ``close`` disconnects but leaves the
    server running. Reward is always 0 and terminated / truncated are
    always False, matching ``RealVegaEnv``: the real world has no task
    semantics.
    """

    def __init__(
        self, host: str, port: int = DEFAULT_PORT, poll_period: float = 0.2
    ) -> None:
        self._client = SkillClient(host, port)
        self._poll_period = poll_period

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[VegaObservation, dict[str, Any]]:
        super().reset(seed=seed)
        return self._client.get_observation(), {}

    def step(
        self, action: TrajectoryDirective
    ) -> tuple[VegaObservation, SupportsFloat, bool, bool, dict[str, Any]]:
        result = self._client.execute_directive(action, poll_period=self._poll_period)
        if result.status is not DirectiveStatus.SUCCEEDED:
            print(
                f"Warning: directive finished {result.status.value}: {result.message}"
            )
        observation = self._client.get_observation()
        return observation, 0.0, False, False, {"directive_result": result}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def close(self) -> None:
        """Disconnect from the skill server (which keeps running)."""
        self._client.close()
