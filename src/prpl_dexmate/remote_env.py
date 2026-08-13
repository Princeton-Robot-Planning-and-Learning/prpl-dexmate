"""Gymnasium environment backed by a remote skill server.

The counterpart of ``RealVegaEnv`` for the remote architecture: instead
of streaming one joint command per policy tick, each ``step`` sends a
whole directive across the RPC boundary and blocks until the skill
server reports a terminal result. The result (status plus telemetry) is
surfaced in the info dict under ``"directive_result"``; a non-success is
not an exception, because the outer loop's answer to a failed skill is
to perceive and replan from wherever the robot stopped.

With ``confirm=True`` (the config default for the remote pipeline), a
pause-and-confirm gate in the style of prpl-tidybot's plan preview runs
before every directive: a summary of the exact trajectory about to
execute — including the init move, which never passes through the
planner — is printed and the operator must approve on stdin. Rejection
raises :class:`DirectiveRejected` before anything is sent, so no motion
is commanded.
"""

import tempfile
from pathlib import Path
from typing import Any, Callable, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import RenderFrame

from prpl_dexmate.preview import DirectivePreviewer
from prpl_dexmate.remote.client import SkillClient
from prpl_dexmate.remote.protocol import DirectiveStatus, TrajectoryDirective
from prpl_dexmate.remote.server import DEFAULT_PORT
from prpl_dexmate.structs import VegaObservation

# Prompt indirection so tests can inject answers without monkeypatching
# the `input` builtin (mirrors prpl-tidybot's preview).
PromptFn = Callable[[str], str]


class DirectiveRejected(Exception):
    """The operator rejected a directive at the confirm gate."""


def describe_directive(directive: TrajectoryDirective) -> str:
    """A human-readable summary of what a directive will do."""
    trajectory = directive.as_array()
    duration = len(trajectory) / directive.hz
    start, end = trajectory[0], trajectory[-1]
    excursion = trajectory.max(axis=0) - trajectory.min(axis=0)
    if len(trajectory) > 1:
        peak_velocity = float(
            np.max(np.abs(np.diff(trajectory, axis=0))) * directive.hz
        )
    else:
        peak_velocity = 0.0
    lines = [
        f"Directive: {directive.component}, {len(trajectory)} frames "
        f"at {directive.hz:g} Hz ({duration:.1f}s)",
        f"  start:     {np.round(start, 3).tolist()}",
        f"  end:       {np.round(end, 3).tolist()}",
        f"  excursion: {np.round(excursion, 3).tolist()} rad",
        f"  peak velocity: {peak_velocity:.3f} rad/s",
    ]
    return "\n".join(lines)


def _confirm_or_reject(
    directive: TrajectoryDirective,
    prompt_fn: PromptFn | None,
    preview_path: Path | None,
) -> None:
    # `input` is resolved here, at call time, so tests can monkeypatch the
    # builtin even though Hydra cannot inject a prompt_fn through config.
    if prompt_fn is None:
        prompt_fn = input
    preview_line = (
        f"Preview video: {preview_path}\n" if preview_path is not None else ""
    )
    answer = (
        prompt_fn(
            f"\n{describe_directive(directive)}\n{preview_line}"
            "Execute on the robot? [y/N]: "
        )
        .strip()
        .lower()
    )
    if answer not in ("y", "yes"):
        raise DirectiveRejected(f"Directive rejected by operator (answer={answer!r})")


class RemoteVegaEnv(gymnasium.Env[VegaObservation, TrajectoryDirective]):
    """One env step = one directive executed to completion on the robot.

    Constructing this connects to the skill server (which performs the
    protocol version handshake); ``close`` disconnects but leaves the
    server running. Reward is always 0 and terminated / truncated are
    always False, matching ``RealVegaEnv``: the real world has no task
    semantics.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        poll_period: float = 0.2,
        confirm: bool = False,
        prompt_fn: PromptFn | None = None,
        previewer: DirectivePreviewer | None = None,
    ) -> None:
        self._client = SkillClient(host, port)
        self._poll_period = poll_period
        self._confirm = confirm
        self._prompt_fn = prompt_fn
        self._previewer = previewer
        # Where preview mp4s land; the pipeline points this at the rollout
        # log dir when one exists.
        self.preview_dir = Path(tempfile.gettempdir())

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
        # The gate runs strictly before the client sends anything, so a
        # rejection guarantees no motion was commanded.
        if self._confirm:
            preview_path = None
            if self._previewer is not None:
                print("Rendering preview video (shadow sim)...")
                preview_path = self._previewer.render_directive(
                    action, self.preview_dir
                )
            _confirm_or_reject(action, self._prompt_fn, preview_path)
        duration = len(action.trajectory) / action.hz
        print(
            f"Executing {action.component} directive on the robot ({duration:.1f}s)..."
        )
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
        if self._previewer is not None:
            self._previewer.close()
