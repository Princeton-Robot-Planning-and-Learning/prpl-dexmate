"""Video recording for pipeline rollouts.

RecordingRunner captures one frame from the real env per tick (plus one
at reset) via the Runner's ``on_step`` hook; VideoRecorder composes the
captured frames into an mp4 at the end of the rollout. Envs whose
``render`` returns None (e.g. RealVegaEnv, which has no renderer yet)
simply produce no frames and no video.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from prpl_utils.real_sim import Runner


class VideoRecorder:
    """Accumulate RGB frames and write them to an mp4."""

    def __init__(self) -> None:
        self._frames: list[NDArray[np.uint8]] = []

    @property
    def num_frames(self) -> int:
        """The number of frames captured so far."""
        return len(self._frames)

    def add_frame(self, frame: NDArray[np.uint8]) -> None:
        """Append one RGB frame (height, width, 3)."""
        assert frame.ndim == 3 and frame.shape[2] == 3
        self._frames.append(frame)

    def save(self, path: Path, fps: int) -> Path | None:
        """Write the frames to ``path`` and return it; None if no frames."""
        if not self._frames:
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        height, width = self._frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
        )
        # cv2.VideoWriter reports failure (e.g. an unwritable path) by
        # never opening rather than by raising; without this check save
        # would silently return a path to a file that was never written.
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        for frame in self._frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        return path


class RecordingRunner(Runner):  # type: ignore[type-arg]
    """A Runner that captures a frame from the real env every tick."""

    def __init__(self, recorder: VideoRecorder, **runner_kwargs: Any) -> None:
        super().__init__(**runner_kwargs)
        self._recorder = recorder
        self._render_env = runner_kwargs["real_env"]

    def reset(self, **kwargs: Any) -> Any:
        state = super().reset(**kwargs)
        self._capture()
        return state

    def on_step(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._capture()

    def _capture(self) -> None:
        frame = self._render_env.render()
        if frame is not None:
            self._recorder.add_frame(np.asarray(frame, dtype=np.uint8))
