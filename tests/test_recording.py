"""Tests for prpl_dexmate.recording."""

from pathlib import Path

import numpy as np

from prpl_dexmate.recording import VideoRecorder


def test_video_recorder_writes_mp4(tmp_path: Path) -> None:
    """Frames are composed into a nonempty mp4 file."""
    recorder = VideoRecorder()
    for i in range(5):
        frame = np.full((32, 48, 3), i * 40, dtype=np.uint8)
        recorder.add_frame(frame)
    assert recorder.num_frames == 5
    path = recorder.save(tmp_path / "video.mp4", fps=10)
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_video_recorder_without_frames_writes_nothing(tmp_path: Path) -> None:
    """Saving with no frames returns None and creates no file."""
    recorder = VideoRecorder()
    path = recorder.save(tmp_path / "video.mp4", fps=10)
    assert path is None
    assert not (tmp_path / "video.mp4").exists()
