"""Shadow-sim video previews for remote directives.

Ported from prpl-tidybot's plan preview, adapted to the directive-level
confirm gate: rather than rendering the planner's state sequence, the
previewer renders the exact dense trajectory a ``TrajectoryDirective``
will execute — which also covers the init move, which never passes
through the planner. The gate in ``RemoteVegaEnv`` writes the mp4 next
to the rollout logs and includes its path in the confirmation prompt.

Arm directives are renderable; head and gripper directives fall back to
the text-only summary. ``base_obs`` — the robot's observation at gate
time — supplies the non-moving components (the other arm, the grippers)
so the shadow scene matches reality; without it they default to zeros,
which suffices for the single-arm VegaMotion3D sim.
"""

from pathlib import Path

import numpy as np
from prpl_utils.real_sim import Perceiver
from relational_structs import ObjectCentricState

from prpl_dexmate.recording import VideoRecorder
from prpl_dexmate.remote.protocol import TrajectoryDirective
from prpl_dexmate.sim_env import KinderSimEnv
from prpl_dexmate.structs import NUM_ARM_JOINTS, NUM_HEAD_JOINTS, VegaObservation

_RENDERABLE_COMPONENTS = ("right_arm", "left_arm")


class DirectivePreviewer:
    """Render right-arm trajectory directives through a shadow sim.

    ``shadow_env`` must be a KinderSimEnv constructed with
    ``allow_state_access=True`` so states can be teleported in. The
    perceiver lifts synthesized VegaObservations (one per rendered
    frame) into sim states; the target it perceives is drawn too, so
    the preview shows the motion relative to the goal.
    """

    def __init__(
        self,
        shadow_env: KinderSimEnv,
        perceiver: Perceiver[VegaObservation, ObjectCentricState],
        fps: int = 10,
        seed: int = 0,
    ) -> None:
        self._shadow_env = shadow_env
        self._perceiver = perceiver
        self._fps = fps
        self._seed = seed
        self._reset_done = False
        self._perceiver_reset_done = False
        self._counter = 0

    def render_directive(
        self,
        directive: TrajectoryDirective,
        out_dir: Path,
        base_obs: VegaObservation | None = None,
    ) -> Path | None:
        """Write a preview mp4 for ``directive``; None if not renderable.

        The trajectory is subsampled to ``fps`` so the video plays at
        roughly the wall-clock speed of the real execution.
        """
        if directive.component not in _RENDERABLE_COMPONENTS:
            return None
        if not self._reset_done:
            # One reset so set_state lands on a clean env (same protocol
            # as tidybot's preview); set_state is idempotent after that.
            self._shadow_env.reset(seed=self._seed)
            self._reset_done = True
        if base_obs is None:
            base_obs = VegaObservation(
                right_arm_conf=[0.0] * NUM_ARM_JOINTS,
                left_arm_conf=[0.0] * NUM_ARM_JOINTS,
                head_conf=[0.0] * NUM_HEAD_JOINTS,
            )
        trajectory = directive.as_array()
        stride = max(1, round(directive.hz / self._fps))
        indices = list(range(0, len(trajectory), stride))
        if indices[-1] != len(trajectory) - 1:
            indices.append(len(trajectory) - 1)
        recorder = VideoRecorder()
        for i in indices:
            frame_conf = [float(v) for v in trajectory[i]]
            observation = VegaObservation(
                right_arm_conf=(
                    frame_conf
                    if directive.component == "right_arm"
                    else list(base_obs.right_arm_conf)
                ),
                left_arm_conf=(
                    frame_conf
                    if directive.component == "left_arm"
                    else list(base_obs.left_arm_conf)
                ),
                head_conf=list(base_obs.head_conf),
                right_gripper_pos=base_obs.right_gripper_pos,
                left_gripper_pos=base_obs.left_gripper_pos,
            )
            if not self._perceiver_reset_done:
                # Stateful perceivers (the pickplace belief tracker) need
                # a reset to stage their scene before stepping.
                state = self._perceiver.reset(observation, {})
                self._perceiver_reset_done = True
            else:
                state = self._perceiver.step(observation, {})
            self._shadow_env.set_state(state)
            frame = self._shadow_env.render()
            if frame is None:
                return None
            recorder.add_frame(np.asarray(frame, dtype=np.uint8))
        out_path = Path(out_dir) / f"preview_{self._counter:03d}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._counter += 1
        return recorder.save(out_path, self._fps)

    def close(self) -> None:
        """Tear down the shadow sim."""
        self._shadow_env.close()
