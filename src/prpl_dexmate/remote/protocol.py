"""Wire messages for the orchestrator <-> skill server RPC boundary.

Directives are the orchestrator-to-server message family: execute a
joint trajectory, or (later) roll out a policy. Results flow back the
other way. All messages are JSON-serializable dataclasses so that the
two sides can run different Python environments (x86 orchestrator,
aarch64 Jetson) without pickle compatibility concerns.

PROTOCOL_VERSION is a hash of the message schemas. The client and server
exchange it at connect time, so an orchestrator and a server checked out
on different branches fail loudly at startup instead of subtly at
runtime.
"""

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray

from prpl_dexmate.structs import NUM_ARM_JOINTS, NUM_HEAD_JOINTS, VegaObservation

# Joint counts per commandable component, keyed by the component names
# used in directives.
COMPONENT_NUM_JOINTS = {
    "right_arm": NUM_ARM_JOINTS,
    "left_arm": NUM_ARM_JOINTS,
    "head": NUM_HEAD_JOINTS,
}


@dataclass(frozen=True)
class TrajectoryDirective:
    """Execute a joint trajectory on one component.

    The whole trajectory travels in one message; the server interpolates
    nothing and executes it frame-by-frame locally (see
    ``motion.follow_joint_trajectory``, whose defaults these mirror).
    The component must already be at the first frame within
    ``max_start_error``.
    """

    component: str
    trajectory: list[list[float]]
    hz: float = 100.0
    max_start_error: float = 0.05
    max_tracking_error: float = 0.5

    def __post_init__(self) -> None:
        assert self.component in COMPONENT_NUM_JOINTS
        num_joints = COMPONENT_NUM_JOINTS[self.component]
        assert len(self.trajectory) >= 1
        assert all(len(frame) == num_joints for frame in self.trajectory)
        assert self.hz > 0
        assert self.max_start_error > 0
        assert self.max_tracking_error > 0

    @classmethod
    def from_array(
        cls,
        component: str,
        trajectory: NDArray[np.floating],
        **kwargs: float,
    ) -> "TrajectoryDirective":
        """Build a directive from a (num_frames, num_joints) array."""
        frames = [[float(v) for v in frame] for frame in np.atleast_2d(trajectory)]
        return cls(component=component, trajectory=frames, **kwargs)

    def as_array(self) -> NDArray[np.float64]:
        """The trajectory as a (num_frames, num_joints) array."""
        return np.asarray(self.trajectory, dtype=float)


@dataclass(frozen=True)
class GripperDirective:
    """Open or close one gripper.

    Semantic rather than positional: the server drives the gripper to
    dexcontrol's predefined open/close pose for the mounted end
    effector. ``timeout`` bounds how long the server blocks on the
    motion (the real gripper call is a short blocking command that
    cannot be aborted mid-way, unlike trajectory streaming).
    """

    side: str
    action: str
    timeout: float = 5.0

    def __post_init__(self) -> None:
        assert self.side in ("left", "right")
        assert self.action in ("open", "close")
        assert self.timeout > 0


@dataclass(frozen=True)
class PolicyRolloutDirective:
    """Run a policy on the server until termination or timeout.

    A placeholder for closed-loop visuomotor skills: it fixes the shape
    of the boundary now so adding policy execution later does not
    reshape the protocol. No server implements it yet.
    """

    policy: str
    policy_args: dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0

    def __post_init__(self) -> None:
        assert self.policy
        assert self.timeout > 0


class DirectiveStatus(Enum):
    """The lifecycle of a directive on the server."""

    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    STOPPED = "stopped"

    @property
    def is_terminal(self) -> bool:
        """Whether the directive has finished executing."""
        return self is not DirectiveStatus.RUNNING


@dataclass(frozen=True)
class DirectiveResult:
    """The server's report on a directive, terminal or in flight.

    ``message`` is human-readable context (e.g. the abort reason on
    FAILED). ``max_tracking_error`` and ``duration`` are telemetry from
    trajectory execution, absent while RUNNING or when not applicable.
    """

    status: DirectiveStatus
    message: str = ""
    max_tracking_error: float | None = None
    duration: float | None = None


Message = Union[
    TrajectoryDirective,
    GripperDirective,
    PolicyRolloutDirective,
    DirectiveResult,
    VegaObservation,
]

_MESSAGE_TYPES: dict[str, type] = {
    "trajectory_directive": TrajectoryDirective,
    "gripper_directive": GripperDirective,
    "policy_rollout_directive": PolicyRolloutDirective,
    "directive_result": DirectiveResult,
    "vega_observation": VegaObservation,
}
_TYPE_NAMES = {cls: name for name, cls in _MESSAGE_TYPES.items()}


def encode_message(message: Message) -> dict[str, Any]:
    """Convert a message to a JSON-safe dict tagged with its type."""
    wire = dataclasses.asdict(message)
    if isinstance(message, DirectiveResult):
        wire["status"] = message.status.value
    wire["type"] = _TYPE_NAMES[type(message)]
    return wire


def decode_message(wire: dict[str, Any]) -> Message:
    """Reconstruct a message from its wire dict.

    Raises ValueError on an unknown or missing type tag, and TypeError
    on fields that do not match the message schema, so version skew that
    slips past the handshake still fails loudly.
    """
    fields = dict(wire)
    type_name = fields.pop("type", None)
    if type_name not in _MESSAGE_TYPES:
        raise ValueError(f"Unknown message type: {type_name}")
    cls = _MESSAGE_TYPES[type_name]
    if cls is DirectiveResult:
        fields["status"] = DirectiveStatus(fields["status"])
    message: Message = cls(**fields)
    return message


def serialize_message(message: Message) -> str:
    """Encode a message to a JSON string."""
    return json.dumps(encode_message(message))


def deserialize_message(data: str) -> Message:
    """Decode a message from a JSON string."""
    return decode_message(json.loads(data))


def _schema_description() -> str:
    parts = []
    for name in sorted(_MESSAGE_TYPES):
        cls = _MESSAGE_TYPES[name]
        fields_desc = ",".join(f"{f.name}:{f.type}" for f in dataclasses.fields(cls))
        parts.append(f"{name}({fields_desc})")
    parts.append("status:" + ",".join(status.value for status in DirectiveStatus))
    return ";".join(parts)


# Hash of the message schemas above; exchanged at connect time. Any
# change to a message's fields, field types, or the status values yields
# a different version.
PROTOCOL_VERSION = hashlib.sha256(_schema_description().encode()).hexdigest()[:12]
