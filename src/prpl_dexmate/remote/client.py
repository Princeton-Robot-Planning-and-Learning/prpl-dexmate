"""The orchestrator-side client for the skill server.

Connecting performs the protocol version handshake immediately, so a
version mismatch (orchestrator and Jetson on different branches) fails
at construction. ``execute_directive`` is the per-skill entry point: it
sends one directive and polls until a terminal result. The polls double
as the heartbeat that keeps the server's watchdog fed, so the poll
period must stay well under the server's watchdog timeout.
"""

import json
import socket
import time
from typing import Any

from prpl_dexmate.remote.protocol import (
    PROTOCOL_VERSION,
    DirectiveResult,
    GripperDirective,
    Message,
    decode_message,
    encode_message,
)
from prpl_dexmate.structs import VegaObservation


class ProtocolMismatchError(ConnectionError):
    """The server speaks a different protocol version (branch skew)."""


class SkillServerError(RuntimeError):
    """The server rejected a request."""


class SkillClient:
    """A connection to one skill server."""

    def __init__(
        self,
        host: str,
        port: int,
        connect_timeout: float = 5.0,
        request_timeout: float = 10.0,
    ) -> None:
        self._socket = socket.create_connection((host, port), timeout=connect_timeout)
        self._socket.settimeout(request_timeout)
        self._reader = self._socket.makefile("r", encoding="utf-8")
        response = self._request(
            {"op": "hello", "protocol_version": PROTOCOL_VERSION}, check=False
        )
        if not response["ok"]:
            self.close()
            raise ProtocolMismatchError(response.get("error", "Handshake failed"))

    def close(self) -> None:
        """Close the connection. The server keeps running."""
        self._reader.close()
        self._socket.close()

    def start_directive(self, directive: Message) -> int:
        """Start a directive on the server and return its id."""
        response = self._request(
            {"op": "start", "directive": encode_message(directive)}
        )
        return int(response["directive_id"])

    def get_result(self, directive_id: int) -> DirectiveResult:
        """Fetch the current result of a directive (RUNNING if in flight)."""
        response = self._request({"op": "status", "directive_id": directive_id})
        result = decode_message(response["result"])
        assert isinstance(result, DirectiveResult)
        return result

    def open_gripper(self, side: str) -> DirectiveResult:
        """Open one gripper and wait for the result."""
        return self.execute_directive(GripperDirective(side=side, action="open"))

    def close_gripper(self, side: str) -> DirectiveResult:
        """Close one gripper and wait for the result."""
        return self.execute_directive(GripperDirective(side=side, action="close"))

    def get_observation(self) -> VegaObservation:
        """Fetch the robot's current joint observation."""
        response = self._request({"op": "observe"})
        observation = decode_message(response["observation"])
        assert isinstance(observation, VegaObservation)
        return observation

    def stop(self) -> None:
        """Stop any running directive; the robot holds position."""
        self._request({"op": "stop"})

    def execute_directive(
        self, directive: Message, poll_period: float = 0.2
    ) -> DirectiveResult:
        """Run one directive to completion: start, then poll (heartbeat)
        until the result is terminal."""
        directive_id = self.start_directive(directive)
        while True:
            result = self.get_result(directive_id)
            if result.status.is_terminal:
                return result
            time.sleep(poll_period)

    def _request(self, payload: dict[str, Any], check: bool = True) -> dict[str, Any]:
        self._socket.sendall(json.dumps(payload).encode("utf-8") + b"\n")
        line = self._reader.readline()
        if not line:
            raise ConnectionError("Skill server closed the connection")
        response: dict[str, Any] = json.loads(line)
        if check and not response["ok"]:
            raise SkillServerError(response.get("error", "Request failed"))
        return response
