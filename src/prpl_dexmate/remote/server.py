"""The skill server that runs on the robot's Jetson.

The server owns the inner control loop: it receives one directive per
skill invocation from the orchestrator (see ``protocol``), executes it
locally at the full control rate against an ``Interface``, and reports a
``DirectiveResult``. Requests are newline-delimited JSON over TCP; each
connection must open with a hello carrying ``PROTOCOL_VERSION``, and a
mismatch is rejected before any directive can be sent.

Safety: every trajectory directive is validated against the URDF
position and velocity limits (``limits.validate_trajectory``) before
anything moves; a violating directive is rejected at start. Only one
directive runs at a time. While one is running, the orchestrator's
status polls double as a heartbeat; if no request arrives for
``watchdog_timeout`` seconds, execution stops and the robot holds
position (the safe-stop for streamed position control). An explicit stop
request does the same immediately.

Run on the robot with ``python -m prpl_dexmate.remote.server`` (or
``--fake`` for a benchless smoke test).
"""

import argparse
import json
import socketserver
import threading
import time
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from prpl_dexmate.interfaces.interface import (
    FakeInterface,
    Interface,
    RealInterface,
)
from prpl_dexmate.interfaces.joint_interface import JointInterface
from prpl_dexmate.limits import validate_trajectory
from prpl_dexmate.motion import follow_joint_trajectory
from prpl_dexmate.remote.protocol import (
    PROTOCOL_VERSION,
    DirectiveResult,
    DirectiveStatus,
    PolicyRolloutDirective,
    TrajectoryDirective,
    decode_message,
    encode_message,
)

DEFAULT_PORT = 8790
DEFAULT_WATCHDOG_TIMEOUT = 2.0


class _JointInterfaceComponent:
    """Adapt a JointInterface to the JointComponent protocol that
    ``follow_joint_trajectory`` expects."""

    def __init__(self, joint_interface: JointInterface) -> None:
        self._joint_interface = joint_interface

    def get_joint_pos(self) -> ArrayLike:
        """Return the component's current joint positions."""
        return self._joint_interface.get_joint_state()

    def set_joint_pos(self, joint_pos: NDArray[np.float64]) -> None:
        """Command joint positions (streaming, non-blocking)."""
        self._joint_interface.execute_action([float(v) for v in joint_pos])


class SkillServer:
    """Serve directives against an Interface.

    The caller owns the interface's lifecycle; ``close`` stops the
    network server and any running directive but does not close the
    interface.
    """

    def __init__(
        self,
        interface: Interface,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        watchdog_timeout: float = DEFAULT_WATCHDOG_TIMEOUT,
    ) -> None:
        self._interface = interface
        self._watchdog_timeout = watchdog_timeout
        self._lock = threading.Lock()
        self._results: dict[int, DirectiveResult] = {}
        self._next_directive_id = 0
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._stop_reason = ""
        self._last_contact = time.monotonic()
        self._tcp_server = _TCPServer((host, port), self)
        self._serve_thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        """The bound port (useful when constructed with port 0)."""
        return int(self._tcp_server.server_address[1])

    def serve_forever(self) -> None:
        """Serve until close is called. Blocks."""
        self._tcp_server.serve_forever()

    def start(self) -> None:
        """Serve on a background thread (for tests and embedding)."""
        self._serve_thread = threading.Thread(
            target=self._tcp_server.serve_forever, daemon=True
        )
        self._serve_thread.start()

    def close(self) -> None:
        """Stop serving and abort any running directive."""
        self._request_stop("server closed")
        worker = self._worker
        if worker is not None:
            worker.join()
        self._tcp_server.shutdown()
        self._tcp_server.server_close()
        if self._serve_thread is not None:
            self._serve_thread.join()

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        """Dispatch one decoded request and return the response dict."""
        with self._lock:
            self._last_contact = time.monotonic()
        op = request.get("op")
        if op == "start":
            return self._handle_start(request)
        if op == "status":
            return self._handle_status(request)
        if op == "observe":
            observation = self._interface.get_observation()
            return {"ok": True, "observation": encode_message(observation)}
        if op == "stop":
            self._request_stop("stop requested")
            return {"ok": True}
        return {"ok": False, "error": f"Unknown op: {op}"}

    def _handle_start(self, request: dict[str, Any]) -> dict[str, Any]:
        try:
            directive = decode_message(request["directive"])
        except (KeyError, ValueError, TypeError, AssertionError) as e:
            return {"ok": False, "error": f"Bad directive: {e}"}
        if isinstance(directive, PolicyRolloutDirective):
            return {"ok": False, "error": "Policy rollout is not implemented yet"}
        if not isinstance(directive, TrajectoryDirective):
            return {"ok": False, "error": "Not a directive"}
        try:
            validate_trajectory(directive.as_array(), directive.component, directive.hz)
        except ValueError as e:
            return {"ok": False, "error": f"Trajectory rejected: {e}"}
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return {"ok": False, "error": "A directive is already running"}
            directive_id = self._next_directive_id
            self._next_directive_id += 1
            self._results[directive_id] = DirectiveResult(DirectiveStatus.RUNNING)
            self._stop_event.clear()
            self._stop_reason = ""
            self._worker = threading.Thread(
                target=self._run_trajectory, args=(directive_id, directive), daemon=True
            )
            self._worker.start()
        return {"ok": True, "directive_id": directive_id}

    def _handle_status(self, request: dict[str, Any]) -> dict[str, Any]:
        directive_id = request.get("directive_id")
        result = None
        if isinstance(directive_id, int):
            with self._lock:
                result = self._results.get(directive_id)
        if result is None:
            return {"ok": False, "error": f"Unknown directive_id: {directive_id}"}
        return {"ok": True, "result": encode_message(result)}

    def _request_stop(self, reason: str) -> None:
        with self._lock:
            if not self._stop_event.is_set():
                self._stop_reason = reason
            self._stop_event.set()

    def _should_stop(self) -> bool:
        if self._stop_event.is_set():
            return True
        with self._lock:
            silence = time.monotonic() - self._last_contact
        if silence > self._watchdog_timeout:
            self._request_stop(f"watchdog: no orchestrator contact for {silence:.2f}s")
            return True
        return False

    def _run_trajectory(
        self, directive_id: int, directive: TrajectoryDirective
    ) -> None:
        component = _JointInterfaceComponent(
            {
                "right_arm": self._interface.right_arm_interface,
                "left_arm": self._interface.left_arm_interface,
                "head": self._interface.head_interface,
            }[directive.component]
        )
        start_time = time.monotonic()
        try:
            max_error = follow_joint_trajectory(
                component,
                directive.as_array(),
                hz=directive.hz,
                max_start_error=directive.max_start_error,
                max_tracking_error=directive.max_tracking_error,
                should_stop=self._should_stop,
            )
        except (ValueError, RuntimeError) as e:
            result = DirectiveResult(
                DirectiveStatus.FAILED,
                message=str(e),
                duration=time.monotonic() - start_time,
            )
        else:
            if self._stop_event.is_set():
                result = DirectiveResult(
                    DirectiveStatus.STOPPED,
                    message=self._stop_reason,
                    max_tracking_error=max_error,
                    duration=time.monotonic() - start_time,
                )
            else:
                result = DirectiveResult(
                    DirectiveStatus.SUCCEEDED,
                    max_tracking_error=max_error,
                    duration=time.monotonic() - start_time,
                )
        with self._lock:
            self._results[directive_id] = result


class _TCPServer(socketserver.ThreadingTCPServer):
    """TCP server holding a reference back to the SkillServer."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self, server_address: tuple[str, int], skill_server: SkillServer
    ) -> None:
        super().__init__(server_address, _RequestHandler)
        self.skill_server = skill_server


class _RequestHandler(socketserver.StreamRequestHandler):
    """One connection: a hello handshake, then newline-delimited JSON
    requests until EOF."""

    def handle(self) -> None:
        assert isinstance(self.server, _TCPServer)
        skill_server = self.server.skill_server
        hello_done = False
        for line in self.rfile:
            try:
                request = json.loads(line)
            except json.JSONDecodeError as e:
                self._respond({"ok": False, "error": f"Bad JSON: {e}"})
                return
            if not hello_done:
                response = self._handle_hello(request)
                self._respond(response)
                if not response["ok"]:
                    return
                hello_done = True
                continue
            self._respond(skill_server.handle_request(request))

    def _handle_hello(self, request: dict[str, Any]) -> dict[str, Any]:
        if request.get("op") != "hello":
            return {"ok": False, "error": "First request must be a hello"}
        client_version = request.get("protocol_version")
        if client_version != PROTOCOL_VERSION:
            return {
                "ok": False,
                "error": (
                    f"Protocol version mismatch: client {client_version}, "
                    f"server {PROTOCOL_VERSION}. Are both machines on the "
                    "same branch?"
                ),
                "protocol_version": PROTOCOL_VERSION,
            }
        return {"ok": True, "protocol_version": PROTOCOL_VERSION}

    def _respond(self, response: dict[str, Any]) -> None:
        self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument(
        "--watchdog-timeout", type=float, default=DEFAULT_WATCHDOG_TIMEOUT
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Serve a FakeInterface instead of connecting to the robot",
    )
    args = parser.parse_args()
    interface: Interface = FakeInterface() if args.fake else RealInterface()
    server = SkillServer(
        interface,
        host=args.host,
        port=args.port,
        watchdog_timeout=args.watchdog_timeout,
    )
    print(
        f"Skill server on {args.host}:{server.port} "
        f"(protocol {PROTOCOL_VERSION}, fake={args.fake})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.close()
        interface.close()


if __name__ == "__main__":
    _main()
