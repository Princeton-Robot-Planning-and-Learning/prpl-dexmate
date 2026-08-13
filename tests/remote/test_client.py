"""Tests for prpl_dexmate.remote.client (handshake behavior)."""

import json
import socket
from typing import Iterator

import pytest

from prpl_dexmate.interfaces.interface import FakeInterface
from prpl_dexmate.remote.client import ProtocolMismatchError, SkillClient
from prpl_dexmate.remote.server import SkillServer


@pytest.fixture(name="server")
def _server_fixture() -> Iterator[SkillServer]:
    interface = FakeInterface()
    server = SkillServer(interface, host="127.0.0.1", port=0)
    server.start()
    yield server
    server.close()
    interface.close()


def test_version_mismatch_fails_at_connect(
    server: SkillServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A client on a different protocol version cannot connect."""
    monkeypatch.setattr("prpl_dexmate.remote.client.PROTOCOL_VERSION", "0000deadbeef")
    with pytest.raises(ProtocolMismatchError, match="version mismatch"):
        SkillClient("127.0.0.1", server.port)


def test_hello_is_required_first(server: SkillServer) -> None:
    """A request before the hello handshake is rejected."""
    with socket.create_connection(("127.0.0.1", server.port), timeout=5.0) as sock:
        sock.sendall(json.dumps({"op": "stop"}).encode("utf-8") + b"\n")
        response = json.loads(sock.makefile("r").readline())
    assert not response["ok"]
    assert "hello" in response["error"]


def test_matching_versions_connect(server: SkillServer) -> None:
    """The default handshake succeeds and the connection is usable."""
    client = SkillClient("127.0.0.1", server.port)
    client.stop()  # A no-op round trip proving the connection works.
    client.close()
