"""Proof that the autouse network guard in conftest actually blocks the network.

The characterization suite is only trustworthy if it is hermetic. In particular
a stray call to the local LiteLLM gateway (localhost:18400) would spend real
money on the `inbox-monitor` virtual key and make the suite non-deterministic.
These tests assert the guard raises rather than assuming it does.
"""

import socket
import subprocess

import httpx
import pytest

from tests.conftest import NetworkAccessAttempted

pytestmark = pytest.mark.unit

GATEWAY_HOST = "localhost"
GATEWAY_PORT = 18400


def test_guard_blocks_raw_socket_connect_to_gateway():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(NetworkAccessAttempted):
            sock.connect((GATEWAY_HOST, GATEWAY_PORT))
    finally:
        sock.close()


def test_guard_blocks_create_connection_to_gateway():
    with pytest.raises(NetworkAccessAttempted):
        socket.create_connection((GATEWAY_HOST, GATEWAY_PORT))


def test_guard_blocks_http_post_to_gateway():
    """The realistic shape of an accidental spend: an HTTP call to the gateway."""
    with pytest.raises(NetworkAccessAttempted):
        httpx.post(
            f"http://{GATEWAY_HOST}:{GATEWAY_PORT}/v1/chat/completions",
            json={"model": "tier-smart", "messages": []},
        )


def test_guard_blocks_dns_resolution_of_provider():
    with pytest.raises(NetworkAccessAttempted):
        socket.getaddrinfo("api.openai.com", 443)


def test_guard_blocks_readwise():
    with pytest.raises(NetworkAccessAttempted):
        httpx.get("https://readwise.io/api/v3/list/")


def test_guard_permits_subprocess():
    """The persona tier's judge is an out-of-process subscription CLI.

    The guard patches sockets in the pytest process only, so a subprocess is
    outside it by construction. That is the single sanctioned route to a
    network, and it costs no metered budget: subscription CLIs do not route
    through the LiteLLM gateway.
    """
    completed = subprocess.run(["echo", "judge"], capture_output=True, text=True, check=True)
    assert completed.stdout.strip() == "judge"


def test_guard_permits_unix_sockets():
    """AF_UNIX must stay open: asyncio internals and aiosqlite rely on it."""
    left, right = socket.socketpair(socket.AF_UNIX)
    try:
        left.send(b"ok")
        assert right.recv(2) == b"ok"
    finally:
        left.close()
        right.close()
