from __future__ import annotations

from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from httpx import Headers
from starlette.websockets import WebSocketDisconnect

from volta_api import app as api
from volta_api.config import _parse_origins


@pytest.fixture
def client(monkeypatch):
    class SessionStub:
        def __init__(self, websocket, *args):
            self.websocket = websocket

        async def handle(self, message):
            await self.websocket.send_json({"received": "json"})

        async def handle_binary_audio(self, data):
            await self.websocket.send_json({"received": "binary"})

        async def close(self):
            pass

    monkeypatch.setattr(api, "Session", SessionStub)
    monkeypatch.setattr(api, "PublicRealtimeSession", SessionStub)
    return TestClient(api.app)


@pytest.mark.parametrize("path", ["/ws", "/v1/realtime"])
@pytest.mark.parametrize(
    "origin",
    [
        "https://attacker.example",
        "null",
        "http://localhost:4173.attacker.example",
        "http://localhost:4173/",
        "http://localhost:4173, https://attacker.example",
        "",
        " ",
        "http://localhost:4173@attacker.example",
        "ws://localhost:4173",
    ],
)
def test_untrusted_origin_rejected_before_session_creation(
    client, monkeypatch, path, origin
):
    def unexpected_session(*args):
        pytest.fail("Rejected connection created a session")

    monkeypatch.setattr(api, "Session", unexpected_session)
    monkeypatch.setattr(api, "PublicRealtimeSession", unexpected_session)
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(path, headers={"origin": origin}):
            pytest.fail("Untrusted browser origin connected")
    assert error.value.code == 1008


@pytest.mark.parametrize("path", ["/ws", "/v1/realtime"])
@pytest.mark.parametrize("other", ["https://attacker.example", "http://localhost:4173"])
def test_duplicate_origin_rejected(client, path, other):
    with pytest.raises(WebSocketDisconnect) as error:
        with client.websocket_connect(
            path,
            headers=Headers(
                [
                    ("origin", "http://localhost:4173"),
                    ("origin", other),
                ]
            ),
        ):
            pytest.fail("Multiple origins connected")
    assert error.value.code == 1008


@pytest.mark.parametrize("path", ["/ws", "/v1/realtime"])
@pytest.mark.parametrize(
    "origin",
    [
        None,
        "http://localhost:4173",
        "http://localhost:4174",
        "http://127.0.0.1:4173",
        "http://127.0.0.1:4174",
    ],
)
def test_trusted_browser_and_native_clients_connect(client, path, origin):
    headers = {"origin": origin} if origin is not None else {}
    with client.websocket_connect(
        path, headers=headers, subprotocols=["realtime"]
    ) as websocket:
        if path == "/v1/realtime":
            assert websocket.accepted_subprotocol == "realtime"


def test_configured_browser_origin_connects(client, monkeypatch):
    monkeypatch.setattr(
        api,
        "settings",
        replace(api.settings, allowed_origins=("http://localhost:5000",)),
    )
    with client.websocket_connect(
        "/v1/realtime", headers={"origin": "http://localhost:5000"}
    ):
        pass


@pytest.mark.parametrize("path", ["/ws", "/v1/realtime"])
def test_allowed_json_messages_reach_session(client, path):
    with client.websocket_connect(
        path, headers={"origin": "http://localhost:4173"}
    ) as websocket:
        websocket.send_json({"type": "session.update", "session": {}})
        assert websocket.receive_json() == {"received": "json"}


def test_allowed_binary_audio_reaches_realtime_session(client):
    with client.websocket_connect(
        "/v1/realtime", headers={"origin": "http://localhost:4173"}
    ) as websocket:
        websocket.send_bytes(b"synthetic-audio")
        assert websocket.receive_json() == {"received": "binary"}


@pytest.mark.parametrize(
    "value",
    [
        "*",
        "null",
        "",
        "http://localhost:4173/",
        "ws://localhost:4173",
        "http://user@localhost:4173",
        "http://localhost:99999",
        "https://example.com?x=1",
    ],
)
def test_invalid_origin_configuration_rejected(value):
    with pytest.raises(ValueError):
        _parse_origins(value)


def test_origin_configuration_is_explicit_and_deduplicated():
    assert _parse_origins(
        "http://localhost:5000, https://voice.example, http://localhost:5000"
    ) == (
        "http://localhost:5000",
        "https://voice.example",
    )
