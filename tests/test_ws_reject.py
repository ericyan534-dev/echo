"""WebSocket reject protocol: /ws {"type":"reject"}.

TestClient gotcha: each websocket_connect runs on its own event loop (genai
sessions would bind to the first caller's loop), so these tests force the mock
predictor and disable prefetch. TestClient is used without the app-level
context manager so the lifespan pre-warm (VAD download) never runs in CI.
"""
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.session import reset_session


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PREDICTOR_PROVIDER", "mock")
    monkeypatch.setenv("PREFETCH", "off")
    reset_session()
    yield TestClient(app)
    reset_session()


def _speak_until_stall(ws):
    """Drive 'I made some toast in the' + pause -> stall (mock: toaster/oven)."""
    t = 0
    for w in ["I", "made", "some", "toast", "in", "the"]:
        ws.send_json({"type": "word", "text": w, "start_ms": t, "end_ms": t + 280,
                      "is_final": True})
        t += 400
    ws.send_json({"type": "silence", "at_ms": t + 1500})
    return ws.receive_json()


def test_ws_reject_serves_replacement(client):
    with client.websocket_connect("/ws") as ws:
        pred = _speak_until_stall(ws)
        assert pred["type"] == "prediction"
        assert pred["candidates"][0]["word"] == "toaster"

        ws.send_json({"type": "reject", "rejected": ["toaster"]})
        rep = ws.receive_json()
        assert rep["type"] == "prediction"
        assert rep["served"] == "reject"
        assert rep["fragment"] == pred["fragment"]
        assert [c["word"] for c in rep["candidates"]] == ["oven"]


def test_ws_reject_accumulates_between_messages(client):
    with client.websocket_connect("/ws") as ws:
        _speak_until_stall(ws)
        ws.send_json({"type": "reject", "rejected": ["toaster"]})
        assert ws.receive_json()["candidates"][0]["word"] == "oven"
        # frontend sends the FULL rejected list each time; either way the
        # pipeline's per-stall union keeps earlier rejections excluded
        ws.send_json({"type": "reject", "rejected": ["toaster", "oven"]})
        rep = ws.receive_json()
        assert rep["served"] == "reject"
        assert rep["candidates"] == []  # both mock words excluded


def test_ws_reject_malformed_and_stall_less_never_kill_socket(client):
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "reject", "rejected": "toaster"})   # not a list
        ws.send_json({"type": "reject"})                          # missing field
        ws.send_json({"type": "reject", "rejected": ["x"]})       # no stall yet
        ws.send_json({"type": "word", "text": "hi", "start_ms": "bad"})  # int() blows
        ws.send_json({"type": "ping"})
        msg = ws.receive_json()  # none of the above produced output or a crash
        assert msg["type"] == "pong"
