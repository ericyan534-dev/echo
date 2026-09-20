"""Capture-hardware registry (backend/audio_sources.py) and its endpoints.

The matcher must find the ASCII endpoint name inside a localised Windows
label, prefer the specific wireless-lav profile over the generic onboard
patterns, and return None for hardware it has never heard of rather than
guessing. The endpoints must store what the page reported, resolve a null
profile_id from the label, refuse an unknown profile_id loudly, and surface
the result in /healthz and /api/config -- all without changing a threshold.

TestClient is used without the app-level context manager so the lifespan
pre-warm never runs in CI (same convention as tests/test_ws_reject.py).
"""
import math

import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.audio_sources import (
    KINDS, PROFILES, AudioSource, get_profile, match_profile, profiles_payload,
)
from backend.session import reset_session


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("PREDICTOR_PROVIDER", "mock")
    monkeypatch.setenv("PREFETCH", "off")
    reset_session()
    yield TestClient(app)
    reset_session()


# ---------------------------------------------------------------- matcher
def test_registry_is_well_formed():
    ids = [p.id for p in PROFILES]
    assert len(ids) == len(set(ids)), "duplicate profile id"
    for p in PROFILES:
        assert p.kind in KINDS
        assert set(p.constraints) == {"echoCancellation", "noiseSuppression", "autoGainControl"}
        assert all(isinstance(v, bool) for v in p.constraints.values())
        assert p.channel_count >= 1
        assert p.match, "a profile with no patterns can never be matched"
        assert p.notes


@pytest.mark.parametrize("label", [
    "Wireless Mic Rx",
    "Microphone (Wireless Mic Rx)",
    # Chinese-locale Windows: CJK prefix, ASCII endpoint name in parentheses.
    "\u9ea6\u514b\u98ce (Wireless Mic Rx)",
    "wireless mic rx",                       # case-insensitive
    "DJI MIC",                               # macOS / Chrome label
    "Default - DJI Mic 2S (2ca3:4015)",      # Chrome's default-device label form
])
def test_matcher_flags_dji_receiver(label):
    p = match_profile(label)
    assert p is not None and p.id == "dji-mic-2s"
    assert p.kind == "lav-wireless"
    assert p.wearer_gate is True


@pytest.mark.parametrize("label", [
    "Microphone Array (Realtek(R) Audio)",
    "\u9ea6\u514b\u98ce\u9635\u5217 (Realtek(R) Audio)",
    "MacBook Pro Microphone",
    "Built-in Microphone",
])
def test_matcher_flags_onboard_array(label):
    p = match_profile(label)
    assert p is not None and p.id == "laptop-array"
    assert p.kind == "onboard-array"
    assert p.wearer_gate is False


def test_matcher_prefers_dji_over_generic_words():
    # A label mentioning both must resolve to the specific hardware, not to
    # whichever generic pattern happens to appear first in the string.
    assert match_profile("Realtek passthrough of Wireless Mic Rx").id == "dji-mic-2s"


def test_matcher_returns_none_for_unknown_hardware():
    assert match_profile("USB Audio Device") is None
    assert match_profile("") is None
    assert match_profile(None) is None
    # Generic English words that happen to name audio gear must not match:
    # the DJI patterns are the receiver's endpoint name and the brand word.
    assert match_profile("paper clip holder") is None
    assert match_profile("wireless headset") is None
    assert match_profile("Microphone (USB Wireless Mic)") is None  # no "Rx"


def test_dji_constraints_disable_browser_processing():
    """The recommendation that matters: AGC normalises the level the speaker
    gate reads, so every processing stage is off for the lav."""
    p = get_profile("dji-mic-2s")
    assert p.constraints == {
        "echoCancellation": False, "noiseSuppression": False, "autoGainControl": False}
    assert p.channel_count == 1   # both receiver channels are bit-identical


def test_laptop_constraints_match_what_the_page_requests_today():
    """frontend/app.js startAudioChannel asks for AEC on / NS off; AGC is
    Chrome's default (on). This profile must not silently move that."""
    p = get_profile("laptop-array")
    assert p.constraints["echoCancellation"] is True
    assert p.constraints["noiseSuppression"] is False
    assert p.constraints["autoGainControl"] is True


def test_get_profile_is_lenient_about_case_and_blank():
    assert get_profile("DJI-MIC-2S").id == "dji-mic-2s"
    assert get_profile("") is None
    assert get_profile(None) is None
    assert get_profile("nope") is None


# ------------------------------------------------------------ AudioSource
def test_from_report_resolves_null_profile_from_label():
    src = AudioSource.from_report({"label": "Wireless Mic Rx", "deviceId": "abc",
                                   "profile_id": None, "floor_dbfs": -61.2})
    assert src.profile_id == "dji-mic-2s"
    assert src.device_id == "abc"
    assert src.floor_dbfs == pytest.approx(-61.2)
    d = src.to_dict()
    assert d["kind"] == "lav-wireless"
    assert d["wearer_gate_recommended"] is True
    assert d["display_name"] == "DJI Mic 2S (wireless lav)"


def test_from_report_keeps_unknown_hardware_as_unknown():
    src = AudioSource.from_report({"label": "USB Audio Device"})
    assert src.profile_id is None
    d = src.to_dict()
    assert d["kind"] is None and d["wearer_gate_recommended"] is None


def test_from_report_rejects_bad_input():
    with pytest.raises(ValueError):
        AudioSource.from_report({})
    with pytest.raises(ValueError):
        AudioSource.from_report({"label": "   "})
    with pytest.raises(ValueError):
        AudioSource.from_report({"label": "x", "profile_id": "not-a-profile"})


@pytest.mark.parametrize("raw", ["nan", float("nan"), float("inf"), "loud", True, None])
def test_from_report_floor_never_raises(raw):
    src = AudioSource.from_report({"label": "Wireless Mic Rx", "floor_dbfs": raw})
    assert src.floor_dbfs is None


def test_from_report_floor_accepts_numeric_string():
    src = AudioSource.from_report({"label": "Wireless Mic Rx", "floor_dbfs": "-58.5"})
    assert src.floor_dbfs == pytest.approx(-58.5)
    assert math.isfinite(src.floor_dbfs)


# -------------------------------------------------------------- endpoints
def test_get_sources_lists_every_profile(client):
    body = client.get("/api/audio/sources").json()
    assert body == profiles_payload()
    ids = [p["id"] for p in body["profiles"]]
    assert ids == ["dji-mic-2s", "laptop-array"]   # the lav must sit first
    dji = next(p for p in body["profiles"] if p["id"] == "dji-mic-2s")
    assert dji["match"] and isinstance(dji["match"], list)
    assert dji["constraints"]["autoGainControl"] is False


def test_healthz_and_config_report_no_source_before_any_post(client):
    assert client.get("/healthz").json()["audio_source"] is None
    assert client.get("/api/config").json()["audio_source"] is None


def test_post_source_is_stored_and_surfaced(client):
    r = client.post("/api/audio/source", json={
        "label": "\u9ea6\u514b\u98ce (Wireless Mic Rx)", "deviceId": "dev-1",
        "profile_id": None, "floor_dbfs": -61.0})
    assert r.status_code == 200, r.text
    echoed = r.json()
    assert echoed["profile_id"] == "dji-mic-2s"
    assert echoed["kind"] == "lav-wireless"
    assert echoed["floor_dbfs"] == pytest.approx(-61.0)

    h = client.get("/healthz").json()
    assert h["audio_source"]["profile_id"] == "dji-mic-2s"
    assert h["audio_source"]["deviceId"] == "dev-1"
    assert h["audio_source"]["wearer_gate_recommended"] is True
    cfg = client.get("/api/config").json()
    assert cfg["audio_source"] == h["audio_source"]
    assert "wearer_gate" in cfg and isinstance(cfg["wearer_gate"], bool)


def test_post_source_replaces_previous(client):
    client.post("/api/audio/source", json={"label": "Wireless Mic Rx"})
    client.post("/api/audio/source", json={
        "label": "Microphone Array (Realtek(R) Audio)", "deviceId": "dev-2"})
    src = client.get("/healthz").json()["audio_source"]
    assert src["profile_id"] == "laptop-array"
    assert src["wearer_gate_recommended"] is False


def test_post_source_explicit_profile_wins_over_label(client):
    r = client.post("/api/audio/source", json={
        "label": "USB Audio Device", "profile_id": "dji-mic-2s"})
    assert r.status_code == 200
    assert r.json()["profile_id"] == "dji-mic-2s"


def test_post_source_rejects_unknown_profile_and_missing_label(client):
    r = client.post("/api/audio/source", json={"label": "x", "profile_id": "bogus"})
    assert r.status_code == 400
    assert "bogus" in r.json()["detail"]
    r = client.post("/api/audio/source", json={"deviceId": "only"})
    assert r.status_code == 400
    # a rejected report must not clobber the stored source
    assert client.get("/healthz").json()["audio_source"] is None


def test_post_source_does_not_touch_thresholds(client):
    before = client.get("/api/config").json()
    client.post("/api/audio/source", json={"label": "Wireless Mic Rx"})
    after = client.get("/api/config").json()
    for key in ("stall_pause_ms", "stall_min_gap_ms", "asr_provider", "wearer_gate"):
        assert before[key] == after[key]
