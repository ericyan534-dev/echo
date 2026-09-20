"""SpeakerGate: proximity-based wearer confidence, and its fail-open contract.

The tests are ordered by importance, and the FAIL-OPEN cases come first on
purpose.  `observe()` returns `float | None`; `None` means "I cannot tell" and
must never lead to suppression anywhere in the system.  Wrongly muting the
wearer is a far worse product failure than admitting a bystander, so every
"unknown" path is tested before any "suppress" path.
"""
import math

import pytest
import torch

from backend.acoustic.features import SR
from backend.acoustic.speaker_gate import SpeakerGate
from backend.acoustic.stream import FRAME, AcousticStream

VOICED = 0.9      # speech_prob well above the VAD threshold
UNVOICED = 0.05   # speech_prob well below it


def mk_frame(dbfs: float, hf: float = 1.0, n: int = FRAME, sr: int = SR) -> torch.Tensor:
    """A voiced-ish frame at exactly `dbfs` RMS, with adjustable high-frequency
    content (`hf` scales the 2.5-6 kHz harmonics; hf<1 mimics distant speech,
    which loses highs to air absorption and off-axis mic response).

    RMS is normalized AFTER the hf scaling, so `dbfs` and `hf` are independent:
    a tilt test cannot accidentally be a level test.
    """
    t = torch.arange(n, dtype=torch.float32) / sr
    x = torch.zeros(n)
    for f, a in ((150.0, 1.0), (300.0, 0.7), (600.0, 0.5), (1000.0, 0.4)):
        x = x + a * torch.sin(2 * math.pi * f * t)
    for f, a in ((2500.0, 0.5), (4000.0, 0.4), (6000.0, 0.3)):
        x = x + hf * a * torch.sin(2 * math.pi * f * t)
    x = x / x.pow(2).mean().sqrt()          # unit RMS (= 0 dBFS)
    return x * (10.0 ** (dbfs / 20.0))


def calibrated(level_dbfs: float = -20.0, hf: float = 1.0, **kw) -> SpeakerGate:
    """A gate whose baseline is established at `level_dbfs`."""
    g = SpeakerGate(**kw)
    for _ in range(g.calibration_frames + 5):
        g.observe(mk_frame(level_dbfs, hf=hf), VOICED)
    return g


# --- FAIL OPEN (the contract that matters most) ---------------------------
def test_cold_gate_returns_none_never_a_low_score():
    """A gate that has not calibrated yet must say 'unknown', not 'quiet'.
    Returning a low confidence at cold start would mute the wearer's first
    words -- exactly when Echo is most needed."""
    g = SpeakerGate()
    out = [g.observe(mk_frame(-20.0), VOICED) for _ in range(g.calibration_frames)]
    assert all(c is None for c in out), f"cold gate leaked a score: {out}"


def test_cold_gate_returns_none_for_quiet_frames_too():
    """Same contract for genuinely quiet input: with no baseline there is
    nothing to compare against, so the answer is None, not 0.0."""
    g = SpeakerGate()
    out = [g.observe(mk_frame(-45.0), VOICED) for _ in range(g.calibration_frames)]
    assert all(c is None for c in out)


def test_non_speech_frames_return_none():
    """The gate answers 'is this the wearer', NOT 'is this speech'. Silero VAD
    already owns the speech question; a non-speech frame is unknown, and it must
    not be counted toward the baseline either."""
    g = calibrated()
    assert g.observe(mk_frame(-20.0), UNVOICED) is None
    assert g.observe(mk_frame(-45.0), UNVOICED) is None


def test_non_speech_frames_do_not_calibrate():
    """Room noise must not be able to establish the wearer's baseline."""
    g = SpeakerGate()
    for _ in range(200):
        g.observe(mk_frame(-20.0), UNVOICED)
    assert g.baseline_dbfs is None
    assert g.observe(mk_frame(-20.0), VOICED) is None  # still uncalibrated


def test_silence_and_zero_frames_never_raise():
    """log10(0) is -inf: digital silence, a muted mic, and a dropped buffer must
    all degrade to None rather than blowing up the audio socket."""
    g = calibrated()
    for frame in (torch.zeros(FRAME), torch.zeros(0), torch.full((FRAME,), 1e-12)):
        assert g.observe(frame, VOICED) is None
        assert g.observe(frame, UNVOICED) is None


def test_non_finite_frames_return_none():
    """A NaN/inf frame (bad decode) is unknown, not suppressed."""
    g = calibrated()
    bad = mk_frame(-20.0).clone()
    bad[10] = float("nan")
    assert g.observe(bad, VOICED) is None
    bad[10] = float("inf")
    assert g.observe(bad, VOICED) is None


def test_gate_never_suppresses_continuously_past_max_suppress_ms():
    """Hard safety cap. Even if the level evidence keeps saying 'not the wearer',
    the gate must give up and admit it cannot tell -- otherwise a wearer who
    moves the mic (or whose voice weakens mid-conversation) is muted forever."""
    g = calibrated(max_suppress_ms=500)
    run = worst = 0
    for _ in range(200):
        c = g.observe(mk_frame(-40.0), VOICED)
        if c is not None and c < 0.5:
            run += 1
            worst = max(worst, run)
        else:
            run = 0
    frames_allowed = 500 // (FRAME * 1000 // SR)
    assert worst <= frames_allowed, (
        f"suppressed for {worst} frames in a row, cap is {frames_allowed}"
    )


def test_tilt_alone_can_never_suppress():
    """Spectral tilt is corroborating evidence, not a verdict: a frame at the
    calibrated LEVEL but with the highs stripped is softened, never zeroed."""
    g = calibrated(hf=1.0)
    conf = g.observe(mk_frame(-20.0, hf=0.02), VOICED)
    assert conf is not None
    assert conf < 0.99, "tilt term is not wired in at all"
    assert conf >= 1.0 - g.tilt_weight - 1e-6, "tilt penalty exceeded its weight"
    assert conf > 0.5, "tilt alone must not be able to suppress"


# --- discrimination -------------------------------------------------------
def test_frame_at_calibrated_level_is_high_confidence():
    g = calibrated()
    conf = g.observe(mk_frame(-20.0), VOICED)
    assert conf is not None and conf >= 0.9


def test_frame_12db_quieter_is_low_confidence():
    g = calibrated()
    conf = g.observe(mk_frame(-32.0), VOICED)
    assert conf is not None and conf < 0.2


def test_louder_than_baseline_is_high_confidence():
    """Above the baseline is at least as likely to be the wearer -- never
    penalize loudness."""
    g = calibrated()
    conf = g.observe(mk_frame(-12.0), VOICED)
    assert conf is not None and conf >= 0.9


def test_confidence_is_monotone_in_level_separation():
    """0 / 3 / 6 / 9 / 12 dB below baseline -> non-increasing confidence, all in
    0..1. A fresh gate per level so the baseline can't drift between probes."""
    seps = [0.0, 3.0, 6.0, 9.0, 12.0]
    confs = []
    for sep in seps:
        g = calibrated()
        c = g.observe(mk_frame(-20.0 - sep), VOICED)
        assert c is not None and 0.0 <= c <= 1.0
        confs.append(c)
    assert confs == sorted(confs, reverse=True), dict(zip(seps, confs))
    assert confs[0] >= 0.9 and confs[-1] < 0.2


def test_baseline_adapts_to_a_sustained_new_level():
    """The wearer moving the mic (the lav shifting on the collar) drops the
    level permanently. That must become the new normal, not a permanent mute."""
    g = calibrated()
    out = [g.observe(mk_frame(-32.0), VOICED) for _ in range(160)]
    assert out[0] is not None and out[0] < 0.2      # suppressed at first
    assert None in out                               # then admits uncertainty
    tail = out[-10:]
    assert all(c is not None and c > 0.8 for c in tail), (
        f"baseline never adapted; tail={tail}"
    )


def test_baseline_tracks_the_loudest_recent_voiced_frames():
    """Baseline = high percentile of the rolling voiced window, so a bystander
    interjecting cannot immediately drag the wearer's baseline down."""
    g = calibrated(-20.0)
    before = g.baseline_dbfs
    for _ in range(5):
        g.observe(mk_frame(-35.0), VOICED)
    assert g.baseline_dbfs is not None
    assert g.baseline_dbfs > before - 3.0, "5 quiet frames moved the baseline too far"


def test_confidence_always_in_unit_range():
    g = calibrated()
    for dbfs in (-3.0, -10.0, -20.0, -25.0, -30.0, -50.0, -65.0):
        c = g.observe(mk_frame(dbfs), VOICED)
        assert c is None or 0.0 <= c <= 1.0


def test_accepts_odd_frame_lengths_and_shapes():
    """The stream slices fixed 800-sample frames, but eval harnesses and the
    browser worklet's 100 ms buffers must not be able to crash the gate."""
    for n in (256, 800, 1600):
        g = SpeakerGate(calibration_frames=4)
        for _ in range(6):
            g.observe(mk_frame(-20.0, n=n), VOICED)
        c = g.observe(mk_frame(-20.0, n=n), VOICED)
        assert c is not None and c >= 0.9, f"n={n} -> {c}"
    g2 = calibrated()                                     # (2, 800) -> flattened
    c = g2.observe(mk_frame(-20.0, n=1600).reshape(2, 800), VOICED)
    assert c is not None and 0.0 <= c <= 1.0


# --- AcousticStream wiring ------------------------------------------------
class _FixedGate:
    """Stand-in for SpeakerGate that always answers the same thing."""

    def __init__(self, value):
        self.value = value
        self.calls = 0

    def observe(self, frame, speech_prob):
        self.calls += 1
        return self.value


def _voiced_stream(gate_value, **kw) -> AcousticStream:
    s = AcousticStream(model_path=None, wearer_gate=True, **kw)
    s.vad = lambda chunk, sr: torch.tensor(0.9)          # force "speech"
    s.prolong.observe_frame = lambda frame, now_ms: True  # force a candidate event
    s.speaker_gate = _FixedGate(gate_value)
    return s


def _feed_2s(s: AcousticStream) -> list:
    pcm = (mk_frame(-20.0, n=1600) * 32767).to(torch.int16).numpy().tobytes()
    events = []
    for _ in range(20):   # 20 x 100 ms = 2 s, past min_voiced_ms
        events.extend(s.feed(pcm))
    return events


def test_stream_emits_when_gate_returns_none():
    """FAIL OPEN through the whole stream: unknown confidence must not suppress."""
    s = _voiced_stream(None)
    assert len(_feed_2s(s)) >= 1
    assert s.speaker_gate.calls > 0, "gate was never consulted"


def test_stream_suppresses_events_below_the_threshold():
    s = _voiced_stream(0.05)
    assert _feed_2s(s) == []


def test_stream_emits_when_gate_is_confident():
    s = _voiced_stream(0.95)
    assert len(_feed_2s(s)) >= 1


def test_stream_suppression_does_not_consume_the_refractory():
    """A suppressed event must not arm the refractory: if it did, one bystander
    frame would blank the wearer for the next refractory_ms."""
    s = AcousticStream(model_path=None, wearer_gate=True)
    s._voiced_ms = 900
    s._wearer_conf = 0.05
    assert s._gate("prolongation") is False
    s._wearer_conf = None                 # unknown -> fail open
    assert s._gate("prolongation") is True


def test_stream_gate_disabled_by_default_and_conf_is_none():
    """Off unless asked for: eval harnesses construct AcousticStream directly
    and must keep measuring the FillerNet path unconfounded. The integrator
    turns it on from Settings."""
    s = AcousticStream(model_path=None)
    assert s.speaker_gate is None
    assert s.wearer_conf is None
    s._voiced_ms = 900
    assert s._gate("filler") is True


def test_stream_ships_the_measured_conf_thresh_operating_point():
    """0.75 is the published operating point (docs/EVAL + eval/run_noise_stress).
    Adding a kwarg must not move it, and neither must a refactor.

    Formerly this asserted __init__.__defaults__[1] == 0.75 -- a POSITIONAL
    index into the defaults tuple. Inserting any parameter before conf_thresh
    repointed it at a neighbour's value, so the pin could go on passing while
    guarding the wrong number. Read by name instead.
    """
    import inspect

    params = inspect.signature(AcousticStream.__init__).parameters
    assert params["conf_thresh"].default == 0.75
    # and the live instance actually adopts it (a default nobody reads is not
    # an operating point)
    assert AcousticStream(model_path=None).conf_thresh == 0.75


def test_stream_with_real_gate_survives_real_frames():
    """End-to-end smoke: the real SpeakerGate inside the real feed() path."""
    s = AcousticStream(model_path=None, wearer_gate=True)
    s.vad = lambda chunk, sr: torch.tensor(0.9)
    s.prolong.observe_frame = lambda frame, now_ms: True
    assert len(_feed_2s(s)) >= 1          # constant level -> wearer, not muted
    assert s.wearer_conf is None or 0.0 <= s.wearer_conf <= 1.0


# --- Word.wearer_conf + /ws parsing --------------------------------------
def test_word_has_optional_wearer_conf_defaulting_to_none():
    from backend.schemas import Word

    assert Word(text="hi").wearer_conf is None
    assert Word(text="hi", wearer_conf=0.4).wearer_conf == 0.4


def test_opt_conf_parser_never_raises_and_clamps():
    from backend.app import _opt_conf

    assert _opt_conf(None) is None
    assert _opt_conf("junk") is None          # unparseable -> unknown
    assert _opt_conf([]) is None
    assert _opt_conf(float("nan")) is None
    assert _opt_conf(float("inf")) is None
    assert _opt_conf(0.42) == pytest.approx(0.42)
    assert _opt_conf("0.42") == pytest.approx(0.42)
    assert _opt_conf(5.0) == 1.0              # clamped, not trusted
    assert _opt_conf(-2.0) == 0.0


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.app import app
    from backend.session import reset_session

    monkeypatch.setenv("PREDICTOR_PROVIDER", "mock")
    monkeypatch.setenv("PREFETCH", "off")
    reset_session()
    yield TestClient(app)
    reset_session()


def test_ws_word_accepts_wearer_conf(client):
    with client.websocket_connect("/ws") as ws:
        t = 0
        for w in ["I", "made", "some", "toast", "in", "the"]:
            ws.send_json({"type": "word", "text": w, "start_ms": t, "end_ms": t + 280,
                          "is_final": True, "wearer_conf": 0.9})
            t += 400
        ws.send_json({"type": "silence", "at_ms": t + 1500})
        pred = ws.receive_json()
        assert pred["type"] == "prediction"
        assert pred["fragment"] == "I made some toast in the"


def test_ws_bad_wearer_conf_is_ignored_and_word_survives(client):
    """A malformed hint must degrade to 'unknown' -- it must neither kill the
    socket nor silently drop the word it rode in on."""
    with client.websocket_connect("/ws") as ws:
        t = 0
        for i, w in enumerate(["I", "made", "some", "toast", "in", "the"]):
            bad = ["nope", None, {}, [], float("nan"), "1e999999"][i]
            ws.send_json({"type": "word", "text": w, "start_ms": t, "end_ms": t + 280,
                          "is_final": True, "wearer_conf": bad})
            t += 400
        ws.send_json({"type": "silence", "at_ms": t + 1500})
        pred = ws.receive_json()
        assert pred["type"] == "prediction"
        assert pred["fragment"] == "I made some toast in the"
