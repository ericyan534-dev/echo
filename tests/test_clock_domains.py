"""Two clocks, one detector.

Repro (demo rehearsal, 2026-09-20: "the stutter was detected but no
recommendation came out, 7 rounds in a row"). In browser-ASR mode the words
and silence ticks carry the BROWSER clock -- ms since *Start listening*,
restarted at 0 on every Start and every page reload -- while acoustic events
carry the AUDIO clock, ms since the session's first-ever audio connection,
never restarted. StallDetector keeps one `_last_fire_ms` across all triggers
for the min-gap refractory, so:

  * an acoustic fire at audio-clock 180 000 made every browser-clock pause at
    ~25 000 compute as "too soon" (25000 - 180000 < 4000) -- for minutes;
  * any fire at all, followed by a Start/reload, suppressed every pause and
    filler until the fresh browser clock caught up with the old one.

The UI still mirrored the acoustic detection, so it LOOKED detected.
"""
from __future__ import annotations

import asyncio

from backend.schemas import AcousticEvent, SilenceTick, Word
from backend.stall_detector import StallDetector


def _say(det, words, t0=0, step=400):
    t = t0
    for w in words:
        det.observe_word(Word(text=w, start_ms=t, end_ms=t + 280, is_final=True))
        t += step
    return t


# --- StallDetector: a clock that jumps backwards is a restarted clock ---------

def test_a_restarted_clock_does_not_inherit_the_old_refractory():
    det = StallDetector(pause_ms=1300, min_gap_ms=4000)
    t = _say(det, ["I", "put", "the", "bread", "in", "the"], t0=60000)
    assert det.observe_silence(t + 1500).trigger == "pause"      # fires at ~63.9 s
    # The user presses Start again: the browser clock restarts at 0.
    det.reset()
    t = _say(det, ["and", "then", "I", "opened", "the"], t0=0)
    ev = det.observe_silence(t + 1500)
    assert ev is not None and ev.trigger == "pause", \
        "a fresh clock must not be measured against a fire from the old one"


def test_an_acoustic_fire_on_another_clock_does_not_mute_the_transcript():
    det = StallDetector(pause_ms=1300, min_gap_ms=4000)
    _say(det, ["I", "put", "the", "bread"], t0=20000)
    # Acoustic channel on the audio clock (session-old, 3 minutes ahead).
    assert det.observe_acoustic(AcousticEvent(kind="block", at_ms=180000)) is not None
    det.reset()
    t = _say(det, ["in", "the", "kitchen", "near", "the"], t0=24000)
    ev = det.observe_silence(t + 1500)
    assert ev is not None, "a browser-clock pause must still fire after an audio-clock fire"


def test_a_small_backwards_step_is_still_inside_the_gap():
    """Jitter between channels is not a restart: an event stamped 300 ms
    before the last fire is inside the 4 s gap and stays suppressed."""
    det = StallDetector(pause_ms=1300, min_gap_ms=4000)
    _say(det, ["I", "put", "the", "bread", "in", "the"], t0=0)
    assert det.observe_silence(4000) is not None                 # fire at 4000
    det.reset()
    _say(det, ["toaster", "and"], t0=4200)
    assert det.observe_acoustic(AcousticEvent(kind="block", at_ms=3700)) is None
    assert det.observe_acoustic(AcousticEvent(kind="block", at_ms=6000)) is None
    assert det.observe_acoustic(AcousticEvent(kind="block", at_ms=8100)) is not None


# --- EchoSession: one session clock; browser timestamps are lifted onto it --

def _mock_settings(monkeypatch, **env):
    import backend.config as config_mod
    monkeypatch.setenv("PREDICTOR_PROVIDER", "mock")
    monkeypatch.setenv("ACOUSTIC_MODEL", "")
    monkeypatch.setenv("STUTTER_MODEL", "")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return config_mod.get_settings()


class _FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _session(monkeypatch):
    import backend.session as session_mod
    clock = _FakeClock()
    monkeypatch.setattr(session_mod.time, "monotonic", clock)
    return session_mod.EchoSession(_mock_settings(monkeypatch)), clock


def test_browser_timestamps_are_lifted_onto_the_session_clock(monkeypatch):
    sess, clock = _session(monkeypatch)          # session epoch at t=1000
    clock.t = 1180.0                             # 180 s later ...
    # ... the browser, which pressed Start 25 s ago, reports 25000.
    assert sess.ui_to_session(25000) == 180000
    # An acoustic event "now" on the session clock is at 180000 too: the
    # two channels finally agree.
    assert sess.new_audio_channels().clock_offset_ms == 180000


def test_consecutive_browser_timestamps_keep_their_exact_spacing(monkeypatch):
    """The pin is held while the browser clock is continuous, so jitter in
    message arrival never changes the gap between two words -- the pause
    detector measures exactly that gap."""
    sess, clock = _session(monkeypatch)
    clock.t = 1100.0
    a = sess.ui_to_session(5000)
    clock.t = 1100.6                     # the next tick arrives 600 ms late
    b = sess.ui_to_session(5250)         # (a busy loop), still 250 ms later
    assert b - a == 250                  # by the browser's own clock


def test_a_restarted_browser_clock_is_repinned(monkeypatch):
    """Start pressed again / page reloaded: the browser goes back to 0 but
    the session clock keeps counting, so a stall 400 ms into the new
    session is NOT 'too soon' after a fire 3.9 s into the old one."""
    sess, clock = _session(monkeypatch)
    clock.t = 1010.0
    assert sess.ui_to_session(3900) == 10000     # old session: fire at 3.9 s
    clock.t = 1070.0                             # a minute later, after reload
    t = sess.ui_to_session(400)
    assert t == 70000
    assert t - 10000 >= 4000                     # clears min_gap_ms=4000


def test_a_synthetic_browser_clock_keeps_its_spacing(monkeypatch):
    """The Simulate tab (and every test) drives timestamps that do not track
    wall-clock time: a 1.6 s jump sent instantly must still be a 1.6 s pause,
    never mistaken for a restart and collapsed onto arrival time."""
    sess, clock = _session(monkeypatch)
    clock.t = 1100.0
    a = sess.ui_to_session(5000)
    b = sess.ui_to_session(6600)          # same instant, clock jumped 1.6 s
    c = sess.ui_to_session(20000)         # a whole simulated conversation later
    assert (b - a, c - b) == (1600, 13400)


def test_ws_words_and_ticks_arrive_on_the_session_clock(monkeypatch):
    """End to end over the real /ws JSON protocol: a word's duration and the
    tick's distance from it survive the lift, and both are on the session
    clock (not the browser's)."""
    import backend.app as app_mod
    from fastapi.testclient import TestClient
    sess, clock = _session(monkeypatch)
    monkeypatch.setattr(app_mod, "get_session", lambda: sess)
    seen = []

    async def record(item):
        seen.append(item)

    sess.pipeline.handle = record
    clock.t = 1050.0
    with TestClient(app_mod.app) as client, client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "word", "text": "bread", "start_ms": 1000, "end_ms": 1280,
                      "is_final": True})
        ws.send_json({"type": "silence", "at_ms": 2800})
        ws.send_json({"type": "ping"})
        ws.receive_json()                        # pong: the two above are handled
    word = next(i for i in seen if isinstance(i, Word))
    tick = next(i for i in seen if isinstance(i, SilenceTick))
    assert word.end_ms - word.start_ms == 280
    assert tick.at_ms - word.end_ms == 2800 - 1280
    assert word.end_ms == 50000                  # 50 s into the session, not 1.28 s
