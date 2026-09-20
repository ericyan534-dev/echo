"""Acoustic channel -> StallDetector fusion, and the prolongation tracker."""
import torch

from backend.acoustic.prolongation import ProlongationTracker
from backend.schemas import AcousticEvent, Word
from backend.stall_detector import StallDetector


def feed_words(det, words, base=0):
    t = base
    for w in words:
        det.observe_word(Word(text=w, start_ms=t, end_ms=t + 280))
        t += 400
    return t


# --- fusion ---------------------------------------------------------------
def test_acoustic_filler_fires_after_content():
    det = StallDetector(pause_ms=1300)
    feed_words(det, ["I", "want", "the"])
    ev = det.observe_acoustic(AcousticEvent("filler", at_ms=1500, confidence=0.9))
    assert ev is not None
    assert ev.trigger == "filler_acoustic"
    assert ev.fragment == "I want the"


def test_acoustic_event_blocked_without_content():
    det = StallDetector(pause_ms=1300)
    assert det.observe_acoustic(AcousticEvent("filler", at_ms=100)) is None


def test_prolongation_maps_to_trigger():
    det = StallDetector(pause_ms=1300)
    feed_words(det, ["pass", "the"])
    ev = det.observe_acoustic(AcousticEvent("prolongation", at_ms=900))
    assert ev is not None and ev.trigger == "prolongation"


def test_acoustic_shares_debounce_with_text_triggers():
    det = StallDetector(pause_ms=1300)
    feed_words(det, ["I", "want", "the", "um"])  # text filler fires
    # immediately after, the acoustic channel hears the same um -> debounced
    assert det.observe_acoustic(AcousticEvent("filler", at_ms=2000)) is None


def test_acoustic_rearms_after_recovery():
    det = StallDetector(pause_ms=1300)
    feed_words(det, ["I", "want", "the", "um"])          # fire 1 (text)
    feed_words(det, ["book"], base=3000)                  # 1 recovery word
    ev = det.observe_acoustic(AcousticEvent("prolongation", at_ms=4000))
    assert ev is not None and ev.trigger == "prolongation"


# --- prolongation tracker --------------------------------------------------
def _frames(vec, n):
    return [vec.clone() for _ in range(n)]


def test_prolongation_fires_on_sustained_identical_frames():
    t = ProlongationTracker(min_ms=700)
    frame = torch.sin(torch.linspace(0, 220 * 6.28, 800)) * 0.3
    fired = [t.observe_frame(frame.clone(), now_ms=i * 50) for i in range(20)]
    assert any(fired)
    assert fired.index(True) * 50 >= 700  # not before the threshold


def test_prolongation_refractory_blocks_double_fire():
    t = ProlongationTracker(min_ms=700, refractory_ms=1500)
    frame = torch.sin(torch.linspace(0, 220 * 6.28, 800)) * 0.3
    fires = sum(t.observe_frame(frame.clone(), now_ms=i * 50) for i in range(28))
    assert fires == 1  # 1.4 s window: second fire suppressed by refractory


def test_prolongation_resets_on_silence():
    t = ProlongationTracker(min_ms=700)
    frame = torch.sin(torch.linspace(0, 220 * 6.28, 800)) * 0.3
    for i in range(10):  # 500 ms of sustain (below threshold)
        t.observe_frame(frame.clone(), now_ms=i * 50)
    t.observe_frame(torch.zeros(800), now_ms=500)  # silence resets streak
    assert t._streak_ms == 0
