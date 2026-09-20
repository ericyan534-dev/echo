"""Transcript-channel speaker filtering.

Agent C gated the ACOUSTIC channel, but the transcript arrived ungated: app.py
parsed `wearer_conf` onto the Word and stall_detector dropped it, so a
bystander's transcribed words still drove predictions. This was observed live
on 2026-08-17 -- a stray browser tab with an open mic injected "water",
"passport", "husband" into an active session.

POLICY, and the reason for it: a low-confidence word is KEPT in the timeline
record but EXCLUDED from the wearer's utterance and from the trigger content
count. Keeping the record is the whole lesson of this overhaul -- destroying
data to serve one consumer is what produced the amputation defect. Excluding it
from the utterance is what stops someone else's speech becoming the fragment we
ask the model to complete.

FAIL OPEN: wearer_conf None means unknown and is always included.
"""
from backend.schemas import Word
from backend.stall_detector import StallDetector
from backend.timeline import Timeline


def w(text, i, conf=None):
    return Word(text=text, start_ms=i * 400, end_ms=i * 400 + 300,
                is_final=True, wearer_conf=conf)


# --- fail open ---------------------------------------------------------------

def test_unknown_confidence_is_always_included():
    t = Timeline(wearer_conf_min=0.35)
    for i, x in enumerate("I need the".split()):
        t.add_word(w(x, i, conf=None))
    assert t.utterance_text() == "I need the"
    assert t.content_count() == 3


def test_a_gate_that_never_reports_changes_nothing():
    """The whole system must behave exactly as before when no gate is running."""
    a, b = Timeline(), Timeline(wearer_conf_min=0.35)
    for i, x in enumerate("Every morning I make some toast".split()):
        a.add_word(w(x, i))
        b.add_word(w(x, i))
    assert a.utterance_text() == b.utterance_text()
    assert a.content_count() == b.content_count()


def test_high_confidence_words_are_included():
    t = Timeline(wearer_conf_min=0.35)
    for i, x in enumerate("I need the".split()):
        t.add_word(w(x, i, conf=0.9))
    assert t.utterance_text() == "I need the"


# --- suppression -------------------------------------------------------------

def test_low_confidence_words_are_excluded_from_the_utterance():
    t = Timeline(wearer_conf_min=0.35)
    t.add_word(w("I", 0, conf=0.9))
    t.add_word(w("need", 1, conf=0.9))
    t.add_word(w("passport", 2, conf=0.05))     # someone else across the room
    t.add_word(w("the", 3, conf=0.9))
    assert t.utterance_text() == "I need the"


def test_low_confidence_words_are_still_kept_in_the_record():
    """Never destroy data to serve one consumer -- that is the amputation bug."""
    t = Timeline(wearer_conf_min=0.35)
    t.add_word(w("I", 0, conf=0.9))
    t.add_word(w("passport", 1, conf=0.05))
    assert [x.text for x in t.words] == ["I", "passport"]
    assert t.words[1].wearer_conf == 0.05


def test_low_confidence_words_do_not_count_toward_triggers():
    """A bystander must not be able to arm the pause trigger, which needs >= 2
    content words."""
    t = Timeline(wearer_conf_min=0.35)
    t.add_word(w("passport", 0, conf=0.05))
    t.add_word(w("husband", 1, conf=0.05))
    assert t.content_count() == 0


def test_low_confidence_words_are_excluded_from_completed_turns():
    t = Timeline(wearer_conf_min=0.35)
    t.add_word(w("I", 0, conf=0.9))
    t.add_word(w("water", 1, conf=0.02))
    t.add_word(w("agree.", 2, conf=0.9))
    t.mark_turn_boundary()
    assert t.completed_turns() == ["I agree."]


# --- end to end through the detector ----------------------------------------

def test_bystander_words_cannot_fire_a_pause_stall():
    det = StallDetector(pause_ms=1300, timeline=Timeline(wearer_conf_min=0.35))
    for i, x in enumerate(["passport", "husband", "water"]):
        det.observe_word(w(x, i, conf=0.05))
    assert det.observe_silence(99000) is None


def test_the_fragment_sent_to_the_model_excludes_the_bystander():
    det = StallDetector(pause_ms=1300, timeline=Timeline(wearer_conf_min=0.35))
    spoken = [("Every", 0.9), ("morning", 0.9), ("I", 0.9), ("make", 0.9),
              ("passport", 0.03), ("some", 0.9), ("toast", 0.9), ("in", 0.9),
              ("the", 0.9)]
    for i, (x, c) in enumerate(spoken):
        det.observe_word(w(x, i, conf=c))
    ev = det.observe_silence(99000)
    assert ev is not None
    assert ev.fragment == "Every morning I make some toast in the"
    assert "passport" not in ev.fragment


def test_a_stall_still_fires_normally_when_the_gate_is_silent():
    """Regression guard: adding the filter must not break the ordinary path."""
    det = StallDetector(pause_ms=1300, timeline=Timeline(wearer_conf_min=0.35))
    for i, x in enumerate("Every morning I make some toast in the".split()):
        det.observe_word(w(x, i))
    ev = det.observe_silence(99000)
    assert ev is not None and ev.trigger == "pause"
    assert ev.fragment == "Every morning I make some toast in the"
