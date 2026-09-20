"""Timeline substrate: append-only word/event log with derived views.

The point of this module is that nothing truncates. See
docs/superpowers/specs/2026-08-17-echo-context-overhaul-design.md section 1.1
for the defect that motivated it.
"""
from backend.schemas import SearchEpisode, StallEvent, Word
from backend.timeline import Timeline


def w(text, i=0, is_final=True):
    return Word(text=text, start_ms=i * 400, end_ms=i * 400 + 300, is_final=is_final)


def test_utterance_text_is_words_since_last_boundary():
    t = Timeline()
    for i, x in enumerate("I need the".split()):
        t.add_word(w(x, i))
    assert t.utterance_text() == "I need the"


def test_turn_boundary_preserves_history_instead_of_discarding():
    t = Timeline()
    for i, x in enumerate("hello there".split()):
        t.add_word(w(x, i))
    t.mark_turn_boundary()
    for i, x in enumerate("second turn".split()):
        t.add_word(w(x, i + 5))
    # the boundary starts a new utterance WITHOUT destroying the first
    assert t.utterance_text() == "second turn"
    assert t.completed_turns() == ["hello there"]


def test_interim_words_are_not_committed():
    t = Timeline()
    t.add_word(w("I", 0))
    t.add_word(w("neeed", 1, is_final=False))
    assert t.utterance_text() == "I"


def test_content_count_excludes_fillers():
    t = Timeline()
    for i, x in enumerate(["I", "want", "um"]):
        t.add_word(w(x, i))
    assert t.content_count() == 2


def test_events_are_appended_with_timestamps():
    t = Timeline()
    t.add_event("served", 1200, {"word": "toaster"})
    assert len(t.events) == 1
    assert t.events[0].kind == "served"
    assert t.events[0].payload["word"] == "toaster"


def test_search_episode_defaults_to_unresolved():
    ep = SearchEpisode(started_at_ms=100, trigger="pause", fragment_at_fire="I need the")
    assert ep.resolved is False
    assert ep.served_word is None
    assert ep.rejected == []


def test_already_served_is_turn_scoped_and_does_not_leak_across_a_reset():
    """StallEvent.already_served says "this turn" -- so a word served in turn
    N must NOT come back as a spent hint in turn N+1.

    This replaces a test that constructed a StallEvent with
    already_served=["toaster"] and asserted it read back as ["toaster"]: a
    dataclass echoing its own constructor argument, which no mutation of the
    code could ever break. The wiring that FILLS the field is asserted in
    tests/test_episode_regression.py; the turn-scoping asserted here was
    covered nowhere, and `episodes` (where served words live) is deliberately
    kept alive by reset()'s neighbours for the refractory, so it is exactly
    the kind of state that could start leaking.
    """
    from backend.stall_detector import StallDetector

    det = StallDetector(pause_ms=1300)
    for i, x in enumerate(["I", "need", "the", "um"]):
        first = det.observe_word(w(x, i))
    assert first is not None and first.already_served == []   # nothing served yet
    det.record_served("toaster")

    # second stall, SAME turn -> the spent word is named
    second = None
    for j, x in enumerate(["a", "sandwich", "um"]):
        second = det.observe_word(w(x, j + 10))
    assert second is not None
    assert second.already_served == ["toaster"]

    # turn boundary -> the hint list starts empty again
    det.reset()
    third = None
    for k, x in enumerate(["pass", "me", "the", "um"]):
        third = det.observe_word(w(x, k + 30))
    assert third is not None
    assert third.already_served == [], "served words leaked across the turn boundary"


def test_already_served_defaults_empty_and_is_not_shared():
    a = StallEvent(fragment="x", trigger="pause")
    b = StallEvent(fragment="y", trigger="pause")
    a.already_served.append("leak")
    assert b.already_served == []   # default_factory, not a shared mutable
