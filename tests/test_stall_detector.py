"""Unit tests for the stall detector — the correctness-critical core.

Covers: pause trigger, filler trigger, hedge trigger, no-fire on fluent speech,
no-fire on short pause, debounce (no double-fire), re-arming after recovery,
the across-every-trigger refractory, and what the pause is measured FROM.
"""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import SilenceTick, Word
from backend.stall_detector import StallDetector


def feed(det, words, base=0, dur=280, gap=120):
    """Feed words sequentially; return list of emitted StallEvents."""
    events = []
    t = base
    for w in words:
        ev = det.observe_word(Word(text=w, start_ms=t, end_ms=t + dur))
        if ev:
            events.append(ev)
        t += dur + gap
    return events, t


def test_pause_trigger_fires():
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "really", "need", "the"])
    ev = det.observe_silence(t + 1500)
    assert ev is not None
    assert ev.trigger == "pause"
    assert ev.fragment == "I really need the"


def test_filler_trigger_fires():
    det = StallDetector(pause_ms=1300)
    events, _ = feed(det, ["I", "want", "the", "um"])
    assert len(events) == 1
    assert events[0].trigger == "filler"
    assert events[0].fragment.endswith("um")


def test_hedge_trigger_fires():
    det = StallDetector(pause_ms=1300)
    events, _ = feed(det, ["pass", "me", "the", "thing"])  # genuine circumlocution
    assert len(events) == 1
    assert events[0].trigger == "hedge"


def test_no_fire_on_fluent_short_pauses():
    det = StallDetector(pause_ms=1300)
    events, t = feed(det, ["I", "feel", "really", "good", "today"])
    # a short pause well under threshold must not fire
    assert det.observe_silence(t + 400) is None
    assert events == []


def test_no_fire_below_pause_threshold():
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["the", "cat"])
    assert det.observe_silence(t + 900) is None


def test_debounce_no_double_fire():
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "need", "the"])
    first = det.observe_silence(t + 1500)
    second = det.observe_silence(t + 3000)  # still stuck
    assert first is not None
    assert second is None  # debounced


def test_rearm_after_recovery():
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "need", "the"])
    first = det.observe_silence(t + 1500)
    assert first is not None
    # speaker recovers with 2+ new content words, then stalls again
    ev2, t2 = feed(det, ["big", "red"], base=t + 1600)
    assert ev2 == []  # those words alone don't trigger
    third = det.observe_silence(t2 + 1500)
    assert third is not None  # re-armed and fires again


def test_sentence_final_resets():
    det = StallDetector(pause_ms=1300)
    feed(det, ["all", "done."])
    assert det.fragment == ""  # reset on terminal punctuation


def test_leading_filler_does_not_fire():
    det = StallDetector(pause_ms=1300)
    events, _ = feed(det, ["um"])  # no content word yet
    assert events == []


def test_interim_words_ignored():
    det = StallDetector(pause_ms=1300)
    assert det.observe_word(Word(text="I", start_ms=0, end_ms=200, is_final=False)) is None
    assert det.fragment == ""  # interim hypothesis dropped, not committed
    det.observe_word(Word(text="I", start_ms=0, end_ms=200, is_final=True))
    assert det.fragment == "I"


def test_fluent_trailing_you_know_does_not_fire():
    # "you know" as a fluent discourse marker must NOT trigger (demo-fatal FP)
    det = StallDetector(pause_ms=1300)
    events, _ = feed(det, ["I", "went", "to", "the", "store", "you", "know"])
    assert events == []


def test_punctuated_filler_still_fires():
    det = StallDetector(pause_ms=1300)
    events, _ = feed(det, ["I", "want", "the", "um."])
    assert len(events) == 1 and events[0].trigger == "filler"


def test_rearm_after_single_recovered_word():
    det = StallDetector(pause_ms=1300)
    e1, _ = feed(det, ["I", "want", "the", "um"])        # filler fires
    assert len(e1) == 1
    e2, _ = feed(det, ["book", "uh"], base=2000)         # 1 recovery word + new filler
    assert len(e2) == 1 and e2[0].trigger == "filler"


def test_context_preserved_across_a_second_stall_in_one_turn():
    """Replaces test_fragment_scoped_to_clause_after_recovery.

    The old test asserted the earlier attempt was DELETED from the fragment.
    That deletion is the amputation defect (spec section 1.1): it took the
    sentence with it, so the model received two-word stubs. The requirement it
    protected -- don't re-serve an abandoned attempt as the live answer -- is
    now met by naming the spent candidate on StallEvent.already_served, which
    is asserted end-to-end in tests/test_episode_regression.py (the detector
    alone has no predictor, so no word has been served here).
    """
    det = StallDetector(pause_ms=1300)
    feed(det, ["I", "need", "the", "um"])                 # fire 1
    e2, _ = feed(det, ["a", "sandwich", "um"], base=3000)  # recover, then fire 2
    assert len(e2) == 1
    assert "I need" in e2[0].fragment                     # context preserved
    assert e2[0].fragment.endswith("a sandwich um")       # and current search included
    assert len(det.episodes) == 2                         # both searches tracked


# --- the refractory is wall clock, not turn-scoped ------------------------
def test_min_gap_survives_a_turn_boundary():
    """min_gap_ms is documented as "minimum time between two fires, ACROSS ALL
    TRIGGERS", and it exists because a served word takes ~1.5-2 s to arrive and
    the wearer needs time to read and use it. A turn boundary does not give
    that time back -- but reset() used to drop `episodes`, which was where the
    refractory state lived, so the next utterance could be interrupted
    immediately. Measured: a pause stall at 2500 ms, TurnEnd, then a filler
    stall at 3650 ms -- two suggestions 1150 ms apart with min_gap_ms=4000.
    """
    det = StallDetector(pause_ms=1300, min_gap_ms=4000)
    for i, w in enumerate(["I", "went", "to", "the"]):
        det.observe_word(Word(text=w, start_ms=i * 300, end_ms=i * 300 + 280))
    first = det.observe_silence(2500)
    assert first is not None and first.trigger == "pause"

    # The wearer stays quiet; the ASR calls the turn at turn_end_ms, which the
    # pipeline turns into detector.reset().
    det.reset()

    det.observe_word(Word(text="the", start_ms=3300, end_ms=3500))
    second = det.observe_word(Word(text="um", start_ms=3550, end_ms=3650))
    assert second is None, (
        "a second suggestion %d ms after the first, with min_gap_ms=%d"
        % ((second.at_ms - first.at_ms) if second else 0, det.min_gap_ms))

    # ... and it is a REFRACTORY, not a mute: once the gap has elapsed the
    # next turn fires normally.
    det.reset()
    det.observe_word(Word(text="the", start_ms=9000, end_ms=9200))
    third = det.observe_word(Word(text="um", start_ms=9250, end_ms=9350))
    assert third is not None and third.at_ms - first.at_ms >= det.min_gap_ms


def test_min_gap_off_by_default_is_unaffected_by_a_turn_boundary():
    """min_gap_ms defaults to 0 (off), which every published bench was measured
    against. Nothing above may change that arm."""
    det = StallDetector(pause_ms=1300)
    for i, w in enumerate(["I", "went", "to", "the"]):
        det.observe_word(Word(text=w, start_ms=i * 300, end_ms=i * 300 + 280))
    assert det.observe_silence(2500) is not None
    det.reset()
    det.observe_word(Word(text="the", start_ms=3300, end_ms=3500))
    assert det.observe_word(Word(text="um", start_ms=3550, end_ms=3650)) is not None


# --- the pause is measured from the SPEAKER -------------------------------
def test_pause_is_measured_from_the_end_of_speech_not_the_last_commit():
    """A streaming recognizer holds the tail of an utterance behind
    LocalAgreement and the silence force-commit, so the last COMMITTED word's
    end trails the real end of speech by the commit lag (p90 700 ms at the
    shipped silence_commit_ms). Measuring the pause from the committed word
    alone turns that lag into pause length and fires on a speaker who stopped
    200 ms ago. The tick carries when speech actually stopped; the detector
    must take the LATER of the two bounds."""
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "went", "to", "the"])
    last_commit = det.timeline.current_utterance()[-1].end_ms
    spoke_until = last_commit + 1200          # 1.2 s of speech not yet committed

    # 200 ms after the speaker stopped: 1400 ms since the last COMMITTED word.
    assert det.observe_silence(spoke_until + 200, spoke_until) is None, (
        "fired 200 ms after the speaker stopped -- that is the commit lag, "
        "not a pause")
    # Still short of the threshold measured from the speaker.
    assert det.observe_silence(spoke_until + 1200, spoke_until) is None
    ev = det.observe_silence(spoke_until + 1300, spoke_until)
    assert ev is not None and ev.trigger == "pause"


def test_a_tick_without_a_speech_end_keeps_the_committed_word_rule():
    """None means "this emitter has no VAD" (a browser or mock timer, or a
    stream recorded before the field existed). It must change nothing."""
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "went", "to", "the"])
    last_commit = det.timeline.current_utterance()[-1].end_ms
    assert det.observe_silence(last_commit + 1299) is None
    assert det.observe_silence(last_commit + 1300) is not None


def test_a_speech_end_can_only_shorten_the_measured_pause():
    """The two bounds are combined with max(), never min(): both say "the
    speaker was still talking at least until here". A speech_end that is
    EARLIER than the last committed word (a stale or coarse VAD reading) must
    never be allowed to stretch the pause and fire early."""
    det = StallDetector(pause_ms=1300)
    _, t = feed(det, ["I", "went", "to", "the"])
    last_commit = det.timeline.current_utterance()[-1].end_ms
    assert det.observe_silence(last_commit + 1299, last_commit - 500) is None
    assert det.observe_silence(last_commit + 1300, last_commit - 500) is not None


def test_the_pipeline_forwards_the_speech_end_on_a_tick():
    """The plumbing between the two halves of the fix. The ASR puts the VAD's
    last voiced moment on every SilenceTick; if the pipeline drops it on the
    way to observe_silence, the detector is back to measuring commit lag and
    nothing else in either half would notice."""
    det = StallDetector(pause_ms=1300)
    pipe = EchoPipeline(det, MockPredictor())
    for i, w in enumerate(["I", "went", "to", "the"]):
        det.observe_word(Word(text=w, start_ms=i * 300, end_ms=i * 300 + 280))
    last_commit = det.timeline.current_utterance()[-1].end_ms
    spoke_until = last_commit + 1200

    async def go():
        # 1400 ms since the last COMMITTED word, 200 ms since the speaker.
        first = await pipe.handle(SilenceTick(at_ms=spoke_until + 200,
                                              speech_end_ms=spoke_until))
        second = await pipe.handle(SilenceTick(at_ms=spoke_until + 1300,
                                               speech_end_ms=spoke_until))
        return first, second

    first, second = asyncio.run(go())
    assert first is None, "the pipeline dropped speech_end_ms from the tick"
    assert second is not None and second.trigger == "pause"
