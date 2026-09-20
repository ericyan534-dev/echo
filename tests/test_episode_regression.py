"""Regression tests for the clause-window amputation defect.

Both cases here were REPRODUCED against the old implementation before the
rewrite; they are not hypothetical. See
docs/superpowers/specs/2026-08-17-echo-context-overhaul-design.md section 1.1.
"""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import Candidate, SilenceTick, TurnEnd, Word
from backend.stall_detector import MAX_SERVED_HINTS, StallDetector


def w(text, i):
    return Word(text=text, start_ms=i * 400, end_ms=i * 400 + 300, is_final=True)


class _AlwaysAnswers(MockPredictor):
    """Returns a fresh word every call, so a test about *bookkeeping* does not
    depend on MockPredictor's fixture lookup happening to cover the fragment."""

    def __init__(self):
        self.n = 0

    async def predict(self, context, fragment, excluded=None, entities=None, **kw):
        self.n += 1
        return [Candidate(word=f"word{self.n}", confidence=0.9)]


def test_stall_after_a_served_suggestion_still_carries_the_sentence():
    """Defect 2/3: after one suggestion, later stalls used to send the model a
    two-word stub ('on some') with the sentence discarded."""
    det = StallDetector(pause_ms=1300)
    sentence = "Every morning I make some toast in the".split()
    for i, t in enumerate(sentence):
        det.observe_word(w(t, i))
    first = det.observe_silence(9000)
    assert first is not None
    assert first.fragment == "Every morning I make some toast in the"

    # speaker recovers and keeps going, then stalls again
    i = len(sentence)
    for t in ["and", "then", "I", "put", "on", "some"]:
        det.observe_word(w(t, i))
        i += 1
    second = det.observe_silence(i * 400 + 5000)
    assert second is not None
    # THE FIX: the sentence survives instead of being amputated to 'on some'
    assert "toast" in second.fragment
    assert second.fragment.startswith("Every morning")


def test_second_stall_keeps_context_and_knows_what_was_already_served():
    """Preserves the intent of the old test_fragment_scoped_to_clause_after_
    recovery: an abandoned attempt must not be re-served as the live answer.

    The old code achieved that by DELETING the earlier text, which is the
    amputation defect. Now the sentence is kept (the model can see the earlier
    attempt as context) and the spent candidate is named explicitly.
    """
    async def run():
        pipe = EchoPipeline(StallDetector(), _AlwaysAnswers(), prefetch=False)
        for i, t in enumerate(["I", "need", "the", "um"]):
            await pipe.handle(w(t, i))
        for j, t in enumerate(["a", "sandwich", "um"]):
            await pipe.handle(w(t, j + 10))
        return pipe.detector.drain_events(), pipe.predictions

    events, preds = asyncio.run(run())
    assert len(events) == 2
    assert "I need" in events[1].fragment                 # context preserved
    assert preds[0].candidates                            # something was served
    assert preds[0].candidates[0].word in events[1].already_served


def test_already_served_stays_bounded_across_many_stalls():
    """The first cut of this field held prior FRAGMENTS, which were prefixes of
    the current one -- six stalls produced six copies of the same sentence."""
    async def run():
        pipe = EchoPipeline(StallDetector(), _AlwaysAnswers(), prefetch=False)
        sentence = "Every morning I make some toast in the".split()
        for i, t in enumerate(sentence):
            await pipe.handle(w(t, i))
        await pipe.handle(SilenceTick(at_ms=99000))
        i = len(sentence)
        for t in ["and", "then", "I", "put", "on", "some"]:
            await pipe.handle(w(t, i))
            await pipe.handle(SilenceTick(at_ms=i * 400 + 90000))
            i += 1
        return pipe.detector.drain_events()

    events = asyncio.run(run())
    last = events[-1]
    # one entry per distinct served WORD -- never one sentence copy per stall
    for entry in last.already_served:
        assert " " not in entry.strip()
    # and capped, so a long turn full of stalls cannot grow the hint line
    assert len(last.already_served) <= MAX_SERVED_HINTS
    # The newest entry is the answer to the PREVIOUS stall, not this one: the
    # hint list is built when the event is emitted, before its own prediction
    # exists. Oldest entries are dropped by the cap.
    assert last.already_served[-1] == "word6"
    assert "word1" not in last.already_served


def test_a_turn_containing_a_stall_is_recorded_in_full():
    """Defect 1: a stalled turn used to be stored as its tail clause only --
    'every day at home' -- losing the proper noun and the object. Turns
    containing a stall are exactly the turns worth remembering."""
    async def run():
        pipe = EchoPipeline(StallDetector(), MockPredictor(), prefetch=False)
        spoken = "My physical therapist Sarah said I should use the walker".split()
        for i, t in enumerate(spoken):
            await pipe.handle(w(t, i))
        await pipe.handle(SilenceTick(at_ms=99000))       # stall mid-sentence
        for j, t in enumerate("every day at home".split()):
            await pipe.handle(w(t, len(spoken) + j))
        await pipe.handle(TurnEnd())
        return pipe.conversation.turns

    turns = asyncio.run(run())
    assert len(turns) == 1
    assert "Sarah" in turns[0]
    assert "walker" in turns[0]
    assert turns[0].startswith("My physical therapist")
