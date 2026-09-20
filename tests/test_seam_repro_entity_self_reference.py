"""REGRESSION: entity memory x the demo script's own words (fixed).

EntityTracker carries a disclosed self-reference denylist
(backend/entities.py _SELF_REFERENCE_DENYLIST): the product/stack names the
presenter says out loud while demoing ("Hi, I'm demoing Echo...") must never
be learned as salient entities -- before the fix, "Echo" was captured from
Beat 1's scripted opening line (capitalized, mid-sentence) and, once scrolled
out of the context window, could be SERVED as a predicted candidate word:
the system suggesting its own product name as the word the speaker is
groping for, live, in front of a judge.
"""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import TurnEnd, Word
from backend.stall_detector import StallDetector

BEAT_1_LINE = "Hi I'm demoing Echo it's a co-pilot for people with aphasia"


async def _speak_turn(pipe, text, base):
    t = base
    for w in text.split():
        await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
        t += 400
    await pipe.handle(TurnEnd())
    return t


def test_demo_beat_1_line_never_becomes_a_served_candidate():
    async def run():
        p = MockPredictor()  # default table -- no keyword match for the probe stall
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p,
                             entity_memory=True, context_turns=6)
        t = 0
        t = await _speak_turn(pipe, BEAT_1_LINE, t)
        for i in range(6):  # push Beat 1's turn out of the 6-turn window
            t = await _speak_turn(pipe, f"filler turn number {i}", t)
        # The denylist keeps the product's own name out of the tracker even
        # though "Echo" sits capitalized mid-sentence in Beat 1's line.
        assert "Echo" not in pipe._entity_tracker.out_of_window(pipe.conversation.turns, 6)

        for w in ["I", "really", "need", "to"]:
            await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
            t += 400
        return await pipe.handle(Word(text="um", start_ms=t, end_ms=t + 280))

    pred = asyncio.run(run())
    served_words = [c.word for c in pred.candidates] if pred else []
    assert "Echo" not in served_words, (
        f"the product's own name leaked into served candidates: {served_words}")
