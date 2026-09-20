"""REGRESSION: reject x entity-memory (was a demo-killer, now fixed).

EchoPipeline.reject() must forward the entity hint (via _entities_kwargs(),
which also filters out the just-rejected words) exactly like _predict()/
_shadow_predict() do. Before the fix, rejecting an entity-served word threw
the entity hint away entirely: when the original serve came from the
entity-memory fallback (no keyword/context cue in the fragment -- exactly
Beat 3's out-of-window name-recall shape), the re-predict had nothing left
and returned an empty candidate list, blanking the card mid-demo.
"""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import TurnEnd, Word
from backend.stall_detector import StallDetector


async def _speak_turn(pipe, text, base):
    t = base
    for w in text.split():
        await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
        t += 400
    await pipe.handle(TurnEnd())
    return t


def test_reject_after_entity_served_word_still_offers_remaining_entities():
    async def run():
        # Empty keyword table: the ONLY way MockPredictor can answer is the
        # entities fallback (mirrors an out-of-window-name stall with no
        # other lexical cue in the fragment -- exactly the Beat 3 shape).
        p = MockPredictor(table={})
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p,
                             entity_memory=True, context_turns=2)
        t = 0
        t = await _speak_turn(pipe, "I met Maria yesterday", t)
        t = await _speak_turn(pipe, "I also saw Boston today", t)
        t = await _speak_turn(pipe, "filler one", t)
        t = await _speak_turn(pipe, "filler two", t)
        assert pipe._entity_tracker.out_of_window(pipe.conversation.turns, 2) == \
            ["Boston", "Maria"]

        for w in ["I", "really", "need", "to"]:
            await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
            t += 400
        pred = await pipe.handle(Word(text="um", start_ms=t, end_ms=t + 280))
        assert [c.word for c in pred.candidates] == ["Boston", "Maria"]

        return await pipe.reject(["Boston"])

    rep = asyncio.run(run())
    # Rejecting "Boston" re-serves from the SAME entity hint minus the
    # rejected word: ["Maria"]. (Pre-fix behavior: entities never reached
    # the re-predict call, so the card blanked with [].)
    assert rep is not None and [c.word for c in rep.candidates] == ["Maria"]
