import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import SilenceTick, TurnEnd, Word
from backend.stall_detector import StallDetector
from backend.stt.mock import MockSTT, script_from_spec
from backend.transcript import Conversation


def run(pipe, stream):
    asyncio.run(pipe.run(stream))


def test_pipeline_fires_prediction_on_pause():
    script = script_from_spec(["I", "really", "need", "the"], stall_after=3)
    pipe = EchoPipeline(
        StallDetector(pause_ms=1300),
        MockPredictor(table={"need the": ["medicine"]}),
    )
    run(pipe, MockSTT(script).stream())
    assert len(pipe.predictions) == 1
    assert pipe.predictions[0].trigger == "pause"
    assert pipe.predictions[0].candidates[0].word == "medicine"
    assert pipe.predictions[0].fragment == "I really need the"


def test_pipeline_silent_on_fluent_speech():
    script = script_from_spec(["I", "feel", "good", "today."])
    pipe = EchoPipeline(StallDetector(pause_ms=1300), MockPredictor())
    run(pipe, MockSTT(script).stream())
    assert pipe.predictions == []


def test_pipeline_uses_conversation_context():
    captured = {}

    class Spy(MockPredictor):
        async def predict(self, context, fragment):
            captured["context"] = context
            captured["fragment"] = fragment
            return await super().predict(context, fragment)

    convo = Conversation()
    convo.add_turn("So what did you have for breakfast?")
    pipe = EchoPipeline(StallDetector(pause_ms=1300), Spy(), conversation=convo)
    script = script_from_spec(["I", "made", "some", "toast", "in", "the"], stall_after=5)
    run(pipe, MockSTT(script).stream())
    assert captured["context"] == ["So what did you have for breakfast?"]
    assert captured["fragment"].startswith("I made some toast")


def test_turn_end_records_context_and_resets():
    pipe = EchoPipeline(StallDetector(pause_ms=1300), MockPredictor())
    asyncio.run(pipe.handle(Word(text="hello", start_ms=0, end_ms=200)))
    asyncio.run(pipe.handle(Word(text="there", start_ms=300, end_ms=500)))
    asyncio.run(pipe.handle(TurnEnd()))
    assert pipe.conversation.turns == ["hello there"]
    assert pipe.detector.fragment == ""


def test_on_prediction_callback_invoked():
    seen = []

    async def cb(pred):
        seen.append(pred)

    pipe = EchoPipeline(
        StallDetector(pause_ms=1300),
        MockPredictor(table={"the": ["toaster"]}),
        on_prediction=cb,
    )
    asyncio.run(pipe.handle(Word(text="I", start_ms=0, end_ms=200)))
    asyncio.run(pipe.handle(Word(text="want", start_ms=300, end_ms=500)))
    asyncio.run(pipe.handle(Word(text="the", start_ms=600, end_ms=800)))
    asyncio.run(pipe.handle(Word(text="um", start_ms=900, end_ms=1100)))
    assert len(seen) == 1
    assert seen[0].trigger == "filler"


def test_completed_punctuated_turn_is_recorded():
    # A turn that ends with terminal punctuation must still land in context,
    # even though the detector resets on the period.
    pipe = EchoPipeline(StallDetector(pause_ms=1300), MockPredictor())
    for w in [Word("I", 0, 200), Word("feel", 300, 500), Word("good.", 600, 800)]:
        asyncio.run(pipe.handle(w))
    asyncio.run(pipe.handle(TurnEnd()))
    assert pipe.conversation.turns == ["I feel good."]


class _EntityRecordingPredictor(MockPredictor):
    """Records the `entities` kwarg (or its absence) on every predict() call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.calls: list[tuple[str, object]] = []  # (fragment, entities_or_SENTINEL)

    async def predict(self, context, fragment, excluded=None, entities="__unset__"):
        self.calls.append((fragment, entities))
        return await super().predict(context, fragment, excluded=excluded,
                                      entities=None if entities == "__unset__" else entities)


def test_entity_memory_off_never_passes_entities_kwarg():
    p = _EntityRecordingPredictor()
    convo = Conversation()
    for i in range(8):
        convo.add_turn(f"I met Frank at turn {i}.")
    pipe = EchoPipeline(
        StallDetector(pause_ms=1300), p, conversation=convo,
        context_turns=3, entity_memory=False,  # default, explicit for clarity
    )
    script = script_from_spec(["I", "need", "the"], stall_after=2)
    run(pipe, MockSTT(script).stream())
    assert p.calls  # the stall did fire a predict call
    assert all(entities == "__unset__" for _, entities in p.calls), (
        "entity_memory=False must never pass the entities kwarg at all "
        "(byte-identical call site to before this feature existed)")


def test_entity_memory_on_injects_out_of_window_entities_on_live_predict():
    p = _EntityRecordingPredictor()
    convo = Conversation()
    convo.add_turn("My neighbor Frank fixed the fence yesterday.")  # turn 0: out of window
    for i in range(5):
        convo.add_turn(f"Small talk turn {i}.")  # turns 1-5
    pipe = EchoPipeline(
        StallDetector(pause_ms=1300), p, conversation=convo,
        context_turns=3, entity_memory=True, prefetch=False,
    )
    script = script_from_spec(["I", "should", "call", "him"], stall_after=3)
    run(pipe, MockSTT(script).stream())
    assert p.calls
    fragment, entities = p.calls[-1]
    assert entities == ["Frank"]


def test_entity_memory_on_injects_on_shadow_prefetch_too():
    """The prefetch cache must benefit from the same entity hint the live
    path gets -- entity memory is threaded into _shadow_predict as well."""
    p = _EntityRecordingPredictor()
    convo = Conversation()
    convo.add_turn("My neighbor Frank fixed the fence yesterday.")
    for i in range(5):
        convo.add_turn(f"Small talk turn {i}.")
    pipe = EchoPipeline(
        StallDetector(pause_ms=1300), p, conversation=convo,
        context_turns=3, entity_memory=True, prefetch=True, prefetch_every=3,
    )

    async def _run():
        await pipe.handle(Word(text="I", start_ms=0, end_ms=200))
        await asyncio.sleep(0)  # let the shadow task run

    asyncio.run(_run())
    assert p.calls, "cold-cache shadow should have fired on the first content word"
    _, entities = p.calls[0]
    assert entities == ["Frank"]


def test_entity_memory_on_with_no_out_of_window_entities_omits_kwarg():
    # Entities exist but are still INSIDE the context window -> nothing to
    # inject; the kwarg must not be sent (mirrors the off case).
    p = _EntityRecordingPredictor()
    convo = Conversation()
    convo.add_turn("My neighbor Frank fixed the fence yesterday.")
    pipe = EchoPipeline(
        StallDetector(pause_ms=1300), p, conversation=convo,
        context_turns=6, entity_memory=True, prefetch=False,
    )
    script = script_from_spec(["I", "should", "call", "him"], stall_after=3)
    run(pipe, MockSTT(script).stream())
    assert p.calls
    _, entities = p.calls[-1]
    assert entities == "__unset__"


def test_predict_timeout_degrades_gracefully():
    """A hung predictor must not block the loop -- Fix 2.

    Uses a very short timeout (50 ms) so the test stays fast.  The pipeline
    must produce a Prediction without hanging, and the timed-out predictor's
    word must NOT appear in the result (the call was cancelled).
    """
    from backend.predictor.base import WordPredictor
    from backend.schemas import Candidate

    class SlowPredictor(WordPredictor):
        async def predict(self, context, fragment):
            await asyncio.sleep(10)   # hangs far longer than the test timeout
            return [Candidate("never", 1.0)]

    pipe = EchoPipeline(
        StallDetector(pause_ms=1300),
        SlowPredictor(),
        predict_timeout=0.05,   # 50 ms -- fast test, still exercises the timeout
    )

    async def _run():
        await pipe.handle(Word(text="I", start_ms=0, end_ms=200))
        await pipe.handle(Word(text="need", start_ms=300, end_ms=500))
        await pipe.handle(Word(text="the", start_ms=600, end_ms=800))
        # Silence tick well past pause_ms=1300 to trigger the stall path
        await pipe.handle(SilenceTick(at_ms=3000))

    asyncio.run(_run())
    assert len(pipe.predictions) == 1, "pipeline must yield a (degraded) prediction, not hang"
    pred = pipe.predictions[0]
    # The SlowPredictor was cancelled by the timeout, so its word must not appear
    assert not any(c.word == "never" for c in pred.candidates), \
        "timed-out predictor result must not leak into the prediction"


def test_summary_retains_an_out_of_window_name_as_the_conversation_grows():
    """The compounding-context property: a name from turn 1 is still reachable
    after it has scrolled far out of the verbatim window."""
    from backend.context import ContextBuilder
    from backend.summarizer import ExtractiveSummarizer

    async def run():
        pipe = EchoPipeline(
            StallDetector(), MockPredictor(), prefetch=False,
            context_builder=ContextBuilder(ExtractiveSummarizer(), verbatim_turns=3),
        )
        await pipe.handle_text_turn("My sister Maria visited Boston yesterday.")
        for i in range(8):
            await pipe.handle_text_turn(f"Then we talked about topic {i}.")
        return pipe.rolling_summary

    summary = asyncio.run(run())
    assert "Maria" in summary


def test_pipeline_without_a_context_builder_is_unchanged():
    """The builder is additive: absent one, behaviour is exactly as before."""
    async def run():
        pipe = EchoPipeline(StallDetector(), MockPredictor(), prefetch=False)
        for i, t in enumerate("I really need the".split()):
            await pipe.handle(Word(text=t, start_ms=i * 400, end_ms=i * 400 + 300))
        pred = await pipe.handle(SilenceTick(at_ms=9000))
        return pipe, pred

    pipe, pred = asyncio.run(run())
    assert pred is not None
    assert pipe.rolling_summary == ""
