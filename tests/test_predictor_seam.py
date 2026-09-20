"""The PredictionRequest seam (Phase 4 / plan step B1).

WHY THIS FILE EXISTS
--------------------
`docs/ROADMAP.md` promises that a local model drops in behind `WordPredictor`
with **zero refactor**. `predict_request` is the richer entry point the local
provider wants (it needs the summary and a declared context budget), so the
promise only holds if the BASE implementation unpacks a `PredictionRequest`
back onto the old `predict(...)` signature. These tests pin that: Gemini,
Claude, mock and demo_fallback must answer `predict_request` correctly with no
edit of their own.
"""
import asyncio
import inspect

import pytest

from backend.predictor.base import (
    PredictionRequest,
    PredictorCapabilities,
    WordPredictor,
)
from backend.predictor.mock import MockPredictor
from backend.schemas import Candidate


class RecordingPredictor(WordPredictor):
    """Full-signature predictor that records exactly what the shim passed."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def predict(self, context, fragment, excluded=None, entities=None,
                      already_served=None):
        self.calls.append({
            "context": context, "fragment": fragment, "excluded": excluded,
            "entities": entities, "already_served": already_served,
        })
        return [Candidate(word="toaster", confidence=0.9)]


class TwoArgPredictor(WordPredictor):
    """A hand-rolled predictor that implements ONLY the original 2-arg
    `predict`. Nothing in the repo forces the optional hints on an
    implementer, so the shim must not blow up on one that ignores them."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def predict(self, context, fragment):  # noqa: D102 - deliberate 2-arg
        self.calls.append((context, fragment))
        return [Candidate(word="oven", confidence=0.4)]


def _req(**over):
    base = dict(
        utterance="I put the bread in the, um",
        recent_turns=["What did you have for breakfast?"],
        summary="They talked about the kitchen renovation.",
        entities=["Frank"],
        excluded=["spoon"],
        already_served=["fork"],
    )
    base.update(over)
    return PredictionRequest(**base)


# --- dataclass shapes --------------------------------------------------------

def test_capabilities_defaults_are_conservative():
    caps = PredictorCapabilities()
    assert caps.max_context_tokens == 8192
    assert caps.typical_latency_ms == 1500
    # Default False: only a provider that actually constrains decoding may
    # claim grammar support, otherwise callers would pass a grammar to a
    # provider that silently ignores it.
    assert caps.supports_grammar is False


def test_prediction_request_optional_fields_default_empty():
    req = PredictionRequest(utterance="the um", recent_turns=[])
    assert req.summary == ""
    assert req.entities == [] and req.excluded == [] and req.already_served == []


def test_base_capabilities_is_the_conservative_default():
    caps = MockPredictor().capabilities
    assert isinstance(caps, PredictorCapabilities)
    assert caps.supports_grammar is False


# --- the shim ----------------------------------------------------------------

def test_predict_request_threads_every_optional_hint_through():
    p = RecordingPredictor()
    out = asyncio.run(p.predict_request(_req()))
    assert [c.word for c in out] == ["toaster"]

    call = p.calls[0]
    assert call["fragment"] == "I put the bread in the, um"
    assert call["excluded"] == ["spoon"]
    assert call["entities"] == ["Frank"]
    assert call["already_served"] == ["fork"]
    # recent turns survive verbatim, and the rolling summary is not silently
    # dropped -- a request field that vanishes is worse than no field.
    assert "What did you have for breakfast?" in call["context"]
    assert any("kitchen renovation" in line for line in call["context"])


def test_predict_request_without_summary_passes_recent_turns_unchanged():
    p = RecordingPredictor()
    asyncio.run(p.predict_request(_req(summary="")))
    assert p.calls[0]["context"] == ["What did you have for breakfast?"]


def test_predict_request_does_not_mutate_the_request_or_its_lists():
    turns = ["a", "b"]
    req = PredictionRequest(utterance="x", recent_turns=turns, summary="s")
    p = RecordingPredictor()
    asyncio.run(p.predict_request(req))
    assert turns == ["a", "b"]           # caller's list untouched
    assert req.recent_turns == ["a", "b"]


def test_predict_request_matches_predict_on_the_real_mock_provider():
    """Same inputs, two entry points, identical answer -- that equivalence IS
    the zero-refactor promise."""
    p = MockPredictor()
    req = _req(utterance="I made some toast in the, um, the thing",
               excluded=[], already_served=[], entities=[])
    via_request = asyncio.run(p.predict_request(req))
    via_predict = asyncio.run(p.predict(
        ["Earlier in this conversation (summary): "
         "They talked about the kitchen renovation.",
         "What did you have for breakfast?"],
        "I made some toast in the, um, the thing",
        [], [], []))
    assert [(c.word, c.confidence) for c in via_request] == \
           [(c.word, c.confidence) for c in via_predict]
    assert via_request and via_request[0].word == "toaster"


def test_predict_request_honours_excluded_on_the_real_mock_provider():
    p = MockPredictor()
    out = asyncio.run(p.predict_request(PredictionRequest(
        utterance="I made some toast in the, um", recent_turns=[],
        excluded=["toaster"])))
    assert "toaster" not in [c.word for c in out]


def test_predict_request_tolerates_a_two_arg_predictor():
    """A predictor that never opted into the optional hints must still answer
    predict_request -- dropping hints it cannot use, not raising TypeError."""
    p = TwoArgPredictor()
    out = asyncio.run(p.predict_request(_req()))
    assert [c.word for c in out] == ["oven"]
    context, fragment = p.calls[0]
    assert fragment == "I put the bread in the, um"
    assert "What did you have for breakfast?" in context


def test_existing_providers_answer_predict_request_without_modification():
    """Gemini / Claude / demo_fallback inherit the shim; none of them defines
    predict_request itself. If one ever does, this test is the reminder that
    the override must stay behaviourally identical."""
    from backend.predictor.claude import ClaudePredictor
    from backend.predictor.demo_fallback import DemoFallbackPredictor
    from backend.predictor.gemini import GeminiPredictor

    for cls in (GeminiPredictor, ClaudePredictor, MockPredictor,
                DemoFallbackPredictor):
        assert "predict_request" not in cls.__dict__, (
            f"{cls.__name__} overrides predict_request; the base shim is what "
            "keeps the zero-refactor promise")
        assert cls.predict_request is WordPredictor.predict_request
        assert "capabilities" not in cls.__dict__ or isinstance(
            cls.capabilities, property)


def test_demo_fallback_predict_request_reaches_the_inner_predictor():
    from backend.predictor.demo_fallback import DemoFallbackPredictor

    inner = RecordingPredictor()
    p = DemoFallbackPredictor(inner=inner, timeout_s=2.0)
    out = asyncio.run(p.predict_request(_req()))
    assert [c.word for c in out] == ["toaster"]
    assert inner.calls, "the wrapper must have delegated to the inner provider"


def test_predict_request_signature_is_one_positional_request():
    params = list(inspect.signature(WordPredictor.predict_request).parameters)
    assert params == ["self", "req"]


def test_abstract_predict_is_still_the_only_abstract_method():
    """predict_request must NOT be abstract: making it abstract would force
    every existing provider to implement it, which is the refactor we promised
    would not be needed."""
    assert WordPredictor.__abstractmethods__ == frozenset({"predict"})

    class OnlyPredict(WordPredictor):
        async def predict(self, context, fragment, excluded=None, entities=None,
                          already_served=None):
            return []

    OnlyPredict()  # must be instantiable


def test_cannot_instantiate_without_predict():
    class Nothing(WordPredictor):
        pass

    with pytest.raises(TypeError):
        Nothing()
