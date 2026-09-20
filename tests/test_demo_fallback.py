"""Demo-day network fallback predictor: strict fire-only-on-failure semantics."""
import asyncio

import pytest

from backend.config import Settings
from backend.predictor import get_predictor
from backend.predictor.base import WordPredictor
from backend.predictor.demo_fallback import DemoFallbackPredictor
from backend.predictor.mock import MockPredictor
from backend.schemas import Candidate


class SlowPredictor(WordPredictor):
    """Never resolves inside the fallback's timeout."""

    def __init__(self, delay: float = 10.0) -> None:
        self.delay = delay
        self.calls: list[tuple] = []

    async def predict(self, context, fragment, excluded=None, entities=None, **kwargs):
        self.calls.append((context, fragment, excluded))
        await asyncio.sleep(self.delay)
        return [Candidate(word="should-never-arrive", confidence=0.99)]


class RaisingPredictor(WordPredictor):
    def __init__(self) -> None:
        self.calls = 0

    async def predict(self, context, fragment, excluded=None, entities=None, **kwargs):
        self.calls += 1
        raise RuntimeError("simulated network failure")


class EmptyPredictor(WordPredictor):
    """Succeeds fast, legitimately returns no candidates."""

    def __init__(self) -> None:
        self.calls = 0

    async def predict(self, context, fragment, excluded=None, entities=None, **kwargs):
        self.calls += 1
        return []


class RecordingPredictor(WordPredictor):
    """Succeeds fast, returns a fixed answer, records what it was called with."""

    def __init__(self, answer):
        self.answer = answer
        self.calls: list[tuple] = []

    async def predict(self, context, fragment, excluded=None, entities=None, **kwargs):
        self.calls.append((context, fragment, excluded, entities, kwargs))
        return self.answer


def _settings(**overrides):
    base = dict(
        predictor_provider="demo_fallback",
        gemini_model="gemini-3.5-flash",
        gemini_api_key="dummy",
        claude_model="claude-haiku-4-5",
        claude_api_key="dummy",
        deepseek_model="deepseek-flash",
        deepseek_api_key="dummy",
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="mock",
        demo_fallback_timeout_s=1.5,
        stt_provider="mock",
        deepgram_api_key=None,
        max_candidates=3,
        pause_ms=1300,
        context_turns=6,
        acoustic_model="",
        acoustic_conf=0.7,
        prefetch=False,
        prefetch_every=3,
        entity_memory=False,
    )
    base.update(overrides)
    return Settings(**base)


# --- fires on timeout / exception -----------------------------------------

def test_fallback_fires_on_timeout():
    inner = SlowPredictor(delay=10.0)
    fb = DemoFallbackPredictor(inner, timeout_s=0.05)
    out = asyncio.run(fb.predict([], "I made some toast in the"))
    assert out and out[0].word == "toaster"


def test_fallback_fires_on_exception():
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict([], "I made some toast in the"))
    assert inner.calls == 1
    assert out and out[0].word == "toaster"


# --- does NOT fire on successful-empty ------------------------------------

def test_fallback_does_not_fire_on_successful_empty():
    inner = EmptyPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    # Fragment matches the local table ("toast") but the inner call succeeded
    # (just with no candidates) -- the canned answer must NOT be substituted.
    out = asyncio.run(fb.predict([], "I made some toast in the"))
    assert out == []
    assert inner.calls == 1


def test_fallback_returns_successful_inner_result_unchanged():
    answer = [Candidate(word="anything", confidence=0.4)]
    inner = RecordingPredictor(answer)
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict(["ctx"], "totally unrelated to any table key"))
    assert out == answer


# --- excluded threading -----------------------------------------------------

def test_fallback_threads_excluded_to_inner():
    inner = RecordingPredictor([])
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    asyncio.run(fb.predict(["ctx"], "some fragment", excluded=["rejected"]))
    assert inner.calls[0][:3] == (["ctx"], "some fragment", ["rejected"])


def test_entities_and_future_kwargs_forwarded_to_inner():
    # A wrapper that silently drops optional predictor params would disable
    # entity memory (or any future feature) whenever demo_fallback is active.
    inner = RecordingPredictor([Candidate(word="ok", confidence=0.9)])
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    asyncio.run(fb.predict(["ctx"], "frag", excluded=["x"],
                           entities=["Marcus"], someday="future"))
    ctx, frag, excluded, entities, kwargs = inner.calls[0]
    assert excluded == ["x"]
    assert entities == ["Marcus"]
    assert kwargs == {"someday": "future"}


def test_fallback_serves_live_entity_when_no_table_match():
    # During an outage, a name-stall with no canned table match is served from
    # the REAL conversation's entity list -- live data, not a canned answer.
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict([], "I should text, um, that guy",
                                 entities=["Marcus"]))
    assert out and out[0].word == "Marcus"


def test_fallback_local_lookup_respects_excluded():
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict([], "I made some toast in the", excluded=["toaster"]))
    words = [c.word for c in out]
    assert "toaster" not in words
    assert "oven" in words


# --- lookup normalization + miss --------------------------------------------

def test_lookup_normalization_case_and_punctuation_insensitive():
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict([], "TOAST!!! in theeee"))
    assert out and out[0].word == "toaster"


def test_lookup_miss_returns_empty():
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    out = asyncio.run(fb.predict([], "a totally unrelated banana smoothie"))
    assert out == []


def test_lookup_covers_all_four_rehearsed_beats():
    inner = RaisingPredictor()
    fb = DemoFallbackPredictor(inner, timeout_s=1.5)
    toast = asyncio.run(fb.predict([], "Every morning I make some toast in the"))
    maria = asyncio.run(fb.predict([], "I really need to call, um"))
    dishes = asyncio.run(fb.predict([], "After dinner I washed all of the dirty"))
    tokyo = asyncio.run(
        fb.predict([], "we're flying to the big city in Japan called")
    )
    assert [c.word for c in toast] == ["toaster", "oven"]
    assert [c.word for c in maria] == ["Maria"]
    assert [c.word for c in dishes] == ["dishes", "dishwasher"]
    assert [c.word for c in tokyo] == ["Tokyo"]


# --- config / factory wiring -------------------------------------------------

def test_factory_builds_demo_fallback_wrapping_mock_inner():
    p = get_predictor(_settings(demo_fallback_inner="mock"))
    assert isinstance(p, DemoFallbackPredictor)
    assert isinstance(p.inner, MockPredictor)
    assert p.timeout_s == 1.5


def test_factory_demo_fallback_inner_cannot_recurse():
    with pytest.raises(ValueError):
        get_predictor(_settings(demo_fallback_inner="demo_fallback"))
