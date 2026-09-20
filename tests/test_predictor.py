import asyncio

import pytest

from backend.config import Settings
from backend.predictor import get_predictor
from backend.predictor.base import WordPredictor
from backend.predictor.claude import ClaudePredictor
from backend.predictor.gemini import GeminiPredictor, _parse
from backend.predictor.mock import MockPredictor


def _settings(provider="mock"):
    return Settings(
        predictor_provider=provider,
        gemini_model="gemini-3.5-flash",
        gemini_api_key="dummy",
        claude_model="claude-haiku-4-5",
        claude_api_key="dummy",
        deepseek_model="deepseek-flash",
        deepseek_api_key="dummy",
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="gemini",
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


def test_all_providers_implement_interface():
    # Importing the classes must not require google-genai / anthropic to be
    # installed (lazy imports inside __init__).
    assert issubclass(MockPredictor, WordPredictor)
    assert issubclass(GeminiPredictor, WordPredictor)
    assert issubclass(ClaudePredictor, WordPredictor)


def test_factory_returns_mock():
    assert isinstance(get_predictor(_settings("mock")), MockPredictor)


def test_factory_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_predictor(_settings("nope"))


def test_mock_predict_keyword_hit():
    p = MockPredictor(table={"bread": ["toaster", "oven"]})
    out = asyncio.run(p.predict(["breakfast?"], "I put it in the, um, bread thing"))
    assert out[0].word == "toaster"
    assert out[0].confidence > 0


def test_mock_predict_miss_returns_empty():
    p = MockPredictor(table={"bread": ["toaster"]})
    out = asyncio.run(p.predict([], "totally unrelated sentence"))
    assert out == []


def test_mock_predict_word_boundary_no_false_substring():
    p = MockPredictor(table={"city": ["Tokyo"]})
    assert asyncio.run(p.predict([], "the cost of electricity")) == []


def test_parser_tolerates_schema_deviant_output():
    # bare-string items are skipped (not a crash)
    assert _parse('{"candidates":["toaster","oven"]}', 3) == []
    # non-numeric confidence degrades to 0.0 (not a ValueError)
    out = _parse('{"candidates":[{"word":"x","confidence":"high"}]}', 3)
    assert len(out) == 1 and out[0].word == "x" and out[0].confidence == 0.0
    # non-list candidates / non-dict top-level -> []
    assert _parse('{"candidates":"nope"}', 3) == []
    assert _parse('["a","b"]', 3) == []


def test_gemini_json_parser_is_tolerant():
    good = '{"candidates":[{"word":"toaster","confidence":0.9},{"word":"oven","confidence":0.5}]}'
    cands = _parse(good, max_candidates=3)
    assert [c.word for c in cands] == ["toaster", "oven"]
    assert _parse("not json", 3) == []
    assert _parse(None, 3) == []


# --- provider prompt assembly ------------------------------------------------
# These build the real request payloads WITHOUT any network call. They exist
# because a live run caught a NameError in GeminiPredictor._build_contents that
# the whole suite missed: every other test uses MockPredictor, so the concrete
# providers had no coverage of prompt assembly at all.

def _bypass_init(cls, **attrs):
    """Construct a predictor without its __init__ (which needs an API key and
    an SDK client). We are testing pure prompt assembly, not transport."""
    obj = object.__new__(cls)
    for k, v in attrs.items():
        setattr(obj, k, v)
    return obj


def test_gemini_build_contents_accepts_every_optional_hint():
    from backend.predictor.gemini import GeminiPredictor

    p = _bypass_init(GeminiPredictor, model="gemini-3.5-flash", max_candidates=3)
    contents = p._build_contents(
        ["ctx line"], "I need the um", ["spoon"], ["Frank"], ["fork"])
    text = contents[-1]["parts"][0]["text"]
    assert "I need the um" in text
    assert "Do NOT suggest: spoon" in text
    assert "Frank" in text
    assert "fork" in text and "already offered" in text.lower()


def test_claude_build_messages_accepts_every_optional_hint():
    from backend.predictor.claude import ClaudePredictor

    p = _bypass_init(ClaudePredictor, model="claude-haiku-4-5", max_candidates=3)
    messages = p._build_messages(
        ["ctx line"], "I need the um", ["spoon"], ["Frank"], ["fork"])
    text = messages[-1]["content"]
    assert "I need the um" in text
    assert "Do NOT suggest: spoon" in text
    assert "Frank" in text
    assert "fork" in text and "already offered" in text.lower()


def test_every_provider_predict_accepts_the_same_optional_kwargs():
    """The pipeline passes these by keyword to whichever provider is active, so
    a signature drift in ANY provider is a runtime break for that deployment."""
    import inspect

    from backend.predictor.claude import ClaudePredictor
    from backend.predictor.gemini import GeminiPredictor
    from backend.predictor.mock import MockPredictor

    for cls in (GeminiPredictor, ClaudePredictor, MockPredictor):
        params = inspect.signature(cls.predict).parameters
        for name in ("excluded", "entities", "already_served"):
            assert name in params, f"{cls.__name__}.predict is missing {name}"
