"""GEMINI_STREAM: the streaming transport must be invisible to the pipeline.

No network. A fake genai client records which endpoint was hit and returns
structured-output chunks the way the real API does (valid partial JSON strings
that concatenate to the full object, sometimes split mid-token).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from backend.config import Settings, get_settings
from backend.predictor import get_predictor
from backend.predictor.gemini import GeminiPredictor


def _part(text):
    return SimpleNamespace(text=text)


def _resp(parts):
    return SimpleNamespace(candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts))])


class _FakeModels:
    def __init__(self, chunks: list[str], full: str) -> None:
        self.chunks = chunks
        self.full = full
        self.calls: list[str] = []

    async def generate_content(self, **kw):
        self.calls.append("generate_content")
        return _resp([_part(self.full)])

    async def generate_content_stream(self, **kw):
        self.calls.append("generate_content_stream")

        async def gen():
            for c in self.chunks:
                # a thought-signature-only chunk, like the real API emits
                yield _resp([SimpleNamespace(thought_signature=b"x")])
                yield _resp([_part(c)])
        return gen()


def _predictor(stream: bool, chunks, full) -> tuple[GeminiPredictor, _FakeModels]:
    p = object.__new__(GeminiPredictor)
    models = _FakeModels(chunks, full)
    p._client = SimpleNamespace(aio=SimpleNamespace(models=models))
    p._genai = None
    p.model = "gemini-3.5-flash"
    p.max_candidates = 3
    p.stream = stream
    return p, models


FULL = '{"candidates":[{"word":"Maria","confidence":0.95},{"word":"sister","confidence":0.6}]}'
# Split mid-key and mid-value on purpose: the parser must only ever see the join.
CHUNKS = ['{"candidates":[{"wo', 'rd":"Maria","confidence":0.9', '5},{"word":"sister","confidence":0.6}]}']


def _predict(p):
    # google.genai.types is imported lazily inside predict(); the SDK is an
    # installed dependency, and the config object it builds is never sent.
    return asyncio.run(p.predict(["My sister Maria visited."], "I need to call, um, the"))


def test_stream_off_is_the_shipped_single_call():
    p, models = _predictor(False, CHUNKS, FULL)
    cands = _predict(p)
    assert models.calls == ["generate_content"]
    assert [c.word for c in cands] == ["Maria", "sister"]


def test_stream_on_concatenates_chunks_before_parsing():
    p, models = _predictor(True, CHUNKS, FULL)
    cands = _predict(p)
    assert models.calls == ["generate_content_stream"]
    assert [(c.word, c.confidence) for c in cands] == [("Maria", 0.95), ("sister", 0.6)]


def test_stream_on_and_off_agree():
    off, _ = _predictor(False, CHUNKS, FULL)
    on, _ = _predictor(True, CHUNKS, FULL)
    assert _predict(off) == _predict(on)


def test_stream_with_no_text_chunks_degrades_to_empty():
    p, _ = _predictor(True, [], FULL)
    assert _predict(p) == []


def test_constructor_default_is_off():
    p = object.__new__(GeminiPredictor)
    GeminiPredictor.__init__.__defaults__  # exists
    import inspect
    sig = inspect.signature(GeminiPredictor.__init__)
    assert sig.parameters["stream"].default is False


def test_settings_default_off_and_env_flag(monkeypatch):
    monkeypatch.delenv("GEMINI_STREAM", raising=False)
    assert get_settings().gemini_stream is False
    monkeypatch.setenv("GEMINI_STREAM", "on")
    assert get_settings().gemini_stream is True


def test_factory_wires_the_flag(monkeypatch):
    seen = {}

    class _Spy(GeminiPredictor):
        def __init__(self, api_key, model="m", max_candidates=3, stream=False):
            seen["stream"] = stream

    monkeypatch.setattr("backend.predictor.gemini.GeminiPredictor", _Spy)
    s = Settings(
        predictor_provider="gemini", gemini_model="gemini-3.5-flash", gemini_api_key="dummy",
        claude_model="c", claude_api_key=None,
        deepseek_model="deepseek-flash", deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="gemini",
        demo_fallback_timeout_s=1.5, stt_provider="mock", deepgram_api_key=None,
        max_candidates=3, pause_ms=1300, context_turns=6, acoustic_model="",
        acoustic_conf=0.7, prefetch=False, prefetch_every=3, entity_memory=False,
        gemini_stream=True,
    )
    get_predictor(s)
    assert seen["stream"] is True
    # Positional construction without the field still means off.
    from dataclasses import replace
    get_predictor(replace(s, gemini_stream=False))
    assert seen["stream"] is False
