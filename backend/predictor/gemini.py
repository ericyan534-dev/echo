"""Gemini predictor (current default).

Uses the google-genai SDK. The SDK is imported lazily inside __init__ so that
simply importing this class (e.g. in the provider factory or tests) never
requires google-genai or an API key — only instantiating the live provider
does.

Default model id is `gemini-3.5-flash` (verified live against the API's model
list). Override with GEMINI_MODEL (e.g. gemini-2.5-flash) if needed.
"""
from __future__ import annotations

import json

from ..prompts import FEW_SHOTS, SYSTEM_PROMPT, build_user_text
from ..schemas import Candidate
from .base import WordPredictor

# google-genai-compatible response schema (OpenAPI subset — no additionalProperties)
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["word", "confidence"],
            },
        }
    },
    "required": ["candidates"],
}


def _shot_answer(words: list[str]) -> str:
    return json.dumps(
        {"candidates": [{"word": w, "confidence": round(1.0 - 0.15 * i, 2)} for i, w in enumerate(words)]}
    )


class GeminiPredictor(WordPredictor):
    def __init__(self, api_key: str | None, model: str = "gemini-3.5-flash", max_candidates: int = 3,
                 stream: bool = False) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is required for the Gemini predictor.")
        from google import genai  # lazy

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model
        self.max_candidates = max_candidates
        # GEMINI_STREAM. Same request, same schema, same parser; only the
        # transport differs. The whole candidate JSON is ~25-45 output tokens,
        # so it arrives in one or two chunks and a partial parse buys nothing
        # -- what streaming buys is the end-of-response wait the non-streaming
        # endpoint adds before it returns. Measured A/B in
        # eval/results/predictor_latency_bench.json.
        self.stream = bool(stream)

    def _build_contents(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[dict]:
        contents: list[dict] = []
        for inp, out in FEW_SHOTS:
            contents.append({"role": "user", "parts": [{"text": build_user_text(inp["context"], inp["fragment"])}]})
            contents.append({"role": "model", "parts": [{"text": _shot_answer(out)}]})
        contents.append({"role": "user", "parts": [{"text": build_user_text(context, fragment, excluded, entities, already_served)}]})
        return contents

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        from google.genai import types  # lazy

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=_RESPONSE_SCHEMA,
            temperature=0.2,
            max_output_tokens=256,
            # gemini-3.5-flash is a thinking model; without this it spends the
            # whole output budget on thoughts and returns empty text. Thinking
            # off also cuts latency — essential for the ~1-2s stall->word loop.
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        )
        contents = self._build_contents(context, fragment, excluded, entities,
                                        already_served)
        if self.stream:
            text = await self._generate_streamed(contents, config)
        else:
            resp = await self._client.aio.models.generate_content(
                model=self.model, contents=contents, config=config)
            text = _text_of(resp)
        return _parse(text, self.max_candidates)

    async def _generate_streamed(self, contents: list[dict], config) -> str | None:
        """generate_content_stream, concatenated. Structured-output chunks are
        valid partial JSON strings that concatenate to the full object, so the
        parser sees exactly what the non-streaming path would have seen."""
        pieces: list[str] = []
        stream = await self._client.aio.models.generate_content_stream(
            model=self.model, contents=contents, config=config)
        async for chunk in stream:
            piece = _text_of(chunk)
            if piece:
                pieces.append(piece)
        return "".join(pieces) or None


def _text_of(resp) -> str | None:
    """Extract concatenated text parts, skipping thought signatures etc.
    (avoids the SDK's noisy non-text-parts warning that resp.text emits)."""
    try:
        parts = resp.candidates[0].content.parts or []
        text = "".join(p.text for p in parts if getattr(p, "text", None))
        return text or None
    except (AttributeError, IndexError, TypeError):
        return None


def _parse(text: str | None, max_candidates: int) -> list[Candidate]:
    """Tolerant parser shared by the Gemini and Claude providers.

    Never raises on schema-deviant LLM output: non-JSON, a non-dict top level, a
    non-list `candidates`, non-dict items, missing words, or non-numeric
    confidence all degrade gracefully to skipping that item / returning [].
    """
    if not text:
        return []
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return []
    if not isinstance(data, dict):
        return []
    items = data.get("candidates", [])
    if not isinstance(items, list):
        return []
    out: list[Candidate] = []
    for c in items:
        if not isinstance(c, dict):
            continue
        word = str(c.get("word", "")).strip()
        if not word:
            continue
        try:
            conf = float(c.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        out.append(Candidate(word=word, confidence=conf))
    return out[:max_candidates]
