"""Claude predictor — the swap-back target.

Implemented behind the same WordPredictor interface as Gemini. To reintegrate
Claude, set PREDICTOR_PROVIDER=claude and ANTHROPIC_API_KEY; no other code
changes. The anthropic SDK is imported lazily so this module imports cleanly
even when anthropic is not installed.

Uses Claude structured outputs (output_config.format) with the shared
CANDIDATES_SCHEMA, and claude-haiku-4-5 for low latency. The `effort` parameter
is intentionally omitted (it is not supported on Haiku).
"""
from __future__ import annotations

from ..prompts import CANDIDATES_SCHEMA, FEW_SHOTS, SYSTEM_PROMPT, build_user_text
from ..schemas import Candidate
from .base import WordPredictor
from .gemini import _parse, _shot_answer  # reuse tolerant parser and few-shot formatter


class ClaudePredictor(WordPredictor):
    def __init__(self, api_key: str | None, model: str = "claude-haiku-4-5", max_candidates: int = 3) -> None:
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required for the Claude predictor.")
        import anthropic  # lazy

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.max_candidates = max_candidates

    def _build_messages(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[dict]:
        messages: list[dict] = []
        for inp, out in FEW_SHOTS:
            messages.append({"role": "user", "content": build_user_text(inp["context"], inp["fragment"])})
            messages.append({"role": "assistant", "content": _shot_answer(out)})
        messages.append({"role": "user", "content": build_user_text(context, fragment, excluded, entities, already_served)})
        return messages

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        resp = await self._client.messages.create(
            model=self.model,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=self._build_messages(context, fragment, excluded, entities,
                                          already_served),
            output_config={"format": {"type": "json_schema", "schema": CANDIDATES_SCHEMA}},
        )
        text = next((b.text for b in resp.content if getattr(b, "type", None) == "text"), None)
        return _parse(text, self.max_candidates)
