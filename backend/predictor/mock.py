"""Deterministic predictor for offline tests, CI, and the no-API-key demo path.

Matches the stalled fragment against a keyword table using word-boundary
matching, preferring the key that occurs closest to the END of the fragment
(the word the speaker just stalled on). Returns [] on no match (so the UI shows
nothing rather than a bogus candidate). No network, no dependencies.
"""
from __future__ import annotations

import re

from ..schemas import Candidate
from .base import WordPredictor

# Default table tuned to the bundled sample utterances (samples/utterances.json).
_DEFAULT_TABLE: dict[str, list[str]] = {
    "bread": ["toaster", "oven"],
    "toast": ["toaster", "oven"],
    "japan": ["Tokyo", "Osaka", "Kyoto"],
    "city": ["Tokyo", "Osaka", "Kyoto"],
    "pressure": ["medication", "blood pressure pills"],
    "pills": ["medication", "prescription"],
    "cutting": ["scissors", "knife"],
    "drink": ["water", "coffee"],
}


class MockPredictor(WordPredictor):
    def __init__(self, table: dict[str, list[str]] | None = None, max_candidates: int = 3) -> None:
        self.table = _DEFAULT_TABLE if table is None else table
        self.max_candidates = max_candidates

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        f = fragment.lower()
        best_words: list[str] | None = None
        best_pos = -1
        for key, words in self.table.items():
            pattern = r"\b" + re.escape(key.lower()) + r"\b"
            last = None
            for m in re.finditer(pattern, f):
                last = m
            if last is not None and last.start() > best_pos:
                best_pos = last.start()
                best_words = words
        if best_words is None and entities:
            # No keyword match in the fragment itself: fall back to the
            # injected out-of-window entity hint (most-recent-first), the
            # same way an LLM would lean on the hint when the fragment gives
            # no other cue. Lets tests/eval exercise the entity-memory path
            # deterministically without a live LLM.
            best_words = list(entities)
        if best_words is None:
            return []
        if excluded:
            best_words = [w for w in best_words if w not in excluded]
        return [
            Candidate(word=w, confidence=round(max(0.1, 1.0 - 0.15 * i), 2))
            for i, w in enumerate(best_words)
        ][: self.max_candidates]
