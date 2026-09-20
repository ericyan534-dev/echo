"""Demo-day network fallback predictor.

Wraps a live inner predictor (Gemini by default) with a strict, disclosed
fallback: if the inner predictor times out or raises (venue Wi-Fi drops, API
hiccup, etc.), serve a candidate from a committed local lookup keyed on the
four rehearsed demo-script fragments (see docs/DEMO_SCRIPT.md's "3-tier
fallback drill"). This composes UNDER backend/pipeline.py's own 4 s timeout +
stale-cache degradation -- it is a faster, disclosed safety net in front of
that, not a replacement for it.

STRICTNESS (judge panel rule): the local lookup fires ONLY on an inner
timeout/exception -- never as a first choice, and never on a successful-but-
empty inner result (an ad-libbed judge question that legitimately returns []
must surface as [], not a canned answer). Every fallback serve logs one ASCII
line to the server console so the behaviour is presenter-visible and can
never be silently passed off as unqualified "live" -- see the disclosure rule
in docs/DEMO_SCRIPT.md.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from ..schemas import Candidate
from .base import WordPredictor
from .mock import MockPredictor

log = logging.getLogger("echo.demo_fallback")

_DEFAULT_WORDS_PATH = Path(__file__).resolve().parent / "demo_fallback_words.json"


def _load_table(path: Path) -> dict[str, list[str]]:
    """Load the committed JSON lookup. Never raises: a missing/malformed file
    degrades to an empty table (every lookup then misses -> [])."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(key): [str(w) for w in words]
        for key, words in data.items()
        if isinstance(words, list)
    }


class DemoFallbackPredictor(WordPredictor):
    """predict() tries `inner` under its own timeout; only a timeout or
    exception falls through to the local backup lookup. A successful inner
    call is returned unchanged, even when it's an empty list.

    The local lookup itself is a MockPredictor built from the committed JSON
    table: its word-boundary, closest-to-end-of-fragment matching (case-
    insensitive) IS the normalization, and it already respects `excluded` by
    filtering matched words before returning -- both requirements come for
    free from tested code instead of being reimplemented here.
    """

    def __init__(
        self,
        inner: WordPredictor,
        timeout_s: float = 1.5,
        max_candidates: int = 3,
        words_path: Path | str | None = None,
    ) -> None:
        self.inner = inner
        self.timeout_s = timeout_s
        table = _load_table(Path(words_path) if words_path else _DEFAULT_WORDS_PATH)
        self._local = MockPredictor(table=table, max_candidates=max_candidates)

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        **kwargs,
    ) -> list[Candidate]:
        # excluded/entities (and any future optional predictor kwargs) are
        # forwarded verbatim to the inner predictor -- a wrapper that silently
        # drops parameters would disable features like entity memory whenever
        # the fallback provider is active. The local path also receives
        # entities: during an outage a name-stall can then be served from the
        # REAL conversation's entity list (live data, not a canned answer).
        try:
            return await asyncio.wait_for(
                self.inner.predict(
                    context, fragment, excluded=excluded, entities=entities, **kwargs
                ),
                timeout=self.timeout_s,
            )
        except Exception:
            # Covers asyncio.TimeoutError (the wait_for timeout) and any
            # inner-predictor failure (network error, SDK exception, bad
            # response, etc.) -- both are "the network path failed" for
            # disclosure purposes.
            log.warning(
                "[demo-fallback] network path failed; served local backup for fragment: %r",
                fragment,
            )
            return await self._local.predict(
                context, fragment, excluded=excluded, entities=entities
            )
