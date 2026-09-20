"""Folding scrolled-out conversation turns into a compact retained record.

DEFAULT IS EXTRACTIVE, AND THAT IS A DELIBERATE CHOICE
------------------------------------------------------
An LLM summary would be richer. It would also put an LLM call on the path of
every long conversation -- and the roadmap's on-device predictor (EchoLM,
Qwen2.5-1.5B via llama.cpp) is both slower and weaker at summarization than the
cloud model. A local deployment would inherit that cost with no way to opt out.

So the default is a RETENTION rule, not compression: a scrolled-out turn is
kept verbatim if and only if it holds the SOLE mention of a salient entity --
i.e. dropping it would lose the only record of a name or place. Everything else
is discarded, including turns whose entities recur elsewhere: a name mentioned
in five turns is not at risk of being lost, and keeping all five would make the
summary grow linearly with the conversation, which is the very problem this is
meant to solve.

Because the output is a literal subset of what was said, it cannot hallucinate,
and it is exactly unit-testable.

LLMSummarizer remains available for deployments that want richer compression;
it must run off the stall path (during fluent speech), never at serve time.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from .entities import EntityTracker


class Summarizer(ABC):
    @abstractmethod
    async def fold(self, turns: list[str], existing: str = "") -> str:
        """Fold `turns` (scrolling out of the verbatim window) into `existing`.
        Returns the new summary text."""
        raise NotImplementedError


class ExtractiveSummarizer(Summarizer):
    def __init__(self, tracker: EntityTracker | None = None) -> None:
        self.tracker = tracker or EntityTracker()

    async def fold(self, turns: list[str], existing: str = "") -> str:
        kept: list[str] = []
        if existing.strip():
            kept.append(existing.strip())

        # Entity -> indices of the turns mentioning it. A turn earns retention
        # only when it is the ONLY turn mentioning some entity: that is exactly
        # the case where dropping it loses information permanently.
        unique_turn_indices: set[int] = set()
        for entity, _last in self.tracker.extract(turns):
            hits = [i for i, t in enumerate(turns) if entity in t]
            if len(hits) == 1:
                unique_turn_indices.add(hits[0])

        for i in sorted(unique_turn_indices):
            text = turns[i].strip()
            if text and text not in kept:
                kept.append(text)
        return "\n".join(kept)
