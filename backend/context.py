"""Assembles the predictor's context under a token budget.

WHY A BUDGET AND NOT A TURN COUNT
---------------------------------
CONTEXT_TURNS=6 is a count. It says nothing about size, and it cannot adapt to
a predictor with a small window. The roadmap's on-device model (EchoLM,
Qwen2.5-1.5B via llama.cpp) has a far tighter effective window than Gemini, so
context has to be assembled against a budget the predictor declares rather than
against a number of turns that happened to feel right for a cloud model.

LAYER PRIORITY (highest first). Under pressure, lower layers are dropped:
  1. current utterance + already-served hints  -- INVIOLABLE, never dropped
  2. verbatim recent turns                     -- short-term fidelity
  3. rolling summary of scrolled-out turns     -- long-horizon memory
  4. salient entities                          -- cheap recall hints

Token counting is APPROXIMATE (len/4). Deliberately no tokenizer dependency:
the budget is a safety margin, not an exact accounting, and it is documented as
such wherever it is surfaced.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .entities import EntityTracker
from .summarizer import Summarizer

# ~4 characters per token. A rough English average, good enough for a margin.
_CHARS_PER_TOKEN = 4


def approx_tokens(text: str) -> int:
    """Approximate token count. Documented heuristic, never exact."""
    return len(text or "") // _CHARS_PER_TOKEN


@dataclass
class ContextPayload:
    utterance: str
    recent_turns: list[str] = field(default_factory=list)
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    already_served: list[str] = field(default_factory=list)

    def approx_total_tokens(self) -> int:
        parts = [self.utterance, self.summary]
        parts.extend(self.recent_turns)
        parts.extend(self.entities)
        parts.extend(self.already_served)
        return sum(approx_tokens(p) for p in parts)


class ContextBuilder:
    def __init__(
        self,
        summarizer: Summarizer,
        tracker: EntityTracker | None = None,
        budget_tokens: int = 2048,
        verbatim_turns: int = 6,
    ) -> None:
        self.summarizer = summarizer
        self.tracker = tracker or EntityTracker()
        self.budget_tokens = budget_tokens
        self.verbatim_turns = verbatim_turns

    def build(
        self,
        conversation_turns: list[str],
        utterance: str,
        already_served: list[str] | None = None,
        excluded: list[str] | None = None,
        summary: str = "",
    ) -> ContextPayload:
        already_served = list(already_served or [])
        excluded = list(excluded or [])

        payload = ContextPayload(utterance=utterance, already_served=already_served,
                                 excluded=excluded)

        # Layer 1 is already in and is never dropped -- if it alone exceeds the
        # budget the caller gets an over-budget payload rather than a mutilated
        # utterance, because a truncated utterance is the defect this whole
        # design exists to remove.
        spent = approx_tokens(utterance) + sum(approx_tokens(a) for a in already_served)
        remaining = self.budget_tokens - spent

        # Layer 2: verbatim recent turns. Walk backwards so the NEWEST survive
        # budget pressure, then restore chronological order -- a model reading
        # the conversation backwards is worse than one reading less of it.
        recent = conversation_turns[-self.verbatim_turns:] if self.verbatim_turns > 0 else []
        kept: list[str] = []
        for turn in reversed(recent):
            cost = approx_tokens(turn)
            if cost > remaining:
                break
            kept.append(turn)
            remaining -= cost
        payload.recent_turns = list(reversed(kept))

        # Layer 3: rolling summary of everything older.
        if summary and approx_tokens(summary) <= remaining:
            payload.summary = summary
            remaining -= approx_tokens(summary)

        # Layer 4: entities whose last mention already scrolled out of layer 2.
        entities = self.tracker.out_of_window(conversation_turns, self.verbatim_turns)
        for e in entities:
            cost = approx_tokens(e)
            if cost > remaining:
                break
            payload.entities.append(e)
            remaining -= cost

        return payload
