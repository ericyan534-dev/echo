"""Append-only transcript timeline -- the single substrate every consumer
derives a view over.

WHY THIS EXISTS
---------------
The previous design let StallDetector own a mutable cursor (`_clause_start`)
that served three unrelated purposes at once: debounce state, the text sent to
the predictor, and the text recorded as conversation memory. Re-arming the
debounce therefore truncated the prediction input AND the conversation record.
Measured effect: after one served suggestion the model received 'on some'
instead of the sentence, and a turn containing a stall was stored as
'every day at home' -- losing the proper noun and the object.

The fix is structural, not a tuning change: nothing here truncates. Consumers
take derived views; no consumer can destroy another's view.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .schemas import Word

FILLERS = {"um", "umm", "uhm", "uh", "uhh", "er", "err", "erm", "hmm", "mm", "mmm"}

_WORD_RE = re.compile(r"[^a-z' ]+")


def norm(text: str) -> str:
    """Lowercase, strip punctuation. Shared by the timeline and the detector so
    'content word' means exactly one thing across the system."""
    return _WORD_RE.sub("", text.lower()).strip()


@dataclass(frozen=True)
class TimelineWord:
    text: str
    start_ms: int
    end_ms: int
    is_final: bool
    turn_index: int
    wearer_conf: float | None = None   # None = unknown; never suppress on None


@dataclass(frozen=True)
class TimelineEvent:
    kind: str          # "acoustic" | "stall" | "served" | "rejected" | "accepted"
    at_ms: int
    payload: dict = field(default_factory=dict)


class Timeline:
    """Append-only word/event log.

    SPEAKER FILTERING. `wearer_conf_min` decides which words count as the
    wearer's. A word believed to come from someone else is KEPT in `words` --
    destroying the record to serve one consumer is precisely the amputation
    defect this module exists to prevent -- but it is excluded from every
    derived TEXT view: the utterance we ask the model to complete, the content
    count that arms triggers, and the turn record.

    FAIL OPEN is a contract, not a default: `wearer_conf is None` means "no
    gate ran, or the gate could not tell", and such words are ALWAYS included.
    Wrongly dropping the wearer's own words is a far worse failure than
    admitting a bystander's.
    """

    def __init__(self, wearer_conf_min: float = 0.0) -> None:
        self.words: list[TimelineWord] = []
        self.events: list[TimelineEvent] = []
        self.wearer_conf_min = wearer_conf_min
        self._turn_index = 0

    # --- ingest -------------------------------------------------------
    def add_word(self, word: Word, wearer_conf: float | None = None) -> TimelineWord | None:
        """Append a word. Interim hypotheses are dropped, not stored: committing
        them would double-count when streaming STT revises a word.

        `wearer_conf` overrides `word.wearer_conf` when given, so a server-side
        gate can supply the value for a transport that carries none (the
        server-side ASR path sees audio only). Otherwise the browser's per-word
        estimate is used.
        """
        if not word.text or not word.text.strip():
            return None
        if not word.is_final:
            return None
        conf = wearer_conf if wearer_conf is not None else getattr(word, "wearer_conf", None)
        tw = TimelineWord(
            text=word.text.strip(),
            start_ms=word.start_ms,
            end_ms=word.end_ms,
            is_final=True,
            turn_index=self._turn_index,
            wearer_conf=conf,
        )
        self.words.append(tw)
        return tw

    # --- speaker filtering --------------------------------------------
    def is_wearer(self, word: TimelineWord) -> bool:
        """True unless the gate positively says this was somebody else."""
        if word.wearer_conf is None:
            return True                      # unknown -> include (fail open)
        return word.wearer_conf >= self.wearer_conf_min

    def add_event(self, kind: str, at_ms: int, payload: dict | None = None) -> None:
        self.events.append(TimelineEvent(kind=kind, at_ms=at_ms, payload=payload or {}))

    def mark_turn_boundary(self) -> None:
        """Close the current utterance. Appends a boundary; discards nothing."""
        self._turn_index += 1

    # --- views --------------------------------------------------------
    # Every text view below filters on is_wearer(). `all_words=True` returns
    # the unfiltered record, which is what a transcript display or a debug dump
    # wants -- the raw log is never mutated, only viewed.
    def current_utterance(self, all_words: bool = False) -> list[TimelineWord]:
        return [w for w in self.words
                if w.turn_index == self._turn_index and (all_words or self.is_wearer(w))]

    def utterance_text(self, all_words: bool = False) -> str:
        return " ".join(w.text for w in self.current_utterance(all_words)).strip()

    def completed_turns(self, all_words: bool = False) -> list[str]:
        out: list[str] = []
        for idx in range(self._turn_index):
            text = " ".join(w.text for w in self.words
                            if w.turn_index == idx and (all_words or self.is_wearer(w))).strip()
            if text:
                out.append(text)
        return out

    def content_count(self) -> int:
        """Content words attributable to the WEARER. Gating this is what stops
        a bystander arming the pause trigger, which needs >= 2 content words."""
        return sum(1 for w in self.current_utterance()
                   if norm(w.text) and norm(w.text) not in FILLERS)
