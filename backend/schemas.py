"""Shared data types for the Echo pipeline.

Plain dataclasses (no pydantic) so the core logic and tests have zero
third-party dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# --- prediction output ---------------------------------------------------
@dataclass
class Candidate:
    """A predicted intended word and the model's confidence in it."""
    word: str
    confidence: float = 0.0


@dataclass
class Prediction:
    candidates: list[Candidate]
    fragment: str
    trigger: str
    latency_ms: float = 0.0
    served: str = "live"   # "live" | "prefetch" | "prefetch-stale" (live call failed; cache served as fallback) | "reject" (re-predicted after the speaker rejected word(s))


# --- transcript stream items (what an STT provider yields) ---------------
@dataclass
class Word:
    """A recognized word with timing (ms). is_final=False for interim STT."""
    text: str
    start_ms: int = 0
    end_ms: int = 0
    is_final: bool = True
    # Confidence (0..1) that the WEARER spoke this word, computed browser-side
    # where transcript and mic PCM share one clock. None = unknown, which must
    # never suppress: see backend/acoustic/speaker_gate.py. New field goes LAST
    # -- Word is built positionally in places.
    wearer_conf: float | None = None


@dataclass
class SilenceTick:
    """Emitted by a timer between words so the detector can notice a pause.

    `speech_end_ms` is when the audio channel last heard the speaker, which is
    NOT the same instant as the last committed word's end. A streaming ASR
    holds the tail of an utterance back until agreement or a silence-forced
    commit (LocalAgreement + silence_commit_ms: ~700 ms at the shipped
    setting), so a detector that measures the pause from the last COMMITTED
    word measures the pipeline's commit lag on top of the speaker's actual
    pause -- and fires a stall on someone who has been quiet for 200 ms.
    Measured: continuous speech 0-2464 ms, committed words ending at 1290 ms,
    speaker stops at 2464, first tick at 2664 -> the detector computed
    2664-1290 = 1374 >= pause_ms and fired on a 200 ms pause.

    None means "the emitter does not know" -- a browser/mock timer, or a
    replayed stream recorded before this field existed. It must never make the
    pause LOOK longer, so the detector takes the later of the two bounds and a
    missing value simply leaves the old behaviour in place.
    """
    at_ms: int
    # New field goes LAST -- SilenceTick is built positionally in places.
    speech_end_ms: int | None = None


@dataclass
class TurnEnd:
    """Marks the end of the speaker's utterance (STT endpointing)."""
    pass


# --- acoustic channel ------------------------------------------------------
@dataclass
class AcousticEvent:
    """Event from the raw-audio channel (things transcripts can't see)."""
    kind: str              # "filler" | "prolongation"
    at_ms: int
    confidence: float = 1.0


# --- detector output -----------------------------------------------------
@dataclass
class StallEvent:
    fragment: str          # the FULL current utterance (never truncated)
    trigger: str           # "pause" | "filler" | "hedge" | "filler_acoustic" | "prolongation"
    at_ms: int = 0
    # Words ALREADY OFFERED earlier in this same turn, so the model does not
    # loop on a suggestion the speaker has moved past.
    #
    # This deliberately carries served WORDS, not the earlier fragments. The
    # first cut of this field held the prior fragments, which turned out to be
    # prefixes of the current one -- by the sixth stall the list was six copies
    # of the same sentence, and telling the model "you already tried: <the
    # sentence it is currently reading>" is worse than saying nothing. The
    # fragment is never truncated, so the earlier attempt is already visible;
    # what the model cannot infer is which candidates were spent.
    already_served: list[str] = field(default_factory=list)


@dataclass
class SearchEpisode:
    """One word-search by the speaker. The debounce unit.

    Debounce asks 'is there an unresolved episode?' -- a question about search
    state. It deliberately does NOT truncate text; that conflation is the bug
    this design removes.
    """
    started_at_ms: int
    trigger: str
    fragment_at_fire: str
    served_word: str | None = None
    rejected: list[str] = field(default_factory=list)
    resolved: bool = False
