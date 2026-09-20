"""Salient-entity memory: tracks names/places/things mentioned earlier in the
conversation so they can be recalled after they fall out of the recent-turns
context window (see backend.pipeline / backend.config.ENTITY_MEMORY).

This is a CAPITALIZATION HEURISTIC, not a named-entity recognizer (no spaCy/
NLTK, no model, no new dependency). It will miss lowercase-styled names and
can false-positive on capitalized common nouns (place names, calendar words,
etc.). That tradeoff is deliberate: zero new dependencies, fully
deterministic, and easy to unit-test. Disclosed plainly wherever this
mechanism's output is surfaced (prompt line, eval report).
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z']*")
_SENT_BOUNDARY_RE = re.compile(r"([.!?]+)(?:\s+|$)")
_WORD_BEFORE_RE = re.compile(r"([A-Za-z]+)$")
_TITLE_RE = re.compile(r"^[A-Z][a-z]*(?:'s)?$")

# Disclosed, short, English-specific abbreviation list: a "." immediately
# after one of these does NOT end the sentence (so "Dr. Patel" tokenizes as
# one sentence, keeping "Dr" and "Patel" as a single consecutive-capitals
# run instead of splitting "Patel" onto a false sentence-start). This is a
# small hand list, not a real abbreviation detector: an abbreviation ending
# in "." that ISN'T in this list (or a genuine sentence-final use of one
# that IS, e.g. "...down on Main St." as the last sentence) still splits --
# a known, accepted tradeoff of this heuristic.
_ABBREVIATIONS = {"dr", "mr", "mrs", "ms", "st"}

# Self-reference denylist (compared case-insensitively, on the full entity
# text): the system's own product/stack names, which the presenter says out
# loud while demoing ("Hi, I'm demoing Echo...") and which the capitalization
# heuristic would otherwise learn as salient entities -- later SERVING the
# product's own name as "the word you're trying to say". A person genuinely
# named one of these is a knowingly accepted miss; disclosed here and in the
# eval report.
_SELF_REFERENCE_DENYLIST = {
    "echo",
    "chrome", "gemini", "claude", "fillernet", "silero", "hackmit",
}


def _split_sentences(text: str) -> list[str]:
    """Split `text` into sentences on . ! ? boundaries, except a lone "."
    directly after a titlecase abbreviation in _ABBREVIATIONS (see above)."""
    sentences: list[str] = []
    start = 0
    for m in _SENT_BOUNDARY_RE.finditer(text):
        punct_start = m.start()
        if m.group(1) == ".":  # a single "." (not "...", "?!", etc.) may be an abbreviation
            word_before = _WORD_BEFORE_RE.search(text[start:punct_start])
            if word_before and word_before.group(1).lower() in _ABBREVIATIONS:
                continue  # abbreviation period -- keep accumulating this sentence
        sentences.append(text[start:m.end()].strip())
        start = m.end()
    if start < len(text):
        sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def _is_title_token(tok: str) -> bool:
    """A "Title" or "Title's" shaped token, with at least 2 alphabetic
    characters (excludes bare single letters like the pronoun "I")."""
    if not _TITLE_RE.match(tok):
        return False
    base = tok[:-2] if tok.endswith("'s") else tok
    return len(base) >= 2


class EntityTracker:
    """Deterministic, dependency-free salient-entity extraction over a list
    of finalized conversation turns.

    Heuristic (applied per turn, in turn order):
      1. A capitalized token ("Title" / "Title's") that is NOT the first
         word of its sentence is an entity mention immediately (e.g.
         "...met Frank..." captures "Frank").
      2. Two or more CONSECUTIVE capitalized tokens, anywhere (including
         sentence-start), merge into one multi-word entity (e.g.
         "Pete's Diner", "Lake Winnipesaukee") -- a run of 2+ capitalized
         words in English prose is essentially never just sentence-start
         capitalization, so position doesn't matter for runs.
      3. A single capitalized word seen ONLY at sentence-start is promoted
         to an entity once it has recurred across >= 2 distinct turns (a
         one-off capitalized sentence-opener is probably just grammar; a
         word that keeps reopening sentences is probably a name).

    Recency is the turn index an entity was LAST seen at (via any of the
    three rules above); a re-mention always refreshes it forward. The
    tracked list is capped at `cap` entries, most-recent-first.
    """

    def __init__(self, cap: int = 12) -> None:
        self.cap = cap

    def extract(self, turns: list[str]) -> list[tuple[str, int]]:
        """Recompute the salient-entity list fresh from `turns` (0-indexed).

        Returns (entity_text, last_turn_index) pairs, most-recent-first,
        capped at self.cap. Pure function of `turns` -- no state carried
        between calls, so re-running over a growing conversation is always
        correct and trivial to reason about.
        """
        confident_last: dict[str, int] = {}
        word_turns: dict[str, set[int]] = {}

        for idx, turn in enumerate(turns):
            stripped = (turn or "").strip()
            if not stripped:
                continue
            for sentence in _split_sentences(stripped):
                tokens = _TOKEN_RE.findall(sentence)
                i = 0
                n = len(tokens)
                while i < n:
                    if _is_title_token(tokens[i]):
                        j = i + 1
                        while j < n and _is_title_token(tokens[j]):
                            j += 1
                        run = tokens[i:j]
                        if len(run) >= 2:
                            text = " ".join(run)
                            confident_last[text] = idx
                        else:
                            word = run[0]
                            word_turns.setdefault(word, set()).add(idx)
                            if i != 0:  # not sentence-start -> confident immediately
                                confident_last[word] = idx
                        i = j
                    else:
                        i += 1

        # Promote sentence-start-only words that recurred across >= 2 turns.
        for word, seen in word_turns.items():
            if word in confident_last:
                confident_last[word] = max(confident_last[word], max(seen))
            elif len(seen) >= 2:
                confident_last[word] = max(seen)

        ordered = sorted(
            ((text, last) for text, last in confident_last.items()
             if text.lower() not in _SELF_REFERENCE_DENYLIST),
            key=lambda kv: -kv[1],
        )
        return ordered[: self.cap]

    def out_of_window(self, turns: list[str], context_turns: int) -> list[str]:
        """Entity texts (most-recent-first, already capped) whose last
        mention falls OUTSIDE the most recent `context_turns` turns -- i.e.
        the same window `Conversation.recent(context_turns)` would send as
        live context. Entities still inside that window are excluded: they
        are already visible to the model, so re-injecting them would waste
        tokens and muddy the ablation.
        """
        n = len(turns)
        effective = max(context_turns, 0)
        window_start = max(0, n - effective)
        return [text for text, last_turn in self.extract(turns) if last_turn < window_start]
