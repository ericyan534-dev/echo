"""Conversation context: the recent completed utterances used to ground
the word prediction.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Conversation:
    turns: list[str] = field(default_factory=list)

    def add_turn(self, text: str) -> None:
        text = (text or "").strip()
        if text:
            self.turns.append(text)

    def recent(self, n: int) -> list[str]:
        if n <= 0:
            return []
        return self.turns[-n:]
