"""STT provider interface.

A provider yields a stream of transcript items: Word (interim or final),
SilenceTick (timer-driven, lets the detector notice pauses), or TurnEnd.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from ..schemas import SilenceTick, TurnEnd, Word

StreamItem = Word | SilenceTick | TurnEnd


class STTProvider(ABC):
    @abstractmethod
    def stream(self) -> AsyncIterator[StreamItem]:
        """Async-iterate transcript items as they arrive."""
        raise NotImplementedError
