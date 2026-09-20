"""Deepgram streaming STT (real-time path).

SKELETON / NEEDS LIVE VALIDATION. The deepgram-sdk is imported lazily. This
wires interim results + endpointing to Word / TurnEnd items and runs a silence
timer to emit SilenceTick. It is not exercised by the offline test suite (needs
a key + audio); validate against the live API before the demo.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from .base import STTProvider, StreamItem


class DeepgramSTT(STTProvider):
    def __init__(self, api_key: str | None, audio_source=None, tick_ms: int = 200) -> None:
        if not api_key:
            raise ValueError("DEEPGRAM_API_KEY is required for the Deepgram STT provider.")
        self.api_key = api_key
        self.audio_source = audio_source  # async iterator of PCM chunks
        self.tick_ms = tick_ms
        self._queue: asyncio.Queue[StreamItem] = asyncio.Queue()

    async def stream(self) -> AsyncIterator[StreamItem]:  # pragma: no cover - live only
        # TODO(live): open Deepgram websocket with interim_results=True,
        # smart_format=True, endpointing enabled. On each transcript word push a
        # Word(text, start_ms, end_ms, is_final). On UtteranceEnd push TurnEnd().
        # Run a parallel asyncio task that emits SilenceTick every tick_ms using a
        # monotonic clock so the StallDetector can see mid-utterance pauses.
        raise NotImplementedError(
            "DeepgramSTT.stream is a skeleton — implement against the live "
            "deepgram-sdk websocket before using STT_PROVIDER=deepgram."
        )
