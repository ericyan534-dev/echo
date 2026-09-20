"""Scripted STT provider — drives the pipeline deterministically from a list of
StreamItems. Used by tests, the replay harness, and the keyless demo path.

Helper `script_from_spec` turns a compact spec into timed Word/SilenceTick/
TurnEnd items so samples are easy to author.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from ..schemas import SilenceTick, TurnEnd, Word
from .base import STTProvider, StreamItem


class MockSTT(STTProvider):
    def __init__(self, script: list[StreamItem] | None = None, realtime: bool = False) -> None:
        self.script = script or []
        self.realtime = realtime  # if True, sleep to roughly mimic timing

    async def stream(self) -> AsyncIterator[StreamItem]:
        last_ms = 0
        for item in self.script:
            if self.realtime:
                # Explicit None checks — a legitimate timestamp of 0 must not be
                # treated as falsy and fall through to last_ms.
                ts = getattr(item, "end_ms", None)
                if ts is None:
                    ts = getattr(item, "at_ms", None)
                now = last_ms if ts is None else ts
                await asyncio.sleep(max(0.0, (now - last_ms) / 1000.0))
                last_ms = now
            yield item


def script_from_spec(
    words: list[str],
    *,
    word_ms: int = 280,
    gap_ms: int = 120,
    stall_after: int | None = None,
    stall_ms: int = 1500,
    end: bool = True,
) -> list[StreamItem]:
    """Build a timed script.

    `stall_after` = index after which to insert a SilenceTick long enough to
    trigger a pause stall (simulates the speaker getting stuck).
    """
    items: list[StreamItem] = []
    t = 0
    for i, w in enumerate(words):
        items.append(Word(text=w, start_ms=t, end_ms=t + word_ms))
        t += word_ms
        if stall_after is not None and i == stall_after:
            items.append(SilenceTick(at_ms=t + stall_ms))
            t += stall_ms
        else:
            t += gap_ms
    if end:
        items.append(TurnEnd())
    return items
