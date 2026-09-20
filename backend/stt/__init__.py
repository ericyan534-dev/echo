"""STT provider factory."""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import STTProvider


def get_stt(settings: Settings | None = None, **kwargs) -> STTProvider:
    s = settings or get_settings()
    if s.stt_provider in ("mock", "replay"):
        from .mock import MockSTT
        return MockSTT(**kwargs)
    if s.stt_provider == "deepgram":
        from .deepgram import DeepgramSTT
        return DeepgramSTT(api_key=s.deepgram_api_key, **kwargs)
    raise ValueError(f"Unknown STT_PROVIDER: {s.stt_provider!r} (expected mock | deepgram)")


__all__ = ["get_stt", "STTProvider"]
