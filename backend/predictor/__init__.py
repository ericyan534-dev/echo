"""Predictor factory — selects the provider from Settings."""
from __future__ import annotations

from ..config import Settings, get_settings
from .base import WordPredictor


def get_predictor(settings: Settings | None = None) -> WordPredictor:
    s = settings or get_settings()
    provider = s.predictor_provider
    primary = _single(s, provider)

    # PREDICTOR_FALLBACKS: race the primary against real fallback providers
    # (predictor/failover.py). A fallback that repeats the primary is dropped;
    # an empty list returns the bare provider, byte-for-byte the old behaviour.
    fallbacks = [f for f in getattr(s, "predictor_fallbacks", ()) if f != provider]
    if not fallbacks:
        return primary
    from .failover import FailoverPredictor
    chain = [(provider, primary)] + [(f, _single(s, f)) for f in fallbacks]
    return FailoverPredictor(chain)


def _single(s: Settings, provider: str) -> WordPredictor:
    """One provider by name, no failover wrapping."""
    if provider == "gemini":
        from .gemini import GeminiPredictor
        # getattr: Settings is constructed positionally in tests and older call
        # sites, so a missing field must read as the default (off), never raise.
        return GeminiPredictor(api_key=s.gemini_api_key, model=s.gemini_model,
                               max_candidates=s.max_candidates,
                               stream=bool(getattr(s, "gemini_stream", False)))

    if provider == "claude":
        from .claude import ClaudePredictor
        return ClaudePredictor(api_key=s.claude_api_key, model=s.claude_model, max_candidates=s.max_candidates)

    if provider == "deepseek":
        from .deepseek import DeepSeekPredictor
        return DeepSeekPredictor(api_key=s.deepseek_api_key, model=s.deepseek_model,
                                 base_url=s.deepseek_base_url,
                                 max_candidates=s.max_candidates)

    if provider == "local":
        from .local import LocalPredictor

        # getattr with defaults on purpose: the local_* settings are added by
        # the integrator (plan step I1) and this must work BEFORE that lands --
        # the constructor defaults are the single source of truth until then,
        # and a missing setting must never become an AttributeError here.
        return LocalPredictor(
            base_url=getattr(s, "local_llm_url", "http://127.0.0.1:8080"),
            model=getattr(s, "local_llm_model", "Qwen3.8-27B-Q4_K_M"),
            ctx_size=getattr(s, "local_llm_ctx", 4096),
            max_candidates=s.max_candidates,
        )

    if provider in ("mock", "stub"):
        from .mock import MockPredictor
        return MockPredictor(max_candidates=s.max_candidates)

    if provider == "demo_fallback":
        from .demo_fallback import DemoFallbackPredictor

        inner_provider = s.demo_fallback_inner
        if inner_provider == "demo_fallback":
            raise ValueError("DEMO_FALLBACK_INNER cannot be 'demo_fallback' (would recurse).")
        inner = _single(s, inner_provider)
        return DemoFallbackPredictor(
            inner=inner,
            timeout_s=s.demo_fallback_timeout_s,
            max_candidates=s.max_candidates,
        )

    raise ValueError(
        f"Unknown PREDICTOR_PROVIDER: {provider!r} "
        "(expected gemini | claude | deepseek | local | mock | demo_fallback)"
    )


__all__ = ["get_predictor", "WordPredictor"]
