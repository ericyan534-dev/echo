"""Provider failover: a hedged race across a chain of real predictors.

Motivating failure (demo rehearsal, 2026-09-20): DeepSeek returned HTTP 503 /
timed out for 30+ consecutive stalls. DeepSeekPredictor "never raises", so
every one of those became an EMPTY candidate list -> detection fired, no word
came out. The failover treats an empty or failed answer from one provider as
"ask the next one", and hedges a slow provider by starting the next before
the first has given up.
"""
import asyncio
from dataclasses import replace

from backend.config import Settings
from backend.predictor import get_predictor
from backend.predictor.base import WordPredictor
from backend.predictor.failover import FailoverPredictor
from backend.schemas import Candidate


class Fixed(WordPredictor):
    """Answers `answer` after `delay` seconds; records calls; raises if asked."""

    def __init__(self, answer, delay=0.0, raises=False):
        self.answer, self.delay, self.raises = answer, delay, raises
        self.calls: list[dict] = []
        self.cancelled = 0

    async def predict(self, context, fragment, excluded=None, entities=None,
                      already_served=None):
        self.calls.append(dict(context=context, fragment=fragment, excluded=excluded,
                               entities=entities, already_served=already_served))
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.raises:
            raise RuntimeError("simulated provider failure")
        return self.answer


TOASTER = [Candidate(word="toaster", confidence=0.9)]
OVEN = [Candidate(word="oven", confidence=0.6)]


def _run(fo, fragment="I put the bread in the"):
    return asyncio.run(fo.predict(["Every morning"], fragment))


# --- happy path: primary answers, fallback never consulted ------------------

def test_primary_answer_is_served_and_fallback_is_not_called():
    primary, backup = Fixed(TOASTER), Fixed(OVEN)
    fo = FailoverPredictor([("deepseek", primary), ("gemini", backup)],
                           hedge_after_s=1.0, timeout_s=3.0)
    assert _run(fo) == TOASTER
    assert backup.calls == []
    assert fo.last_served_by == "deepseek"


# --- the demo failure: primary "succeeds" with [] (DeepSeek 503 -> []) ------

def test_empty_primary_falls_through_to_the_next_provider():
    primary, backup = Fixed([]), Fixed(TOASTER)
    fo = FailoverPredictor([("deepseek", primary), ("gemini", backup)],
                           hedge_after_s=1.0, timeout_s=3.0)
    assert _run(fo) == TOASTER
    assert len(primary.calls) == 1 and len(backup.calls) == 1
    assert fo.last_served_by == "gemini"


def test_raising_primary_falls_through_to_the_next_provider():
    primary, backup = Fixed(None, raises=True), Fixed(TOASTER)
    fo = FailoverPredictor([("deepseek", primary), ("gemini", backup)],
                           hedge_after_s=1.0, timeout_s=3.0)
    assert _run(fo) == TOASTER


# --- hedging: a slow primary does not delay the word --------------------------

def test_slow_primary_is_hedged_and_the_fast_backup_wins():
    primary, backup = Fixed(TOASTER, delay=5.0), Fixed(OVEN, delay=0.01)
    fo = FailoverPredictor([("deepseek", primary), ("gemini", backup)],
                           hedge_after_s=0.05, timeout_s=3.0)
    loop = asyncio.new_event_loop()
    try:
        t0 = loop.time()
        out = loop.run_until_complete(fo.predict([], "bread in the"))
        elapsed = loop.time() - t0
    finally:
        loop.close()
    assert out == OVEN
    assert elapsed < 1.0          # did not wait for the 5 s primary
    assert primary.cancelled == 1  # the loser was cancelled, not leaked


def test_primary_that_answers_before_the_hedge_wins_without_hedging():
    primary, backup = Fixed(TOASTER, delay=0.01), Fixed(OVEN)
    fo = FailoverPredictor([("deepseek", primary), ("gemini", backup)],
                           hedge_after_s=0.5, timeout_s=3.0)
    assert _run(fo) == TOASTER
    assert backup.calls == []


# --- exhaustion: every provider empty/slow -> [] within the budget -----------

def test_all_empty_returns_empty():
    fo = FailoverPredictor([("a", Fixed([])), ("b", Fixed([]))],
                           hedge_after_s=1.0, timeout_s=3.0)
    assert _run(fo) == []
    assert fo.last_served_by is None


def test_everything_slow_returns_empty_at_the_deadline_not_never():
    fo = FailoverPredictor([("a", Fixed(TOASTER, delay=5.0)), ("b", Fixed(OVEN, delay=5.0))],
                           hedge_after_s=0.02, timeout_s=0.1)
    loop = asyncio.new_event_loop()
    try:
        t0 = loop.time()
        out = loop.run_until_complete(fo.predict([], "x"))
        elapsed = loop.time() - t0
    finally:
        loop.close()
    assert out == []
    assert elapsed < 1.0


# --- the optional kwargs reach every provider unchanged ------------------------

def test_kwargs_are_forwarded_to_every_provider():
    primary, backup = Fixed([]), Fixed(TOASTER)
    fo = FailoverPredictor([("a", primary), ("b", backup)], hedge_after_s=1.0, timeout_s=3.0)
    asyncio.run(fo.predict(["ctx"], "frag", excluded=["oven"], entities=["Maria"],
                           already_served=["kettle"]))
    for p in (primary, backup):
        assert p.calls[0] == dict(context=["ctx"], fragment="frag", excluded=["oven"],
                                  entities=["Maria"], already_served=["kettle"])


# --- factory wiring: PREDICTOR_FALLBACKS ---------------------------------------

def _settings(**overrides):
    base = dict(
        predictor_provider="mock",
        gemini_model="gemini-3.5-flash", gemini_api_key="dummy",
        claude_model="claude-haiku-4-5", claude_api_key="dummy",
        deepseek_model="deepseek-flash", deepseek_api_key="dummy",
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="mock", demo_fallback_timeout_s=1.5,
        stt_provider="mock", deepgram_api_key=None,
        max_candidates=3, pause_ms=1300, context_turns=6,
        acoustic_model="", acoustic_conf=0.7, prefetch=False, prefetch_every=3,
        entity_memory=False,
    )
    base.update(overrides)
    return Settings(**base)


def test_settings_default_has_no_fallbacks_and_factory_returns_the_bare_provider():
    s = _settings()
    assert s.predictor_fallbacks == ()
    assert type(get_predictor(s)).__name__ == "MockPredictor"


def test_factory_wraps_the_provider_in_a_failover_chain_when_fallbacks_are_set():
    s = _settings(predictor_provider="deepseek", predictor_fallbacks=("gemini", "mock"))
    p = get_predictor(s)
    assert isinstance(p, FailoverPredictor)
    assert [name for name, _ in p.chain] == ["deepseek", "gemini", "mock"]
    assert [type(q).__name__ for _, q in p.chain] == \
        ["DeepSeekPredictor", "GeminiPredictor", "MockPredictor"]


def test_factory_drops_a_fallback_that_repeats_the_primary():
    s = _settings(predictor_provider="mock", predictor_fallbacks=("mock",))
    assert type(get_predictor(s)).__name__ == "MockPredictor"


def test_env_parses_a_comma_list(monkeypatch):
    from backend.config import get_settings
    monkeypatch.setenv("PREDICTOR_PROVIDER", "deepseek")
    monkeypatch.setenv("PREDICTOR_FALLBACKS", " Gemini, mock ,, ")
    get_settings.cache_clear() if hasattr(get_settings, "cache_clear") else None
    s = get_settings()
    assert s.predictor_fallbacks == ("gemini", "mock")
    s2 = replace(s, predictor_fallbacks=())
    assert s2.predictor_fallbacks == ()
