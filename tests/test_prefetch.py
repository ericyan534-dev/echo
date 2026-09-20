"""Speculative prefetch: shadow prediction, cache serving, drift, staleness.

NO WALL-CLOCK BARRIERS. Every "wait for the shadow" point here is either
`settle()` (drive the event loop until nothing is pending) or an explicit
`hold`/`release()` gate on the predictor. The earlier version slept for real
milliseconds and asserted `pred.latency_ms < 50`, so a loaded machine turned a
scheduling hiccup into what read like a logic bug. Timing is now a state we
set, not a quantity we measure -- and the "no LLM round-trip" claim is checked
by the code path taken (served=="prefetch", predictor not called), which is
what the property actually is.
"""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.mock import MockPredictor
from backend.schemas import Candidate, SilenceTick, TurnEnd, Word
from backend.stall_detector import StallDetector


class CountingPredictor(MockPredictor):
    """Deterministic predictor with EXPLICIT completion control.

    A call made while `hold_all` is set parks on an asyncio.Event until
    `release()`; calls made after `hold_all` is cleared return immediately.
    That makes "a shadow is still in flight" a fact the test establishes,
    rather than a race it hopes to win by sleeping.
    """

    def __init__(self, *a, hold_all: bool = False, **kw):
        super().__init__(*a, **kw)
        self.calls: list[str] = []
        self.hold_all = hold_all
        self._gate = asyncio.Event()

    def release(self) -> None:
        self._gate.set()

    async def predict(self, context, fragment):
        self.calls.append(fragment)
        if self.hold_all:
            await self._gate.wait()
        return [Candidate(word="toaster", confidence=0.9)]


async def spin(n: int = 3) -> None:
    """Yield to the event loop n times: enough for a just-created task to
    start and reach its first await. A zero-delay yield, not a sleep."""
    for _ in range(n):
        await asyncio.sleep(0)


async def settle(max_rounds: int = 40) -> None:
    """Run every pending task to completion. Bounded, so a regression that
    chase-relaunches shadows forever terminates (and then fails on the call
    count) instead of hanging the suite. Must not be called while a predictor
    call is held."""
    for _ in range(max_rounds):
        pending = {t for t in asyncio.all_tasks()
                   if t is not asyncio.current_task() and not t.done()}
        if not pending:
            return
        await asyncio.wait(pending)


def make_pipe(predictor, prefetch=True, every=3):
    return EchoPipeline(
        StallDetector(pause_ms=1300), predictor,
        prefetch=prefetch, prefetch_every=every,
    )


async def speak(pipe, words, base=0):
    t = base
    for w in words:
        await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
        t += 400
    await spin()  # let any shadow task created above actually start
    return t


def test_cold_cache_shadows_on_first_word_then_every_n():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, every=3)
        await speak(pipe, ["I"])
        await settle()
        first = list(p.calls)
        await speak(pipe, ["really", "need", "the"], base=400)
        await settle()
        return first, p.calls

    first, calls = asyncio.run(run())
    assert first == ["I"]  # cold cache: shadow on the very first content word
    assert calls == ["I", "I really need the"]  # warm cache: refresh every 3


def test_cache_rewarms_immediately_after_served_stall():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, every=3)
        t = await speak(pipe, ["I", "really", "need"])
        await settle()                                 # first shadow cached
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))  # consumes cache
        n_after_stall = len(p.calls)
        await speak(pipe, ["big"], base=t + 2000)      # 1st word after stall
        await settle()
        return pred, n_after_stall, p.calls

    pred, n_after_stall, calls = asyncio.run(run())
    assert pred is not None and pred.served == "prefetch"
    # cache went cold when the stall consumed it -> next content word must
    # relaunch a shadow immediately, not wait out the every-3 cadence
    assert len(calls) == n_after_stall + 1


def test_stall_served_from_prefetch_cache():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, every=3)
        t = await speak(pipe, ["I", "really", "need", "the"])
        await settle()  # shadow completes
        n_before_stall = len(p.calls)
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))
        return pred, n_before_stall, p.calls

    pred, n_before_stall, calls = asyncio.run(run())
    assert pred is not None
    assert pred.served == "prefetch"
    assert pred.candidates[0].word == "toaster"
    # THE claim: no LLM round-trip at stall time. Asserted as the code path
    # taken -- the predictor was not called while serving the stall -- rather
    # than as elapsed wall-clock, which measured the machine, not the code.
    assert len(calls) == n_before_stall, "a live predictor call was made despite a warm cache"
    # two shadows: cold-cache warm-up on "I", cadence refresh at word 4
    assert len(calls) == 2


def test_drift_beyond_limit_falls_back_to_live():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, every=3)
        await speak(pipe, ["I", "really", "need"])
        await settle()                                   # cache warm
        cached_count = pipe._cache.content_count
        # Speak past the drift limit while every shadow is HELD, so the cache
        # provably cannot refresh. The live stall call is made after hold_all
        # is cleared, so it returns normally -- no timeouts, no sleeps.
        p.hold_all = True
        t = await speak(pipe, ["a", "shiny", "new", "red"], base=2000)
        assert pipe._cache.content_count == cached_count  # still the stale entry
        p.hold_all = False                                # only the live call runs
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))
        p.release()
        await settle()
        return pred, cached_count

    pred, cached_count = asyncio.run(run())
    assert pred is not None
    assert pred.served == "live"  # drift > 2 content words past the cache


def test_turn_end_clears_cache():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, every=3)
        await speak(pipe, ["I", "really", "need"])
        await settle()
        assert pipe._cache.candidates is not None        # turn 1 cache is warm
        await pipe.handle(TurnEnd())
        assert pipe._cache.candidates is None            # cleared at the boundary
        # hold the new turn's shadow so "is the cache warm?" has one answer
        p.hold_all = True
        t = await speak(pipe, ["pass", "me", "the"], base=5000)
        p.hold_all = False
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))
        p.release()
        await settle()
        return pred

    pred = asyncio.run(run())
    assert pred is not None
    assert pred.fragment == "pass me the"
    # the previous turn's cache must not be reachable from this turn
    assert pred.served == "live"


class FailingPredictor(CountingPredictor):
    async def predict(self, context, fragment):
        self.calls.append(fragment)
        raise RuntimeError("boom")


class EmptyPredictor(CountingPredictor):
    async def predict(self, context, fragment):
        self.calls.append(fragment)
        return []


def test_failing_shadow_does_not_busy_loop():
    """Regression: with a cold cache, a shadow that raises (or returns no
    candidates) must NOT be chase-relaunched at the same content position --
    before the guard, this busy-looped at wire speed (~200k calls/s) against
    a failing live predictor. Bound: one attempt per new content word.

    settle() drains every relaunch the loop is willing to make (bounded, so a
    true busy loop terminates the test instead of hanging it), which is a
    stricter check than the old fixed 50 ms sleep.
    """
    async def run(pred_cls):
        p = pred_cls()
        pipe = make_pipe(p, every=3)
        await speak(pipe, ["I"])
        await settle()
        one_word = len(p.calls)
        await speak(pipe, ["really"], base=400)
        await settle()
        return one_word, len(p.calls)

    for cls in (FailingPredictor, EmptyPredictor):
        one_word, two_words = asyncio.run(run(cls))
        assert one_word == 1, f"{cls.__name__}: relaunched at same position"
        assert two_words == 2, f"{cls.__name__}: retry not bounded per word"


def test_prefetch_off_never_shadows():
    async def run():
        p = CountingPredictor()
        pipe = make_pipe(p, prefetch=False)
        t = await speak(pipe, ["I", "really", "need", "the", "big"])
        await settle()
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))
        return pred, p.calls

    pred, calls = asyncio.run(run())
    assert pred.served == "live"
    assert len(calls) == 1  # only the stall-time call


def test_inflight_shadow_does_not_refill_cache_after_turn_end():
    """Regression: a shadow launched during turn N must not repopulate the
    cache *after* turn N ended -- its fragment is stale. Before the fix, the
    turn-end cleared the cache but left the in-flight shadow's sequence intact,
    so its completion wrote the previous turn's word back into the cache."""
    async def run():
        p = CountingPredictor(hold_all=True)     # shadow parks, provably in flight
        pipe = make_pipe(p, every=3)
        await speak(pipe, ["I", "really", "need"])  # launches shadow A
        assert p.calls == ["I"], "shadow A never started"
        assert pipe._cache.candidates is None, "A completed early; the test is not testing"
        await pipe.handle(TurnEnd())                 # clears; A is orphaned
        p.release()                                  # A lands AFTER the turn ended
        await settle()
        return pipe._cache.candidates

    cached = asyncio.run(run())
    assert cached is None     # the orphaned shadow must not refill a cleared cache


def test_stale_shadow_completion_dropped():
    async def run():
        p = CountingPredictor(hold_all=True)
        pipe = make_pipe(p, every=2)
        await speak(pipe, ["one", "two"])            # shadow A, held in flight
        assert p.calls == ["one"], "shadow A never started"
        # bump the sequence so A's completion is stale when it lands
        pipe._shadow_seq += 1
        pipe._shadow_inflight = False
        p.release()                                   # A completes, must be dropped
        await settle()
        return pipe._cache.candidates

    cached = asyncio.run(run())
    assert cached is None
