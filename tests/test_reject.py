"""Reject path: re-predict for the last served stall with rejected words excluded."""
import asyncio

from backend.pipeline import EchoPipeline
from backend.predictor.base import WordPredictor
from backend.predictor.mock import MockPredictor
from backend.schemas import Candidate, SilenceTick, TurnEnd, Word
from backend.stall_detector import StallDetector

TABLE = {"the": ["toaster", "oven", "grill"]}


class SpyPredictor(MockPredictor):
    """MockPredictor that records the excluded list of every call."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.excluded_seen: list[list[str]] = []

    async def predict(self, context, fragment, excluded=None):
        self.excluded_seen.append(list(excluded) if excluded else [])
        return await super().predict(context, fragment, excluded)


class SlowPredictor(WordPredictor):
    async def predict(self, context, fragment, excluded=None):
        await asyncio.sleep(10)
        return [Candidate("never", 1.0)]


class BoomPredictor(WordPredictor):
    async def predict(self, context, fragment, excluded=None):
        raise RuntimeError("boom")


async def _stall(pipe, base=0):
    """Speak 'I need the' then 'um' -> filler stall, fragment 'I need the um'."""
    t = base
    for w in ["I", "need", "the"]:
        await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
        t += 400
    return await pipe.handle(Word(text="um", start_ms=t, end_ms=t + 280))


def test_reject_happy_path_served_like_a_prediction():
    seen = []

    async def cb(pred):
        seen.append(pred)

    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p, on_prediction=cb)
        first = await _stall(pipe)
        rep = await pipe.reject(["toaster"])
        return p, pipe, first, rep

    p, pipe, first, rep = asyncio.run(run())
    assert first is not None and first.candidates[0].word == "toaster"
    assert rep is not None and rep.served == "reject"
    assert rep.candidates[0].word == "oven"
    assert rep.fragment == first.fragment and rep.trigger == first.trigger
    # routed exactly like a normal prediction: recorded AND fanned out, so the
    # session broadcast updates every connected browser uniformly
    assert pipe.predictions[-1] is rep
    assert len(seen) == 2 and seen[-1] is rep
    assert p.excluded_seen[-1] == ["toaster"]


def test_reject_exclusions_accumulate_within_one_stall():
    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        await _stall(pipe)
        r1 = await pipe.reject(["toaster"])
        r2 = await pipe.reject(["oven"])
        return p, r1, r2

    p, r1, r2 = asyncio.run(run())
    assert r1.candidates[0].word == "oven"
    assert r2.candidates[0].word == "grill"
    assert p.excluded_seen[-1] == ["toaster", "oven"]  # union of both rejects


def test_reject_without_active_stall_is_noop():
    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        return p, await pipe.reject(["toaster"])

    p, rep = asyncio.run(run())
    assert rep is None
    assert p.excluded_seen == []  # predictor never called


def test_reject_timeout_degrades_without_raising():
    async def run():
        pipe = EchoPipeline(
            StallDetector(pause_ms=1300), SpyPredictor(table=TABLE),
            predict_timeout=0.05,
        )
        await _stall(pipe)
        pipe.predictor = SlowPredictor()  # the reject re-call hangs
        return await pipe.reject(["toaster"])

    assert asyncio.run(run()) is None


def test_reject_exception_degrades_and_keeps_exclusions_for_retry():
    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        await _stall(pipe)
        pipe.predictor = BoomPredictor()
        failed = await pipe.reject(["toaster"])
        pipe.predictor = p  # provider recovers; the retry must still exclude
        retried = await pipe.reject(["oven"])
        return p, failed, retried

    p, failed, retried = asyncio.run(run())
    assert failed is None
    assert retried is not None and retried.candidates[0].word == "grill"
    assert p.excluded_seen[-1] == ["toaster", "oven"]


def test_rejections_cleared_on_new_stall():
    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        await _stall(pipe)
        await pipe.reject(["toaster"])
        # recovery word re-arms the detector; a fresh filler = a NEW stall
        await pipe.handle(Word(text="big", start_ms=2000, end_ms=2280))
        second = await pipe.handle(Word(text="um", start_ms=2400, end_ms=2680))
        assert second is not None
        rep = await pipe.reject(["grill"])
        return p, rep

    p, rep = asyncio.run(run())
    assert rep is not None
    assert p.excluded_seen[-1] == ["grill"]  # 'toaster' did not leak across stalls


def test_rejections_cleared_on_turn_end():
    async def run():
        p = SpyPredictor(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        await _stall(pipe)
        await pipe.reject(["toaster"])
        await pipe.handle(TurnEnd())
        return await pipe.reject(["oven"])  # no stall to re-serve anymore

    assert asyncio.run(run()) is None


def test_concurrent_rejects_serialized():
    class SlowSpy(SpyPredictor):
        async def predict(self, context, fragment, excluded=None):
            self.excluded_seen.append(list(excluded) if excluded else [])
            await asyncio.sleep(0.05)
            return [Candidate("oven", 0.8)]

    async def run():
        p = SlowSpy(table=TABLE)
        pipe = EchoPipeline(StallDetector(pause_ms=1300), p)
        await _stall(pipe)
        n_after_stall = len(p.excluded_seen)
        # rapid double press: the second reject must no-op, not double-call
        r1, r2 = await asyncio.gather(
            pipe.reject(["toaster"]), pipe.reject(["toaster"])
        )
        return p, n_after_stall, r1, r2

    p, n_after_stall, r1, r2 = asyncio.run(run())
    assert (r1 is None) != (r2 is None)  # exactly one served, one dropped
    assert len(p.excluded_seen) == n_after_stall + 1  # one predictor call


def test_reject_does_not_corrupt_prefetch_state():
    class CountingPredictor(MockPredictor):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls: list[str] = []

        async def predict(self, context, fragment, excluded=None):
            self.calls.append(fragment)
            return [c for c in [Candidate("toaster", 0.9), Candidate("oven", 0.8)]
                    if not excluded or c.word not in excluded]

    async def run():
        p = CountingPredictor()
        pipe = EchoPipeline(
            StallDetector(pause_ms=1300), p, prefetch=True, prefetch_every=3,
        )
        t = 0
        for w in ["I", "need", "the"]:
            await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 280))
            t += 400
        await asyncio.sleep(0.01)  # shadow completes -> cache warm
        pred = await pipe.handle(SilenceTick(at_ms=t + 1500))
        assert pred.served == "prefetch"
        seq = pipe._shadow_seq
        baseline = pipe._last_shadow_count

        rep = await pipe.reject(["toaster"])
        assert rep is not None and rep.candidates[0].word == "oven"
        # reject must not write the cache, bump the shadow sequence, orphan an
        # in-flight shadow, or move the one-attempt-per-position baseline
        assert pipe._cache.candidates is None
        assert pipe._shadow_seq == seq
        assert pipe._shadow_inflight is False
        assert pipe._last_shadow_count == baseline

        # prefetch keeps working: next content word re-warms the cold cache
        n = len(p.calls)
        await pipe.handle(Word(text="big", start_ms=t + 2000, end_ms=t + 2280))
        await asyncio.sleep(0.01)
        assert len(p.calls) == n + 1
        assert pipe._cache.candidates is not None
        return True

    assert asyncio.run(run())
