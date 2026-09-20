"""Provider failover -- a hedged race across a chain of REAL predictors.

Why this exists (demo rehearsal, 2026-09-20): the live provider was bare
DeepSeek, which for 30+ consecutive stalls answered HTTP 503 or blew its 3 s
timeout. Every cloud provider here "never raises" (see deepseek.py) -- a
failure is returned as an EMPTY candidate list -- so each of those stalls was
detected correctly and then produced no word at all. Nothing in the stack
noticed: demo_fallback.py was not active, serves a canned four-fragment
table rather than a model, and by design ignores an empty-but-successful
result -- which is exactly what a swallowed 503 looks like.

This wrapper fixes it at the seam, not the symptom:

  * A provider that fails, raises, or returns [] is "ask the next one",
    immediately. Falling through to ANOTHER model answering the REAL fragment
    is not a canned answer, so the strictness rule that keeps demo_fallback
    honest does not apply here.
  * A provider that is merely SLOW is hedged: after `hedge_after_s` the next
    provider is started as well, and the first non-empty answer wins. The
    loser is cancelled, never leaked. In the common case the primary answers
    inside the hedge window and the fallback is never even called.
  * The whole race is bounded by `timeout_s` so it composes under the
    pipeline's own predict timeout; at the deadline it returns [] rather than
    hanging the stall->word loop.

Every fallback serve logs one ASCII line naming the provider that answered
(presenter-visible, same disclosure rule as demo_fallback.py), and
`last_served_by` records it for /healthz-style inspection.
"""
from __future__ import annotations

import asyncio
import logging

from ..schemas import Candidate
from .base import WordPredictor

log = logging.getLogger("echo.predictor.failover")


class FailoverPredictor(WordPredictor):
    def __init__(
        self,
        chain: list[tuple[str, WordPredictor]],
        hedge_after_s: float = 1.2,
        timeout_s: float = 3.5,
    ) -> None:
        if not chain:
            raise ValueError("FailoverPredictor needs at least one provider.")
        self.chain = list(chain)
        self.hedge_after_s = hedge_after_s
        self.timeout_s = timeout_s
        self.last_served_by: str | None = None

    async def _call(self, name: str, predictor: WordPredictor, context, fragment,
                    kwargs) -> list[Candidate]:
        """One provider, normalized: any failure is [] plus a log line. A
        CancelledError is the race cancelling a loser (or the server shutting
        down) and is never swallowed."""
        try:
            return list(await predictor.predict(context, fragment, **kwargs) or [])
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a provider failure is routine here
            log.warning("[failover] %s failed (%s: %s)", name, type(exc).__name__, exc)
            return []

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        kwargs = dict(excluded=excluded, entities=entities, already_served=already_served)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_s
        pending: dict[asyncio.Task, str] = {}
        next_idx = 0
        next_hedge = loop.time()  # set for real by the first launch()
        self.last_served_by = None

        def launch() -> None:
            nonlocal next_idx, next_hedge
            name, predictor = self.chain[next_idx]
            next_idx += 1
            task = asyncio.ensure_future(
                self._call(name, predictor, context, fragment, kwargs))
            pending[task] = name
            next_hedge = loop.time() + self.hedge_after_s

        launch()
        try:
            while pending:
                now = loop.time()
                if now >= deadline:
                    break
                more = next_idx < len(self.chain)
                wait_until = min(deadline, next_hedge) if more else deadline
                done, _ = await asyncio.wait(
                    set(pending), timeout=max(0.0, wait_until - now),
                    return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    name = pending.pop(task)
                    result = task.result()  # _call never raises
                    if result:
                        self.last_served_by = name
                        if name != self.chain[0][0]:
                            log.warning("[failover] served by %s (%s gave nothing) for fragment: %r",
                                        name, self.chain[0][0], fragment)
                        return result
                    # empty or failed: ask the next provider right away
                    if next_idx < len(self.chain):
                        launch()
                if not done and next_idx < len(self.chain) and loop.time() >= next_hedge:
                    log.warning("[failover] %s slow (> %.1fs); hedging with %s",
                                pending and next(iter(pending.values())) or self.chain[0][0],
                                self.hedge_after_s, self.chain[next_idx][0])
                    launch()
            if pending:
                log.warning("[failover] no provider answered within %.1fs for fragment: %r",
                            self.timeout_s, fragment)
            return []
        finally:
            # Cancel the losers AND wait for the cancellation to land, so no
            # half-finished HTTP call outlives the stall it was raced for.
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
