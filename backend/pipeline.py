"""EchoPipeline — wires STT stream -> stall detector -> predictor -> output.

v2 adds two input/serving paths:
  * AcousticEvent items (from the raw-audio channel) flow into the detector's
    observe_acoustic — fillers/prolongations the transcript never shows.
  * Speculative prefetch: while the speaker is fluent, the pipeline shadow-
    predicts — immediately on the first content word whenever the cache is
    cold (turn start, or right after a stall consumed it), then every
    `prefetch_every` new content words to keep it fresh; when a stall fires, a
    cached prediction is served instantly (served="prefetch") instead of
    waiting ~1.5s for the LLM round-trip. The cache is only used if the spoken
    fragment hasn't drifted more than `_PREFETCH_DRIFT` words past it.

Pure orchestration, independent of FastAPI/websockets, fully unit-testable.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Awaitable, Callable

from .context import ContextBuilder
from .entities import EntityTracker
from .predictor.base import WordPredictor
from .schemas import AcousticEvent, Candidate, Prediction, SilenceTick, StallEvent, TurnEnd, Word
from .stall_detector import StallDetector
from .transcript import Conversation

OnPrediction = Callable[[Prediction], Awaitable[None]]

_PREFETCH_DRIFT = 2  # cached fragment may trail the live one by at most N content words
_PREDICT_TIMEOUT_S = 4.0  # max seconds to wait for a live LLM call before degrading


class _PrefetchCache:
    def __init__(self) -> None:
        self.fragment = ""
        self.content_count = -1
        self.candidates: list[Candidate] | None = None

    def clear(self) -> None:
        self.fragment = ""
        self.content_count = -1
        self.candidates = None


class EchoPipeline:
    def __init__(
        self,
        detector: StallDetector,
        predictor: WordPredictor,
        conversation: Conversation | None = None,
        context_turns: int = 6,
        on_prediction: OnPrediction | None = None,
        prefetch: bool = False,
        prefetch_every: int = 3,
        predict_timeout: float = _PREDICT_TIMEOUT_S,
        entity_memory: bool = False,
        context_builder: "ContextBuilder | None" = None,
    ) -> None:
        self.detector = detector
        self.predictor = predictor
        self.conversation = conversation or Conversation()
        self.context_turns = context_turns

        # Long-horizon memory. Absent a builder the pipeline keeps its previous
        # behaviour exactly (raw recent-turns tail), so this is purely additive
        # for existing callers.
        self.context_builder = context_builder
        self.rolling_summary = ""
        self.on_prediction = on_prediction
        self.predictions: list[Prediction] = []  # captured for inspection/tests
        self._predict_timeout = predict_timeout

        # Salient-entity memory (ENTITY_MEMORY): when enabled, injects names/
        # things mentioned earlier in the conversation but already outside
        # the `context_turns` window into the predictor prompt. The tracker
        # itself is cheap/stateless to construct, so it's always built; the
        # flag just gates whether it's consulted (see _entities_for_prompt).
        self.entity_memory = entity_memory
        self._entity_tracker = EntityTracker()

        self.prefetch_enabled = prefetch
        self.prefetch_every = max(1, prefetch_every)
        self._cache = _PrefetchCache()
        self._shadow_seq = 0          # stale-completion guard (bumped on invalidate)
        self._shadow_inflight = False
        # Strong refs to in-flight shadow tasks; the loop keeps only weak ones.
        self._shadow_tasks: set = set()
        self._last_shadow_count = 0   # content count at last shadow launch

        # Reject path: the last served stall (so a reject can re-predict for
        # the same fragment) + words the speaker rejected during that stall.
        self._last_stall_fragment: str | None = None
        self._last_stall_trigger = ""
        self._rejected_this_stall: list[str] = []
        self._reject_inflight = False  # serialize rapid double-rejects

    async def run(self, stream: AsyncIterator) -> None:
        async for item in stream:
            await self.handle(item)

    async def handle(self, item: Word | SilenceTick | AcousticEvent | TurnEnd) -> Prediction | None:
        event: StallEvent | None = None
        if isinstance(item, Word):
            event = self.detector.observe_word(item)
            if event is None:
                self._maybe_shadow_predict()
        elif isinstance(item, SilenceTick):
            # speech_end_ms carries WHEN THE SPEAKER STOPPED, which is not the
            # end of the last committed word: the streaming ASR holds the
            # utterance tail behind LocalAgreement. Passing it through is what
            # keeps the pause trigger measuring the wearer instead of the
            # commit lag. None (browser/mock timer) leaves the old rule.
            event = self.detector.observe_silence(item.at_ms, item.speech_end_ms)
        elif isinstance(item, AcousticEvent):
            event = self.detector.observe_acoustic(item)
        elif isinstance(item, TurnEnd):
            # Record what the speaker ACTUALLY said this turn. This used to read
            # detector.fragment, which was clause-windowed -- so any turn
            # containing a stall was stored truncated, losing exactly the content
            # worth remembering (spec section 1.1). full_turn_text() names that
            # intent explicitly so the two cannot silently diverge again.
            text = self.detector.full_turn_text() or self.detector.pop_last_turn()
            self.conversation.add_turn(text)
            await self._fold_scrolled_turns()
            self.detector.reset()
            self.detector.pop_last_turn()  # clear any stale stashed sentence
            self._invalidate_shadow()  # an in-flight shadow is for the old turn
            self._last_stall_fragment = None  # a reject can't target the old turn
            self._rejected_this_stall = []
            return None

        if event is None:
            return None
        return await self._predict(event)

    # --- long-horizon memory ---------------------------------------------
    async def _fold_scrolled_turns(self) -> None:
        """Fold turns that have scrolled out of the verbatim window into the
        rolling summary.

        The default summarizer is extractive, so this is cheap and adds no
        latency. An LLM-backed summarizer must NOT be called from here on the
        stall path -- schedule it during fluent speech, like the prefetch
        shadow, or it will add seconds at exactly the wrong moment.
        """
        if self.context_builder is None:
            return
        keep = self.context_builder.verbatim_turns
        turns = self.conversation.turns
        scrolled = turns[:-keep] if keep > 0 else list(turns)
        if not scrolled:
            return
        self.rolling_summary = await self.context_builder.summarizer.fold(
            scrolled, self.rolling_summary)

    async def handle_text_turn(self, text: str) -> None:
        """Record a complete utterance as conversation context (eval/test
        convenience -- the live path builds turns word by word via TurnEnd)."""
        self.conversation.add_turn(text)
        await self._fold_scrolled_turns()

    # --- speculative prefetch -------------------------------------------
    def _invalidate_shadow(self) -> None:
        """Drop the cache AND orphan any in-flight shadow so a completion that
        was launched for a now-consumed fragment/turn cannot repopulate the
        cache after it was cleared. Bumping the sequence makes the running
        task fail its seq guard (no cache write, no chase-relaunch); clearing
        the inflight flag lets a fresh shadow launch on the next content word.
        Re-baselines the shadow cadence to the current content position."""
        self._shadow_seq += 1
        self._shadow_inflight = False
        self._cache.clear()
        self._last_shadow_count = self.detector._content_count

    def _maybe_shadow_predict(self) -> None:
        if not self.prefetch_enabled or self._shadow_inflight:
            return
        count = self.detector._content_count
        if count < 1:
            return
        if count < self._last_shadow_count:
            # The detector's clause windowing reset its content count (post-
            # stall re-arm / punctuation): re-baseline so the new clause's
            # first word can re-warm the cold cache.
            self._last_shadow_count = 0
        # Never retry the same content position: one shadow attempt per new
        # content word, regardless of whether the last attempt succeeded,
        # raised, or returned no candidates. Without this, the chase-relaunch
        # in _shadow_predict's finally block busy-loops at wire speed against
        # a failing/empty predictor (cache stays None -> relaunch same spot).
        if count == self._last_shadow_count:
            return
        # Cold cache (turn start, or a stall just consumed it): shadow on the
        # very next content word, so the first stall of a turn is a cache hit
        # instead of a full LLM round-trip (the measured 2.5-3.3 s worst-case
        # dead air). Warm cache: refresh every `prefetch_every` content words.
        if (self._cache.candidates is not None
                and count - self._last_shadow_count < self.prefetch_every):
            return
        fragment = self.detector.fragment
        if not fragment:
            return
        self._last_shadow_count = count
        self._shadow_seq += 1
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop (sync tests without prefetch assertions) — skip
        self._shadow_inflight = True
        # Hold a strong reference until it finishes. The event loop keeps only
        # a WEAK one, so a bare create_task can be collected mid-flight -- the
        # prefetch would silently never land and the stall would fall back to a
        # live call. app.py does the same thing for the prewarm task.
        task = asyncio.create_task(
            self._shadow_predict(self._shadow_seq, fragment, count))
        self._shadow_tasks.add(task)
        task.add_done_callback(self._shadow_tasks.discard)

    async def _shadow_predict(self, seq: int, fragment: str, count: int) -> None:
        try:
            candidates = await self.predictor.predict(
                self.conversation.recent(self.context_turns), fragment,
                **self._entities_kwargs(),
            )
            if seq == self._shadow_seq and candidates:  # drop stale completions
                self._cache.fragment = fragment
                self._cache.content_count = count
                self._cache.candidates = candidates
        except Exception:
            pass  # shadow failures are silent; the live path still works
        finally:
            if seq == self._shadow_seq:
                self._shadow_inflight = False
                # Chase: if the speaker advanced past this shadow while it was
                # in flight (fast talker / word burst), immediately refresh so
                # the cache is current when the stall eventually comes.
                self._maybe_shadow_predict()

    def _entities_kwargs(self) -> dict:
        """kwargs dict for the `entities=` predictor.predict argument: empty
        when entity memory is off or there's nothing out-of-window yet, else
        {"entities": [...]}. Returning an empty dict (rather than always
        passing entities=None) keeps every existing 2-arg predictor.predict
        call site byte-identical when the feature is off/inactive -- the
        same backward-compatible-extension approach `excluded` uses. Shared
        by both the live predict path and the shadow prefetch path so the
        prefetch cache benefits from the same hint the live call would get."""
        if not self.entity_memory:
            return {}
        entities = self._entity_tracker.out_of_window(self.conversation.turns, self.context_turns)
        # Never hint a word the speaker rejected this stall: an entity line
        # saying "mentioned earlier: X" alongside "do NOT suggest: X" is a
        # contradictory prompt, and for the mock/lookup paths the raw entity
        # list would literally re-serve the rejected word.
        if self._rejected_this_stall:
            lowered = {w.lower() for w in self._rejected_this_stall}
            entities = [e for e in entities if e.lower() not in lowered]
        return {"entities": entities} if entities else {}

    def _served_kwargs(self, event: StallEvent) -> dict:
        """Same empty-dict-when-absent pattern as _entities_kwargs, so every
        existing predictor call site stays byte-identical when unused."""
        return {"already_served": list(event.already_served)} if event.already_served else {}

    def _cache_usable(self, fragment: str) -> bool:
        c = self._cache
        if not c.candidates:
            return False
        if not fragment.startswith(c.fragment):
            return False
        return self.detector._content_count - c.content_count <= _PREFETCH_DRIFT

    # --- serving ----------------------------------------------------------
    async def _predict(self, event: StallEvent) -> Prediction:
        # New stall: it becomes the reject target, with a fresh rejected list.
        self._last_stall_fragment = event.fragment
        self._last_stall_trigger = event.trigger
        self._rejected_this_stall = []
        t0 = time.perf_counter()
        if self.prefetch_enabled and self._cache_usable(event.fragment):
            pred = Prediction(
                candidates=list(self._cache.candidates or []),
                fragment=event.fragment,
                trigger=event.trigger,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                served="prefetch",
            )
        else:
            try:
                candidates = await asyncio.wait_for(
                    self.predictor.predict(
                        self.conversation.recent(self.context_turns), event.fragment,
                        **self._entities_kwargs(),
                        **self._served_kwargs(event),
                    ),
                    timeout=self._predict_timeout,
                )
                served = "live"
            except Exception:
                # A failed live call must NEVER kill the socket loop at the
                # stall moment. Degrade: serve the prefetch cache even if it
                # drifted (stale word > no word > crash), else empty.
                candidates = list(self._cache.candidates or [])
                served = "prefetch-stale" if candidates else "live"
            pred = Prediction(
                candidates=candidates,
                fragment=event.fragment,
                trigger=event.trigger,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                served=served,
            )
        self._invalidate_shadow()  # a stall consumes the cache; orphan any
        # in-flight shadow so it can't refill it, then refresh on next words
        # Tell the detector what went out, so a later stall in this same turn
        # doesn't loop back to the same word.
        if pred.candidates:
            self.detector.record_served(pred.candidates[0].word)
        self.predictions.append(pred)
        if self.on_prediction is not None:
            await self.on_prediction(pred)
        return pred

    # --- reject path --------------------------------------------------------
    async def reject(self, rejected: list[str]) -> Prediction | None:
        """The speaker rejected served word(s): re-predict for the last served
        stall with everything rejected this stall excluded.

        Never raises: no active stall, a concurrent reject in flight, or a
        failed/timed-out predictor call all return None (the UI keeps its
        locally promoted candidate). Deliberately does NOT touch the detector
        or any prefetch state (_cache / _shadow_*): a reject is a re-serve of
        the same stall, not a new speech event.
        """
        if self._last_stall_fragment is None or self._reject_inflight:
            return None
        self._reject_inflight = True
        try:
            for w in rejected:
                w = str(w).strip()
                if w and w not in self._rejected_this_stall:
                    self._rejected_this_stall.append(w)
            t0 = time.perf_counter()
            try:
                candidates = await asyncio.wait_for(
                    self.predictor.predict(
                        self.conversation.recent(self.context_turns),
                        self._last_stall_fragment,
                        excluded=list(self._rejected_this_stall),
                        # entity hints must survive a reject, or rejecting an
                        # entity-served word (Beat 3's name recall) re-predicts
                        # blind and can blank the card; _entities_kwargs also
                        # filters out the just-rejected words themselves.
                        **self._entities_kwargs(),
                    ),
                    timeout=self._predict_timeout,
                )
            except Exception:
                return None  # rejected words stay recorded for the next try
            pred = Prediction(
                candidates=candidates,
                fragment=self._last_stall_fragment,
                trigger=self._last_stall_trigger,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                served="reject",
            )
            self.predictions.append(pred)
            if self.on_prediction is not None:
                await self.on_prediction(pred)
            return pred
        finally:
            self._reject_inflight = False
