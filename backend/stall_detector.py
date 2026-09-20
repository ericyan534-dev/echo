"""Stall detector -- decides WHEN the speaker is stuck searching for a word.

Five triggers: pause / filler / hedge / filler_acoustic / prolongation.

WHAT CHANGED IN v3
------------------
The detector no longer owns text. It reads a Timeline (append-only) and owns
only SearchEpisode state. Debounce asks 'is there an unresolved episode?'.

Previously one cursor (`_clause_start`) served as debounce state, as the
predictor's input, and as the conversation record -- so re-arming truncated
both consumers. Measured: post-suggestion stalls sent 'on some' to the model,
and a stalled turn was recorded as 'every day at home'. Abandoned attempts are
now ANNOTATED on the event (`StallEvent.abandoned`) rather than deleted, which
preserves the original intent (do not re-send an abandoned attempt as if it
were the live search) without destroying the sentence.

Pure Python, no audio deps -- fully deterministic and unit-testable.
"""
from __future__ import annotations

from .schemas import AcousticEvent, SearchEpisode, StallEvent, Word
from .timeline import FILLERS, Timeline, norm

# Suffix phrases that signal "talking around" a word. Longer phrases first.
# NOTE: bare "you know" is intentionally NOT here -- it's a common fluent
# discourse marker and firing on it is a demo-fatal false positive.
HEDGES = [
    "what do you call it",
    "what's it called",
    "what you call it",
    "whatchamacallit",
    "you know the",
    "the thingy",
    "the thing",
    "that thing",
    "this thing",
    "the whatsit",
]

# Content words past a fire before the episode resolves and the detector
# re-arms. Kept at 1 -- identical to the old _REARM_CONTENT -- so re-arm
# TIMING is unchanged by this rewrite; only the text handling changes.
REARM_CONTENT_WORDS = 1

# Most recent served words carried into the prompt as "already offered" hints.
MAX_SERVED_HINTS = 5


class StallDetector:
    def __init__(self, pause_ms: int = 1300, timeline: Timeline | None = None,
                 min_gap_ms: int = 0, scorer=None,
                 rearm_content_words: int = REARM_CONTENT_WORDS) -> None:
        self.pause_ms = pause_ms
        self.timeline = timeline or Timeline()
        # Minimum time between two fires, ACROSS ALL TRIGGERS. Defaults to 0
        # (off) so every existing test and published bench keeps its exact
        # behaviour; the app layer sets it from Settings.
        #
        # This exists because AcousticStream's refractory is per-KIND, and the
        # five-type StutterNet turned that into five independent 1.2 s
        # refractories that can interleave. Measured on real aphasic speech,
        # the stack fired 17-25 times per minute -- aphasic speech is dense
        # with dysfluency, and firing on each one is the "it nags" failure.
        # The constraint is a product fact, not a fitted parameter: a served
        # word takes ~1.5-2 s to arrive and the speaker needs time to read and
        # use it, so a second suggestion before that is noise by construction.
        self.min_gap_ms = min_gap_ms
        # Optional calibrated scorer (backend/stall_scorer.py). When present it
        # REPLACES the any-of trigger rule: filler, hedge and acoustic events
        # stop firing on their own and become features of one score instead.
        # Defaults to None, which leaves every existing test and published
        # bench on exactly the rule they were measured against.
        self.scorer = scorer
        # Content words the wearer must produce after a fire before the search
        # counts as recovered and the detector re-arms. At 1 -- the shipped
        # value -- a fire can be followed by another two words later, so a
        # dysfluency-dense utterance absorbs several suggestions while the next
        # utterance gets none. Since recall is per UTTERANCE, those extra fires
        # are spent without buying anything.
        self.rearm_content_words = rearm_content_words
        self.episodes: list[SearchEpisode] = []
        # When the detector last fired, in the SAME clock as `at_ms`. Held
        # separately from `episodes` because `reset()` drops the episode list
        # at a turn boundary and the refractory must survive it: min_gap_ms is
        # a wall-clock product constraint (a served word takes ~1.5-2 s to
        # arrive and be read), and a turn boundary does not give the wearer
        # that time back. Measured before this was split out: a pause stall at
        # 2500 ms, TurnEnd, then a filler stall at 3650 ms -- two suggestions
        # 1150 ms apart with min_gap_ms=4000.
        self._last_fire_ms: int | None = None
        self._content_at_fire = -1
        self._last_turn = ""
        self._pending: list[StallEvent] = []   # events raised during observe_*

    # --- introspection ---------------------------------------------------
    @property
    def fragment(self) -> str:
        """The FULL current utterance. Never windowed."""
        return self.timeline.utterance_text()

    @property
    def _content_count(self) -> int:
        return self.timeline.content_count()

    @property
    def _open_episode(self) -> SearchEpisode | None:
        return next((e for e in reversed(self.episodes) if not e.resolved), None)

    def full_turn_text(self) -> str:
        """What the conversation record should store for this turn: everything
        said, not the tail clause."""
        return self.timeline.utterance_text()

    def drain_events(self) -> list[StallEvent]:
        """Events raised since the last drain (test/eval convenience)."""
        out, self._pending = self._pending, []
        return out

    # --- internal --------------------------------------------------------
    def _resolve_if_recovered(self) -> None:
        ep = self._open_episode
        if ep is None:
            return
        if self._content_count - self._content_at_fire >= self.rearm_content_words:
            ep.resolved = True

    def _emit(self, trigger: str, at_ms: int) -> StallEvent | None:
        self._resolve_if_recovered()
        if self._open_episode is not None:
            return None            # debounce: a search is already in progress
        if self.min_gap_ms and self._last_fire_ms is not None:
            # A clock that has jumped BACKWARDS by more than one gap is a
            # restarted clock, not an early event: the browser stamps its
            # words and ticks in ms since *Start listening* and restarts at 0
            # on every Start and every reload, while this detector -- and the
            # last fire -- live for the whole process. Measured before this:
            # one fire at 63.9 s, then a reload, then every pause and filler
            # suppressed for the next 64 s, and indefinitely once an acoustic
            # fire on the audio clock (never restarted) had stamped the last
            # fire minutes ahead of the browser. Jitter between channels is a
            # few hundred ms and stays inside the gap, so it is unaffected.
            if at_ms < self._last_fire_ms - self.min_gap_ms:
                self._last_fire_ms = None
            # Measured against the LAST fire, not the last unresolved one: the
            # point is how often the wearer is interrupted, and a resolved
            # episode interrupted them just as much. Across turns too -- see
            # `_last_fire_ms`.
            elif at_ms - self._last_fire_ms < self.min_gap_ms:
                return None
        fragment = self.fragment
        self.episodes.append(SearchEpisode(
            started_at_ms=at_ms, trigger=trigger, fragment_at_fire=fragment))
        self._last_fire_ms = at_ms
        self._content_at_fire = self._content_count
        self.timeline.add_event("stall", at_ms, {"trigger": trigger})
        event = StallEvent(
            fragment=fragment,
            trigger=trigger,
            at_ms=at_ms,
            already_served=self.served_this_turn(),
        )
        self._pending.append(event)
        return event

    def served_this_turn(self) -> list[str]:
        """Words offered during earlier episodes of this turn, oldest first,
        de-duplicated, capped at the most recent MAX_SERVED_HINTS.

        Two bounds, both deliberate. Each entry is a WORD, so an entry cannot
        grow with sentence length (see StallEvent.already_served). And the list
        is capped, because a speaker who stalls many times in one long turn
        would otherwise accumulate an ever-growing hint line -- the oldest
        suggestions are also the least relevant to the word being sought now.
        """
        out: list[str] = []
        for ep in self.episodes:
            w = (ep.served_word or "").strip()
            if w and w not in out:
                out.append(w)
        return out[-MAX_SERVED_HINTS:]

    def record_served(self, word: str) -> None:
        """Tell the detector which word was served for the open episode, so the
        next stall in this turn can avoid re-offering it. Called by the
        pipeline once a prediction resolves; a no-op when nothing is open."""
        ep = self._open_episode or (self.episodes[-1] if self.episodes else None)
        if ep is not None and word:
            ep.served_word = word.strip()

    # --- inputs ----------------------------------------------------------
    def observe_word(self, word: Word) -> StallEvent | None:
        """Feed a recognized word. Returns a StallEvent if this word triggers one."""
        tw = self.timeline.add_word(word)
        if tw is None:
            return None            # empty or interim -- not committed
        self._resolve_if_recovered()

        # Triggers only ever fire on the WEARER's own words. The timeline
        # already filters who counts when it assembles the fragment and the
        # content count, but the filler and hedge tests read this word's text
        # directly -- so without this check the conversation partner saying
        # "um" fired a stall and Echo offered the wearer a word for someone
        # else's sentence. Measured on real two-speaker audio: Echo fired
        # during the clinician's utterances up to 0.465 of the time.
        #
        # `is_wearer` returns True for unknown confidence, so this cannot mute
        # anyone when speaker attribution is absent or undecided.
        if not self.timeline.is_wearer(tw):
            return None

        raw = tw.text
        n = norm(raw)

        # With a scorer, a word never fires by itself -- it only changes what
        # the next evaluation instant sees. Sentence-final punctuation still
        # closes the utterance, because that is segmentation, not detection.
        if self.scorer is not None:
            if raw.endswith((".", "?", "!")):
                self._last_turn = self.timeline.utterance_text()
                self.reset()
            return None

        # Filler / hedge are evaluated BEFORE the terminal-punctuation reset so
        # a punctuated stall token (e.g. "um.") still fires.
        if n in FILLERS and self._content_count >= 1:
            return self._emit("filler", tw.end_ms)

        frag = norm(self.fragment)
        for h in HEDGES:
            if frag.endswith(h) and self._content_count >= 1:
                return self._emit("hedge", tw.end_ms)

        # Sentence-final punctuation => fluent completion; stash then reset.
        if raw.endswith((".", "?", "!")):
            self._last_turn = self.timeline.utterance_text()
            self.reset()
            return None

        return None

    def observe_acoustic(self, event: AcousticEvent) -> StallEvent | None:
        """Feed an event from the raw-audio channel (filler heard acoustically,
        prolongation). These signals are invisible to transcripts -- Chrome
        suppresses 'um/uh' and ASR normalizes 'theeee' to 'the' -- so they
        arrive only via this path. Shares the debounce/episode state with the
        transcript triggers via _emit.

        Gated on >= 1 transcribed content word so the predictor has a fragment
        to work with (the AcousticStream additionally gates on accumulated
        voiced time, so a lone leading 'um' never fires).
        """
        if self.scorer is not None:
            self.scorer.observe_acoustic(event.kind, event.at_ms, event.confidence)
            if self._content_count < 1:
                return None
            if not self.scorer.fires(self.timeline, event.at_ms):
                return None
            return self._emit("score:%s" % event.kind, event.at_ms)
        if self._content_count < 1:
            return None
        trigger = "filler_acoustic" if event.kind == "filler" else event.kind
        return self._emit(trigger, event.at_ms)

    def observe_silence(self, now_ms: int,
                        speech_end_ms: int | None = None) -> StallEvent | None:
        """Call on a timer tick. Returns a StallEvent if a mid-utterance pause
        has exceeded the threshold.

        `speech_end_ms` is when the audio channel last heard the speaker (see
        SilenceTick). The pause the wearer actually took is bounded by that
        instant, not by when the ASR got around to committing a word: a
        streaming recognizer holds the tail of the utterance behind
        LocalAgreement and the silence-forced commit, so the last COMMITTED
        word's end can trail the real end of speech by ~700 ms at the shipped
        silence_commit_ms. Measuring from the committed word alone turns that
        lag into pause length and fires a stall on a speaker who paused for
        200 ms -- the "it nags" failure.

        The two bounds are combined with max(), never min(): both say "the
        speaker was still talking at least until here", and the later of them
        is the tighter truth. None (an emitter with no VAD -- a browser timer,
        a replayed stream) therefore changes nothing.
        """
        words = self.timeline.current_utterance()
        if not words:
            return None
        last_ms = words[-1].end_ms
        if speech_end_ms is not None:
            last_ms = max(last_ms, speech_end_ms)
        if self.scorer is not None:
            # Evidence OR timeout, not evidence alone. The scorer fires on
            # things the transcript can show -- fillers, cut-off words, hedges,
            # acoustic dysfluency -- and measured on real aphasic speech that
            # evidence reaches the text for 0.560 of clinician-coded word
            # searches. Good, and 22x what a browser transcript manages, but it
            # is not all of them. For the rest there is nothing to see and only
            # the silence itself to go on, so the classic pause rule stays as
            # the floor. Dropping it (an early fit put a NEGATIVE weight on
            # pause length) cost recall on exactly those utterances.
            if self._content_count < 1:
                return None
            if self.scorer.fires(self.timeline, now_ms):
                return self._emit("score", now_ms)
            if (now_ms - last_ms >= self.pause_ms
                    and self._content_count >= 2):
                return self._emit("pause", now_ms)
            return None
        if now_ms - last_ms >= self.pause_ms and self._content_count >= 2:
            return self._emit("pause", now_ms)
        return None

    def pop_last_turn(self) -> str:
        """Return and clear the most recently completed (punctuation-reset) turn."""
        t = self._last_turn
        self._last_turn = ""
        return t

    def reset(self) -> None:
        """Close the current utterance. The timeline keeps the history.

        `_last_fire_ms` deliberately survives: the refractory is about how
        often the WEARER is interrupted, and a turn boundary is a 2 s pause in
        one conversation, not a new session.
        """
        self.timeline.mark_turn_boundary()
        self.episodes = []
        self._content_at_fire = -1
        if self.scorer is not None:
            self.scorer.reset()
