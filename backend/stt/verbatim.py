"""VerbatimASR -- server-side streaming transcription that keeps the stalls in.

WHY THIS REPLACES THE BROWSER RECOGNIZER
----------------------------------------
Echo's v2 thesis was "consumer ASR deletes the evidence", and it is measured:
on 5,044 annotated filler clips the Chrome path recovered the filler 0 times.
Echo worked around that with a parallel acoustic channel. The workaround was
necessary because no ASR in the stack would emit "um".

CrisperWhisper does, on purpose, and it is CONTROLLABLE -- mode="verbatim"
keeps fillers, false starts, repetitions and prolongations; mode="intended"
cleans them up. Measured on 50 SEP-28k annotated events, same model, same
audio: verbatim preserves the dysfluency in 0.900 of cases, intended in 0.060.
"Intended" is what every consumer recognizer silently gives you.

WHAT THIS DOES *NOT* DO
-----------------------
It does not replace the acoustic channel. Both channels now consume the SAME
PCM from the SAME socket, which is a real gain rather than redundancy: before
this, the transcript arrived on the browser's clock and the acoustic events on
the audio clock, and the two were not comparable. Now `end_ms` on a Word and
`at_ms` on an AcousticEvent are the same sample counter, so fusing them is
sound instead of approximate.

STREAMING A NON-STREAMING MODEL
-------------------------------
Whisper transcribes a whole window; it has no notion of partial output. The
standard fix is LocalAgreement-n (Liu et al. 2020; Machacek et al. 2023,
"Turning Whisper into a Real-Time Transcription System"): transcribe a growing
buffer every `step_ms`, and commit only the prefix on which the last two
hypotheses agree. Agreement is the confidence signal -- a word that survives
seeing more audio is a word the model is not going to revise.

Two Echo-specific additions:

  * COMMIT ON SILENCE. When the VAD says the speaker stopped, the audio for
    the words already spoken is complete, so there is nothing left to revise:
    the whole hypothesis is force-committed without waiting for agreement.
    This matters because a stall IS a silence -- without it, the words right
    before a stall would still be uncommitted at the exact moment the detector
    needs them, and the prompt would be a sentence missing its last word.

    The trigger was swept twice, and the second sweep overturned the first.
    Against the original wall-time word placement, 280 ms looked like the
    best value on BOTH accuracy and latency. Once the word times were fixed
    the ranking inverted and 700 ms won on accuracy; see WHAT THE WRAPPER
    COSTS below. 700 ms is the shipped value, and it IS a trade: it buys WER
    with commit lag (p90 300 -> 700 ms, median 200 -> 400 ms), which the
    stall budget can afford because the pause trigger fires at 1300 ms.

  * SILENCE TICKS ARE VAD-GATED AND CARRY THE STOP TIME. If ticks were
    emitted unconditionally, any commit delay would look exactly like a pause
    and fire a false stall -- the transcript would be "late", not the speaker.
    (This is a defect in the browser path, where transcript arrival gaps were
    the only available clock.)

    Gating alone does not finish the job, and for a while this docstring
    claimed it did. Gating stops ticks DURING speech; the pause is still
    evaluated against the last COMMITTED word, and LocalAgreement plus the
    silence force-commit hold the utterance tail behind the audio -- the
    forced commit does not even submit until `silence_commit_ms` after the
    speaker stops, so there is a guaranteed ~700 ms window in which a stale
    committed end is the only thing the detector can see. Measured: speech
    0-2464 ms, committed words ending at 1290 ms, speaker stops at 2464, first
    tick at 2664; the detector computed 2664-1290 = 1374 >= pause_ms (1300)
    and fired -- on a real pause of 200 ms. That is the "it nags" failure,
    Echo interrupting every time the wearer breathes between clauses.

    So the tick carries `speech_end_ms`, the VAD's last voiced moment, and the
    detector bounds the pause by the LATER of that and the last committed
    word. The VAD already knows this exactly and for free; it simply was not
    on the wire. `speech_end_ms=None` (a browser or mock timer with no VAD)
    leaves the old committed-word rule untouched.

WHAT THE WRAPPER COSTS, AND WHAT OF IT IS RECOVERABLE
-----------------------------------------------------
Same model, same audio, same scorer, on APROCSA (6 speakers with chronic
post-stroke aphasia, 300 s each, fillers stripped from both sides, scored
against the CHAT clinician transcripts):

    offline -- one pass over the whole 300 s        WER 0.288
    this wrapper, as first shipped                  WER 0.402
    this wrapper, with word times over voiced span  WER 0.383
    this wrapper, now (silence_commit_ms=700)       WER 0.375

The silence force-commit was the obvious suspect. Aphasic pauses run to a
672 ms median INSIDE word-search utterances (eval/results/aphasia_pause_fit
.json), so a 280 ms trigger fires inside nearly every hesitation and commits a
hypothesis formed from a fraction of the utterance. Swept against the original
wall-time word placement it looked exonerated -- raising it appeared to cost
accuracy AND latency at once:

    silence_commit_ms    280    500    700    900   1000
    WER (wall times)   0.402  0.409  0.418  0.426  0.419
    commit lag p90       300    500    700    900   1000  ms

That conclusion did not survive fixing the word times. The knob was being
scored THROUGH the timing bug: what the rows above measure is the sum of a
recognition effect and a word-placement effect, and re-swept with the
placement fixed the slope changes sign. (The reading -- not itself measured --
is that a longer commit window carries more audio per commit, so a word
mis-placed by even spreading drifted further; that penalty scaled with the
knob and swamped whatever the knob did to recognition.) With
word_time_policy="incremental":

    silence_commit_ms    280    500    700
    WER                0.383  0.375  0.375
    WER concatenated   0.339  0.329  0.322
    commit lag p90       300    500    700  ms
    commit lag median    200    400    400  ms

700 is the shipped value. It ties 500 on the per-utterance metric and is 0.7
points ahead once the utterance boundaries are removed -- the concatenated
score is the half of the metric that is pure recognition, so the tie-break
comes from the part that is not about timestamps at all.

One side effect of 700 ms is worth knowing about before it is met in a log.
A window is only retired inside silence, so requiring a longer silence means
the window sometimes fails to find one: measured on 180 s of APROCSA
(eval/bench_asr_stream.py), the buffer reached the 2 x max_window_s hard cap
twice at 700 ms and never at 280 ms, and each time `_maybe_turn_end` took the
last-resort break that logs "no pause in N s of speech". That path can clip
one word at the seam. It is not free, and it is not hidden -- but it did not
cost WER on this corpus, which is why the default moved anyway.

The window-retirement threshold is flat over the same kind of range
(reset_window_s 3.5 -> 0.402, 6.0 -> 0.401, 12.0 with a 20 s cap -> 0.413, all
measured at the old word placement). That knob still has no win in it.

What the gap is actually made of was found by re-timing the wrapper's OWN
committed words with the offline pass's timestamps and rescoring: 0.402 ->
0.341. Six of the eleven points were never recognition error. They are word
TIMES. The scorer assigns a hypothesis word to a reference utterance by its
timestamp, and with word_timestamps off this class was interpolating those
times evenly over WALL time -- so on speech that is mostly pause, a correctly
recognised word drifted into the neighbouring utterance and was counted
twice, as a deletion here and an insertion there. Spreading over the VAD's
VOICED time instead, bounded below by the words already committed, is free
(the VAD has already run) and recovers 1.9 of those points; see
_assign_times.

The residual gap is 0.375 against the offline 0.288. Part of it is still word
times -- the offline-timestamp re-scoring showed that these same committed
words score 0.341 when given perfect ones -- and the rest is recognition, and
that part is structural: this wrapper decodes 3.5-8 s windows, the offline pass
decodes 300 s with Whisper's own 30 s longform continuation and 12 words of
carried context. Model size is NOT in it -- turbo 0.402, large 0.408, medium
0.409, all within noise.

Carrying the decoder context across a window reset was the obvious next move,
and it is built (`context_prompt`, off by default) and measured, and it makes
things WORSE: 0.375 -> 0.435, on 5 of 6 speakers. The offline pass is not
winning because of the prompt. Its chunks OVERLAP by 4 s and the prompt is what
stops it transcribing that overlap twice; strip the overlap out of the offline
arm and the prompt becomes a small loss there too (0.2792 -> 0.2918). In a
stream whose windows never overlap, the model does what it was trained to do
-- skip the context words at the head of the window -- and deletes speech
nobody has transcribed. See eval/bench_asr_context_prompt.py and
backend/stt/context_prompt.py.

THREADING
---------
`feed()` must never block the audio websocket: a 300-600 ms transcribe on the
event loop would stall the mic's PCM stream and the UI at once. Transcription
runs on a single background worker; `feed()` submits a snapshot and harvests
whatever the previous call finished. One worker, so results stay ordered.

Nothing in the live path waits on that worker. A turn end still has to drain
it -- the tail of the turn belongs to this turn's transcript, not the next
one's -- but it drains by DEFERRING: if the worker is still running, the turn
is declared one `feed()` later instead, and `_turn_end_pending` keeps
`_harvest` off the result so it is still consumed as the forced full commit
the drain needs. (An earlier version called `Future.result(timeout=5.0)` right
there, which paid the transcribe latency on the event loop at every turn end
and up to five seconds in the worst case -- the exact thing this section says
must not happen.) A worker that never returns is bounded the same way the
timeout bounded it: after 5 s of audio the turn breaks anyway and the orphaned
job is dropped as stale by `_consume`.

`sync=True` is the deliberate exception and is offline-only, for replays that
feed audio far faster than real time; see the constructor.
"""
from __future__ import annotations

import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..schemas import SilenceTick, TurnEnd, Word
from .base import StreamItem

log = logging.getLogger("echo.asr")

SR = 16_000
VAD_CHUNK = 512                  # silero requirement @16k (32 ms)

_MODEL_CACHE: dict[tuple, object] = {}

# Compared on normalized text so punctuation/casing churn between hypotheses
# ("well," -> "well") does not block agreement. Apostrophes are kept: "don't"
# and "dont" are different words to the detector's filler/hedge matching.
_NORM_RE = re.compile(r"[^a-z' ]+")


def _norm(w: str) -> str:
    return _NORM_RE.sub("", w.lower()).strip()


def get_model(model_name: str = "turbo", device: str = "auto",
              compute_type: str = "float16", backend: str = "transformers"):
    """Load-once CrisperWhisper cache.

    backend defaults to "transformers", not "auto": the ct2 backend needs a
    nyra fork of ctranslate2 and fails to import against stock 4.8.1 on
    Windows. "auto" would pick it and raise at load time.
    """
    key = (model_name, device, compute_type, backend)
    model = _MODEL_CACHE.get(key)
    if model is None:
        from crisperwhisper import CrisperWhisperModel

        log.info("loading CrisperWhisper %s (%s, %s)", model_name, backend, compute_type)
        model = CrisperWhisperModel(model_name, backend=backend,
                                    compute_type=compute_type, device=device)
        _MODEL_CACHE[key] = model
    return model


def warm_asr(model_name: str = "turbo", device: str = "auto",
             compute_type: str = "float16") -> None:
    """Preload weights at startup so the first utterance doesn't pay the load.

    Also runs one tiny decode: the first CUDA generate() call compiles kernels
    and allocates workspace, which is seconds, and it would otherwise land on
    the speaker's first sentence.
    """
    model = get_model(model_name, device, compute_type)
    try:
        model.transcribe(np.zeros(SR, dtype="float32"), sr=SR, language="en")
    except Exception as exc:  # pragma: no cover - warmup is best effort
        log.warning("ASR warmup decode failed (%s); model is still loaded", exc)


def _get_vad():
    """Fresh Silero instance with its own recurrent state.

    Deliberately NOT shared with AcousticStream's VAD. Silero is stateful, and
    a shared instance would interleave two independent consumers through one
    hidden state -- the exact bug AcousticStream's _get_vad_instance() docstring
    describes. The duplicate compute is ~1 ms per 32 ms chunk against a
    transcribe that costs hundreds of ms; it is not worth coupling them.

    The loader is shared (weights are cached once on disk and in memory and
    then deep-copied per consumer) but the INSTANCE is not, which is exactly
    the distinction that matters for a recurrent model.
    """
    from ..acoustic.stream import _get_vad_instance

    return _get_vad_instance()


@dataclass
class _Job:
    audio: np.ndarray
    window_start_ms: int
    force: bool          # commit the whole hypothesis, don't wait for agreement
    # Words already committed from audio that has been RETIRED -- never words
    # still inside this job's window. See `context_prompt` in VerbatimASR.
    context: str | None = None


@dataclass
class _W:
    """One hypothesis word. start/end are seconds into the job's window, or
    None when word timestamps were not computed."""
    word: str
    start: float | None
    end: float | None


class VerbatimASR:
    """PCM16 in -> Word / SilenceTick / TurnEnd out."""

    def __init__(
        self,
        model_name: str = "turbo",
        device: str = "auto",
        compute_type: str = "float16",
        language: str = "en",
        mode: str = "verbatim",
        step_ms: int = 700,
        tick_ms: int = 200,
        silence_commit_ms: int = 700,
        turn_end_ms: int = 2000,
        max_window_s: float = 8.0,
        reset_window_s: float = 3.5,
        max_turn_s: float = 30.0,
        min_window_ms: int = 700,
        model=None,
        vad=None,
        word_time_policy: str = "incremental",
        context_prompt: bool = False,
        context_words: int = 12,
        context_log: bool = False,
        sentence_reset: bool = False,
        word_timestamps: bool = False,
        sync: bool = False,
    ) -> None:
        self.language = language
        self.mode = mode
        self.step = int(SR * step_ms / 1000)
        self.tick_ms = tick_ms
        self.silence_commit_ms = silence_commit_ms
        self.turn_end_ms = turn_end_ms
        self.max_window = int(SR * max_window_s)
        self.reset_window = int(SR * reset_window_s)
        # A speaker who never leaves a `turn_end_ms` gap (an interview answer,
        # a monologue) would otherwise accumulate one unbounded "turn": the
        # detector's fragment is deliberately never truncated, so the whole
        # monologue would be sent as the current utterance. Force a boundary.
        self.max_turn_ms = int(max_turn_s * 1000)
        self.min_window = int(SR * min_window_ms / 1000)
        # Whisper's word-level timestamps come from cross-attention DTW, and
        # on turbo that costs a FLAT ~1.0 s per call regardless of window
        # length (measured: 1229 ms at 1 s of audio, 1468 ms at 10 s, versus
        # 197/528 ms with it off). That is the whole streaming latency budget
        # spent on precision the live path does not use: the only timestamp
        # the detector reads is the last committed word's end_ms, and the VAD
        # already knows when speech stopped, exactly and for free. Off live,
        # on for offline evaluation where per-word alignment is the point.
        self.word_timestamps = word_timestamps
        # How the times of words BETWEEN the window's endpoints are estimated
        # when word_timestamps is off (which is the live default -- the DTW
        # alignment costs a flat ~1 s per call). "linear" spreads them evenly
        # over wall time, "voiced" over the VAD's voiced time only,
        # "incremental" over the voiced time that is still unaccounted for
        # after the words already committed. Measured on APROCSA at identical
        # commit lag and RTF -- at the shipped silence_commit_ms=700,
        # "voiced" 0.398 vs "incremental" 0.375; at the older 280 ms,
        # "linear" 0.402 vs "incremental" 0.383.
        self.word_time_policy = word_time_policy
        # Decode each window with the words already committed from RETIRED
        # audio as the checkpoint's `<ctx> ... <ectx>` continuation prompt --
        # the prompt format CrisperWhisper2 was trained with, which
        # `model.transcribe()` leaves empty on anything under 30 s (i.e. on
        # every window this class ever decodes). See
        # `backend/stt/context_prompt.py`.
        #
        # The context is deliberately restricted to words whose audio is GONE.
        # Words committed from the CURRENT window are still in the audio the
        # next hypothesis sees, and LocalAgreement here assumes every
        # hypothesis of a window covers that window from its origin: telling
        # the model to continue past words it is about to re-hear would make
        # the hypothesis lose its committed prefix, and `_ncommit` slicing
        # would emit the wrong words. Only a silence-forced full commit
        # retires audio (`_drop_snapshot`), and that is the only place the
        # context grows.
        #
        # OFF, and it stays off: measured on APROCSA by
        # eval/bench_asr_context_prompt.py it costs WER 0.3752 -> 0.4350, worse
        # on 5 of 6 speakers. Not because the prompt makes the decoder babble
        # (echo 0, loops 57 -> 43) but because it makes it SKIP -- the training
        # condition re-presents the context words as audio at the head of the
        # next chunk and here they are not there, so the skip deletes real
        # speech. It is kept, off, because the measurement has to stay
        # re-runnable and because the next person to have this idea should find
        # the answer rather than the idea.
        self.context_prompt = context_prompt
        self.context_words = context_words
        # Diagnostic tap for the bench: every (context, hypothesis) pair the
        # decoder actually saw, which is what a degeneracy count needs. Off
        # live -- it would grow without bound over a long session.
        self.context_log: list[dict] = []
        self._context_log_on = context_log
        self._ctx_model = None
        # Words committed from audio that has been retired (the context), and
        # words committed from the window still in the buffer (which are NOT
        # context -- the model is about to re-hear that audio).
        self._retired_words: list[str] = []
        self._window_words: list[str] = []
        # Whisper punctuates; the browser recognizer did not. The detector
        # treats sentence-final punctuation as a fluent completion and resets
        # the utterance, so leaving this on would silently change turn
        # segmentation the moment the ASR swapped. Off by default, measured
        # separately (eval/run_asr_integration_eval.py).
        self.sentence_reset = sentence_reset
        # Offline replays feed audio far faster than real time, so an async
        # worker would still be on the first window when the file ends and the
        # run would score zero words -- a property of the harness, not of the
        # model. sync=True blocks feed() until the transcribe returns, making
        # replay deterministic and complete. It must stay OFF live: a blocking
        # transcribe on the audio socket stalls the PCM stream and the UI.
        self.sync = sync
        self.model = model if model is not None else get_model(model_name, device, compute_type)
        if self.context_prompt:
            if hasattr(self.model, "set_context"):
                # Already a context-capable facade (an injected test double,
                # or a ContextModel the caller built itself).
                self._ctx_model = self.model
            else:
                from .context_prompt import ContextModel

                self._ctx_model = ContextModel(context_words=context_words,
                                               model=self.model)

        # Injectable so the commit/turn logic can be tested deterministically
        # against a scripted speech/silence pattern. Every interesting bug in
        # this file (phantom pauses, stale jobs, duplicate emission after a
        # trim) is a timing bug, and timing bugs do not reproduce when the
        # oracle is a real VAD on real audio.
        self._vad = vad if vad is not None else _get_vad()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="echo-asr")
        self._future: Future | None = None

        self._window = np.zeros(0, dtype="float32")   # audio not yet committed
        self._window_start_ms = 0
        self._turn_start_ms = 0
        self._consumed = 0            # total samples ever fed (the clock)
        self._since_step = 0
        self._vad_pending = np.zeros(0, dtype="float32")
        self._speech_prob = 0.0
        self._silence_ms = 0
        self._voiced_ms = 0
        self._next_tick_ms = 0
        # A commit is armed: the next submit takes the whole hypothesis and may
        # retire audio. `_force_silence` records that the VAD's silence run is
        # what armed it, which is the arming that must be CANCELLED the moment
        # speech resumes -- the single worker may still have been busy when the
        # silence crossed the threshold, and a flag left set from that silence
        # fires mid-utterance, retires the window across a word and re-emits
        # the word straddling the seam. The hard-cap arming in _maybe_turn_end
        # is deliberately NOT cancelled: it exists precisely for speech with no
        # pause in it, and it is followed immediately by a turn break.
        self._force_next = False
        self._force_silence = False
        # A turn end is waiting on the worker (see _maybe_turn_end), and the
        # pending result must be consumed as a forced full commit rather than
        # harvested as an ordinary agreement step.
        self._turn_end_pending = False
        self._turn_end_pending_ms = 0
        # Absolute ms of the last chunk the VAD called speech. The submit gate
        # asks "does the CURRENT WINDOW contain voice", not "has this turn ever
        # contained voice" -- after a trim the window can be pure silence, and
        # transcribing silence is where Whisper invents "Thank you." / "Bye."
        self._last_voiced_ms = -1
        # Absolute [start_ms, end_ms) of every run the VAD called speech,
        # pruned to the current window. This is the only thing the pipeline
        # knows for free about WHERE inside a window the words are.
        self._voiced_spans: list[list[int]] = []
        self._vad_ms = 0              # audio the VAD has actually consumed
        # End of the last word actually emitted. A word committed at step s
        # cannot have been spoken before the words committed at step s-1, and
        # that bound is free -- see _assign_times.
        self._last_commit_end_ms = 0
        self._committed_this_turn = 0
        self._ncommit = 0             # words of the current hypothesis already emitted
        self._prev_hyp: list[str] = []
        self._interim: list[str] = []
        self._closed = False

    # --- clock ------------------------------------------------------------
    @property
    def now_ms(self) -> int:
        return int(self._consumed * 1000 / SR)

    @property
    def speech_prob(self) -> float:
        return self._speech_prob

    @property
    def interim_text(self) -> str:
        """Uncommitted tail -- for UI display only. Never fed to the detector:
        an interim word can be revised, and a revised filler would fire a stall
        for a word the speaker never said."""
        return " ".join(self._interim)

    # --- main entry point --------------------------------------------------
    def feed(self, pcm16: bytes) -> list[StreamItem]:
        if self._closed:
            return []
        x = np.frombuffer(pcm16[:len(pcm16) - (len(pcm16) % 2)], dtype="<i2")
        if x.size == 0:
            return []
        x = (x.astype("float32") / 32768.0)
        self._consumed += x.size
        self._since_step += x.size
        self._window = np.concatenate([self._window, x])

        items: list[StreamItem] = []
        items.extend(self._harvest())          # results from the previous step
        self._run_vad(x)
        self._maybe_submit()
        if self.sync and self._future is not None:
            fut, self._future = self._future, None
            try:
                items.extend(self._consume(fut.result(), forced=False))
            except Exception:
                pass
        # Ticks last, so a word committed in this same call is already on the
        # timeline when the detector evaluates the pause that follows it.
        items.extend(self._ticks())
        items.extend(self._maybe_turn_end())
        return items

    def close(self) -> None:
        self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    # --- VAD ---------------------------------------------------------------
    def _run_vad(self, x: np.ndarray) -> None:
        import torch

        self._vad_pending = np.concatenate([self._vad_pending, x])
        chunk_ms = VAD_CHUNK * 1000 // SR
        while self._vad_pending.size >= VAD_CHUNK:
            chunk = self._vad_pending[:VAD_CHUNK]
            self._vad_pending = self._vad_pending[VAD_CHUNK:]
            with torch.no_grad():
                self._speech_prob = float(self._vad(torch.from_numpy(chunk.copy()), SR).item())
            # The VAD runs on a contiguous 32 ms grid of its own, which lags
            # `now_ms` by up to one fed buffer. Span edges use the VAD's own
            # clock so a word is never placed in audio the VAD never saw.
            t0, self._vad_ms = self._vad_ms, self._vad_ms + chunk_ms
            if self._speech_prob >= 0.5:
                self._voiced_ms += chunk_ms
                self._last_voiced_ms = self.now_ms
                if self._voiced_spans and self._voiced_spans[-1][1] == t0:
                    self._voiced_spans[-1][1] = self._vad_ms
                else:
                    self._voiced_spans.append([t0, self._vad_ms])
                # Speech resumed: the next silence gets its own commit, so
                # cancel one armed by the silence that just ended. Without
                # this the flag survives a busy worker and the forced job is
                # submitted DURING speech -- which commits without agreement
                # and slides the window origin mid-utterance, breaking both
                # "only a silence-forced full commit may retire audio" and
                # "every window boundary must sit inside silence".
                self._silence_ms = 0
                if self._force_silence:
                    self._force_next = False
                    self._force_silence = False
            else:
                was = self._silence_ms
                self._silence_ms += chunk_ms
                # Crossing the threshold exactly once arms a forced commit.
                # Re-arming every chunk would re-transcribe the same silence
                # at wire speed for the whole pause.
                if was < self.silence_commit_ms <= self._silence_ms:
                    self._force_next = True
                    self._force_silence = True
        # Spans that end before the window starts can never place a word in
        # it; dropping them keeps the list bounded on a long monologue.
        while len(self._voiced_spans) > 1 and self._voiced_spans[0][1] < self._window_start_ms:
            self._voiced_spans.pop(0)

    # --- silence ticks -----------------------------------------------------
    def _ticks(self) -> list[SilenceTick]:
        """Tick only while the audio is silent, and say WHEN speech stopped.

        The gating alone is not enough, which is the thing this docstring used
        to claim. It stops ticks DURING speech, but the pause is evaluated
        against the last COMMITTED word, and that trails the real end of speech
        by the commit lag (p90 700 ms at the shipped silence_commit_ms) --
        the silence-forced commit that flushes the tail does not even submit
        until silence_commit_ms after the speaker stopped, so there is a
        guaranteed window in which the stale committed end is what the detector
        sees. Measured: speech 0-2464 ms, committed words ending at 1290, first
        tick at 2664, and a 1300 ms pause trigger fired on a 200 ms pause.

        `speech_end_ms` is the VAD's last voiced moment -- exact, already
        computed, and the only place the information exists.
        """
        out: list[SilenceTick] = []
        if self._speech_prob >= 0.5:
            self._next_tick_ms = self.now_ms + self.tick_ms
            return out
        # -1 means the VAD has never heard speech, which is not a stop.
        end = self._last_voiced_ms if self._last_voiced_ms >= 0 else None
        while self._next_tick_ms <= self.now_ms:
            out.append(SilenceTick(at_ms=self._next_tick_ms, speech_end_ms=end))
            self._next_tick_ms += self.tick_ms
        return out

    # --- turn segmentation --------------------------------------------------
    def _maybe_turn_end(self) -> list[StreamItem]:
        if self._committed_this_turn == 0:
            self._turn_end_pending = False
            return []
        # An over-long turn is broken at the first real pause rather than
        # mid-word: cutting on the clock alone would split a sentence and hand
        # the predictor half a fragment.
        over_length = (self.now_ms - self._turn_start_ms >= self.max_turn_ms
                       and self._silence_ms >= self.silence_commit_ms)
        # Same rule for the window cap, and for the same reason. An earlier
        # version force-committed the moment the window hit the cap, which
        # happens DURING speech -- so the boundary landed mid-word, the next
        # window still contained the first half of that word, and the model
        # re-proposed it. Every window boundary must sit inside silence.
        over_window = (self._window.size > self.max_window
                       and self._silence_ms >= self.silence_commit_ms)
        hard_cap = self._window.size > 2 * self.max_window
        if hard_cap and self._future is None and len(self._prev_hyp) > self._ncommit:
            # Words are still uncommitted and no pause is coming to flush
            # them. Arm a forced commit and end the turn one cycle later,
            # so the break never silently swallows the tail of the turn.
            self._force_next = True
            return []
        if (self._silence_ms < self.turn_end_ms
                and not over_length and not over_window and not hard_cap):
            self._turn_end_pending = False
            return []
        items: list[StreamItem] = []
        # Drain anything the worker still owes before declaring the turn over,
        # or the last words of the turn land in the NEXT turn's transcript.
        #
        # WAITING for it is not an option. feed() runs on the asyncio event loop
        # that also serves the /ws/audio PCM websocket and the UI socket, and a
        # transcribe costs 300-600 ms (worst case, the `result(timeout=5.0)`
        # that used to be here, five seconds) -- blocking stalls the audio
        # stream and the UI at once, at every single turn end. So the turn is
        # declared one feed() later instead, once the worker has returned;
        # `_turn_end_pending` keeps `_harvest` off the result in the meantime
        # so it is still consumed as the forced full commit the drain needs.
        if self._future is not None:
            if not self._future.done():
                if not self._turn_end_pending:
                    self._turn_end_pending = True
                    self._turn_end_pending_ms = self.now_ms
                # Same last-resort bound the blocking wait had, on the audio
                # clock: a worker that never returns must not wedge turn
                # segmentation for ever.
                if self.now_ms - self._turn_end_pending_ms < 5000:
                    return []
                self._future = None       # orphan it; _consume drops it stale
            else:
                fut, self._future = self._future, None
                try:
                    items.extend(self._consume(fut.result(), forced=True))
                except Exception:
                    pass
        self._turn_end_pending = False
        if hard_cap:
            # Last resort for speech with no measurable pause at all. Reaching
            # here can clip one word at the seam, so it is logged rather than
            # silently absorbed.
            log.warning("no pause in %.0f s of speech; forcing a turn break",
                        self._window.size / SR)
        items.append(TurnEnd())
        self._reset_turn()
        return items

    def _promote_context(self) -> None:
        self._retired_words.extend(self._window_words)
        self._window_words = []
        if len(self._retired_words) > self.context_words:
            del self._retired_words[:-self.context_words]

    def _reset_turn(self) -> None:
        # The turn's audio is discarded here, so its committed words are behind
        # the new window exactly as they are after a snapshot drop. The context
        # carries ACROSS a turn boundary: a turn end is a 2 s pause in one
        # conversation, not a new recording, and the continuation objective is
        # about what was said before, not about turn structure.
        self._promote_context()
        self._window = np.zeros(0, dtype="float32")
        self._window_start_ms = self.now_ms
        self._turn_start_ms = self.now_ms
        self._last_commit_end_ms = self.now_ms
        self._prev_hyp = []
        self._ncommit = 0
        self._interim = []
        self._committed_this_turn = 0
        self._since_step = 0
        self._force_next = False
        self._force_silence = False
        self._voiced_ms = 0

    # --- worker plumbing ----------------------------------------------------
    def _maybe_submit(self) -> None:
        if self._future is not None:
            return                      # one transcribe in flight at a time
        due = self._since_step >= self.step or self._force_next
        if not due or self._window.size < self.min_window:
            return
        # Nothing voiced in this window at all: transcribing pure silence is
        # where Whisper hallucinates ("Thank you.", "Bye."), so don't.
        if self._last_voiced_ms < self._window_start_ms:
            self._force_next = False
            self._force_silence = False
            return
        job = _Job(audio=self._window.copy(),
                   window_start_ms=self._window_start_ms,
                   force=self._force_next,
                   context=self._context_text())
        self._since_step = 0
        self._force_next = False
        self._force_silence = False
        self._future = self._pool.submit(self._transcribe, job)

    def _context_text(self) -> str | None:
        """The continuation prompt for the next window, or None.

        Only words whose audio has been RETIRED -- see `context_prompt`. The
        list is already capped at `context_words`, so this is O(1).
        """
        if not self.context_prompt or not self._retired_words:
            return None
        return " ".join(self._retired_words) or None

    def _transcribe(self, job: _Job) -> tuple[_Job, list]:
        try:
            if self._ctx_model is not None:
                self._ctx_model.set_context(job.context)
                res = self._ctx_model.transcribe(
                    job.audio, sr=SR, language=self.language, mode=self.mode,
                    word_timestamps=self.word_timestamps)
            else:
                res = self.model.transcribe(
                    job.audio, sr=SR, language=self.language, mode=self.mode,
                    word_timestamps=self.word_timestamps)
            if self._context_log_on:
                self.context_log.append({
                    "window_start_ms": job.window_start_ms,
                    "audio_s": round(job.audio.size / SR, 3),
                    "force": bool(job.force),
                    "context": job.context,
                    "hyp": (res.text or "").strip(),
                })
            if res.words:
                return job, [_W(w.word, w.start, w.end) for w in res.words]
            # word_timestamps=False returns text only. Whitespace tokens are
            # still what LocalAgreement compares, so segmentation is intact;
            # only the timings are missing, and _assign_times fills those from
            # the VAD rather than inventing precision.
            return job, [_W(t, None, None) for t in (res.text or "").split()]
        except Exception as exc:  # pragma: no cover - live path
            log.warning("transcribe failed: %s", exc)
            return job, []

    def _harvest(self) -> list[StreamItem]:
        # A turn end is waiting on this exact result and must consume it as a
        # forced full commit, or the tail of the turn lands in the next turn's
        # transcript. Leave it alone.
        if self._turn_end_pending:
            return []
        if self._future is None or not self._future.done():
            return []
        fut, self._future = self._future, None
        try:
            payload = fut.result()
        except Exception:
            return []
        return self._consume(payload, forced=False)

    def _consume(self, payload, forced: bool) -> list[StreamItem]:
        job, words = payload
        # A turn boundary may have reset the window while this job was in
        # flight; its offsets no longer mean anything.
        if job.window_start_ms != self._window_start_ms:
            return []
        words = [w for w in words if _norm(w.word)]
        hyp = [_norm(w.word) for w in words]

        if forced or job.force:
            n_agree = len(words)        # audio is complete; nothing to revise
        else:
            n_agree = _common_prefix(self._prev_hyp, hyp)
        self._prev_hyp = hyp

        # Emit only the newly-agreed slice. The window is NOT trimmed on
        # commit, so hypotheses keep the same origin and `_ncommit` is all the
        # bookkeeping needed. An earlier version trimmed the audio at each
        # commit and sliced `_prev_hyp` to match; when the cut landed slightly
        # early -- which it does whenever word timestamps are off and the trim
        # point is interpolated -- the next hypothesis repeated the last
        # committed words and they were emitted twice. Deduplicating instead
        # would have been worse: "you you recently" is a WordRep, the exact
        # signal this whole change exists to preserve, and a deduplicator
        # cannot tell it from a stitching artifact.
        lo, hi = self._ncommit, max(self._ncommit, n_agree)
        new = words[lo:hi]
        self._ncommit = hi
        # The interim line is everything past the COMMIT cursor, not past
        # n_agree. When a hypothesis retracts (n_agree < _ncommit) the words in
        # between have already gone out as final, and slicing at n_agree put
        # them on screen a second time as grey interim text.
        self._interim = [w.word.strip() for w in words[hi:]]

        items: list[StreamItem] = []
        times = self._assign_times(job, hyp, lo, hi)
        for w, (start_s, end_s) in zip(new, times):
            text = w.word.strip()
            if not self.sentence_reset:
                # Strip only sentence-FINAL punctuation, which is what the
                # detector resets on. Internal commas/hyphens are left alone:
                # "w-" is a cut-off word and the single best evidence of a
                # block, so it must survive into the fragment.
                text = text.rstrip(".?!")
            if not text:
                continue
            end_ms = job.window_start_ms + int(end_s * 1000)
            items.append(Word(
                text=text,
                start_ms=job.window_start_ms + int(start_s * 1000),
                end_ms=end_ms,
                is_final=True,
            ))
            self._last_commit_end_ms = max(self._last_commit_end_ms, end_ms)
            self._committed_this_turn += 1
            self._window_words.append(text)

        # The ONLY safe place to move the window origin is a silence-forced
        # commit that emitted the entire hypothesis: at that instant every
        # word in the snapshot is already out, and the boundary sits inside
        # silence so no word straddles it. Sliding anywhere else re-covers
        # committed audio, the model re-proposes those words, and they are
        # emitted twice -- measured on a 30 s replay as "...on f- f- Facebook
        # that you I thought I saw something on f- f- Facebook that you
        # posted...". Deduplicating that away is not an option: "you you
        # recently" is a WordRep, the exact evidence this class exists to
        # preserve, and no deduplicator can tell it from a seam.
        if (forced or job.force) and self._ncommit >= len(words) \
                and self._window.size >= self.reset_window:
            self._drop_snapshot(job)
        return items

    def _drop_snapshot(self, job: _Job) -> None:
        """Retire exactly the audio this job transcribed, keeping the tail that
        arrived after the snapshot was taken (which no hypothesis has seen).

        Not `reset_window_s == 0`: Whisper has a 30 s receptive field and gets
        materially worse on short fragments -- an earlier benchmark scored
        0.217 on 1 s clips purely because of it. So the window is allowed to
        accumulate a few seconds of context before the first pause is taken as
        a reset point.
        """
        drop = min(job.audio.size, self._window.size)
        if drop <= 0:
            return
        # Every word of this job is already emitted (the caller checks
        # `_ncommit >= len(words)`) and the audio it was spoken in is about to
        # go, so those words become context and can never be re-proposed.
        self._promote_context()
        self._window = self._window[drop:]
        self._window_start_ms += int(drop * 1000 / SR)
        self._prev_hyp = []
        self._ncommit = 0
        self._last_commit_end_ms = self._window_start_ms

    def _assign_times(self, job: _Job, words: list, lo: int, hi: int) -> list[tuple[float, float]]:
        """Seconds-into-window (start, end) for hypothesis words [lo:hi).

        With real word timestamps this is a passthrough of the model's own
        alignment. Without them -- the live default, because the DTW alignment
        costs a flat ~1 s per call -- the honest position is that we know three
        things: every word in this hypothesis was spoken somewhere inside this
        window, speech stopped when the VAD said it did, and the VAD also said
        WHICH parts of the window were speech at 32 ms resolution. The third
        one used to be thrown away: words were spread evenly over WALL time,
        so on speech that is mostly pauses -- which is precisely the speech
        this product is for -- a word landed in the middle of a silence it was
        never spoken in, and drifted into the neighbouring utterance.

        `word_time_policy="voiced"` spreads them evenly over VOICED time
        instead: word k of n is placed where k/n of the window's voiced
        duration has elapsed. It is still ORDERING rather than alignment and
        nothing should read per-word precision out of it, but it is ordering
        that cannot put a word inside a pause. `"incremental"` additionally
        floors the slice at the last committed word's end. Measured on APROCSA
        against the clinician's utterance boundaries, at the shipped
        silence_commit_ms=700: "voiced" 0.398, "incremental" 0.375. It costs
        nothing, because the VAD has already run.

        The last committed word's end stays pinned to the VAD's last voiced
        moment, which is exact and is the single timestamp the pause trigger
        actually reads.

        Spreading uses each word's index in the FULL hypothesis, not in the
        emitted slice, so a word's time does not depend on which step happened
        to commit it.
        """
        n = len(words)
        if n == 0 or hi <= lo:
            return []
        if self.word_timestamps and isinstance(words[0], _W):
            out, last = [], 0.0
            for w in words[lo:hi]:
                s = w.start if w.start is not None else last
                e = w.end if w.end is not None else s
                last = e
                out.append((s, e))
            if out and out[-1][1] is not None:
                return out

        window_s = job.audio.size / SR
        voiced_end_ms = self._last_voiced_ms - job.window_start_ms
        span = max(0.05, min(window_s, voiced_end_ms / 1000.0))

        out = None
        if self.word_time_policy in ("voiced", "incremental"):
            floor = 0.0
            if self.word_time_policy == "incremental":
                # Words already emitted bound this slice from below. Whatever
                # step committed word lo-1 had already seen the audio it was
                # spoken in, so nothing after it can start earlier.
                floor = (self._last_commit_end_ms - job.window_start_ms) / 1000.0
                floor = min(max(0.0, floor), max(0.0, span - 0.05))
                n, lo, hi = n - lo, 0, hi - lo
            out = self._spread_over_voiced(job, n, lo, hi, span, floor)
        if out is None:
            step = span / n
            out = [(i * step, (i + 1) * step) for i in range(lo, hi)]
        if hi >= n:
            # Last word of the hypothesis: pin its end to the measured end of
            # speech rather than to the interpolation.
            out[-1] = (out[-1][0], span)
        return out

    def _spread_over_voiced(self, job: _Job, n: int, lo: int, hi: int,
                            span: float,
                            floor: float = 0.0) -> list[tuple[float, float]] | None:
        """Place n words at equal intervals of VOICED time inside the window.

        Returns None when the VAD spans do not cover this window (a replayed
        job whose spans were pruned, or an injected VAD in a test), so the
        caller falls back to the old even-over-wall-time spread rather than
        inventing a position.
        """
        iv: list[tuple[float, float]] = []
        for s0, s1 in self._voiced_spans:
            a = max(floor, (s0 - job.window_start_ms) / 1000.0)
            b = min(span, (s1 - job.window_start_ms) / 1000.0)
            if b > a:
                iv.append((a, b))
        total = sum(b - a for a, b in iv)
        if total <= 0:
            return None
        edges = [self._at_voiced_frac(iv, total, k / n) for k in range(n + 1)]
        return [(edges[i], edges[i + 1]) for i in range(lo, hi)]

    @staticmethod
    def _at_voiced_frac(iv: list[tuple[float, float]], total: float,
                        frac: float) -> float:
        x = frac * total
        for a, b in iv:
            d = b - a
            if x <= d:
                return a + x
            x -= d
        return iv[-1][1]


def _common_prefix(a: list[str], b: list[str]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n
