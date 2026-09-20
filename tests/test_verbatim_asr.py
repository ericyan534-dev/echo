"""VerbatimASR streaming/commit logic, with the model and VAD scripted.

Everything interesting in this module is a TIMING property -- when a word is
safe to commit, when a silence is a pause versus the end of a turn, whether a
job that outlived its window can still emit. Those do not reproduce reliably
against a real VAD on real audio, so both oracles are injected and the audio
is meaningless: what the fake VAD says about a frame is the only thing that
decides speech vs silence.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from backend.schemas import SilenceTick, TurnEnd, Word
from backend.stall_detector import StallDetector
from backend.stt.verbatim import SR, VerbatimASR, _Job, _W
from backend.timeline import FILLERS, norm


class FakeVAD:
    """Returns whatever probability the test currently wants."""

    def __init__(self, prob: float = 1.0) -> None:
        self.prob = prob

    def __call__(self, chunk, sr):
        class _P:
            def __init__(self, v):
                self._v = v

            def item(self):
                return self._v

        return _P(self.prob)


class ScriptedModel:
    """Yields a preset hypothesis per transcribe() call."""

    def __init__(self, hypotheses: list[str]) -> None:
        self.hypotheses = list(hypotheses)
        self.calls = 0

    def transcribe(self, audio, **kw):
        text = self.hypotheses[min(self.calls, len(self.hypotheses) - 1)]
        self.calls += 1

        class _R:
            def __init__(self, t):
                self.text = t
                self.words = None

        return _R(text)


def frame(ms: int = 100) -> bytes:
    return np.zeros(int(SR * ms / 1000), dtype="<i2").tobytes()


def make(hyps, vad, **kw):
    kw.setdefault("sync", True)
    kw.setdefault("min_window_ms", 100)
    return VerbatimASR(model=ScriptedModel(hyps), vad=vad, **kw)


def words_of(items):
    return [i.text for i in items if isinstance(i, Word)]


# --- LocalAgreement -------------------------------------------------------
def test_word_is_not_committed_until_two_hypotheses_agree():
    """The whole point of LocalAgreement: a word only ships once seeing more
    audio has failed to change it. A first hypothesis commits nothing."""
    vad = FakeVAD(1.0)
    asr = make(["I need the", "I need the thing"], vad, step_ms=100)
    first = asr.feed(frame(200))
    assert words_of(first) == [], "nothing may commit on a single hypothesis"
    second = asr.feed(frame(200))
    assert words_of(second) == ["I", "need", "the"], (
        "the agreed prefix commits; 'thing' is still revisable")
    asr.close()


def test_disagreement_after_prefix_stops_the_commit_there():
    vad = FakeVAD(1.0)
    asr = make(["I saw a boat", "I saw a bird"], vad, step_ms=100)
    asr.feed(frame(200))
    out = words_of(asr.feed(frame(200)))
    assert out == ["I", "saw", "a"], "commit must stop at the first disagreement"
    asr.close()


def test_silence_forces_a_commit_without_agreement():
    """A stall IS a silence. If the words before it waited for agreement, the
    predictor would be handed a sentence missing its last word at exactly the
    moment it matters."""
    vad = FakeVAD(1.0)
    asr = make(["I need the"], vad, step_ms=10_000, silence_commit_ms=100)
    asr.feed(frame(200))               # speech, no step due
    vad.prob = 0.0
    out = words_of(asr.feed(frame(200)))
    assert out == ["I", "need", "the"], "silence must force the whole hypothesis out"
    asr.close()


def test_committed_words_are_never_emitted_twice_after_a_trim():
    """The trim slides the window forward; _prev_hyp is sliced to match. If the
    two ever disagree the same words re-emit and the fragment doubles."""
    vad = FakeVAD(1.0)
    asr = make(["one two", "one two", "one two three", "one two three"],
               vad, step_ms=100)
    seen = []
    for _ in range(4):
        seen.extend(words_of(asr.feed(frame(200))))
    assert seen == sorted(set(seen), key=seen.index), "no word may commit twice"
    assert seen[:2] == ["one", "two"]
    asr.close()


# --- silence ticks --------------------------------------------------------
def test_ticks_only_while_the_audio_is_silent():
    """Ticks drive the pause trigger. Emitting them during speech would make
    ASR commit delay indistinguishable from a speaker's pause, and fire a
    stall on a speaker who never stopped talking."""
    vad = FakeVAD(1.0)
    asr = make(["hello"], vad, step_ms=10_000)
    assert [i for i in asr.feed(frame(400)) if isinstance(i, SilenceTick)] == []
    vad.prob = 0.0
    ticks = [i for i in asr.feed(frame(400)) if isinstance(i, SilenceTick)]
    assert ticks, "silence must produce ticks"
    asr.close()


def test_tick_timestamps_advance_monotonically():
    vad = FakeVAD(0.0)
    asr = make(["x"], vad, step_ms=10_000, tick_ms=100)
    ticks = [i.at_ms for i in asr.feed(frame(500)) if isinstance(i, SilenceTick)]
    assert ticks == sorted(ticks) and len(set(ticks)) == len(ticks)
    asr.close()


# --- turn segmentation ----------------------------------------------------
def test_turn_end_needs_committed_words_and_enough_silence():
    vad = FakeVAD(1.0)
    asr = make(["hello there"], vad, step_ms=100, silence_commit_ms=100,
               turn_end_ms=1000)
    asr.feed(frame(200))
    vad.prob = 0.0
    out = asr.feed(frame(500))
    assert not any(isinstance(i, TurnEnd) for i in out), "500ms < turn_end_ms"
    out = asr.feed(frame(700))
    assert any(isinstance(i, TurnEnd) for i in out), "1200ms >= turn_end_ms"
    asr.close()


def test_silence_alone_never_ends_a_turn_with_no_words():
    """Otherwise a quiet room emits an endless stream of empty turns, each one
    appending a blank line to the conversation record."""
    vad = FakeVAD(0.0)
    asr = make(["ignored"], vad, step_ms=100, turn_end_ms=200)
    out = []
    for _ in range(10):
        out.extend(asr.feed(frame(200)))
    assert not any(isinstance(i, TurnEnd) for i in out)
    asr.close()


def test_turn_end_is_forced_at_max_turn_s_on_a_monologue():
    """A speaker who never leaves a 2 s gap would otherwise accumulate one
    unbounded turn, and the detector's fragment is deliberately never
    truncated -- the whole monologue would be sent as the current utterance."""
    vad = FakeVAD(1.0)
    asr = make(["a", "a", "a b", "a b"], vad, step_ms=100,
               silence_commit_ms=100, turn_end_ms=60_000, max_turn_s=1.0)
    for _ in range(6):
        asr.feed(frame(200))
    vad.prob = 0.0
    out = []
    for _ in range(3):
        out.extend(asr.feed(frame(200)))
    assert any(isinstance(i, TurnEnd) for i in out), (
        "an over-long turn must break at the first real pause")
    asr.close()


# --- text handling --------------------------------------------------------
def test_filler_tokens_survive_and_normalize_to_a_detector_filler():
    """CrisperWhisper emits '[UM]'. The whole swap is pointless if that does
    not reach StallDetector's filler set."""
    vad = FakeVAD(1.0)
    asr = make(["[UM] the"], vad, step_ms=100, silence_commit_ms=100)
    asr.feed(frame(200))
    vad.prob = 0.0
    out = words_of(asr.feed(frame(200)))
    assert out[0] == "[UM]"
    assert norm(out[0]) in FILLERS, "normalization must land it in FILLERS"
    asr.close()


def test_sentence_final_punctuation_is_stripped_by_default():
    """The detector resets the utterance on '.', '?' and '!'. Whisper
    punctuates and the browser recognizer did not, so leaving it in would
    silently change turn segmentation the moment the ASR swapped."""
    vad = FakeVAD(1.0)
    asr = make(["done."], vad, step_ms=100, silence_commit_ms=100)
    asr.feed(frame(200))
    vad.prob = 0.0
    assert words_of(asr.feed(frame(200))) == ["done"]
    asr.close()


def test_cut_off_word_marker_is_preserved():
    """'f-' is the block / sound-repetition evidence. Stripping trailing
    hyphens with the punctuation would delete the single most useful token."""
    vad = FakeVAD(1.0)
    asr = make(["f- Facebook."], vad, step_ms=100, silence_commit_ms=100)
    asr.feed(frame(200))
    vad.prob = 0.0
    assert words_of(asr.feed(frame(200))) == ["f-", "Facebook"]
    asr.close()


def test_sentence_reset_true_keeps_punctuation():
    vad = FakeVAD(1.0)
    asr = make(["done."], vad, step_ms=100, silence_commit_ms=100,
               sentence_reset=True)
    asr.feed(frame(200))
    vad.prob = 0.0
    assert words_of(asr.feed(frame(200))) == ["done."]
    asr.close()


# --- timing ---------------------------------------------------------------
def test_last_word_end_is_pinned_to_the_vads_last_voiced_moment():
    """Without word timestamps this is the one time value the pause trigger
    reads, and the VAD knows it exactly. If it drifted, every pause length
    would be wrong."""
    vad = FakeVAD(1.0)
    asr = make(["one two"], vad, step_ms=100, silence_commit_ms=100)
    asr.feed(frame(500))
    voiced_end = asr._last_voiced_ms
    vad.prob = 0.0
    out = [i for i in asr.feed(frame(200)) if isinstance(i, Word)]
    assert out, "expected a forced commit"
    assert out[-1].end_ms == pytest.approx(voiced_end, abs=40)
    asr.close()


def test_word_times_are_non_decreasing():
    vad = FakeVAD(1.0)
    asr = make(["one two three"], vad, step_ms=100, silence_commit_ms=100)
    asr.feed(frame(600))
    vad.prob = 0.0
    out = [i for i in asr.feed(frame(200)) if isinstance(i, Word)]
    starts = [w.start_ms for w in out]
    assert starts == sorted(starts)
    assert all(w.end_ms >= w.start_ms for w in out)
    asr.close()


# --- robustness -----------------------------------------------------------
def test_pure_silence_is_never_transcribed():
    """Whisper invents 'Thank you.' / 'Bye.' on silence. Those would enter the
    transcript as real words and the predictor would treat them as context."""
    vad = FakeVAD(0.0)
    model = ScriptedModel(["Thank you."])
    asr = VerbatimASR(model=model, vad=vad, sync=True, step_ms=100,
                      min_window_ms=100)
    for _ in range(8):
        asr.feed(frame(200))
    assert model.calls == 0, "no window containing zero voiced audio may be sent"
    asr.close()


def test_a_transcribe_exception_is_swallowed():
    class Boom:
        def transcribe(self, audio, **kw):
            raise RuntimeError("cuda oom")

    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=Boom(), vad=vad, sync=True, step_ms=100,
                      min_window_ms=100)
    assert asr.feed(frame(300)) == [] or True   # must not raise
    asr.close()


class PositionalModel:
    """A perfect ASR: transcribes exactly the voiced audio it is handed.

    Each 100 ms speech frame is stamped with its own position on a word clock
    that advances ONLY during speech; silence frames are zeros and carry no
    position. The model reports every word whose frames are present in the
    window it receives. So it never invents a word, never drops one, and names
    them identically no matter which window they arrive in -- meaning any word
    emitted twice, or lost, is the streaming wrapper's fault and not the
    model's. That is exactly the class of bug window boundaries produce.
    """

    WORD_FRAMES = 5           # one word per 500 ms of speech

    def transcribe(self, audio, **kw):
        marks = np.unique(np.rint(np.asarray(audio) * 32768).astype("int64"))
        marks = marks[marks > 0]
        if marks.size == 0:
            return _Res("")
        ks = sorted({int(m - 1) // self.WORD_FRAMES for m in marks})
        return _Res(" ".join("w%d" % k for k in ks))


class _Res:
    def __init__(self, text):
        self.text = text
        self.words = None


def positional_frame(index: int, ms: int = 100) -> bytes:
    """A speech frame at word-clock position `index` (1-based internally, so
    zero can mean silence)."""
    return np.full(int(SR * ms / 1000), index + 1, dtype="<i2").tobytes()


def silence_frame(ms: int = 100) -> bytes:
    return np.zeros(int(SR * ms / 1000), dtype="<i2").tobytes()


def test_long_speech_past_the_window_cap_emits_no_word_twice():
    """The regression for the measured failure: on a 30 s replay the window
    slid mid-turn, the model re-proposed already-committed words, and the
    transcript read '...on f- f- Facebook that you I thought I saw something
    on f- f- Facebook that you posted...'. Only a silence-forced full commit
    may retire audio, and every window boundary must sit inside silence.

    The speech run per cycle is 3.0 s against a 2.0 s max_window, so the window
    cap is crossed WHILE THE VAD SAYS SPEECH on every cycle -- which is the
    only condition under which the `_silence_ms >= silence_commit_ms` guard on
    `over_window` does anything at all. (Two earlier versions of this fixture
    both missed it. One paused for a single 100 ms frame -- 3 VAD chunks,
    96 ms -- so the silence threshold was never crossed and the whole run went
    through the hard-cap seam instead; lengthening the pause to 300 ms fixed
    that but left the speech run AT 2.0 s, so the cap was never crossed during
    speech either and removing the guard changed nothing.) The 3.0 s run also
    stays under the 4.0 s hard cap, so the two paths do not mask each other.
    """
    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=PositionalModel(), vad=vad, sync=True,
                      step_ms=500, min_window_ms=200, silence_commit_ms=100,
                      max_window_s=2.0, reset_window_s=1.0, turn_end_ms=60_000)
    seen = []
    broke_during_speech = []
    i = 0
    for _ in range(12):
        for _ in range(30):                    # 3 s of speech (6 whole words)
            items = asr.feed(positional_frame(i))
            seen.extend(words_of(items))
            if any(isinstance(it, TurnEnd) for it in items):
                broke_during_speech.append((asr.now_ms, asr._silence_ms))
            i += 1
        # A real pause: 300 ms, which is 9 VAD chunks, comfortably past
        # silence_commit_ms.
        vad.prob = 0.0
        for _ in range(3):
            seen.extend(words_of(asr.feed(silence_frame())))
        vad.prob = 1.0
    asr.close()
    assert not broke_during_speech, (
        "a window boundary landed inside speech: %r" % broke_during_speech)
    dupes = [w for w in set(seen) if seen.count(w) > 1]
    assert not dupes, "words emitted more than once: %s" % dupes[:5]
    assert len(seen) > 20, "sanity: the run must actually commit words"


def test_words_stay_in_order_across_a_window_reset():
    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=PositionalModel(), vad=vad, sync=True,
                      step_ms=500, min_window_ms=200, silence_commit_ms=100,
                      max_window_s=2.0, reset_window_s=1.0, turn_end_ms=60_000)
    seen = []
    i = 0
    for _ in range(8):
        for _ in range(20):
            seen.extend(words_of(asr.feed(positional_frame(i))))
            i += 1
        vad.prob = 0.0
        for _ in range(3):
            seen.extend(words_of(asr.feed(silence_frame())))
        vad.prob = 1.0
    idx = [int(w[1:]) for w in seen]
    assert idx == sorted(idx), "committed words must stay in spoken order"
    asr.close()


def test_empty_and_odd_length_frames_are_tolerated():
    vad = FakeVAD(1.0)
    asr = make(["x"], vad, step_ms=100)
    assert asr.feed(b"") == []
    asr.feed(b"\x01")            # odd byte count must not raise
    asr.close()


# --- word placement without model timestamps -------------------------------
def _two_words_over_a_gap(policy):
    """1 s of speech, a 2 s pause, 1 s of speech, then a forced commit.

    step_ms is set past the end of the run so the ONLY transcribe is the
    silence-forced one, which sees the whole pattern in one window.
    """
    vad = FakeVAD(1.0)
    asr = make(["alpha beta"], vad, step_ms=1_000_000, silence_commit_ms=2500,
               turn_end_ms=600_000, max_window_s=30.0, reset_window_s=30.0,
               word_time_policy=policy)
    out = []
    for prob, frames in ((1.0, 10), (0.0, 20), (1.0, 10), (0.0, 30)):
        vad.prob = prob
        for _ in range(frames):
            out.extend(i for i in asr.feed(frame()) if isinstance(i, Word))
    asr.close()
    return out


def test_word_times_avoid_the_pause_they_were_not_spoken_in():
    """The guarantee is not per-word alignment -- nothing here can give that --
    it is that no word time lands inside a silence the speaker was not talking
    through. Spread over wall time the seam falls at 2.0 s, in the middle of
    the pause; on aphasic speech, which is mostly pauses, that drift is what
    pushes a correctly recognised word into the neighbouring utterance."""
    PAUSE = (1100, 2900)          # the silent run, with a chunk of slack

    def in_pause(ms):
        return PAUSE[0] < ms < PAUSE[1]

    linear = _two_words_over_a_gap("linear")
    assert [w.text for w in linear] == ["alpha", "beta"]
    for policy in ("voiced", "incremental"):
        voiced = _two_words_over_a_gap(policy)
        assert [w.text for w in voiced] == ["alpha", "beta"], policy
        assert not any(in_pause(w.start_ms) or in_pause(w.end_ms)
                       for w in voiced),             (policy, [(w.text, w.start_ms, w.end_ms) for w in voiced])
    # the old policy is the thing being fixed: it puts the seam in the pause
    assert any(in_pause(w.end_ms) for w in linear),         [(w.text, w.start_ms, w.end_ms) for w in linear]


def test_word_time_policy_falls_back_when_the_vad_saw_no_speech_in_window():
    """A window with no voiced span (spans pruned, or an injected VAD that
    never reported speech) must still produce ordered times rather than an
    exception or an invented position."""
    vad = FakeVAD(1.0)
    asr = make(["alpha beta"], vad, step_ms=200)
    asr._voiced_spans = []
    out = [i for _ in range(6) for i in asr.feed(frame()) if isinstance(i, Word)]
    asr.close()
    assert [w.text for w in out] == ["alpha", "beta"]
    assert out[0].end_ms <= out[1].end_ms


# --- continuation context prompt -------------------------------------------
class ContextRecordingModel(ScriptedModel):
    """A ScriptedModel that also duck-types ContextModel's `set_context`.

    VerbatimASR uses an injected model directly when it already carries
    `set_context`, so this records exactly what the live path would put in the
    `<ctx> ... <ectx>` slot without loading any weights.
    """

    def __init__(self, hypotheses):
        super().__init__(hypotheses)
        self.contexts = []

    def set_context(self, text):
        self.contexts.append(text)


def test_context_is_off_by_default_and_no_context_is_ever_built():
    """Nothing ships on an unmeasured mechanism: the default path must not
    even construct a prompt."""
    vad = FakeVAD(1.0)
    asr = make(["one two", "one two"], vad, step_ms=100)
    asr.feed(frame(200))
    asr.feed(frame(200))
    assert asr._ctx_model is None
    assert asr._context_text() is None
    asr.close()


def test_context_holds_only_words_whose_audio_has_been_retired():
    """The invariant the whole design rests on. Words committed from the
    CURRENT window are still in the audio the next hypothesis sees; telling
    the model to continue past them would make the hypothesis lose its
    committed prefix and `_ncommit` would slice the wrong words out.
    Only a silence-forced full commit retires audio."""
    vad = FakeVAD(1.0)
    model = ContextRecordingModel(["one two", "one two", "three four"])
    asr = VerbatimASR(model=model, vad=vad, sync=True, min_window_ms=100,
                      step_ms=100, silence_commit_ms=100, turn_end_ms=100_000,
                      reset_window_s=0.1, context_prompt=True)
    asr.feed(frame(200))
    out = words_of(asr.feed(frame(200)))
    assert out == ["one", "two"], out
    # Agreement committed them, but the audio is still in the window, so they
    # are NOT context yet.
    assert asr._context_text() is None
    assert asr._retired_words == []
    vad.prob = 0.0                       # silence -> forced commit -> retire
    asr.feed(frame(200))
    asr.feed(frame(200))
    assert asr._retired_words == ["one", "two"]
    assert asr._context_text() == "one two"
    asr.close()


def test_context_is_capped_at_context_words():
    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=ContextRecordingModel(["x"]), vad=vad, sync=True,
                      min_window_ms=100, context_prompt=True, context_words=3)
    asr._window_words = ["a", "b", "c", "d", "e"]
    asr._promote_context()
    assert asr._retired_words == ["c", "d", "e"]
    assert asr._context_text() == "c d e"
    asr.close()


def test_context_survives_a_turn_end():
    """A turn end is a 2 s pause in one conversation, not a new recording.
    The continuation objective is about what was said before, so the context
    carries across -- and the turn's audio is gone, so it is safe to."""
    vad = FakeVAD(1.0)
    model = ContextRecordingModel(["hello there"])
    asr = VerbatimASR(model=model, vad=vad, sync=True, min_window_ms=100,
                      step_ms=100, silence_commit_ms=100, turn_end_ms=300,
                      context_prompt=True)
    asr.feed(frame(200))
    vad.prob = 0.0
    items = []
    for _ in range(8):
        items.extend(asr.feed(frame(200)))
    assert any(isinstance(i, TurnEnd) for i in items)
    assert asr._context_text() == "hello there"
    asr.close()


# --- window boundaries must sit in silence, even after a busy worker -------
class _ContentModel:
    """Reads back the "word ids" present in the audio it is handed.

    Each word is a block of a distinct DC value, and a word is reported
    whenever ANY of its samples are in the window -- which is what Whisper does
    with a word cut in half at a window boundary. The first call blocks so a
    test can hold the single ASR worker busy.
    """

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.calls = 0

    def release(self) -> None:
        self.gate.set()

    def transcribe(self, audio, **kw):
        self.calls += 1
        if self.calls == 1:
            self.gate.wait(10.0)
        ids, seen, out = [], set(), []
        for v in audio:
            i = int(round(float(v) * 32768.0))
            if i > 0 and (not ids or ids[-1] != i):
                ids.append(i)
        for i in ids:
            if i not in seen:
                seen.add(i)
                out.append("w%d" % i)
        return _Res(" ".join(out))


def word_frames(word_id: int, ms: int = 250, chunk_ms: int = 25):
    n = SR * chunk_ms // 1000
    buf = np.full(n, word_id, dtype="<i2").tobytes()
    return [buf] * (ms // chunk_ms)


def silence_frames(ms: int, chunk_ms: int = 25):
    n = SR * chunk_ms // 1000
    buf = np.zeros(n, dtype="<i2").tobytes()
    return [buf] * (ms // chunk_ms)


def test_a_forced_commit_armed_by_silence_is_cancelled_when_speech_resumes():
    """"Only a silence-forced full commit may retire audio" and "every window
    boundary must sit inside silence".

    _force_next is armed when the VAD's silence run crosses silence_commit_ms.
    If the single worker is busy at that instant the forced job cannot be
    submitted -- and the flag used to survive the speech that followed, so the
    forced commit landed MID-UTTERANCE: n_agree became len(words) with no
    agreement required, _drop_snapshot slid the window origin across a word,
    and the word straddling the seam was emitted twice. That is the exact
    duplicate-word failure the class exists to prevent.
    """
    vad = FakeVAD(1.0)
    model = _ContentModel()
    asr = VerbatimASR(model=model, vad=vad, sync=False, step_ms=200,
                      silence_commit_ms=200, min_window_ms=100,
                      reset_window_s=0.5, max_window_s=8.0, turn_end_ms=100000)
    retired_during_speech = []
    original = asr._drop_snapshot

    def spy(job):
        if asr._speech_prob >= 0.5 or asr._silence_ms < asr.silence_commit_ms:
            retired_during_speech.append((asr.now_ms, asr._speech_prob,
                                          asr._silence_ms))
        return original(job)

    asr._drop_snapshot = spy
    emitted: list[str] = []

    def pump(bufs, prob):
        vad.prob = prob
        for b in bufs:
            emitted.extend(words_of(asr.feed(b)))

    try:
        # 1.0 s of speech; the first transcribe blocks, holding the worker.
        for wid in (1, 2, 3, 4):
            pump(word_frames(wid), 1.0)
        assert asr._future is not None, "the worker must be busy for this repro"

        # A 300 ms pause crosses silence_commit_ms and arms the forced commit,
        # but _maybe_submit cannot submit it: one transcribe at a time.
        pump(silence_frames(300), 0.0)
        assert asr._force_next is True

        # Speech resumes after 300 ms -- far below the 672 ms median pause
        # INSIDE an aphasic word-search utterance, so this is the common case,
        # not a corner. The arming must be cancelled with the silence.
        for wid in (5, 6):
            pump(word_frames(wid), 1.0)
        assert asr._force_next is False, (
            "a commit armed by a silence that has ended is still pending; it "
            "will retire audio mid-speech")

        model.release()
        deadline = time.time() + 10.0
        while asr._future is not None and not asr._future.done() and time.time() < deadline:
            time.sleep(0.01)
        for wid in (7, 8, 9, 10, 11, 12):
            pump(word_frames(wid), 1.0)
            time.sleep(0.05)
        time.sleep(0.4)
        pump(word_frames(13), 1.0)
        time.sleep(0.4)
        pump(word_frames(14), 1.0)
    finally:
        asr.close()

    dupes = sorted({w for w in emitted if emitted.count(w) > 1})
    assert not retired_during_speech, (
        "window retired while the VAD said speech: %r" % retired_during_speech)
    assert not dupes, "words emitted twice across the seam: %r (%s)" % (
        dupes, " ".join(emitted))


def test_the_hard_cap_break_never_swallows_the_uncommitted_tail():
    """The counterpart, and the reason the hard-cap branch arms a forced commit
    instead of breaking straight away.

    The hard cap is the last resort for speech with no measurable pause in it.
    It ends the turn, and _reset_turn DISCARDS the window -- so any word still
    sitting uncommitted behind LocalAgreement at that instant is gone from the
    transcript for good. Arming a forced commit and breaking one cycle later
    flushes them first. (The arming is deliberately NOT cancelled when speech
    continues, unlike a silence-armed one: there is no silence coming, that is
    the whole situation.) 6 s of unbroken speech, 12 words, a 2 s hard cap:
    the seam may re-propose a word -- that cost is documented -- but it may
    never LOSE one.
    """
    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=PositionalModel(), vad=vad, sync=True,
                      step_ms=500, min_window_ms=200, silence_commit_ms=100,
                      max_window_s=1.0, reset_window_s=1.0, turn_end_ms=60_000)
    seen = []
    for i in range(60):               # 6 s, never a silent frame
        seen.extend(words_of(asr.feed(positional_frame(i))))
    asr.close()
    idx = sorted({int(w[1:]) for w in seen})
    assert idx, "sanity: the run must commit words"
    assert idx[0] == 0 and idx == list(range(idx[0], idx[-1] + 1)), (
        "the hard-cap turn break swallowed words the speaker said: %r" % (seen,))
    assert len(idx) >= 8, "sanity: most of a 12-word run must survive: %r" % (seen,)


# --- the pause trigger must measure the speaker ---------------------------
_VOCAB = ("i went to the shop and bought some of those little green things for "
          "dinner with my wife last night on the way home from work").split()


class _RateModel:
    """A speaker at 2.5 words/s: one word per 400 ms of the window it is given."""

    def transcribe(self, audio, **kw):
        n = max(1, int(audio.size / SR * 2.5))
        return _Res(" ".join(_VOCAB[:min(n, len(_VOCAB))]))


def test_silence_ticks_carry_the_moment_speech_stopped():
    """The VAD knows exactly when the speaker stopped and it is free. Without
    it on the wire the detector can only measure from the last COMMITTED word,
    which trails by the commit lag."""
    vad = FakeVAD(1.0)
    asr = make(["hello there"], vad, step_ms=10_000, tick_ms=100)
    asr.feed(frame(400))
    stopped = asr._last_voiced_ms
    vad.prob = 0.0
    ticks = [i for i in asr.feed(frame(400)) if isinstance(i, SilenceTick)]
    asr.close()
    assert ticks, "silence must produce ticks"
    assert all(t.speech_end_ms == stopped for t in ticks), (
        [(t.at_ms, t.speech_end_ms) for t in ticks])
    assert all(t.at_ms > t.speech_end_ms for t in ticks)


def test_a_tick_before_any_speech_reports_no_stop_time():
    """-1 is "the VAD has never heard speech", which is not a stop -- and
    passing it on would make every pause look 1 s longer than it is."""
    vad = FakeVAD(0.0)
    asr = make(["x"], vad, step_ms=10_000, tick_ms=100)
    ticks = [i for i in asr.feed(frame(500)) if isinstance(i, SilenceTick)]
    asr.close()
    assert ticks and all(t.speech_end_ms is None for t in ticks)


def test_pause_trigger_measures_the_speaker_not_the_commit_lag():
    """The end-to-end property, against the real detector.

    Ticks stop DURING speech, but that alone does not make the trigger measure
    the speaker: the first ticks after the speaker stops used to be compared
    against the last COMMITTED word's end_ms while LocalAgreement was still
    holding the tail of the utterance back. The detector saw (real pause +
    commit lag) and fired a stall ~200 ms after the wearer stopped talking --
    Echo interrupting every time the wearer breathes between clauses.
    """
    vad = FakeVAD(1.0)
    asr = VerbatimASR(model=_RateModel(), vad=vad, sync=True, step_ms=700,
                      silence_commit_ms=700, turn_end_ms=2000,
                      min_window_ms=700, reset_window_s=3.5, max_window_s=8.0)
    det = StallDetector(pause_ms=1300)
    fires: list[tuple[str, int]] = []

    def pump(bufs, prob):
        vad.prob = prob
        for b in bufs:
            for it in asr.feed(b):
                if isinstance(it, Word):
                    det.observe_word(it)
                elif isinstance(it, SilenceTick):
                    ev = det.observe_silence(it.at_ms, it.speech_end_ms)
                    if ev is not None:
                        fires.append((ev.trigger, ev.at_ms))
                elif isinstance(it, TurnEnd):
                    det.reset()

    try:
        pump(silence_frames(2480, 40), 1.0)
        stopped_at = asr.now_ms
        pump(silence_frames(1400, 40), 0.0)
    finally:
        asr.close()

    pause_fires = [at for kind, at in fires if kind == "pause"]
    assert pause_fires, "expected the pause trigger to fire eventually"
    real_pause = pause_fires[0] - stopped_at
    assert real_pause >= det.pause_ms, (
        "stall fired %d ms after the speaker stopped, but pause_ms is %d -- "
        "the trigger measured the commit lag, not the speaker"
        % (real_pause, det.pause_ms))


# --- feed() must never block the event loop -------------------------------
class GatedModel:
    """Returns scripted hypotheses; blocks from call `block_from` onward."""

    def __init__(self, hypotheses, block_from=10_000):
        self.hypotheses = list(hypotheses)
        self.block_from = block_from
        self.gate = threading.Event()
        self.calls = 0

    def release(self):
        self.gate.set()

    def transcribe(self, audio, **kw):
        i, self.calls = self.calls, self.calls + 1
        if i >= self.block_from:
            self.gate.wait(20.0)
        return _Res(self.hypotheses[min(i, len(self.hypotheses) - 1)])


def _drain(asr, timeout=5.0):
    deadline = time.time() + timeout
    while asr._future is not None and not asr._future.done() and time.time() < deadline:
        time.sleep(0.005)


def test_a_turn_end_never_blocks_the_event_loop():
    """feed() runs on the asyncio loop that also serves the PCM socket
    and the UI. It used to drain the worker with Future.result(timeout=5.0) at
    every turn end, paying the transcribe latency (300-600 ms typical, five
    seconds worst case) on that loop -- stalling the audio stream and the UI at
    once. The turn is now declared one feed() later instead."""
    vad = FakeVAD(1.0)
    model = GatedModel(["one two", "one two", "one two three"], block_from=2)
    asr = VerbatimASR(model=model, vad=vad, sync=False, step_ms=200,
                      silence_commit_ms=200, min_window_ms=100,
                      turn_end_ms=600, reset_window_s=10.0, max_window_s=8.0)
    out = []
    try:
        for _ in range(3):            # speech: two hypotheses agree, words ship
            out.extend(asr.feed(frame(200)))
            _drain(asr)
            out.extend(asr.feed(frame(100)))
        assert words_of(out) == ["one", "two"], words_of(out)

        vad.prob = 0.0                # silence: the forced commit blocks
        asr.feed(frame(200))
        _drain(asr, 0.3)
        assert asr._future is not None and not asr._future.done()

        t0 = time.perf_counter()
        held = []
        for _ in range(6):            # well past turn_end_ms
            held.extend(asr.feed(frame(200)))
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, "feed() blocked on the ASR worker for %.2f s" % elapsed
        assert not any(isinstance(i, TurnEnd) for i in held), (
            "the turn cannot be declared before the worker's words are in it")

        model.release()
        _drain(asr)
        late = asr.feed(frame(200))
        out.extend(late)
    finally:
        asr.close()
        model.release()
    assert any(isinstance(i, TurnEnd) for i in late), (
        "the turn must be declared once the worker returns")
    assert words_of(out) == ["one", "two", "three"], (
        "the drained tail belongs to THIS turn's transcript: %r" % words_of(out))


# --- the interim line must never repeat a committed word -------------------
def test_a_retracted_hypothesis_never_shows_a_committed_word_as_interim():
    """The commit cursor is max(_ncommit, n_agree), but the interim slice used
    to start at n_agree. When a hypothesis retracts (n_agree < _ncommit) the
    words in between are already out as final AND were shown again as grey
    interim text -- the same word on screen twice."""
    vad = FakeVAD(1.0)
    asr = make(["one two three", "one two three", "one deux three"], vad,
               step_ms=100, silence_commit_ms=100, turn_end_ms=60_000)
    final = []
    for _ in range(3):
        final.extend(words_of(asr.feed(frame(200))))
    interim = asr.interim_text.split()
    asr.close()
    assert final == ["one", "two", "three"], final
    assert not [w for w in interim if w in final], (
        "already-final words shown again as interim: %r" % interim)


# --- guards that a mutation must not be able to remove ---------------------
def test_a_short_window_is_not_retired_by_a_forced_commit():
    """reset_window_s is not 0 on purpose: Whisper has a 30 s receptive field
    and degrades badly on short fragments (an earlier bench scored 0.217 on 1 s
    clips purely because of it). A forced commit on a window shorter than
    reset_window_s must commit the words and keep the audio."""
    vad = FakeVAD(1.0)
    asr = make(["one two"], vad, step_ms=1_000_000, silence_commit_ms=100,
               reset_window_s=5.0, turn_end_ms=1_000_000)
    asr.feed(frame(300))
    vad.prob = 0.0
    out = words_of(asr.feed(frame(200)))
    assert out == ["one", "two"], out
    assert asr._window_start_ms == 0, (
        "a %.2f s window was retired below the %.2f s reset threshold"
        % (asr._window.size / SR, asr.reset_window / SR))
    assert asr._window.size > 0
    asr.close()


def test_the_forced_commit_arms_the_moment_silence_reaches_the_threshold():
    """Exactly at silence_commit_ms, not one VAD chunk later. The boundary is
    the whole knob: 700 ms is a measured trade of WER against commit lag, and
    an off-by-one chunk silently moves it by 32 ms in the direction that costs
    latency at the stall moment."""
    vad = FakeVAD(1.0)
    # 96 ms == exactly three 32 ms VAD chunks.
    asr = make(["one two"], vad, step_ms=1_000_000, silence_commit_ms=96,
               turn_end_ms=1_000_000, reset_window_s=1_000.0)
    asr.feed(frame(200))
    assert asr._silence_ms == 0
    vad.prob = 0.0
    out = words_of(asr.feed(frame(96)))
    asr.close()
    assert asr._silence_ms == 96, asr._silence_ms
    assert out == ["one", "two"], (
        "silence reached silence_commit_ms exactly and no commit was armed: %r"
        % (out,))


def test_incremental_word_times_are_floored_at_the_last_committed_word():
    """"incremental" is "voiced" plus one thing: the slice is floored at the
    end of the last word already committed, because whatever step committed it
    had already seen the audio it was spoken in. That floor is the entire
    measured difference -- APROCSA WER 0.398 for "voiced" against 0.375 for
    "incremental" at the shipped silence_commit_ms=700 -- so without it the
    policy collapses into the one it beat."""
    hyps = ["alpha", "alpha", "alpha beta gamma", "alpha beta gamma"]

    def run(policy):
        vad = FakeVAD(1.0)
        asr = make(list(hyps), vad, step_ms=100, silence_commit_ms=100_000,
                   turn_end_ms=1_000_000, reset_window_s=1_000.0,
                   word_time_policy=policy)
        out = []
        for _ in range(4):
            out.extend(i for i in asr.feed(frame(200)) if isinstance(i, Word))
        asr.close()
        return out

    inc = run("incremental")
    assert [w.text for w in inc] == ["alpha", "beta", "gamma"], inc
    for a, b in zip(inc, inc[1:]):
        assert b.start_ms >= a.end_ms, (
            "a later commit was placed before the end of an earlier one: %r"
            % [(w.text, w.start_ms, w.end_ms) for w in inc])
    # and that really is the difference: "voiced" spreads the same words from
    # the top of the window and puts beta back inside alpha.
    voiced = run("voiced")
    assert any(b.start_ms < a.end_ms for a, b in zip(voiced, voiced[1:])), (
        "the two policies became indistinguishable: %r"
        % [(w.text, w.start_ms, w.end_ms) for w in voiced])


def test_a_job_that_outlived_its_window_emits_nothing():
    """A turn boundary (or a snapshot drop) moves the window origin while a
    transcribe is in flight. That job's offsets no longer mean anything, and
    its hypothesis covers audio that is gone -- emitting it would replay words
    from the previous window into the new one.

    Exercised directly: every other test in this file runs sync=True, where a
    result is consumed in the same feed() that submitted it, so this guard is
    never reached."""
    vad = FakeVAD(1.0)
    asr = make(["one two"], vad, step_ms=100, turn_end_ms=1_000_000)
    asr.feed(frame(200))
    hyp = [_W("one", None, None), _W("two", None, None)]

    live = _Job(audio=asr._window.copy(), window_start_ms=asr._window_start_ms,
                force=True)
    stale = _Job(audio=asr._window.copy(),
                 window_start_ms=asr._window_start_ms - 1000, force=True)
    assert asr._consume((stale, list(hyp)), forced=True) == [], (
        "a job from a retired window must emit nothing")
    assert words_of(asr._consume((live, list(hyp)), forced=True)) == ["one", "two"], (
        "sanity: the same payload on the CURRENT window does emit")
    asr.close()
