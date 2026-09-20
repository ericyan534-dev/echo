"""Context-biased decoding: give the recogniser the words that came before.

WHY
---
CrisperWhisper2 was TRAINED with a continuation objective. Its longform
strategy decodes each 30 s chunk with the previous chunk's last few words in
the decoder prompt as ``<ctx> ... <ectx>`` -- so this is not generic Whisper
``initial_prompt`` conditioning, which is the thing famous for hallucination
loops; it is the prompt format the checkpoint was fitted to.

The public ``model.transcribe()`` does NOT expose that prompt on short audio.
``context`` is only reachable from inside the longform strategies, which never
run on the windows Echo streams (every window is under the 30 s encoder field,
so the short path is taken and the context slot is left empty).

WHAT THIS IS
------------
The thinnest possible reach into the package to close that gap without forking
it: a PromptBuilder subclass that injects a caller-supplied context string, and
a model facade that calls the library's own ``_transcribe_short`` with it. The
decode path -- greedy generate, n-gram repair, coverage/temperature fallback,
word-timing attention -- is the library's, unmodified. Only the prompt differs.

    m = ContextModel("turbo")
    m.set_context("i went to the store and")
    res = m.transcribe(pcm, sr=16000, mode="verbatim")

ONE MISMATCH WITH TRAINING, AND IT IS THE WHOLE QUESTION
--------------------------------------------------------
In the longform strategy the chunks OVERLAP (30 s window, 26 s stride) and
``context_words`` is deliberately sized so the context text spans that overlap
-- the model re-hears the context words at the head of the next chunk and
continues past them. Echo's stream has no overlap at all: a window is retired
only inside silence and the next window starts on audio nobody has decoded. So
the context here describes audio the model canNOT hear, which is a different
condition from the one the objective was fitted on.

It was measured rather than assumed, and IT LOSES. On APROCSA, same audio,
same scorer (`eval/bench_asr_context_prompt.py`):

    offline, 4 s overlap      no context 0.3909   with context 0.2876
    offline, no overlap       no context 0.2792   with context 0.2918
    Echo streaming            no context 0.3752   with context 0.4350

The prompt's whole offline value is deduplicating the overlap -- without it,
32 of 72 chunks open by transcribing the previous chunk's tail a second time.
Take the overlap away and the prompt is a small loss; put it in Echo's stream,
where the windows never overlap, and it is a 6-point loss on 5 of 6 speakers,
because the model does what it was trained to do and SKIPS the context words
at the head of the window -- except that here nobody has transcribed that
audio, so the skip is a deletion:

    context   "later I couldn't walk for a [UM] I think about four months"
    no ctx    "Three or four months [UH] but [UM]"
    with ctx  "[UH] but [UM]"

The feared failure -- echoing the prompt back, looping -- did not happen at
all (echo 0 in both arms, loops 57 -> 43). The failure is the opposite: the
decoder says LESS. This module stays because the measurement has to be
re-runnable, and `VerbatimASR(context_prompt=True)` stays off.

Something would have to change for this to be worth revisiting: giving the
stream a real overlap (retiring less audio than the window decoded) so the
context describes audio the model can actually hear. That is the condition the
objective was fitted on, and it is not a parameter -- it breaks the invariant
that a window boundary always sits in silence, and it re-opens the
double-emission bug documented in `VerbatimASR._consume`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Result:
    """Duck-types the fields `VerbatimASR._transcribe` reads."""
    text: str
    words: list = field(default_factory=list)


def ctx_prompt_builder(engine, language: str, context: str | None):
    """A PromptBuilder whose verbatim()/intended() always carry `context`.

    Subclassing rather than calling `_build` directly keeps the tag order,
    the decoder prefix and the tokenizer exactly as the package builds them --
    the order is load-bearing (the package notes it must match training).
    """
    from crisperwhisper.prompt import PromptBuilder

    class _CtxPromptBuilder(PromptBuilder):
        def verbatim(self, hotwords=None, context=None):
            return super().verbatim(hotwords=hotwords, context=context or self.ctx)

        def intended(self, hotwords=None, context=None):
            return super().intended(hotwords=hotwords, context=context or self.ctx)

    pb = _CtxPromptBuilder(engine, language=language)
    pb.ctx = context or None
    return pb


class ContextModel:
    """CrisperWhisper with a settable continuation context.

    Wraps an already-loaded model when one is passed, and otherwise shares the
    process-wide model cache with the live path, so running this next to
    `VerbatimASR` loads one copy of the weights, not two.
    """

    def __init__(self, model_name: str = "turbo", device: str = "auto",
                 compute_type: str = "float16", context_words: int = 12,
                 model=None) -> None:
        if model is None:
            from .verbatim import get_model

            model = get_model(model_name, device, compute_type)
        self._model = model
        self.context_words = context_words
        self._context: str | None = None
        self.n_calls = 0
        self.n_with_context = 0

    def set_context(self, text: str | None) -> None:
        """Words spoken BEFORE the audio about to be transcribed.

        Trimmed to `context_words`: the training-time context was a handful of
        words, and a long prompt is exactly the condition under which Whisper
        starts transcribing its own prompt instead of the audio.
        """
        if not text:
            self._context = None
            return
        toks = text.split()[-self.context_words:]
        self._context = " ".join(toks) or None

    @property
    def context(self) -> str | None:
        return self._context

    # `VerbatimASR` calls this with these exact keywords.
    def transcribe(self, audio: np.ndarray, sr: int = 16000, language: str = "en",
                   mode: str = "verbatim", word_timestamps: bool = False,
                   **kw):
        from crisperwhisper.model import SHORT_THRESHOLD_S, CrisperWhisperModel

        self.n_calls += 1
        ctx = self._context
        if ctx:
            self.n_with_context += 1
        # Over 30 s the library's own longform strategies apply (and already
        # carry their own rolling context); nothing to add, so defer to them.
        if len(audio) / sr > SHORT_THRESHOLD_S or not ctx:
            return self._model.transcribe(audio, sr=sr, language=language,
                                          mode=mode, word_timestamps=word_timestamps)

        engine = self._model._engine
        pb = ctx_prompt_builder(engine, language, ctx)
        audio = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            raise ValueError("ContextModel expects 16 kHz audio")
        text, words = CrisperWhisperModel._transcribe_short(
            engine, pb, audio, mode=mode, hotwords=None,
            max_new_tokens=256, hallucination_mitigation=True,
            word_timestamps=word_timestamps, alignment_heads=None,
            temperature_fallback=True, suppress_tokens=None,
        )
        return _Result(text=text, words=words or [])
