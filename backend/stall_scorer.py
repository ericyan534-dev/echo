"""One calibrated score instead of five independent binary triggers.

WHY
---
The detector fires when ANY of pause / filler / hedge / block / prolongation /
sound_rep / word_rep fires. Each is a threshold on one signal, so the only way
to trade recall against interruptions is to move every threshold at once or to
add a refractory that throws away good fires along with bad ones. Measured on
real aphasic speech, that gave 0.843 recall at 0.611 false-alarm and 19
fires/minute -- and the refractory sweep just slid down one flat curve.

The signals are not independent evidence of the same thing, and they are not
equally strong. A retracing plus a 1.5 s pause plus a cut-off word is a
different event from one "um". Aphasic speech is DENSE with weak signals, which
is exactly why any-of firing degenerates there: something is nearly always
true.

So: extract the signals as continuous features, weight them, and fire on the
total. That changes the shape of the curve rather than the point on it. The
weights are fitted by logistic regression on a tune split of speakers and
reported on a held-out split (eval/fit_stall_scorer.py) -- never on all six.

Fires only on the WEARER's evidence. Every feature reads timeline words that
pass `is_wearer`, and acoustic events are gated upstream.

WHAT THIS DELIBERATELY DOES NOT DO
----------------------------------
It does not replace the debounce or the refractory; an unresolved search still
suppresses a second fire, and the minimum gap between suggestions is a product
constraint, not a modelling one. It only decides WHETHER this moment looks like
a word search.
"""
from __future__ import annotations

import math

from .timeline import FILLERS, norm

# Order is the contract: fitted weight vectors are stored against it, so a
# reordering silently repoints every weight. Anything appended goes at the END.
FEATURES = [
    "pause_log",        # log1p(ms of silence since the wearer's last word)
    "fillers_recent",   # filled pauses among the last few wearer words
    "hedge",            # "the thing", "what's it called", ...
    "fragments",        # cut-off words -- "f-", "re-" -- verbatim ASR only
    "repetition",       # immediate word/phrase repetition
    "acou_block",       # strongest recent Block probability
    "acou_prolong",
    "acou_rep",         # max of sound_rep and word_rep
    "acou_filler",
    "content_count",    # how much of the utterance exists to predict from
]

RECENT_WORDS = 6
ACOUSTIC_WINDOW_MS = 2000

# Fitted on the TUNE speakers by eval/fit_stall_scorer.py. Shipped rather than
# loaded from a file so a checkout runs the measured configuration; rerun the
# fitter to change them and paste the block it prints.
DEFAULT_WEIGHTS = {
    "pause_log": -0.5934,
    "fillers_recent": 3.1178,
    "hedge": 0.8322,
    "fragments": 3.4887,
    "repetition": -0.6994,
    "acou_block": 0.0,
    "acou_prolong": 0.9759,
    "acou_rep": 0.1388,
    "acou_filler": 2.8021,
    "content_count": -0.9429,
}
DEFAULT_BIAS = 0.1657

_FRAGMENT_SUFFIX = "-"


class StallScorer:
    """Accumulates evidence; answers "does this instant look like a search?"."""

    def __init__(self, weights: dict | None = None, bias: float = DEFAULT_BIAS,
                 threshold: float = 0.5, hedges: tuple = ()) -> None:
        self.weights = dict(weights or DEFAULT_WEIGHTS)
        self.bias = bias
        self.threshold = threshold
        self.hedges = hedges
        self._acoustic: list = []          # (at_ms, kind, confidence)

    def reset(self) -> None:
        self._acoustic = []

    def observe_acoustic(self, kind: str, at_ms: int, confidence: float) -> None:
        self._acoustic.append((at_ms, kind, confidence))
        # Bounded by time, not by count: a burst of events inside the window is
        # exactly the evidence worth keeping, and a fixed-length ring would
        # drop the earliest of a burst rather than the oldest in time.
        cut = at_ms - ACOUSTIC_WINDOW_MS * 3
        if len(self._acoustic) > 64:
            self._acoustic = [e for e in self._acoustic if e[0] >= cut]

    def _acou(self, now_ms: int, kinds: tuple) -> float:
        best = 0.0
        for at, kind, conf in self._acoustic:
            if kind in kinds and 0 <= now_ms - at <= ACOUSTIC_WINDOW_MS:
                best = max(best, float(conf))
        return best

    def features(self, timeline, now_ms: int) -> dict:
        words = timeline.current_utterance()
        if not words:
            return {k: 0.0 for k in FEATURES}
        texts = [w.text for w in words[-RECENT_WORDS:]]
        lowered = [norm(t) for t in texts]

        pause = max(0, now_ms - words[-1].end_ms)
        frag = "".join(texts)
        hedge = 0.0
        nf = norm(timeline.utterance_text())
        for h in self.hedges:
            if nf.endswith(h):
                hedge = 1.0
                break

        rep = 0.0
        for i in range(len(lowered) - 1):
            if lowered[i] and lowered[i] == lowered[i + 1]:
                rep = 1.0
                break

        return {
            "pause_log": math.log1p(pause) / 8.0,
            "fillers_recent": sum(1 for t in lowered if t in FILLERS) / 3.0,
            "hedge": hedge,
            # A trailing hyphen is CrisperWhisper's cut-off-word marker and the
            # single most specific textual sign of a block. It only exists at
            # all because the transcript is verbatim.
            "fragments": min(3, sum(1 for t in texts if t.endswith(_FRAGMENT_SUFFIX))) / 3.0,
            "repetition": rep,
            "acou_block": self._acou(now_ms, ("block",)),
            "acou_prolong": self._acou(now_ms, ("prolongation",)),
            "acou_rep": self._acou(now_ms, ("sound_rep", "word_rep")),
            "acou_filler": self._acou(now_ms, ("filler",)),
            "content_count": min(12, timeline.content_count()) / 12.0,
        }

    def score(self, timeline, now_ms: int) -> float:
        f = self.features(timeline, now_ms)
        z = self.bias + sum(self.weights.get(k, 0.0) * f[k] for k in FEATURES)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))

    def fires(self, timeline, now_ms: int) -> bool:
        return self.score(timeline, now_ms) >= self.threshold
