"""SpeakerGate -- "is this the wearer talking?" from proximity evidence only.

Silero VAD (in `stream.py`) answers *is this speech*. It never answers *whose*.
Without a second opinion, the acoustic stall channel fires on the conversation
partner's fillers as readily as on the wearer's, and Echo prompts the wrong
person. This gate is that second opinion, built from two cheap physical cues
that both follow from the wearer's mouth being closer to the mic than anyone
else's:

  1. **Level.** Sound pressure falls ~6 dB per doubling of distance. The
     wearer's voiced frames sit near the top of the recent level distribution;
     a talker across the table sits below it.
  2. **Spectral tilt.** Distance costs high frequencies (air absorption,
     off-axis mic response, and the fact that a distant talker is usually not
     aimed at the mic). Frames with the highs stripped are further away.

Level is the primary evidence; tilt can only *soften* a level score, never
create a suppression on its own (see `tilt_weight`).

HONEST LIMITATION -- read this before believing any number this module returns
--------------------------------------------------------------------------
The proximity argument is strong for the **DJI Mic 2S lav**, a capsule on the
speaker's collar at roughly 5 cm: a partner at 1 m is ~26 dB down, which is far
outside normal speech-level variation. It is **materially weaker for a laptop mic on a table**, where the
wearer and a bystander may sit at similar distance and similar angle; there the
separation can collapse to a few dB, which is inside the range a single speaker
covers just by changing loudness. **The laptop mic is Echo's primary demo path**,
so on that path this gate should be read as a soft prior, not a speaker ID. It
is not diarization, it does not model voices, and it cannot separate two people
at the same distance -- with equal-distance talkers it will admit both, which is
the intended failure direction.

The gate is validated on synthetic mixes only. It has never been tested in a
real room with two real speakers.

FAIL-OPEN CONTRACT (the part that must not be "optimized" away)
---------------------------------------------------------------
`observe()` returns `float | None`. `None` means "I cannot tell" and must never
cause suppression anywhere downstream. Wrongly muting the wearer -- a person
who already struggles to be heard -- is a far worse failure than admitting a
bystander. Concretely, `None` is returned when:

  * the baseline is not calibrated yet (cold start): unknown, never "quiet";
  * the frame is not speech (that is the VAD's question, not this one);
  * the frame is digitally silent / non-finite (also guards log10(0));
  * the gate has been suppressing continuously for `max_suppress_ms`, which is a
    hard cap: a wearer who moves the mic, leans back, or tires mid-conversation
    re-anchors the baseline instead of being muted indefinitely.

That last one is not a garnish, it is the main adaptation mechanism. The rolling
window alone is slow: a 0.75 percentile does not move until ~75% of the window
has been replaced, i.e. ~7.5 s at the defaults, and 7.5 s of a muted wearer is
already a failed conversation. `max_suppress_ms` bounds that at 3 s. The price
is explicit: a bystander who holds the floor for longer than `max_suppress_ms`
stops being gated. That trade is chosen deliberately in the wearer's favour.

Two deliberate design choices worth defending:

  * **All voiced frames update the baseline, bystanders included.** Updating only
    from frames the gate already believes are the wearer is a latch: one loud
    transient sets a high baseline that nothing can ever bring down, and the
    wearer is muted for the rest of the session. A high percentile over a short
    rolling window gives bystander-resistance without that positive feedback.
  * **Thresholds are constructor arguments, not config.** `backend/config.py` is
    wired by the integrator; this module ships defensible defaults.

Pure torch + stdlib. No new dependency.
"""
from __future__ import annotations

import math
from collections import deque
from statistics import median

import torch

from .features import SR

# Tilt bands (Hz). The low band covers the voiced fundamental and first formant
# region; the high band the fricative/consonant energy that distance eats first.
TILT_LOW_HZ = (100.0, 1000.0)
TILT_HIGH_HZ = (2000.0, 6500.0)

_EPS = 1e-12


class SpeakerGate:
    """Adaptive proximity gate. One instance per audio stream (it is stateful).

    Args:
      speech_prob_min: VAD probability at or above which a frame counts as
        speech. Matches `stream.py`'s own 0.5 gate.
      calibration_frames: voiced frames required before any score is produced.
        Below this the answer is `None` (fail open). 20 frames = ~1 s at 50 ms.
      history_frames: rolling voiced-frame window backing the baseline.
        200 = ~10 s at 50 ms. This is also the adaptation time constant.
      baseline_percentile: percentile of the window's dBFS taken as "the
        wearer's level". High (0.75) so a bystander in the window cannot drag it
        down; not 1.0, because a single transient must not define the baseline.
      near_margin_db: frames within this many dB of the baseline score 1.0.
        Absorbs ordinary speech-level variation.
      far_margin_db: frames this far below the baseline score 0.0. 12 dB is
        ~4x distance, comfortably outside one speaker's own dynamic range.
      tilt_weight: maximum fraction of the score the tilt term may remove. At
        0.3 a full tilt penalty leaves 0.7, so tilt ALONE never suppresses.
      tilt_range_db: high/low band ratio drop, in dB, that counts as a full
        tilt penalty.
      low_conf: the gate's own notion of "I am saying no", used only to drive
        `max_suppress_ms`. Keep it >= the consumer's threshold so the escape
        hatch fires no later than actual suppression would.
      max_suppress_ms: hard cap on continuous suppression. On hitting it the
        gate returns `None` and re-anchors the baseline to the recent level.
      silence_floor_dbfs: below this a frame is treated as silence -> `None`.
      sample_rate: PCM sample rate, for the tilt band edges.
    """

    def __init__(
        self,
        *,
        speech_prob_min: float = 0.5,
        calibration_frames: int = 20,
        history_frames: int = 200,
        baseline_percentile: float = 0.75,
        near_margin_db: float = 3.0,
        far_margin_db: float = 12.0,
        tilt_weight: float = 0.3,
        tilt_range_db: float = 6.0,
        low_conf: float = 0.5,
        max_suppress_ms: int = 3000,
        silence_floor_dbfs: float = -70.0,
        sample_rate: int = SR,
    ) -> None:
        if far_margin_db <= near_margin_db:
            raise ValueError("far_margin_db must exceed near_margin_db")
        self.speech_prob_min = speech_prob_min
        self.calibration_frames = max(1, int(calibration_frames))
        self.history_frames = max(self.calibration_frames, int(history_frames))
        self.baseline_percentile = min(max(baseline_percentile, 0.0), 1.0)
        self.near_margin_db = near_margin_db
        self.far_margin_db = far_margin_db
        self.tilt_weight = min(max(tilt_weight, 0.0), 1.0)
        self.tilt_range_db = max(tilt_range_db, 1e-6)
        self.low_conf = low_conf
        self.max_suppress_ms = max_suppress_ms
        self.silence_floor_dbfs = silence_floor_dbfs
        self.sample_rate = sample_rate

        self._hist: deque[tuple[float, float]] = deque(maxlen=self.history_frames)
        self._suppress_ms = 0.0
        self._band_cache: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    @property
    def baseline_dbfs(self) -> float | None:
        """Current level baseline, or None while uncalibrated. Observability
        only -- nothing should gate on this."""
        if len(self._hist) < self.calibration_frames:
            return None
        return self._percentile([d for d, _ in self._hist], self.baseline_percentile)

    @property
    def calibrated(self) -> bool:
        return len(self._hist) >= self.calibration_frames

    # ------------------------------------------------------------------
    def observe(self, frame: torch.Tensor, speech_prob: float) -> float | None:
        """Score one frame. Returns wearer confidence in 0..1, or None for
        "cannot tell". None must never suppress -- see the module docstring."""
        if speech_prob is None or speech_prob < self.speech_prob_min:
            return None                      # the VAD's question, not ours
        feats = self._features(frame)
        if feats is None:
            return None                      # silence / non-finite / empty
        dbfs, tilt_db = feats

        if len(self._hist) < self.calibration_frames:
            self._hist.append((dbfs, tilt_db))
            return None                      # cold start: unknown, not "quiet"

        # Baseline is the recent PAST, so a frame never scores against itself.
        levels = [d for d, _ in self._hist]
        baseline_db = self._percentile(levels, self.baseline_percentile)
        baseline_tilt = median(t for _, t in self._hist)
        self._hist.append((dbfs, tilt_db))

        delta = dbfs - baseline_db
        if delta >= -self.near_margin_db:
            level = 1.0                       # at or above baseline: the wearer
        elif delta <= -self.far_margin_db:
            level = 0.0
        else:
            level = ((delta + self.far_margin_db)
                     / (self.far_margin_db - self.near_margin_db))

        # Tilt only ever penalizes, and only up to tilt_weight of the score.
        penalty = min(max((baseline_tilt - tilt_db) / self.tilt_range_db, 0.0), 1.0)
        conf = level * (1.0 - self.tilt_weight * penalty)
        conf = min(max(conf, 0.0), 1.0)

        # Hard cap on continuous suppression: re-anchor rather than mute forever.
        frame_ms = 1000.0 * self._n(frame) / self.sample_rate
        if conf < self.low_conf:
            self._suppress_ms += frame_ms
        else:
            self._suppress_ms = 0.0
        if self._suppress_ms >= self.max_suppress_ms:
            self._suppress_ms = 0.0
            recent = list(self._hist)[-self.calibration_frames:]
            self._hist = deque(recent, maxlen=self.history_frames)
            return None                       # "I cannot tell" -> fail open

        return conf

    def reset(self) -> None:
        """Drop the baseline (e.g. mic changed). The next frames fail open."""
        self._hist.clear()
        self._suppress_ms = 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def _n(frame: torch.Tensor) -> int:
        try:
            return int(frame.numel())
        except AttributeError:
            return 0

    def _features(self, frame: torch.Tensor) -> tuple[float, float] | None:
        """(dBFS, tilt_dB) for one frame, or None if the frame is unusable."""
        if frame is None:
            return None
        x = frame.reshape(-1)
        if x.numel() == 0:
            return None
        if not x.dtype.is_floating_point:
            x = x.float()
        if not bool(torch.isfinite(x).all()):
            return None                      # NaN/inf: unknown, not suppressed
        rms = float(x.pow(2).mean().sqrt())
        if not math.isfinite(rms) or rms <= 0.0:
            return None
        dbfs = 20.0 * math.log10(max(rms, _EPS))   # guards log10(0)
        if not math.isfinite(dbfs) or dbfs < self.silence_floor_dbfs:
            return None
        return dbfs, self._tilt_db(x)

    def _tilt_db(self, x: torch.Tensor) -> float:
        """10*log10(high-band energy / low-band energy). Constant offsets do not
        matter: it is only ever compared against its own rolling median."""
        n = x.numel()
        win, lo_mask, hi_mask = self._bands(n)
        spec = torch.fft.rfft(x * win)
        power = spec.real.pow(2) + spec.imag.pow(2)
        lo = float(power[lo_mask].sum())
        hi = float(power[hi_mask].sum())
        return 10.0 * math.log10((hi + _EPS) / (lo + _EPS))

    def _bands(self, n: int):
        cached = self._band_cache.get(n)
        if cached is None:
            freqs = torch.fft.rfftfreq(n, d=1.0 / self.sample_rate)
            lo = (freqs >= TILT_LOW_HZ[0]) & (freqs < TILT_LOW_HZ[1])
            hi = (freqs >= TILT_HIGH_HZ[0]) & (freqs < TILT_HIGH_HZ[1])
            cached = (torch.hann_window(n), lo, hi)
            self._band_cache[n] = cached
        return cached

    @staticmethod
    def _percentile(values: list[float], q: float) -> float:
        """Linear-interpolated percentile. Pure stdlib: the window is ~200
        floats, so a sort is cheaper than building a tensor per frame."""
        s = sorted(values)
        if len(s) == 1:
            return s[0]
        pos = q * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(s) - 1)
        return s[lo] + (s[hi] - s[lo]) * (pos - lo)
