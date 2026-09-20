"""Rule-based prolongation detector ("theeee...", "sooo...").

Prolongations are invisible to every transcript pipeline (ASR normalizes
duration: 'uhhhh' -> 'uh', 'theeee' -> 'the'), yet they are among the most
frequent hesitation signals (Eklund 2001 found them MORE frequent than filled
pauses). Acoustic signature, per the literature: a sustained phone with a
near-static spectral envelope (Shriberg 1993 also notes a flat-pitch plateau).
This detector keys on the spectral-envelope half only — it does NOT compute
pitch/F0 — unlike running speech where the envelope shifts every ~100-200 ms as
phones change.

Rule: N consecutive 50 ms frames that are (a) voiced/energetic and (b) whose
mel envelopes stay nearly identical (cosine similarity >= SIM_THRESH) for
>= `min_ms` total. Deliberately NOT learned — published prolongation
classifiers only reach ~0.5-0.7 F1, while this rule is transparent and tunable.
"""
from __future__ import annotations

import torch

from .features import logmel

FRAME_MS = 50
# Natural sustained vowels jitter: consecutive-frame envelope sims measured at
# ~0.95-0.98 on real held vowels, while running speech breaks the streak every
# ~100-150 ms as phones change. 0.94 (not 0.95) so the real-vowel jitter band
# sits ABOVE the threshold with margin instead of exactly on it -- chosen by
# the committed sweep (eval/tune_stall_thresholds.py): max detection rate
# subject to 0 false fires on 120 s of real running speech.
SIM_THRESH = 0.94
ENERGY_FLOOR = 0.005   # RMS on [-1,1] float PCM; below this = silence


class ProlongationTracker:
    # min_ms 600 (was 700): same sweep -- fires 100 ms sooner and detects more
    # sustained vowels while keeping 0 speech false fires; the demo hold
    # ("theeee...") only needs ~0.65 s instead of ~0.75 s.
    def __init__(self, min_ms: int = 600, refractory_ms: int = 1500,
                 sim_thresh: float = SIM_THRESH) -> None:
        self.min_ms = min_ms
        self.refractory_ms = refractory_ms
        self.sim_thresh = sim_thresh
        self._prev_env: torch.Tensor | None = None
        self._streak_ms = 0
        self._fired_at: int | None = None

    def observe_frame(self, pcm: torch.Tensor, now_ms: int) -> bool:
        """Feed one 50 ms frame (800 samples @16k). True => prolongation detected."""
        rms = float(pcm.pow(2).mean().sqrt())
        if rms < ENERGY_FLOOR:
            self._reset_streak()
            return False

        env = logmel(pcm).mean(dim=-1).flatten()  # (64,) time-averaged envelope
        fired = False
        if self._prev_env is not None:
            sim = float(torch.cosine_similarity(env, self._prev_env, dim=0))
            if sim >= self.sim_thresh:
                self._streak_ms += FRAME_MS
                in_refractory = (
                    self._fired_at is not None
                    and now_ms - self._fired_at < self.refractory_ms
                )
                if self._streak_ms >= self.min_ms and not in_refractory:
                    self._fired_at = now_ms
                    fired = True
            else:
                self._reset_streak()
        self._prev_env = env
        return fired

    def _reset_streak(self) -> None:
        self._streak_ms = 0
        self._prev_env = None
