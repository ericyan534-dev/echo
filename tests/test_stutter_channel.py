"""StutterNet, its pooling, and the detector-level refractory.

The refractory tests matter most: measured on real aphasic speech the stack
fired 17-25 times a minute, because AcousticStream's refractory is per-KIND and
a five-type model turned one 1.2 s guard into five interleaving ones.
"""
from __future__ import annotations

import pytest
import torch

from backend.acoustic.stutter import (FRAME_MS, TYPES, StutterNet,
                                      linear_softmax_pool)
from backend.schemas import AcousticEvent, Word
from backend.stall_detector import StallDetector
from backend.timeline import Timeline


# --- pooling --------------------------------------------------------------
def test_pooling_is_between_mean_and_max():
    """linear-softmax sits between mean and max by construction. That is the
    reason it is used: mean smears a short event across the clip (a block is
    short and the speech around it is ordinary), max has zero gradient
    everywhere except one frame."""
    p = torch.tensor([[0.1, 0.2, 0.9, 0.1]])
    pooled = float(linear_softmax_pool(p, dim=-1))
    assert float(p.mean()) < pooled < float(p.max())


def test_pooling_is_exact_on_a_constant_sequence():
    p = torch.full((1, 10), 0.42)
    assert float(linear_softmax_pool(p, dim=-1)) == pytest.approx(0.42, abs=1e-5)


def test_pooling_handles_an_all_zero_sequence():
    """A clip with no evidence must pool to ~0, not divide by zero."""
    p = torch.zeros(1, 10)
    assert float(linear_softmax_pool(p, dim=-1)) == pytest.approx(0.0, abs=1e-5)


# --- model ----------------------------------------------------------------
def test_output_is_temporally_resolved_not_one_verdict_per_clip():
    """A 3 s clip-level verdict localizes an event to +/-1.5 s, and the whole
    stall-to-word budget is about 1.5 s. The frame axis is the point."""
    model = StutterNet().eval()
    x = torch.randn(2, 1, 64, 301)          # 3.0 s of log-mel
    with torch.no_grad():
        frames = model(x)
    assert frames.shape[0] == 2
    assert frames.shape[1] == len(TYPES)
    assert frames.shape[2] > 40, "time must survive the CNN, not be pooled away"


def test_frame_rate_matches_the_declared_constant():
    """FRAME_MS is what the live stream uses to decide how many trailing frames
    cover the last hop. If the network's stride changes and this constant does
    not, the stream silently reads the wrong slice of time."""
    model = StutterNet().eval()
    seconds = 3.0
    n_in = int(seconds * 100)               # log-mel is 100 fps
    with torch.no_grad():
        frames = model(torch.randn(1, 1, 64, n_in + 1))
    got_ms = seconds * 1000 / frames.shape[2]
    assert abs(got_ms - FRAME_MS) < FRAME_MS * 0.25


def test_clip_logits_returns_probabilities_for_every_type():
    model = StutterNet().eval()
    with torch.no_grad():
        clip, frames = model.clip_logits(torch.randn(1, 1, 64, 301))
    assert clip.shape == (1, len(TYPES))
    assert bool(((clip >= 0) & (clip <= 1)).all())
    assert bool(((frames >= 0) & (frames <= 1)).all())


def test_types_are_multi_label_not_a_softmax():
    """3,009 SEP-28k clips carry two or more types at >=2/3 agreement, so the
    heads must be independent. If these ever summed to 1 the model would have
    been silently turned back into single-label."""
    model = StutterNet().eval()
    with torch.no_grad():
        clip, _ = model.clip_logits(torch.randn(8, 1, 64, 301))
    sums = clip.sum(dim=1)
    assert not bool(torch.allclose(sums, torch.ones_like(sums), atol=1e-3))


# --- detector refractory --------------------------------------------------
def words(det: StallDetector, n: int, t0: int = 0, step: int = 200) -> int:
    at = t0
    for i in range(n):
        det.observe_word(Word(text="w%d" % i, start_ms=at, end_ms=at + 100))
        at += step
    return at


def test_refractory_suppresses_a_second_fire_inside_the_window():
    det = StallDetector(pause_ms=1000, timeline=Timeline(), min_gap_ms=4000)
    words(det, 3)
    first = det.observe_acoustic(AcousticEvent("block", 1000, 0.9))
    assert first is not None
    words(det, 2, t0=1100)          # recovery, so debounce would allow a refire
    second = det.observe_acoustic(AcousticEvent("filler", 2500, 0.9))
    assert second is None, "1.5 s after the last fire is inside the 4 s guard"


def test_refractory_allows_a_fire_once_the_window_has_passed():
    det = StallDetector(pause_ms=1000, timeline=Timeline(), min_gap_ms=4000)
    words(det, 3)
    assert det.observe_acoustic(AcousticEvent("block", 1000, 0.9)) is not None
    words(det, 2, t0=1100)
    assert det.observe_acoustic(AcousticEvent("block", 5200, 0.9)) is not None


def test_refractory_spans_different_triggers():
    """The bug it fixes: AcousticStream's guard is per-kind, so five types gave
    five independent refractories that interleave."""
    det = StallDetector(pause_ms=1000, timeline=Timeline(), min_gap_ms=4000)
    words(det, 3)
    assert det.observe_acoustic(AcousticEvent("block", 1000, 0.9)) is not None
    for kind, at in (("prolongation", 1500), ("sound_rep", 2000),
                     ("word_rep", 2500), ("filler", 3000)):
        words(det, 1, t0=at - 50)
        assert det.observe_acoustic(AcousticEvent(kind, at, 0.9)) is None, (
            "%s at %d ms escaped the cross-trigger refractory" % (kind, at))


def test_refractory_defaults_off_so_published_benches_are_unchanged():
    """Four eval harnesses and the whole existing test suite construct
    StallDetector directly; a non-zero default would silently move every
    published number."""
    det = StallDetector(pause_ms=1000, timeline=Timeline())
    assert det.min_gap_ms == 0
    words(det, 3)
    assert det.observe_acoustic(AcousticEvent("filler", 1000, 0.9)) is not None
    words(det, 2, t0=1100)
    assert det.observe_acoustic(AcousticEvent("filler", 1500, 0.9)) is not None


def test_refractory_measures_from_the_last_fire_not_the_last_unresolved_one():
    """A resolved episode interrupted the wearer just as much as an open one."""
    det = StallDetector(pause_ms=1000, timeline=Timeline(), min_gap_ms=3000)
    words(det, 3)
    det.observe_acoustic(AcousticEvent("block", 1000, 0.9))
    words(det, 5, t0=1100)          # every episode resolves
    assert det.observe_acoustic(AcousticEvent("block", 2000, 0.9)) is None
