"""Tests for the wave-2 rigor benches: SNR mixing math (eval/run_noise_stress.py)
and stream-construction / attribution bookkeeping (eval/run_dual_channel_ablation.py).

Only pure, mockable logic is tested here. Anything that needs the PFSD dataset
on disk (data/pfsd/clips/test/...) is guarded with pytest.mark.skipif so CI
without the dataset still passes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from backend.schemas import AcousticEvent, Word
from eval.run_dual_channel_ablation import (
    attribute,
    build_stream,
    run_condition,
    spurious_acoustic_fires,
)
from eval.run_noise_stress import (
    CONF_THRESH,
    assign_interferers,
    binary_prf,
    mix_at_snr,
    rms,
    score_condition,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "pfsd" / "clips" / "test"
_no_data = pytest.mark.skipif(not DATA_DIR.is_dir(), reason="PFSD test-split clips not downloaded")


# --- SNR mixing math (eval/run_noise_stress.py) ----------------------------
def test_rms_of_constant_signal():
    x = np.full(1000, 0.5, dtype="float32")
    assert rms(x) == pytest.approx(0.5, abs=1e-6)


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1000, dtype="float32")) == pytest.approx(0.0, abs=1e-6)


def test_mix_at_snr_hits_target_ratio():
    rng = np.random.default_rng(0)
    signal = rng.normal(0, 0.1, 16000).astype("float32")
    noise = rng.normal(0, 0.3, 16000).astype("float32")
    for snr_db in (15.0, 10.0, 5.0, 0.0):
        mixed, clipped = mix_at_snr(signal, noise, snr_db)
        assert not clipped  # small amplitudes here, no clipping expected
        # recover the (approximately) added noise and check its RMS lands on target
        added_noise = mixed - signal
        target_noise_rms = rms(signal) / (10.0 ** (snr_db / 20.0))
        assert rms(added_noise) == pytest.approx(target_noise_rms, rel=1e-3)


def test_mix_at_snr_lower_db_means_louder_noise():
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 0.1, 16000).astype("float32")
    noise = rng.normal(0, 0.1, 16000).astype("float32")
    loud_mix, _ = mix_at_snr(signal, noise, 0.0)     # noise as loud as signal
    quiet_mix, _ = mix_at_snr(signal, noise, 20.0)   # noise 10x quieter than signal
    assert rms(loud_mix - signal) > rms(quiet_mix - signal)


def test_mix_at_snr_clips_and_reports_it():
    signal = np.full(1000, 0.9, dtype="float32")
    noise = np.full(1000, 0.9, dtype="float32")
    mixed, clipped = mix_at_snr(signal, noise, 0.0)  # equal RMS, sum will exceed 1.0
    assert clipped
    assert mixed.max() <= 1.0 and mixed.min() >= -1.0


def test_mix_at_snr_silent_interferer_is_a_noop():
    signal = np.full(1000, 0.3, dtype="float32")
    noise = np.zeros(1000, dtype="float32")
    mixed, clipped = mix_at_snr(signal, noise, 5.0)
    assert not clipped
    np.testing.assert_array_equal(mixed, signal)


def test_assign_interferers_avoids_self_mix():
    # a 1-item pool that IS the eval item's own path: the retry loop should
    # give up after 5 tries and still return deterministically (no crash),
    # documenting the known-degenerate edge case rather than hiding it.
    p = Path("clip.wav")
    items = [(p, 0)]
    out = assign_interferers(items, [p], seed=13)
    assert out == [p]  # only one pool member; retries can't find a different one


def test_assign_interferers_deterministic_for_fixed_seed():
    pool = [Path(f"music_{i}.wav") for i in range(20)]
    items = [(Path(f"clip_{i}.wav"), 0) for i in range(10)]
    a = assign_interferers(items, pool, seed=13)
    b = assign_interferers(items, pool, seed=13)
    assert a == b
    c = assign_interferers(items, pool, seed=7)
    assert a != c  # different seed, different assignment (extremely likely for n=10/20)


# --- binary P/R/F1 (eval/run_noise_stress.py) -------------------------------
def test_binary_prf_perfect_prediction():
    y = torch.tensor([True, True, False, False])
    out = binary_prf(y, y.clone())
    assert out["precision"] == 1.0 and out["recall"] == 1.0 and out["f1"] == 1.0
    assert out["fp"] == 0 and out["fn"] == 0


def test_binary_prf_all_negative_predictions():
    y_true = torch.tensor([True, True, False])
    y_pred = torch.tensor([False, False, False])
    out = binary_prf(y_true, y_pred)
    assert out["tp"] == 0 and out["fn"] == 2
    assert out["recall"] == 0.0
    assert out["precision"] == 0.0  # tp+fp == 0 -> defined as 0, not div-by-zero


def test_score_condition_operating_point_is_subset_of_standard():
    # operating_point ANDs in a confidence gate on top of the argmax decision,
    # so its recall can never exceed standard's recall on the same probs.
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(50, 4), dim=1)  # CLASSES = [uh, um, speech, other]
    y = torch.randint(0, 4, (50,))
    out = score_condition(probs, y)
    assert out["operating_point"]["recall"] <= out["standard"]["recall"]
    assert out["operating_point"]["tp"] + out["operating_point"]["fp"] <= out["n"]


def _shipped_conf_thresh() -> float:
    """AcousticStream's shipped conf_thresh, read BY NAME.

    This used to read __init__.__defaults__[1] -- a positional index into the
    defaults tuple, which silently repoints at a different parameter's value
    the moment anyone inserts an argument mid-signature (stream.py carried a
    comment begging authors to append at the end because of exactly this).
    inspect.signature keys on the name, so a reordered signature cannot make
    this pin quietly assert the wrong thing; a renamed/removed parameter
    raises instead.
    """
    import inspect

    from backend.acoustic.stream import AcousticStream

    params = inspect.signature(AcousticStream.__init__).parameters
    assert "conf_thresh" in params, "AcousticStream lost its conf_thresh parameter"
    default = params["conf_thresh"].default
    assert default is not inspect.Parameter.empty, "conf_thresh must keep a shipped default"
    return default


def test_conf_thresh_matches_shipped_operating_point():
    # pin to the live default so this bench can never silently drift from
    # backend/acoustic/stream.py's shipped conf_thresh
    assert CONF_THRESH == _shipped_conf_thresh()


def test_config_env_default_matches_shipped_operating_point(monkeypatch):
    # Regression: config.py's ACOUSTIC_CONF env default sat at the old 0.70
    # operating point while stream.py/docs shipped 0.75 -- session.py always
    # passes settings.acoustic_conf, so the LIVE server silently ran at 0.70.
    # Pin the env fallback to the same shipped value as the class default.
    from backend.config import get_settings

    monkeypatch.delenv("ACOUSTIC_CONF", raising=False)
    assert get_settings().acoustic_conf == _shipped_conf_thresh()


# --- dual-channel ablation bookkeeping (eval/run_dual_channel_ablation.py) -
def test_attribute_finds_first_fire_within_its_own_window():
    onsets = [1000.0, 5000.0, 9000.0]
    window_ends = [2600.0, 6600.0, 10600.0]  # e.g. onset + ~1s filler + 1.6s gap
    fires = [(1200.0, "filler_acoustic"), (6500.0, "pause"), (10500.0, "pause")]
    out = attribute(fires, onsets, window_ends)
    assert out[0] == (1200.0, "filler_acoustic")
    assert out[1] == (6500.0, "pause")
    assert out[2] == (10500.0, "pause")


def test_attribute_does_not_leak_a_fire_from_the_next_cycles_fluent_tail():
    # this is the exact bug the tightened window fixes: a fire that lands
    # AFTER cycle 0's own window ends (but before cycle 1's onset, i.e. during
    # cycle 1's fluent Words-clip run) must NOT be credited to cycle 0.
    onsets = [1000.0, 8000.0]
    window_ends = [2600.0, 9600.0]
    fires = [(4000.0, "filler_acoustic")]  # inside cycle 1's fluent tail, not either window
    out = attribute(fires, onsets, window_ends)
    assert out == [None, None]


def test_attribute_ignores_fires_before_onset():
    onsets = [5000.0]
    window_ends = [7000.0]
    fires = [(1000.0, "pause")]  # fired before the filler even started
    assert attribute(fires, onsets, window_ends) == [None]


def test_attribute_ignores_fires_after_window_end():
    onsets = [5000.0]
    window_ends = [7000.0]
    fires = [(7500.0, "pause")]  # fired after this cycle's own window closed
    assert attribute(fires, onsets, window_ends) == [None]


def test_attribute_is_order_independent_of_fires_list():
    onsets = [1000.0, 5000.0]
    window_ends = [2600.0, 6600.0]
    fires_a = [(6000.0, "pause"), (1500.0, "filler_acoustic")]
    fires_b = list(reversed(fires_a))
    assert attribute(fires_a, onsets, window_ends) == attribute(fires_b, onsets, window_ends)


def test_spurious_acoustic_fires_excludes_in_window_hits():
    onsets = [1000.0, 8000.0]
    window_ends = [2600.0, 9600.0]
    fires = [(1200.0, "filler_acoustic"),   # inside cycle 0's window -- not spurious
             (4000.0, "filler_acoustic"),   # cycle 1's fluent tail -- spurious
             (5000.0, "pause")]             # not a filler_acoustic fire at all -- ignored
    out = spurious_acoustic_fires(fires, onsets, window_ends)
    assert out == [4000.0]


def test_spurious_acoustic_fires_empty_when_all_fires_are_in_window():
    onsets = [1000.0]
    window_ends = [2600.0]
    fires = [(1200.0, "filler_acoustic")]
    assert spurious_acoustic_fires(fires, onsets, window_ends) == []


def test_run_condition_off_never_fires_filler_acoustic():
    words = [Word(text="I", start_ms=0, end_ms=200, is_final=True),
             Word(text="want", start_ms=300, end_ms=500, is_final=True)]
    ticks = list(range(0, 3000, 100))
    fires = run_condition(words, ticks, acoustic_events=None, pause_ms=1300)
    assert all(trigger != "filler_acoustic" for _, trigger in fires)


def test_run_condition_on_fires_filler_acoustic_before_pause_deadline():
    words = [Word(text="I", start_ms=0, end_ms=200, is_final=True),
             Word(text="want", start_ms=300, end_ms=500, is_final=True)]
    ticks = list(range(0, 3000, 100))
    acoustic = [AcousticEvent(kind="filler", at_ms=700, confidence=0.9)]
    fires = run_condition(words, ticks, acoustic_events=acoustic, pause_ms=1300)
    assert fires and fires[0] == (700, "filler_acoustic")
    # pause (500 + 1300 = 1800) never fires: filler_acoustic already fired and
    # the detector isn't rearmed by any subsequent content word in this test
    assert all(trigger != "pause" for _, trigger in fires)


def test_run_condition_pause_fires_without_acoustic_channel():
    words = [Word(text="I", start_ms=0, end_ms=200, is_final=True),
             Word(text="want", start_ms=300, end_ms=500, is_final=True)]
    ticks = list(range(0, 3000, 100))
    fires = run_condition(words, ticks, acoustic_events=None, pause_ms=1300)
    assert ("pause" in [t for _, t in fires])
    at_ms = next(ms for ms, t in fires if t == "pause")
    assert at_ms >= 500 + 1300  # can't fire before the deadline


# --- data-dependent (guarded) ----------------------------------------------
@_no_data
def test_build_stream_onsets_are_monotonic_and_sample_accurate():
    rng = np.random.default_rng(13)
    built = build_stream(n_fillers=3, words_per_cycle=2, silence_gap_ms=1600, rng=rng)
    assert built is not None
    audio, words, onsets, window_ends, speech_seconds = built
    assert len(onsets) == 3
    assert len(window_ends) == 3
    assert onsets == sorted(onsets)  # strictly increasing across cycles
    assert len(audio) / 16000 * 1000 >= onsets[-1]  # last onset fits inside the stream
    assert len(words) == 3 * 2
    for onset, w_end, next_onset in zip(onsets, window_ends, onsets[1:] + [None]):
        assert w_end > onset  # each cycle's window has positive width
        if next_onset is not None:
            assert w_end < next_onset  # window closes before the next cycle's filler starts
