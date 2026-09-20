"""The speaker-gate harness must be able to fail, and must never crash.

These tests pin the SCORING (especially the fail-open accounting: unknown is
never a suppression) and the degradation paths -- no audio source, no
SpeakerGate, a SpeakerGate whose API drifted -- all of which must produce a
SKIPPED result file instead of a traceback. They deliberately do not pin the
measured rates: that would freeze the number this harness exists to discover.
"""
import json

import numpy as np
import pytest

from eval import run_speaker_gate_eval as sg


# ------------------------------------------------------------------ scoring
def test_unknown_confidence_is_not_a_suppression():
    """The whole fail-open contract in one assertion: None means unknown, and
    unknown must never be scored (or acted on) as a suppression."""
    assert sg.suppression_rate([None, None, None], 0.35) == 0.0
    assert sg.suppression_rate([None, 0.1], 0.35) == 0.5


def test_suppression_rate_is_strictly_below_threshold():
    assert sg.suppression_rate([0.35], 0.35) == 0.0
    assert sg.suppression_rate([0.349], 0.35) == 1.0


def test_suppression_rate_of_nothing_is_zero_not_an_error():
    assert sg.suppression_rate([], 0.35) == 0.0


def test_unknown_rate_counts_only_none():
    assert sg.unknown_rate([None, 0.9, 0.9, None]) == 0.5
    assert sg.unknown_rate([]) == 0.0


def test_word_conf_p90_reads_the_loud_part_median_the_middle():
    frames = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    assert sg.word_conf(frames, 0.9) == pytest.approx(1.0)
    assert sg.word_conf(frames, 0.5) == pytest.approx(0.0)


def test_word_conf_ignores_unknown_frames_and_returns_none_when_all_unknown():
    assert sg.word_conf([None, 0.4, None], 0.5) == pytest.approx(0.4)
    assert sg.word_conf([None, None], 0.9) is None
    assert sg.word_conf([], 0.9) is None


def test_percentile_of_single_value():
    assert sg.percentile_of([0.5], 0.9) == 0.5


# ------------------------------------------------------------------- levels
def test_dbfs_of_digital_silence_does_not_blow_up():
    assert sg.dbfs(np.zeros(512, dtype=np.float32)) == -120.0
    assert sg.dbfs(np.zeros(0, dtype=np.float32)) == -120.0


def test_scale_to_dbfs_hits_the_target():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000).astype(np.float32)
    assert sg.dbfs(sg.scale_to_dbfs(x, -26.0)) == pytest.approx(-26.0, abs=0.01)


def test_scale_to_dbfs_leaves_silence_alone():
    z = np.zeros(64, dtype=np.float32)
    assert np.array_equal(sg.scale_to_dbfs(z, -26.0), z)


# ------------------------------------------------------------------ stimuli
def _clips(n, seed=0):
    rng = np.random.default_rng(seed)
    return [sg.scale_to_dbfs(rng.standard_normal(int(sg.CLIP_S * sg.SR))
                             .astype(np.float32), -26.0) for _ in range(n)]


def test_build_trial_alternates_and_excludes_calibration_from_scoring():
    calib = max(1, int(round(sg.CALIB_S / sg.CLIP_S)))
    audio, words, spans = sg.build_trial(
        _clips(2 + calib), _clips(2, seed=1), 6.0, 0.0, np.random.default_rng(1))
    assert [w["speaker"] for w in words] == ["wearer", "bystander"] * 2
    # calibration produced audio and speech spans but no scored words
    assert len(spans) == len(words) + calib
    assert audio.min() >= -1.0 and audio.max() <= 1.0
    assert np.isfinite(audio).all()


def test_build_trial_puts_the_bystander_exactly_sep_db_down():
    calib = max(1, int(round(sg.CALIB_S / sg.CLIP_S)))
    _, words, _ = sg.build_trial(_clips(1 + calib), _clips(1, seed=1),
                                 12.0, 0.0, np.random.default_rng(2))
    levels = {w["speaker"]: w["level_db"] for w in words}
    assert levels["wearer"] == pytest.approx(sg.WEARER_DBFS)
    assert levels["bystander"] == pytest.approx(sg.WEARER_DBFS - 12.0)


def test_plan_trials_finds_nothing_in_an_empty_directory(tmp_path):
    src, plan = sg.plan_trials(tmp_path, tmp_path / "missing.csv",
                               trials=2, need=4, per_trial=2, seed=1)
    assert (src, plan) == ("", [])


# -------------------------------------------------------------- degradation
def _run(tmp_path, *argv):
    out = tmp_path / "speaker_gate_eval.json"
    code = sg.main(["--out", str(out), *argv])
    return code, json.loads(out.read_text(encoding="utf-8"))


def test_no_audio_source_is_skipped_not_a_crash(tmp_path):
    code, data = _run(tmp_path, "--source", "pfsd", "--clips-root", str(tmp_path))
    assert code == 0
    assert data["status"] == "SKIPPED"
    assert "unvalidated in real rooms" in data["disclaimer"]


def test_missing_speaker_gate_is_skipped_not_a_crash(tmp_path, monkeypatch):
    """Agent-order independence: the harness ships before/without the backend
    gate and must still write a well-formed SKIPPED result."""
    monkeypatch.setattr(sg, "load_gate_class", lambda: None)
    code, data = _run(tmp_path, "--source", "synthetic", "--trials", "1")
    assert code == 0
    assert data["status"] == "SKIPPED"
    assert "SpeakerGate" in data["reason"]
    assert data["source"] == "synthetic"


def test_gate_api_drift_is_skipped_not_reported_as_a_gate_result(tmp_path, monkeypatch):
    class Broken:
        def observe(self, frame):          # wrong arity on purpose
            return 1.0

    monkeypatch.setattr(sg, "load_gate_class", lambda: Broken)
    code, data = _run(tmp_path, "--source", "synthetic", "--trials", "1",
                      "--words-per-speaker", "1")
    assert code == 0
    assert data["status"] == "SKIPPED"
    assert "API mismatch" in data["reason"]


def test_a_gate_that_always_answers_unknown_suppresses_nothing(tmp_path, monkeypatch):
    """End-to-end fail-open: an all-None gate must score zero suppression for
    BOTH speakers, and must pass the wearer-safety gate."""
    class Unknowing:
        def observe(self, frame, speech_prob):
            return None

    monkeypatch.setattr(sg, "load_gate_class", lambda: Unknowing)
    code, data = _run(tmp_path, "--source", "synthetic", "--trials", "1",
                      "--words-per-speaker", "2")
    assert code == 0
    assert data["status"] == "OK"
    assert data["fail_open_respected"] is True
    for cell in data["by_separation_db"].values():
        for agg in cell.values():
            assert agg["bystander_suppression_rate"] == 0.0
            assert agg["wearer_false_suppression_rate"] == 0.0
            assert agg["wearer_unknown_rate"] == 1.0


def test_a_gate_that_mutes_the_wearer_fails_the_harness(tmp_path, monkeypatch):
    """A harness that cannot fail proves nothing: a gate that suppresses
    everything must exit non-zero, however well it suppresses bystanders."""
    class Muting:
        def observe(self, frame, speech_prob):
            return 0.0

    monkeypatch.setattr(sg, "load_gate_class", lambda: Muting)
    code, data = _run(tmp_path, "--source", "synthetic", "--trials", "1",
                      "--words-per-speaker", "2")
    assert code == 1
    assert data["status"] == "OK"
    assert data["fail_open_respected"] is False
    assert data["by_separation_db"]["12"]["p90"]["wearer_false_suppression_rate"] == 1.0
