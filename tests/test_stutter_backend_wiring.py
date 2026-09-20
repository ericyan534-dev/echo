"""The SSL acoustic model must be reachable from the running product.

WavLM was trained, evaluated, calibrated and written up across two versions
while `backend/config.py` had no `stutter_backend` field at all -- so the
server could only ever construct the CNN, and every SSL number described a
model the product could not load. These tests exist so that cannot silently
recur: a backend the config cannot express is a backend that does not ship.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backend.config as config_mod  # noqa: E402


def _settings(monkeypatch, **env):
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    importlib.reload(config_mod)
    return config_mod.get_settings()


def test_default_is_the_cpu_viable_cnn(monkeypatch):
    """SSL costs 156 ms per 3 s window on CPU against a 125 ms hop. Defaulting
    to it would put the acoustic channel above real time on any machine
    without a GPU."""
    s = _settings(monkeypatch, STUTTER_BACKEND=None, STUTTER_MODEL=None)
    assert s.stutter_backend == "cnn"
    assert s.stutter_model.endswith("stutternet.pt")


def test_ssl_selects_the_v2_checkpoint_not_the_destroyed_one(monkeypatch):
    """models/stutternet_ssl.pt was clobbered by a smoke run and now embeds
    n_train=200. The SSL default must point at v2, which is the checkpoint
    whose metrics are published."""
    s = _settings(monkeypatch, STUTTER_BACKEND="ssl", STUTTER_MODEL=None,
                  ACOUSTIC_DEVICE="cuda")
    assert s.stutter_backend == "ssl"
    assert s.stutter_model.endswith("stutternet_ssl_v2.pt")


def test_explicit_model_still_wins(monkeypatch):
    s = _settings(monkeypatch, STUTTER_BACKEND="ssl", STUTTER_MODEL="models/x.pt",
                  ACOUSTIC_DEVICE="cuda")
    assert s.stutter_model == "models/x.pt"


def test_a_typo_is_rejected_rather_than_silently_downgraded(monkeypatch):
    """Falling back to CNN on an unrecognised value would let an operator
    believe WavLM is running while it is not; the two differ by 0.07 AP on
    Block."""
    with pytest.raises(ValueError):
        _settings(monkeypatch, STUTTER_BACKEND="wavlm")


@pytest.mark.parametrize("value", ["SSL", " ssl ", "Cnn"])
def test_case_and_whitespace_tolerated(monkeypatch, value):
    s = _settings(monkeypatch, STUTTER_BACKEND=value, STUTTER_MODEL=None,
                  ACOUSTIC_DEVICE="cuda")
    assert s.stutter_backend == value.strip().lower()


# --- the wiring itself, exercised rather than grepped ----------------------
def _record_streams(monkeypatch):
    """Replace AcousticStream where the session looks it up and record kwargs."""
    import backend.session as session_mod

    calls: list[dict] = []

    class Recorder:
        def __init__(self, **kw):
            calls.append(kw)

    monkeypatch.setattr(session_mod, "AcousticStream", Recorder)
    return session_mod, calls


def test_session_passes_the_backend_through(monkeypatch):
    """A config field the session never reads is the same bug one layer up.

    The previous version of this test asserted that the SOURCE LINE
    "stutter_backend=self.settings.stutter_backend" appeared in session.py. A
    reviewer defeated it in one edit -- `stutter_backend="cnn",  # was
    stutter_backend=self.settings.stutter_backend` -- behaviour broken, literal
    still present, test green. It detected deletion, never breakage. This one
    builds the object graph and reads what actually arrives.
    """
    session_mod, calls = _record_streams(monkeypatch)
    s = _settings(monkeypatch, STUTTER_BACKEND="ssl", STUTTER_MODEL=None,
                  ACOUSTIC_DEVICE="cuda", PREDICTOR_PROVIDER="mock")

    session_mod.EchoSession(settings=s).new_audio_stream()

    assert len(calls) == 1
    kw = calls[0]
    assert kw["stutter_backend"] == "ssl", (
        "the session handed AcousticStream %r while the operator configured "
        "'ssl'" % kw.get("stutter_backend"))
    assert kw["stutter_model"].endswith("stutternet_ssl_v2.pt")
    assert kw["stutter_scale"] == s.stutter_scale


def test_session_passes_the_device_through(monkeypatch):
    """DEFECT: config told the operator to 'use a machine that has a GPU' while
    there was no device setting at all and new_audio_stream() never passed one,
    so AcousticStream took its device='cpu' default. The SSL backend was
    unreachable on the hardware it requires."""
    session_mod, calls = _record_streams(monkeypatch)
    s = _settings(monkeypatch, STUTTER_BACKEND="ssl", STUTTER_MODEL=None,
                  ACOUSTIC_DEVICE="cuda:1", PREDICTOR_PROVIDER="mock")

    session_mod.EchoSession(settings=s).new_audio_stream()

    assert calls[0]["device"] == "cuda:1"


def test_session_passes_the_explicitness_of_the_checkpoint_through(monkeypatch):
    session_mod, calls = _record_streams(monkeypatch)
    s = _settings(monkeypatch, STUTTER_MODEL="models/x.pt", STUTTER_BACKEND=None,
                  ACOUSTIC_DEVICE=None, PREDICTOR_PROVIDER="mock")
    session_mod.EchoSession(settings=s).new_audio_stream()
    assert calls[0]["stutter_required"] is True

    session_mod, calls = _record_streams(monkeypatch)
    s = _settings(monkeypatch, STUTTER_MODEL=None, STUTTER_BACKEND=None,
                  ACOUSTIC_DEVICE=None, PREDICTOR_PROVIDER="mock")
    session_mod.EchoSession(settings=s).new_audio_stream()
    assert calls[0]["stutter_required"] is False, (
        "the built-in default may still degrade: models/*.pt is untracked and "
        "a clean clone has to start")


# --- ACOUSTIC_DEVICE ------------------------------------------------------
def test_device_defaults_to_cpu(monkeypatch):
    s = _settings(monkeypatch, ACOUSTIC_DEVICE=None, STUTTER_BACKEND=None,
                  STUTTER_MODEL=None)
    assert s.acoustic_device == "cpu"


@pytest.mark.parametrize("value", ["cpu", "cuda", "cuda:0", "cuda:3", "mps"])
def test_valid_devices_are_taken_literally(monkeypatch, value):
    s = _settings(monkeypatch, ACOUSTIC_DEVICE=value, STUTTER_BACKEND=None,
                  STUTTER_MODEL=None)
    assert s.acoustic_device == value


def test_a_device_typo_is_rejected_rather_than_silently_downgraded(monkeypatch):
    """Same rule as STUTTER_BACKEND: 'cude' must not read as a working GPU
    config and behave like a stalled server."""
    with pytest.raises(ValueError):
        _settings(monkeypatch, ACOUSTIC_DEVICE="cude", STUTTER_BACKEND=None,
                  STUTTER_MODEL=None)


def test_auto_resolves_rather_than_being_passed_to_torch(monkeypatch):
    import torch

    s = _settings(monkeypatch, ACOUSTIC_DEVICE="auto", STUTTER_BACKEND=None,
                  STUTTER_MODEL=None)
    assert s.acoustic_device == ("cuda" if torch.cuda.is_available() else "cpu")


def test_ssl_on_cpu_is_refused_not_shipped_as_a_footgun(monkeypatch):
    """Measured: 438 ms per _run_stutter on CPU against a 125 ms hop, run from
    an awaited handler. It does not merely miss the budget -- it makes the whole
    session late, once per hop."""
    with pytest.raises(ValueError) as exc:
        _settings(monkeypatch, STUTTER_BACKEND="ssl", ACOUSTIC_DEVICE="cpu",
                  STUTTER_MODEL=None, STUTTER_ALLOW_CPU=None)
    assert "ACOUSTIC_DEVICE" in str(exc.value)


def test_ssl_on_cpu_can_be_forced_for_offline_checks(monkeypatch):
    s = _settings(monkeypatch, STUTTER_BACKEND="ssl", ACOUSTIC_DEVICE="cpu",
                  STUTTER_MODEL=None, STUTTER_ALLOW_CPU="on")
    assert (s.stutter_backend, s.acoustic_device) == ("ssl", "cpu")


def test_the_cnn_default_is_unaffected_by_the_new_refusal(monkeypatch):
    s = _settings(monkeypatch, STUTTER_BACKEND=None, ACOUSTIC_DEVICE=None,
                  STUTTER_MODEL=None, STUTTER_ALLOW_CPU=None)
    assert (s.stutter_backend, s.acoustic_device) == ("cnn", "cpu")


def teardown_module(_mod):
    importlib.reload(config_mod)
