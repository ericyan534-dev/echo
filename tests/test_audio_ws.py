"""PCM16 framing robustness + acoustic stream gating."""
import numpy as np
import pytest
import torch

from backend.acoustic.features import logmel, pcm16_to_float


def test_pcm16_roundtrip():
    x = (np.sin(np.linspace(0, 100, 1600)) * 20000).astype("<i2")
    t = pcm16_to_float(x.tobytes())
    assert t.shape == (1600,)
    assert float(t.abs().max()) <= 1.0
    back = (t.numpy() * 32768.0).astype("<i2")
    assert np.array_equal(back, x)


def test_pcm16_odd_length_tolerated():
    x = (np.zeros(100)).astype("<i2").tobytes() + b"\x7f"  # trailing junk byte
    t = pcm16_to_float(x)
    assert t.shape == (100,)


def test_pcm16_empty_and_tiny():
    assert pcm16_to_float(b"").numel() == 0
    assert pcm16_to_float(b"\x01").numel() == 0  # single byte -> dropped


def test_logmel_shape():
    torch.manual_seed(0)
    x = torch.randn(16000) * 0.1
    assert logmel(x).shape == (64, 101)   # 1.0 s @ 16 kHz, 64 mels, 10 ms hop


def test_logmel_is_gain_invariant():
    """The POINT of the mean/std step: mic gain must not change the features,
    so a podcast-corpus-trained model transfers to the DJI Mic 2S lav.

    Asserting mean~0 / std~1 (the earlier version) tested nothing -- those are
    definitionally true of the function's own last line. Gain invariance is
    the property that normalization exists to buy, and it dies the moment the
    normalization is removed or made non-per-example.
    """
    torch.manual_seed(0)
    x = torch.randn(16000) * 0.1
    base = logmel(x)
    for gain in (0.25, 4.0, 32.0):
        assert torch.allclose(base, logmel(x * gain), atol=0.05), f"gain {gain} shifted features"


def test_logmel_normalizes_each_example_independently_in_a_batch():
    """Per-EXAMPLE, not per-batch: a loud clip batched next to a quiet one
    must get byte-identical features to being run alone. Normalizing over the
    batch dimension would make one clip's level leak into the other's."""
    torch.manual_seed(0)
    loud = torch.randn(16000) * 0.5
    quiet = torch.randn(16000) * 0.002
    batched = logmel(torch.stack([loud, quiet]))
    assert batched.shape == (2, 64, 101)
    assert torch.allclose(batched[0], logmel(loud), atol=1e-4)
    assert torch.allclose(batched[1], logmel(quiet), atol=1e-4)


def test_stream_gate_blocks_until_voiced(monkeypatch):
    from backend.acoustic.stream import AcousticStream

    s = AcousticStream(model_path=None)
    s._voiced_ms = 100                       # below min_voiced_ms (800)
    assert s._gate("prolongation") is False
    s._voiced_ms = 900
    assert s._gate("prolongation") is True
    assert s._gate("prolongation") is False  # refractory blocks repeat


def test_stream_hop_carries_remainder_with_oversized_frames():
    """Oversized 100 ms frames (bigger than the live 20 ms socket chunk) must
    not stretch the effective classifier hop past hop_ms: feed() decrements
    `_since_hop` by the hop (not reset to 0) on fire, so any leftover carries
    into the next call instead of being dropped. Dropping the remainder would
    drag the effective hop toward frame_ms and silently under-count fires."""
    from backend.acoustic.stream import SR, AcousticStream

    hop_ms = 125
    s = AcousticStream(model_path=None, hop_ms=hop_ms)
    fires = 0

    def _count_fire(*_args, **_kw):
        # signature-agnostic on purpose: this test is about hop accounting,
        # not about _run_filler's arguments.
        nonlocal fires
        fires += 1
        return None

    s._run_filler = _count_fire

    frame_ms = 100  # oversized vs. the live 20 ms socket chunk
    n_calls = 20
    frame = np.zeros(int(SR * frame_ms / 1000), dtype="<i2").tobytes()
    for _ in range(n_calls):
        s.feed(frame)

    total_ms = n_calls * frame_ms
    expected_fires = total_ms // hop_ms
    assert fires == expected_fires, (
        f"carry-remainder logic broken: {fires} fires over {total_ms} ms of "
        f"audio, expected exactly {expected_fires} (= total_ms // hop_ms)"
    )


def test_warm_vad_idempotent():
    """warm_vad() populates the cache on first call and is a no-op on second."""
    pytest.importorskip("silero_vad")
    import backend.acoustic.stream as stream_mod
    from backend.acoustic.stream import warm_vad

    original = stream_mod._vad_weights
    stream_mod._vad_weights = None  # simulate cold cache
    try:
        warm_vad()
        first = stream_mod._vad_weights
        assert first is not None, "warm_vad() must populate _vad_weights"
        warm_vad()  # second call must be a no-op (same object)
        assert stream_mod._vad_weights is first, "warm_vad() must not reload on second call"
    finally:
        stream_mod._vad_weights = original  # restore so other tests are unaffected
