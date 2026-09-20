"""Log-mel feature frontend, shared by the training script and the live
stream service so train/serve features can never drift apart.

16 kHz mono PCM -> 64-bin log-mel, 25 ms window / 10 ms hop.
A 1.0 s clip yields (64, 101). Per-example mean/std normalization makes the
features robust to mic gain differences (podcast corpus vs the DJI Mic 2S lav
or the laptop array at demo time).
"""
from __future__ import annotations

import torch
import torchaudio

SR = 16_000
N_MELS = 64
N_FFT = 400      # 25 ms
HOP = 160        # 10 ms

_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, power=2.0
)


def logmel(pcm: torch.Tensor) -> torch.Tensor:
    """(..., samples) float32 in [-1, 1] -> (..., N_MELS, frames) normalized log-mel."""
    m = _melspec(pcm)
    m = torch.log(m + 1e-6)
    mean = m.mean(dim=(-2, -1), keepdim=True)
    std = m.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
    return (m - mean) / std


def pcm16_to_float(pcm16: bytes) -> torch.Tensor:
    """Raw little-endian PCM16 bytes -> float32 tensor in [-1, 1].
    Tolerates odd-length frames (drops the trailing byte) so a malformed
    network frame can never crash the audio socket."""
    import numpy as np

    if len(pcm16) % 2:
        pcm16 = pcm16[:-1]
    if not pcm16:
        return torch.zeros(0)
    x = np.frombuffer(pcm16, dtype="<i2").astype("float32") / 32768.0
    return torch.from_numpy(x)
