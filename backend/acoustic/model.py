"""Compact CNN filler classifier (~136k params).

Input: (B, 1, 64 mels, 101 frames) — one second of log-mel.
Classes: uh, um, speech (lexical words), other (silence/breath/laughter/music).
The stall trigger only acts on uh/um; "speech" as its own class forces the
model to learn the filler-vs-word boundary instead of a blunt voice detector.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

CLASSES = ["uh", "um", "speech", "other"]
# PodcastFillers consolidated-vocab labels -> our classes
LABEL_MAP = {
    "Uh": "uh",
    "Um": "um",
    "Words": "speech",
    "None": "other",
    "Breath": "other",
    "Laughter": "other",
    "Music": "other",
}


class FillerNet(nn.Module):
    def __init__(self, n_classes: int = len(CLASSES)) -> None:
        super().__init__()

        def block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.net = nn.Sequential(
            block(1, 24),    # 64x101 -> 32x50
            block(24, 48),   # -> 16x25
            block(48, 96),   # -> 8x12
            block(96, 96),   # -> 4x6
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_checkpoint(path: str | Path, device: str = "cpu") -> FillerNet:
    model = FillerNet()
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)
    return model
