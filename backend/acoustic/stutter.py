"""StutterNet -- weakly-supervised, multi-label dysfluency detection.

WHY THE OLD MODEL WAS WRONG, NOT JUST WEAK
------------------------------------------
FillerNet's classes are ["uh", "um", "speech", "other"], trained on
PodcastFillers: fluent podcast hosts saying "um". Two consequences, and they
are the two complaints against the shipped system:

  * IT CANNOT DETECT A BLOCK. A block is a silent struggle to initiate a word
    and it is the single strongest evidence that a speaker is stuck. There is
    no class for it, so no threshold can recover it.
  * IT FIRES ON FLUENT SPEECH. It was trained to spot interjections, which
    fluent speakers produce constantly. The model is behaving correctly on the
    task it was given; the task was not Echo's.

This model is trained on SEP-28k -- people who actually stutter -- with Apple's
five types, as five INDEPENDENT labels rather than one softmax. That is forced
by the data: 3,009 clips carry two or more types at >=2/3 annotator agreement,
so a single-label formulation is wrong for ~21% of the labelled set, and it
erases exactly the distinction Echo needs, between a block (silent struggle)
and an interjection ("um"), which mean opposite things for a word-finding aid.

WEAK LABELS, FRAME-LEVEL OUTPUT
-------------------------------
SEP-28k labels a 3 s clip, not a moment. But the live stream must decide
"is the speaker stuck RIGHT NOW", and a 3 s verdict localizes an event to
+/-1.5 s -- far too coarse when the whole stall-to-word budget is ~1.5 s.

So this is multiple-instance learning: the network emits a probability per
40 ms frame, and the clip-level prediction used for training is a pooled
function of those frames. Training sees only the weak clip label; inference
reads the frames. Pooling is linear-softmax, sum(p^2)/sum(p) (Wang et al.
2019, "A Comparison of Five Multiple Instance Learning Pooling Functions for
Sound Event Detection with Weak Labeling") -- differentiable everywhere,
unlike max, and it does not smear a short event across the clip the way mean
does. That last property is what matters here: a block is short and the clip
around it is ordinary speech.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Ordered by how much each type matters to a word-finding aid, strongest
# evidence first. A Block is a silent struggle to initiate -- the clearest sign
# the speaker is stuck. An Interjection is the weakest, because fluent speakers
# produce them constantly, and treating it as strong is what made the old model
# fire on normal speech.
TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]

# Frames per second of model output: 16 kHz / 160-sample hop = 100 fps of
# log-mel, and the network pools time by 4, so 25 fps -> 40 ms per frame.
FRAME_MS = 40


def linear_softmax_pool(frame_probs: torch.Tensor, dim: int = -1,
                        eps: float = 1e-6) -> torch.Tensor:
    """sum(p^2) / sum(p) over the time axis.

    Weights each frame by its own probability, so confident frames dominate
    without the zero-gradient-everywhere problem of a hard max.
    """
    num = (frame_probs * frame_probs).sum(dim=dim)
    den = frame_probs.sum(dim=dim).clamp_min(eps)
    return num / den


class StutterNet(nn.Module):
    """(B, 1, n_mels, T) log-mel -> per-frame logits (B, n_types, T//4).

    Frequency is pooled away completely and time only by 4, because the output
    has to stay temporally resolved. ~430k parameters: it runs on CPU inside
    the audio callback alongside the VAD, which is a hard constraint -- the
    GPU belongs to the ASR.
    """

    def __init__(self, n_types: int = len(TYPES), n_mels: int = 64) -> None:
        super().__init__()

        def block(cin: int, cout: int, pool: tuple[int, int]) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )

        self.features = nn.Sequential(
            block(1, 32, (2, 2)),      # 64 x T   -> 32 x T/2
            block(32, 64, (2, 2)),     #          -> 16 x T/4
            block(64, 128, (2, 1)),    #          ->  8 x T/4
            block(128, 128, (2, 1)),   #          ->  4 x T/4
        )
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Conv1d(128, n_types, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)            # (B, C, F, T')
        h = h.mean(dim=2)               # collapse frequency -> (B, C, T')
        return self.head(self.dropout(h))

    def clip_logits(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (clip_probs, frame_probs). Training uses the first; the live
        stream uses the second."""
        frame_logits = self(x)
        frame_probs = torch.sigmoid(frame_logits)
        return linear_softmax_pool(frame_probs, dim=-1), frame_probs


def load_checkpoint(path: str | Path, device: str = "cpu") -> StutterNet:
    state = torch.load(path, map_location=device, weights_only=False)
    model = StutterNet(n_types=len(state.get("types", TYPES)))
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval().to(device)
    return model


def checkpoint_meta(path: str | Path) -> dict:
    """Types and per-type operating thresholds stored with the weights.

    Thresholds belong in the checkpoint, not in config: they are fitted on a
    validation split alongside the weights and are meaningless against a
    different set of weights. Keeping them together is what stops a retrain
    from silently inheriting the previous model's operating point.
    """
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "types": state.get("types", TYPES),
        # Two different quantities, and confusing them is a real bug this repo
        # shipped: `thresholds` are fitted on the CLIP-level linear-softmax
        # pool, `frame_thresholds` on the per-frame probability the live stream
        # actually reads. Pooling averages a short event down, so a clip
        # threshold is systematically lower than the frame probability the same
        # event produces -- comparing one against the other fired constantly
        # (25/min on real aphasic speech). The stream must use frame_thresholds.
        "thresholds": state.get("thresholds", {}),
        "frame_thresholds": state.get("frame_thresholds", {}),
        "frame_calibration": state.get("frame_calibration", {}),
        "metrics": state.get("metrics", {}),
        "trained_on": state.get("trained_on", "unknown"),
    }
