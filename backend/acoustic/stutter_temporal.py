"""StutterTemporal -- a sequence head that models the SHAPE of a dysfluency.

WHY A TEMPORAL HEAD AT ALL
--------------------------
StutterSSL (see stutter_ssl.py) reads a frozen WavLM frame sequence with a
per-frame convolutional head: kernel 5 then 3 at 50 fps is ~140 ms of local
context, and every frame is otherwise classified on its own. That is exactly
the wrong prior for a Block. A block is not a property of one frame; it is a
TRAJECTORY -- articulators posture, the glottis closes, sub-audible tension
holds for a few hundred milliseconds, and then a word breaks out on a hard
onset. The informative thing is the stall-and-release pattern OVER TIME, and a
head with a ~140 ms receptive field that scores frames near-independently
cannot represent "this silence has been held, under tension, for longer than a
word boundary would be". That is why the frozen conv head recalls ~0.10 of
Blocks at its 2%-clean-fire operating point while it recalls the rest far
better.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
The ONLY change from StutterSSL is the head. Everything that would otherwise
confound the comparison is kept byte-identical by reusing it:

  * the frozen WavLM Base+ backbone and its feature extraction come straight
    from stutter_ssl._load_encoder -- same encoder, same layerdrop-off, same
    unnormalised-waveform contract;
  * the 13 LEARNED LAYER WEIGHTS and the post-sum LayerNorm are the same
    construction, initialised uniform, so the head still reads a learned
    weighted sum of all hidden states rather than the last one;
  * five INDEPENDENT sigmoid heads, multiple-instance learning with
    linear-softmax pooling for the clip label, and the per-frame temperature
    that keeps the operating point representable in float32 -- all identical.

The conv stack (Dropout -> Conv1d 5 -> GELU -> Conv1d 3 -> GELU -> Conv1d 1) is
replaced by a small BiLSTM over the frame sequence followed by a per-frame
linear projection. A BiLSTM frame at time t sees the WHOLE clip on both sides
through its recurrence, so it can carry "how long has this state been held" and
"did a hard onset follow" into the per-frame decision -- the temporal structure
the conv head throws away. The backbone stays FROZEN: this is the light head
that fits 16 GB and trains in minutes; a fine-tuned larger backbone is a GX10
job, not this one.

SERVING CONTRACT
----------------
raw_frame_logits / forward / clip_logits / trainable_state_dict / config and
the checkpoint layout are key-compatible with stutter_ssl, so the same
calibration (frame_calibration in scripts/train_stutter.py sense), the same
eval harness and the same stream loader shape all apply unchanged. The
checkpoint stores only the trainable tensors (layer weights, LayerNorm, LSTM,
projection, temperature); the frozen backbone is rebuilt from the HuggingFace
cache, so the file is a few MB, not 380.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .stutter import TYPES, linear_softmax_pool  # noqa: F401  (re-exported)
from .stutter_ssl import DEFAULT_ENCODER, FRAME_MS, _load_encoder  # noqa: F401


class StutterTemporal(nn.Module):
    """raw waveform (B, N) -> per-frame logits (B, n_types, T), T ~= N/320.

    Frozen WavLM features -> learned layer-weighted sum -> LayerNorm -> BiLSTM
    (temporal sequence model) -> per-frame linear -> 5 dysfluency logits.
    """

    def __init__(self, encoder_name: str = DEFAULT_ENCODER,
                 n_types: int = len(TYPES), hidden: int = 128,
                 lstm_layers: int = 2, dropout: float = 0.3) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.n_unfreeze = 0            # this head NEVER unfreezes the backbone
        self.lstm_layers = int(lstm_layers)
        self.hidden = int(hidden)
        # mask_time_prob=0.0: with a frozen encoder SpecAugment would just inject
        # noise into features we are not learning (same reasoning as stutter_ssl).
        self.encoder = _load_encoder(encoder_name, mask_time_prob=0.0)
        dim = self.encoder.config.hidden_size
        n_layers = self.encoder.config.num_hidden_layers

        # Identical to StutterSSL: uniform-init learned weights over the 13
        # hidden states, then a LayerNorm to stabilise the head's input while
        # the layer weights move.
        self.layer_logits = nn.Parameter(torch.zeros(n_layers + 1))
        self.norm = nn.LayerNorm(dim)

        # THE ARCHITECTURE UNDER TEST. A BiLSTM: each output frame integrates the
        # whole clip on both sides through the recurrence, so it can encode the
        # duration and release shape of a stall rather than a single instant.
        # hidden is PER DIRECTION; the concat is 2*hidden. Two layers, dropout
        # BETWEEN layers (nn.LSTM applies dropout to all but the last layer).
        self.lstm = nn.LSTM(
            input_size=dim, hidden_size=self.hidden, num_layers=self.lstm_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if self.lstm_layers > 1 else 0.0)
        self.head_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(2 * self.hidden, n_types)

        # SERVING-SIDE TEMPERATURE -- see stutter_ssl.py for the full argument.
        # A per-type divisor on the frame logits, fitted on VAL, that moves the
        # operating point back into representable float32 range without changing
        # any frame decision.
        self.register_buffer("frame_temperature", torch.ones(n_types))
        self._freeze()

    # --- freezing ---------------------------------------------------------
    def _freeze(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        # A fully frozen encoder must stay in eval mode inside the training loop,
        # or its dropout makes the "frozen features" stochastic and the run stops
        # measuring the representation.
        self.encoder.eval()
        return self

    # --- forward ----------------------------------------------------------
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) waveform -> (B, T, C) layer-weighted frozen features."""
        with torch.no_grad():
            out = self.encoder(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)          # (L+1, B, T, C)
        w = torch.softmax(self.layer_logits, dim=0).view(-1, 1, 1, 1).to(hs.dtype)
        return (hs * w).sum(dim=0)

    def raw_frame_logits(self, wav: torch.Tensor) -> torch.Tensor:
        """Untempered frame logits -- the training and evaluation path."""
        h = self.norm(self.encode(wav))                     # (B, T, C)
        seq, _ = self.lstm(h)                               # (B, T, 2*hidden)
        logits = self.proj(self.head_drop(seq))            # (B, T, n_types)
        return logits.transpose(1, 2)                       # (B, n_types, T)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Tempered frame logits -- the SERVING path (matches stutter_ssl)."""
        return self.raw_frame_logits(wav) / self.frame_temperature.view(1, -1, 1)

    def clip_logits(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (clip_probs, frame_probs) -- same MIL contract as StutterSSL.

        Cast to fp32 before the sigmoid: under bf16 autocast the pooled
        probability is fed straight into a log, and bf16 has too few mantissa
        bits near p -> 0.
        """
        frame_probs = torch.sigmoid(self.raw_frame_logits(wav).float())
        return linear_softmax_pool(frame_probs, dim=-1), frame_probs

    # --- checkpointing ----------------------------------------------------
    def layer_weights(self) -> list[float]:
        return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()

    def trainable_state_dict(self) -> dict:
        """Only what training changed; the frozen backbone is restored by name
        from the pretrained checkpoint."""
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        keep.add("frame_temperature")
        return {k: v.detach().cpu().clone()
                for k, v in self.state_dict().items() if k in keep}

    def config(self) -> dict:
        return {"encoder_name": self.encoder_name, "n_unfreeze": 0,
                "hidden": self.hidden, "lstm_layers": self.lstm_layers,
                "head": "bilstm", "frame_ms": FRAME_MS}


def load_checkpoint(path: str | Path, device: str = "cpu") -> StutterTemporal:
    """Rebuild the frozen backbone, then overlay the trained tensors."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("ssl", {})
    model = StutterTemporal(
        encoder_name=cfg.get("encoder_name", DEFAULT_ENCODER),
        n_types=len(state.get("types", TYPES)),
        hidden=cfg.get("hidden", 128),
        lstm_layers=cfg.get("lstm_layers", 2),
    )
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError("unexpected tensors in %s: %s" % (path, unexpected[:4]))
    model.eval().to(device)
    return model


def checkpoint_meta(path: str | Path) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "types": state.get("types", TYPES),
        "thresholds": state.get("thresholds", {}),
        "frame_thresholds": state.get("frame_thresholds", {}),
        "frame_calibration": state.get("frame_calibration", {}),
        "metrics": state.get("metrics", {}),
        "trained_on": state.get("trained_on", "unknown"),
        "ssl": state.get("ssl", {}),
        "frame_temperature": state.get("frame_temperature", {}),
    }
