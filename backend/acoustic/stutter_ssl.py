"""StutterSSL -- the same task as StutterNet, on a pretrained speech encoder.

WHY REPLACE THE LOG-MEL CNN AT ALL
---------------------------------
StutterNet (see stutter.py) is a 583k-param CNN over 64-bin log-mel. On the
episode-disjoint test split it reaches ANY AP 0.786, but the per-type table is
carried almost entirely by Interjection (0.734). Block sits at 0.256 against a
0.137 prevalence -- a lift of 1.86 -- and a downstream calibrated fit assigned
the Block feature a weight of exactly 0.0. In other words the head that matters
most to a word-finding aid is the one that carries no usable signal.

That is a representation problem, not a capacity problem. A block is a silent
struggle to initiate a word: the informative thing is a SPECIFIC KIND of
silence -- articulators posturing, glottis closed, sub-audible tension, and
then a hard onset. Log-mel over a 25 ms window throws away almost everything
that distinguishes that silence from an ordinary inter-word pause, and a small
CNN trained on 14.6k weakly-labelled clips has no way to reinvent the missing
representation from scratch.

A self-supervised encoder does have it. WavLM Base+ was pretrained with
denoising masked prediction on 94k hours of speech; its objective forces it to
model exactly the non-lexical structure -- articulatory state, voice quality,
speaker condition -- that a phonetic recogniser is free to discard. This module
keeps the task, the data, the splits, the metrics and the MIL formulation
identical and swaps ONLY the representation, so the difference in the numbers
is attributable.

WHAT IS KEPT IDENTICAL
----------------------
  * Five INDEPENDENT sigmoid heads. ~21% of the labelled set carries two or
    more types; a softmax would erase the block/interjection distinction, which
    is the one distinction Echo actually needs.
  * Multiple-instance learning. The network emits a probability per frame; the
    clip probability used for training is linear-softmax pooling,
    sum(p^2)/sum(p), over those frames. Training sees a 3 s weak label; the
    live stream reads frames. Do NOT collapse this to a clip classifier.
  * Class-balanced BCE on the POOLED probability (so it cannot use
    BCEWithLogits -- the sigmoid is already inside the pool).

WHAT CHANGES
------------
  * Input is raw 16 kHz waveform, not log-mel. WavLM Base+ has
    feat_extract_norm="group" and do_normalize=false, so the waveform is fed in
    [-1, 1] with no utterance normalisation. Getting this backwards is a silent
    accuracy loss, not an error.
  * Frame rate is ~49.7 fps (20 ms) instead of 25 fps (40 ms). Finer, which is
    strictly better for a live stall detector whose whole budget is ~1.5 s.
  * LEARNED LAYER WEIGHTS. The head reads a softmax-weighted sum of all 13
    hidden states, not just the last one. For paralinguistic tasks the middle
    layers are consistently the informative ones -- the top layers of a masked
    -prediction model drift toward phonetic content -- and which layers the
    model chooses is itself a reportable result. Weights are initialised
    uniform (all-zero logits) so the model is not started with a prior.

COST, WHICH IS WHERE THIS MODEL FAILS
-------------------------------------
The encoder is 94.4M parameters against StutterNet's 0.58M, and it was
measured, not estimated (scripts/train_stutter_ssl.py --bench-only):

    per 3.0 s clip        StutterSSL   StutterNet
    CPU, 24 threads          156 ms       11 ms
    CPU, 1 thread            963 ms       46 ms
    CUDA, batch 1             19 ms          --

The acoustic channel is specified to run on CPU next to the VAD, one window
per 125 ms hop, while the GPU serves the ASR. 156 ms exceeds the hop even with
the whole CPU, and the whole CPU is not what this gets. So the honest reading
is: the representation question is settled (see the metrics), the deployment
question is not -- as it stands this model runs offline or on the GPU, and
serving it in the live loop needs distillation into a small model, an ONNX/
int8 export, or a smaller encoder. A model that is more accurate and cannot be
served is a finding, not a win.

LOADING FROM THE LIVE STREAM
----------------------------
The checkpoint stores only the TRAINABLE tensors (head, layer weights, and any
unfrozen transformer layers) plus the encoder id; the frozen backbone is
rebuilt from the HuggingFace cache. That keeps the file at a few MB instead of
380 MB, at the cost of needing the cache present at load time.
`checkpoint_meta` here is key-compatible with stutter.checkpoint_meta, so
backend/acoustic/stream.py needs only its `from .stutter import load_checkpoint`
swapped for `from .stutter_ssl import load_checkpoint` to serve this model.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .stutter import TYPES, linear_softmax_pool  # noqa: F401  (re-exported)

# WavLM/wav2vec2/HuBERT base all stride the waveform by 320 samples: 16 kHz /
# 320 = 50 fps nominal. The conv stack is valid-padded, so 48000 samples give
# 149 frames rather than 150; 20 ms per frame is the number that matters.
FRAME_MS = 20

DEFAULT_ENCODER = "microsoft/wavlm-base-plus"


def _load_encoder(name: str, mask_time_prob: float):
    """AutoModel handles WavLM / wav2vec2 / HuBERT with one code path.

    They share the Wav2Vec2 architecture family, so the head below is
    encoder-agnostic and switching backbones is a flag, not a rewrite.
    """
    from transformers import AutoModel

    enc = AutoModel.from_pretrained(name)
    # layerdrop randomly SKIPS whole transformer layers during training. With a
    # frozen encoder that just injects noise into features we are not learning,
    # and with learned layer weights it makes the weights themselves unstable
    # (a dropped layer's hidden state is its input, so the weighted sum silently
    # changes meaning). Off, always.
    enc.config.layerdrop = 0.0
    enc.config.apply_spec_augment = mask_time_prob > 0
    enc.config.mask_time_prob = mask_time_prob
    return enc


class StutterSSL(nn.Module):
    """raw waveform (B, N) -> per-frame logits (B, n_types, T), T ~= N/320.

    The head is deliberately small (~1.3M params). The experiment is "what is
    the pretrained representation worth", and a heavy head would confound that
    with head capacity -- the frozen run would then be measuring the head.
    """

    def __init__(self, encoder_name: str = DEFAULT_ENCODER,
                 n_types: int = len(TYPES), hidden: int = 256,
                 dropout: float = 0.3, n_unfreeze: int = 0,
                 mask_time_prob: float = 0.0) -> None:
        super().__init__()
        self.encoder_name = encoder_name
        self.n_unfreeze = int(n_unfreeze)
        self.encoder = _load_encoder(encoder_name, mask_time_prob)
        dim = self.encoder.config.hidden_size
        n_layers = self.encoder.config.num_hidden_layers

        # 13 = embedding output + 12 transformer layers. Zero logits => uniform
        # softmax => the model starts with no opinion about which layer is
        # informative and has to earn one.
        self.layer_logits = nn.Parameter(torch.zeros(n_layers + 1))
        # The hidden states of different layers have wildly different scales
        # (the embedding output in particular). Normalising after the weighted
        # sum keeps the head's input distribution stable while the layer
        # weights move during training.
        self.norm = nn.LayerNorm(dim)

        # Kernel 5 then 3 at 50 fps = ~140 ms of local context. The encoder
        # already carries clip-wide context through self-attention, so this is
        # only about smoothing the frame decision, not about seeing further.
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Conv1d(dim, hidden, 5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, 3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, n_types, 1),
        )
        # SERVING-SIDE TEMPERATURE. Measured on the trained frozen-WavLM model:
        # the 98th-percentile peak frame LOGIT on dysfluency-free VAL clips is
        # 13.2 (Block) to 27.7 (SoundRep). sigmoid(27.7) is 1.0 in float32 --
        # not "close to 1", exactly 1.0 -- so the per-frame operating point the
        # live stream compares against is not representable as a probability and
        # every saturated frame fires. That is the same class of bug the repo
        # already ate once (clip thresholds compared against frame probs, 25
        # fires/minute); this time it is float32, not scale.
        #
        # The fix is a per-type divisor on the frame logits, fitted on VAL, that
        # moves the operating point back into representable range. It is a
        # strictly monotone per-type map and each type is thresholded
        # independently, so it changes NO frame decision -- precision and recall
        # at the calibrated point are identical. It is numerical conditioning,
        # not calibration in the reliability sense, and it is not fitted on TEST.
        self.register_buffer("frame_temperature", torch.ones(n_types))
        self._apply_freeze()

    # --- freezing ---------------------------------------------------------
    def _apply_freeze(self) -> None:
        """Freeze everything, then thaw the top n_unfreeze transformer layers.

        The convolutional feature extractor stays frozen in every configuration.
        That is the standard wav2vec2/WavLM fine-tuning recipe: those layers are
        the most data-hungry part of the pretraining and 14.6k clips of weak
        labels will damage them faster than they will improve them.
        """
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.n_unfreeze <= 0:
            return
        layers = self.encoder.encoder.layers
        for layer in layers[-self.n_unfreeze:]:
            for p in layer.parameters():
                p.requires_grad = True
        # Nothing else is thawed. In the non-stable-layer-norm variant that
        # WavLM Base+ uses, encoder.layer_norm sits BELOW the transformer stack
        # (it normalises the positional-conv output), so unfreezing it would
        # quietly move the input to every "frozen" layer as well and the
        # unfreeze-top-N ablation would stop meaning what it says.

    def train(self, mode: bool = True):
        super().train(mode)
        if self.n_unfreeze <= 0:
            # A fully frozen encoder must stay in eval mode even inside a
            # training loop: otherwise its dropout makes the "frozen features"
            # stochastic, and the run stops measuring the representation.
            self.encoder.eval()
        return self

    # --- forward ----------------------------------------------------------
    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """(B, N) waveform -> (B, T, C) layer-weighted features."""
        grad = self.n_unfreeze > 0 and self.training
        with torch.set_grad_enabled(grad):
            out = self.encoder(wav, output_hidden_states=True)
        hs = torch.stack(out.hidden_states, dim=0)          # (L+1, B, T, C)
        w = torch.softmax(self.layer_logits, dim=0).view(-1, 1, 1, 1).to(hs.dtype)
        return (hs * w).sum(dim=0)

    def raw_frame_logits(self, wav: torch.Tensor) -> torch.Tensor:
        """Untempered frame logits -- the training and evaluation path."""
        h = self.norm(self.encode(wav))                     # (B, T, C)
        return self.head(h.transpose(1, 2))                 # (B, n_types, T)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """Tempered frame logits -- the SERVING path.

        backend/acoustic/stream.py does `torch.sigmoid(self.stutter(feats))`,
        so the temperature has to live here to reach it. Training and the
        reported metrics deliberately use raw_frame_logits instead: the
        temperature is fitted after training, and running it back through the
        clip-level pool would change the reported numbers for a change that is
        supposed to be invisible.
        """
        return self.raw_frame_logits(wav) / self.frame_temperature.view(1, -1, 1)

    def clip_logits(self, wav: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (clip_probs, frame_probs) -- same contract as StutterNet.

        Cast to fp32 before the sigmoid: under bf16 autocast the pooled
        probability is fed straight into a log, and bf16 has ~3 decimal digits
        of mantissa, which is not enough near p -> 0.
        """
        frame_probs = torch.sigmoid(self.raw_frame_logits(wav).float())
        return linear_softmax_pool(frame_probs, dim=-1), frame_probs

    # --- checkpointing ----------------------------------------------------
    def layer_weights(self) -> list[float]:
        return torch.softmax(self.layer_logits.detach().cpu(), dim=0).tolist()

    def trainable_state_dict(self) -> dict:
        """Only what training changed.

        Saving the whole module would write 380 MB of weights that are
        byte-identical to the HuggingFace cache. Anything not in here is
        restored from the pretrained checkpoint by name.
        """
        keep = {n for n, p in self.named_parameters() if p.requires_grad}
        # frame_temperature is a buffer, not a parameter, and it is fitted on
        # VAL after training -- it must travel with the weights for the same
        # reason the thresholds do.
        keep.add("frame_temperature")
        return {k: v.detach().cpu().clone()
                for k, v in self.state_dict().items() if k in keep}

    def config(self) -> dict:
        return {"encoder_name": self.encoder_name, "n_unfreeze": self.n_unfreeze,
                "hidden": self.head[1].out_channels, "frame_ms": FRAME_MS}


def load_checkpoint(path: str | Path, device: str = "cpu") -> StutterSSL:
    """Rebuild the pretrained backbone, then overlay the trained tensors."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    cfg = state.get("ssl", {})
    model = StutterSSL(
        encoder_name=cfg.get("encoder_name", DEFAULT_ENCODER),
        n_types=len(state.get("types", TYPES)),
        hidden=cfg.get("hidden", 256),
        n_unfreeze=cfg.get("n_unfreeze", 0),
    )
    missing, unexpected = model.load_state_dict(state["model"], strict=False)
    if unexpected:
        raise RuntimeError("unexpected tensors in %s: %s" % (path, unexpected[:4]))
    model.eval().to(device)
    return model


def checkpoint_meta(path: str | Path) -> dict:
    """Same keys as stutter.checkpoint_meta, plus the encoder identity.

    Thresholds travel with the weights for the reason spelled out in
    stutter.py: they are fitted on a validation split alongside these weights
    and mean nothing against any other weights. `frame_thresholds` (per-frame,
    what the live stream compares against) and `thresholds` (clip-level, fitted
    on the linear-softmax pool) are different scales and must not be swapped.
    """
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
