"""AcousticStream — live sliding-window analysis of raw mic PCM.

feed(pcm16_bytes) -> list[AcousticEvent]. Pipeline per feed:
  - Silero VAD on 512-sample chunks (speech gating + utterance accumulation)
  - 50 ms frames -> ProlongationTracker (rule-based)
  - every `hop_ms`, FillerNet on the last 1.0 s window -> filler event

Guards against decoration-grade false positives:
  - events only after >= `min_voiced_ms` of accumulated speech in the current
    utterance (a lone "um" before any sentence is not a word-search stall)
  - per-kind refractory so the channel can't spam the detector
  - optional SpeakerGate (`wearer_gate=True`): drops events whose frames look
    like a non-wearer (far-field level/tilt). Fails open -- unknown confidence
    always passes. See speaker_gate.py for the honest limits of that evidence.
  - FillerNet is optional: without a checkpoint the stream still emits
    prolongation events (graceful degradation, and the pre-training dev mode)

The clock (`at_ms`) is sample-accurate: ms of audio consumed since start, plus
`clock_offset_ms`. The offset is 0 for a directly-constructed stream (every
eval harness) and non-zero for a /ws/audio connection that joined an already
running session, whose one Timeline all connections share.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

from ..schemas import AcousticEvent
from .features import SR, logmel, pcm16_to_float
from .model import CLASSES, load_checkpoint
from .prolongation import ProlongationTracker
from .speaker_gate import SpeakerGate

log = logging.getLogger("echo.acoustic")

VAD_CHUNK = 512                      # silero requirement @16k (32 ms)
FRAME = 800                          # 50 ms prolongation frames
WINDOW = SR                          # 1.0 s FillerNet window
STUTTER_WINDOW = 3 * SR              # 3.0 s StutterNet window (SEP-28k clip length)
HOP_MS = 125                         # default classifier hop; single source of
                                      # truth so eval scripts derive the same
                                      # value instead of hard-coding it

_vad_weights = None  # cached weights; each stream gets its own deep-copied instance


def _get_vad_instance():
    """Return a fresh per-stream VAD model with independent recurrent state.

    Silero VAD is stateful/recurrent.  A module-level singleton shared by all
    AcousticStream instances (e.g. browser + Clip in the hardware-demo config)
    causes the streams to interleave through one recurrent state: each stream's
    reset_states() call in __init__ wipes the OTHER stream's state mid-utterance.

    Fix: load the weights once (torch.hub caches them on disk), then deep-copy
    the template for each new stream.  deep-copy starts with fresh hidden state
    equivalent to reset_states(), so no extra reset call is needed.
    """
    import copy

    global _vad_weights
    if _vad_weights is None:
        from silero_vad import load_silero_vad

        _vad_weights = load_silero_vad()
    return copy.deepcopy(_vad_weights)


_fillernet_cache: dict[tuple[str, float, str], object] = {}


def _get_fillernet(path: Path, device: str):
    """Load-once FillerNet cache keyed by (path, mtime, device).

    The CNN is stateless (pure feedforward, run in eval mode under no_grad),
    so a single instance is safely shared across AcousticStream instances.
    mtime in the key picks up a retrained checkpoint on the next stream.
    This is what makes the startup pre-warm real: without it every new
    stream paid its own torch.load.
    """
    key = (str(path), path.stat().st_mtime, device)
    model = _fillernet_cache.get(key)
    if model is None:
        model = load_checkpoint(path, device)
        _fillernet_cache.clear()  # drop stale checkpoints (path/mtime changed)
        _fillernet_cache[key] = model
    return model


def warm_vad() -> None:
    """Pre-load Silero VAD weights into the module-level cache.

    Call from a background thread at startup to avoid the cold-cache load
    penalty on the first /ws/audio connection.  Idempotent: a second call
    is a cheap no-op because _vad_weights is already populated.  Raises
    ImportError if silero_vad is not installed (caller should log and swallow).
    """
    _get_vad_instance()


class AcousticStream:
    # conf_thresh 0.75: measured operating point for the aux-binary-loss
    # checkpoint -- minimizes stream false alarms subject to clip-level filler
    # recall staying >= the previously shipped point (0.837); see
    # eval/results/threshold_sweep.json and docs/EVAL.md.
    def __init__(
        self,
        model_path: str | Path | None = None,
        conf_thresh: float = 0.75,
        hop_ms: int = HOP_MS,
        min_voiced_ms: int = 800,
        refractory_ms: int = 1200,
        device: str = "cpu",
        # Prefer appending new arguments at the END: eval scripts pass some of
        # these positionally-adjacent. The stronger reason used to be that
        # tests pinned __init__.__defaults__[1] BY POSITION -- an inserted
        # parameter silently repointed the pin at the wrong value, and a
        # mutation proved the pin still read 0.75 while the shipped threshold
        # had moved to 0.80. Those tests now read the default by name through
        # inspect.signature, so that particular trap is closed.
        wearer_gate: bool = False,
        wearer_conf_min: float = 0.35,
        stutter_model: str | Path | None = None,
        stutter_scale: float = 1.0,
        stutter_backend: str = "cnn",
        stutter_required: bool = False,
        clock_offset_ms: int = 0,
        context_lag_ms: int = 0,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.hop = int(SR * hop_ms / 1000)
        self.min_voiced_ms = min_voiced_ms
        self.refractory_ms = refractory_ms
        # How far behind the window edge the per-frame decision reads, so the
        # scored frames carry right context. 0 = old trailing-edge behaviour.
        # See _run_stutter for the full rationale and the measurement.
        self.context_lag_ms = int(context_lag_ms)
        self.device = device
        self.wearer_conf_min = wearer_conf_min
        # Offset added to every timestamp this stream emits. Zero for a stream
        # constructed directly (every eval harness), non-zero for the second
        # and later /ws/audio connections of one session, which start their own
        # sample counter at connect time but must land on the SHARED Timeline
        # the session's StallDetector owns. See EchoSession.new_audio_channels.
        self.clock_offset_ms = int(clock_offset_ms)

        # StutterNet SUPERSEDES FillerNet when present. They are not merged:
        # FillerNet's classes are uh/um/speech/other trained on fluent podcast
        # hosts, so its "filler" verdict is a different quantity from
        # StutterNet's Interjection head and averaging them would be
        # meaningless. FillerNet stays reachable because four published evals
        # measure it and must remain reproducible.
        self.stutter = None
        self.stutter_types: list[str] = []
        self.stutter_thresholds: dict[str, float] = {}
        self.stutter_scale = stutter_scale
        self.stutter_backend = stutter_backend
        # Non-None when a checkpoint WAS asked for and could not be found. The
        # stream then runs FillerNet, which is a different model with different
        # classes -- so the fact has to survive as state, not only as a log
        # line, or nothing downstream can tell the operator what is running.
        self.stutter_missing: str | None = None
        if stutter_model and not Path(stutter_model).exists():
            # DEFECT: this used to fall through to FillerNet and log only
            # "FillerNet ready from models/fillernet.pt". An operator who asked
            # for the five-type SSL model got a four-class interjection CNN
            # with no diagnostic at all. config._stutter_backend() validates
            # its string precisely to prevent this class of silent downgrade;
            # a checkpoint that is not on disk is the same failure one layer
            # down, so it is reported at the same volume.
            #
            # `stutter_required` is what the OPERATOR asked for (config sets it
            # when STUTTER_MODEL or STUTTER_BACKEND is present in the
            # environment). Named checkpoint missing -> hard failure. Built-in
            # default missing -> loud downgrade, because models/*.pt is
            # untracked apart from fillernet.pt and a clean clone must still
            # start (README: "runs out-of-the-box").
            msg = ("StutterNet checkpoint %r NOT FOUND. The five-type stutter "
                   "channel is DISABLED; %s instead." % (
                       str(stutter_model),
                       "the four-class FillerNet (uh/um/speech/other) will run"
                       if model_path else "no classifier will run"))
            if stutter_required:
                raise FileNotFoundError(
                    msg + " Set STUTTER_MODEL to a checkpoint that exists, or "
                    "unset STUTTER_MODEL/STUTTER_BACKEND to accept the "
                    "default degraded path.")
            self.stutter_missing = str(stutter_model)
            log.error(msg)
        if stutter_model and Path(stutter_model).exists():
            # "ssl" is WavLM Base+ with a small head (backend/acoustic/
            # stutter_ssl.py). It reads RAW WAVEFORM rather than log-mel and
            # emits a frame every 20 ms rather than 40, so both are read from
            # the module instead of assumed. It is markedly better -- ANY AP
            # 0.882 vs 0.786, Block 0.325 vs 0.256 on the v2 test set (an
            # earlier 0.894/0.384 was measured on a smaller, easier test split
            # and against a checkpoint that no longer exists; see
            # docs/VERSIONS.md) -- and markedly slower:
            # 156 ms per 3 s window on a full CPU against a 125 ms hop, so it
            # is only viable on the GPU (19 ms) until it is distilled.
            if stutter_backend in ("ssl", "temporal"):
                if str(device).startswith("cpu"):
                    log.warning(
                        "%s on CPU: the WavLM backbone runs far slower than the "
                        "125 ms hop, so the acoustic channel cannot keep up with "
                        "real time. Use a CUDA device.",
                        "StutterTemporal" if stutter_backend == "temporal" else "StutterSSL")
                if stutter_backend == "temporal":
                    from .stutter_temporal import FRAME_MS, checkpoint_meta, load_checkpoint
                else:
                    from .stutter_ssl import FRAME_MS, checkpoint_meta, load_checkpoint
            else:
                from .stutter import FRAME_MS, checkpoint_meta, load_checkpoint

            self.stutter = load_checkpoint(Path(stutter_model), device)
            meta = checkpoint_meta(Path(stutter_model))
            self.stutter_types = meta["types"]
            # Thresholds travel WITH the weights: they are fitted on that
            # checkpoint's own data and mean nothing against other weights, so
            # a retrain cannot silently inherit the old operating point.
            #
            # frame_thresholds, not thresholds. The clip thresholds are fitted
            # on the linear-softmax POOL over ~75 frames; this code compares
            # against a single frame. Pooling averages a short event down, so
            # the clip threshold is systematically the lower number, and using
            # it here fired on almost everything -- 25 events/minute on real
            # aphasic speech, with the false-alarm rate rising from 0.704 to
            # 0.889 versus the model it replaced. Falling back to the clip
            # thresholds is deliberate but loud: an uncalibrated checkpoint
            # should still run, and should say so.
            self.stutter_thresholds = dict(meta.get("frame_thresholds") or {})
            if not self.stutter_thresholds:
                self.stutter_thresholds = dict(meta["thresholds"])
                log.warning("StutterNet checkpoint has no frame_thresholds; falling "
                            "back to CLIP thresholds, which are the wrong scale for "
                            "per-frame decisions and will over-fire. Run "
                            "eval/calibrate_stutter_frames.py --write.")
            self.stutter_frame_ms = FRAME_MS
            log.info("StutterNet ready from %s (types=%s)", stutter_model,
                     ",".join(self.stutter_types))

        self.model = None
        if self.stutter is None and model_path and Path(model_path).exists():
            self.model = _get_fillernet(Path(model_path), device)
            log.info("FillerNet ready from %s", model_path)
        elif self.stutter is None:
            log.warning("No FillerNet checkpoint (%s) -- filler events disabled, "
                        "prolongation rule active.", model_path)

        self.vad = _get_vad_instance()
        # deep-copy already starts with fresh hidden state; reset_states() is a
        # no-op here but kept for forward-compat with future silero versions.
        try:
            self.vad.reset_states()
        except AttributeError:
            pass
        self.prolong = ProlongationTracker()
        # Off by default: eval harnesses construct AcousticStream directly and
        # must keep measuring the FillerNet/prolongation path unconfounded by a
        # speaker gate. The integrator turns it on from Settings.
        self.speaker_gate = SpeakerGate() if wearer_gate else None
        self._wearer_conf: float | None = None   # None = unknown = never suppress

        # StutterNet reads a 3.0 s window (the SEP-28k clip length it was
        # trained on); FillerNet reads 1.0 s. Sizing the buffer from the model
        # rather than from a constant is what keeps train-time and serve-time
        # input identical -- feeding a 3 s-trained model a 1 s window is the
        # same class of mistake as the 1 s-clip ASR benchmark that scored 0.217.
        self._model_window = STUTTER_WINDOW if self.stutter is not None else WINDOW
        self._buf_keep = self._model_window + WINDOW
        self._buf = torch.zeros(0)        # rolling float PCM
        self._consumed = 0                # total samples ever consumed
        self._since_hop = 0
        self._vad_pending = torch.zeros(0)
        self._voiced_ms = 0               # voiced time in current utterance
        self._silence_ms = 0
        self._frame_pending = torch.zeros(0)
        self._last_emit: dict[str, int] = {}
        self._speech_prob = 0.0

    # ------------------------------------------------------------------
    @property
    def now_ms(self) -> int:
        return self.clock_offset_ms + int(self._consumed * 1000 / SR)

    @property
    def speech_prob(self) -> float:
        return self._speech_prob

    @property
    def wearer_conf(self) -> float | None:
        """Latest wearer confidence, or None for unknown (gate off, not yet
        calibrated, or non-speech). None never suppresses."""
        return self._wearer_conf

    def feed(self, pcm16: bytes) -> list[AcousticEvent]:
        x = pcm16_to_float(pcm16)
        if x.numel() == 0:
            return []
        self._consumed += x.numel()
        self._since_hop += x.numel()
        self._buf = torch.cat([self._buf, x])[-self._buf_keep:]

        events: list[AcousticEvent] = []
        self._run_vad(x)
        events.extend(self._run_prolongation(x))
        # One classifier pass PER HOP contained in this frame, not one per
        # feed(). A single `-=` inside an `if` can never catch up when a frame
        # is longer than the hop: measured, 10 s fed as 1 s buffers ran 10
        # passes instead of 80 and left an 8.75 s backlog in _since_hop that
        # grew without bound. The live path (browser and Clip both send 100 ms,
        # under the 125 ms hop) took exactly one pass either way, so only eval
        # replay and larger-buffer clients saw the 8x undersampling -- silently.
        n_hops = self._since_hop // self.hop
        if n_hops:
            self._since_hop -= n_hops * self.hop   # carry the sub-hop remainder
            # Each pending hop is classified on the window that ENDED at that
            # hop, not on the newest window N times: re-reading the same
            # trailing 3 s would restore the pass count while still looking at
            # one moment, and would stamp every event at the end of the frame.
            # Hops older than the retained buffer cannot be reconstructed --
            # that audio is gone -- so they are dropped rather than faked.
            max_off = max(0, self._buf.numel() - self._model_window // 2)
            for k in range(n_hops):
                end_off = (n_hops - 1 - k) * self.hop   # oldest pending first
                if end_off > max_off:
                    continue
                if self.stutter is not None:
                    events.extend(self._run_stutter(end_off))
                else:
                    ev = self._run_filler(end_off)
                    if ev:
                        events.append(ev)
        return events

    def _window_ending_at(self, end_off: int, size: int) -> torch.Tensor:
        """The trailing `size` samples of the buffer as of `end_off` samples ago,
        left-padded when the stream is younger than the window."""
        buf = self._buf if end_off <= 0 else self._buf[:max(0, self._buf.numel() - end_off)]
        window = buf[-size:]
        if window.numel() < size:
            window = torch.nn.functional.pad(window, (size - window.numel(), 0))
        return window

    def _at_ms(self, end_off: int) -> int:
        return self.now_ms - int(end_off * 1000 / SR)

    # ------------------------------------------------------------------
    def _run_vad(self, x: torch.Tensor) -> None:
        self._vad_pending = torch.cat([self._vad_pending, x])
        while self._vad_pending.numel() >= VAD_CHUNK:
            chunk, self._vad_pending = (
                self._vad_pending[:VAD_CHUNK],
                self._vad_pending[VAD_CHUNK:],
            )
            with torch.no_grad():
                self._speech_prob = float(self.vad(chunk, SR).item())
            chunk_ms = VAD_CHUNK * 1000 // SR
            if self._speech_prob >= 0.5:
                self._voiced_ms += chunk_ms
                self._silence_ms = 0
            else:
                self._silence_ms += chunk_ms
                if self._silence_ms >= 1000:   # utterance ended
                    self._voiced_ms = 0

    def _run_prolongation(self, x: torch.Tensor) -> list[AcousticEvent]:
        events: list[AcousticEvent] = []
        self._frame_pending = torch.cat([self._frame_pending, x])
        while self._frame_pending.numel() >= FRAME:
            frame, self._frame_pending = (
                self._frame_pending[:FRAME],
                self._frame_pending[FRAME:],
            )
            if self.speaker_gate is not None:
                # Overwriting a good value with None is deliberate: stale
                # confidence is worse than admitting we no longer know.
                self._wearer_conf = self.speaker_gate.observe(frame, self._speech_prob)
            if self.prolong.observe_frame(frame, self.now_ms) and self._gate("prolongation"):
                events.append(AcousticEvent("prolongation", self.now_ms, 1.0))
        return events

    def _run_filler(self, end_off: int = 0) -> AcousticEvent | None:
        if self.model is None or self._buf.numel() - end_off < WINDOW // 2:
            return None
        window = self._window_ending_at(end_off, WINDOW)
        with torch.no_grad():
            feats = logmel(window).unsqueeze(0).unsqueeze(0).to(self.device)
            probs = torch.softmax(self.model(feats)[0], dim=0)
        p = {c: float(probs[i]) for i, c in enumerate(CLASSES)}
        filler_p = p["uh"] + p["um"]
        top = max(p, key=p.get)
        at = self._at_ms(end_off)
        if top in ("uh", "um") and filler_p >= self.conf_thresh and self._gate("filler", at):
            return AcousticEvent("filler", at, round(filler_p, 3))
        return None

    # Model type -> the AcousticEvent kind the detector understands. Only
    # Interjection maps to "filler": StallDetector rewrites that one to
    # "filler_acoustic", and the rest arrive as their own trigger names so a
    # served suggestion can be attributed to the evidence that caused it.
    STUTTER_KIND = {
        "Block": "block",
        "Prolongation": "prolongation",
        "SoundRep": "sound_rep",
        "WordRep": "word_rep",
        "Interjection": "filler",
    }

    def _run_stutter(self, end_off: int = 0) -> list[AcousticEvent]:
        """Per-frame dysfluency detection on the trailing 3.0 s window.

        The model is trained on weak clip-level labels but emits a probability
        every 40 ms, so the decision here reads only the frames covering the
        audio since the last hop. Reading the whole window instead would make
        every event fire up to 3 s late and repeat for 3 s -- the clip verdict
        localizes an event only to +/-1.5 s, and the entire stall-to-word
        budget is about 1.5 s.
        """
        if self._buf.numel() - end_off < self._model_window // 2:
            return []
        window = self._window_ending_at(end_off, self._model_window)
        with torch.no_grad():
            if self.stutter_backend == "ssl":
                # Raw waveform in; forward() already applies the per-type
                # temperature that makes the operating point representable.
                wav = window.unsqueeze(0).to(self.device)
                frame_probs = torch.sigmoid(self.stutter(wav))[0]
            else:
                feats = logmel(window).unsqueeze(0).unsqueeze(0).to(self.device)
                frame_probs = torch.sigmoid(self.stutter(feats))[0]   # (types, T)

        n_recent = max(1, int(round(self.hop * 1000 / SR / self.stutter_frame_ms)))
        # CONTEXT LAG -- the fix for "obvious stutter -> nothing".
        #
        # The threshold and temperature are calibrated on the FULL-CONTEXT peak
        # frame probability (eval calibration takes logits.max over the whole
        # 3 s clip). But the live decision above read only the LAST n_recent
        # frames -- the trailing edge of the window, where a frame has no right
        # context. A bidirectional encoder (WavLM) and, through its time
        # pooling/padding, the log-mel CNN both score a dysfluency frame far
        # lower at that edge than a few hundred ms later once the rest of the
        # word has arrived. Measured (eval/diagnose_live_stutter.py): obvious
        # blocks peak ~1.0 mid-window and 0.05-0.6 at the edge, so the operating
        # point was effectively unreachable and blocks fired 0/6 even with the
        # gate open and the model clearing the threshold at full context.
        #
        # So read the band that is `context_lag_ms` BEHIND the edge, which by
        # then carries that much right context. The cost is context_lag_ms of
        # added latency, well inside the ~1.5 s stall-to-word budget, and the
        # event timestamp is shifted back by the same amount so it still lands on
        # the moment it happened. lag == 0 restores the exact trailing-edge
        # behaviour, which is the constructor default the eval harnesses use.
        lag = int(round(self.context_lag_ms / self.stutter_frame_ms))
        T = frame_probs.shape[1]
        end = max(1, T - lag)
        start = max(0, end - n_recent)
        recent = frame_probs[:, start:end].max(dim=1).values

        events: list[AcousticEvent] = []
        at = self._at_ms(end_off + int(self.context_lag_ms * SR / 1000))
        for i, t in enumerate(self.stutter_types):
            kind = self.STUTTER_KIND.get(t)
            if kind is None:
                continue
            # A per-frame probability is not on the same scale as the pooled
            # clip probability the threshold was fitted against, so the scale
            # factor is an explicit, tunable knob rather than a silent 1:1.
            thresh = self.stutter_thresholds.get(t, 0.5) * self.stutter_scale
            p = float(recent[i])
            if p >= thresh and self._gate(kind, at):
                events.append(AcousticEvent(kind, at, round(p, 3)))
        return events

    def _gate(self, kind: str, at_ms: int | None = None) -> bool:
        """Utterance-accumulation + wearer + refractory gating for emission.

        The wearer check is ADDITIONAL to the existing gates, and sits before the
        refractory bookkeeping on purpose: a suppressed event must not arm the
        refractory, or one bystander frame would blank the wearer for the next
        refractory_ms. `None` (unknown) always passes -- fail open.
        """
        now = self.now_ms if at_ms is None else at_ms
        if self._voiced_ms < self.min_voiced_ms:
            return False
        if self._wearer_conf is not None and self._wearer_conf < self.wearer_conf_min:
            return False
        last = self._last_emit.get(kind)
        if last is not None and now - last < self.refractory_ms:
            return False
        self._last_emit[kind] = now
        return True
