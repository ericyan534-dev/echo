"""EchoSession — the shared live session joining both socket types.

  /ws        UI clients (transcript events in, predictions + viz out)
  /ws/audio  raw PCM16 in (browser AudioWorklet capturing the DJI Mic 2S
             lav or the laptop mic)

One session per server process (hackathon-appropriate). Everything flows
through one EchoPipeline so transcript triggers, acoustic triggers, prefetch,
and delivery stay coherent.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .acoustic.stream import AcousticStream
from .audio_sources import AudioSource
from .config import Settings, get_settings
from .context import ContextBuilder
from .pipeline import EchoPipeline
from .predictor import WordPredictor, get_predictor
from .predictor.mock import MockPredictor
from .schemas import Prediction, SilenceTick, Word
from .stall_detector import StallDetector
from .summarizer import ExtractiveSummarizer
from .timeline import Timeline

log = logging.getLogger("echo.session")


class AudioChannels:
    """The per-connection consumers of one raw-PCM stream.

    Owning them together is what guarantees they see identical bytes on an
    identical clock. Splitting the frame between two independently-fed objects
    is how the browser-transcript era ended up with two incomparable clocks.
    """

    def __init__(self, acoustic: AcousticStream, asr=None,
                 clock_offset_ms: int = 0, on_close=None) -> None:
        self.acoustic = acoustic
        self.asr = asr
        # Milliseconds between the session's audio epoch and this connection's
        # first sample. Both consumers count samples from THEIR OWN connect
        # time, so without this a socket that joins 40 s late stamps its first
        # word at 100 ms and interleaves it into the shared Timeline 40 s in
        # the past. See EchoSession.new_audio_channels.
        self.clock_offset_ms = int(clock_offset_ms)
        self._on_close = on_close

    def close(self) -> None:
        """Release the ASR worker thread. Called when the socket drops --
        without it, every reconnect leaks a thread for the process lifetime."""
        if self.asr is not None:
            self.asr.close()
        if self._on_close is not None:
            cb, self._on_close = self._on_close, None   # idempotent
            cb(self)


def make_predictor(settings: Settings) -> WordPredictor:
    try:
        return get_predictor(settings)
    except Exception as exc:  # missing key / SDK -> safe demo fallback
        log.warning("Predictor '%s' unavailable (%s); falling back to MockPredictor.",
                    settings.predictor_provider, exc)
        return MockPredictor(max_candidates=settings.max_candidates)


class EchoSession:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.predictor = make_predictor(self.settings)
        self._ui_clients: set[Any] = set()
        self._last_interim = ""                # dedupe interim broadcasts
        # Which microphone the page is capturing through, as reported by
        # POST /api/audio/source (backend/audio_sources.py). Informational:
        # it is surfaced in /healthz and /api/config and changes no threshold.
        self.audio_source: AudioSource | None = None
        # One clock for every /ws/audio connection this session ever accepts.
        # EchoSession is a process-wide singleton owning ONE Timeline, while
        # new_audio_channels() is per-connection: every browser AudioWorklet
        # connection (a reload, a second tab) starts its own sample counter at
        # 0 when it connects. Stamping each against this epoch is what keeps
        # their events comparable on the shared timeline.
        # THE session clock: ms since this instant. Fixed at creation rather
        # than at the first audio connection so that it exists before any
        # channel does, and every channel -- audio streams, the server ASR,
        # and the browser's words and ticks -- is stamped against it.
        self._epoch: float = time.monotonic()
        # Where the BROWSER's clock sits on ours. The browser stamps words and
        # silence ticks in ms since *Start listening*, restarted at 0 on every
        # Start and every reload, while this session, its Timeline and the
        # StallDetector's refractory live for the whole process. Measured
        # before this: one fire at 3.9 s, a reload, a stall at 4.3 s -- "400 ms
        # after the last fire" by timestamp, minutes later in the room -- and
        # nothing served, round after round. ui_to_session() lifts browser
        # timestamps onto the session clock and re-pins on a restart. Unset
        # until the browser has reported a time.
        self._ui_offset_ms: int | None = None
        self._ui_last_ms: int = 0
        # The connection that owns the transcript. A second server-side ASR on
        # a second socket would put every word on the timeline twice -- the
        # exact failure /api/config guards against for browser SpeechRecognition
        # -- so only one audio connection at a time transcribes.
        self._transcript_owner: "AudioChannels | None" = None
        # The transcript channel needs its own speaker filter. The acoustic gate
        # only sees audio; browser-supplied per-word wearer_conf arrives on the
        # Word and is applied here. Threshold is 0.0 when the gate is off, which
        # admits everything -- identical to pre-gate behaviour.
        self.pipeline = EchoPipeline(
            detector=StallDetector(
                pause_ms=self.settings.pause_ms,
                min_gap_ms=self.settings.stall_min_gap_ms,
                timeline=Timeline(
                    wearer_conf_min=(self.settings.wearer_conf_min
                                     if self.settings.wearer_gate else 0.0)),
            ),
            predictor=self.predictor,
            context_turns=self.settings.context_turns,
            on_prediction=self._on_prediction,
            prefetch=self.settings.prefetch,
            prefetch_every=self.settings.prefetch_every,
            entity_memory=self.settings.entity_memory,
            context_builder=ContextBuilder(
                ExtractiveSummarizer(),
                budget_tokens=self.settings.context_budget_tokens,
                verbatim_turns=self.settings.context_turns,
            ),
        )

    # --- UI clients --------------------------------------------------------
    def attach_ui(self, ws: Any) -> None:
        self._ui_clients.add(ws)

    def detach_ui(self, ws: Any) -> None:
        self._ui_clients.discard(ws)

    async def broadcast_ui(self, payload: dict) -> None:
        for ws in list(self._ui_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                self._ui_clients.discard(ws)

    # --- audio channel -------------------------------------------------------
    def new_audio_channels(self) -> "AudioChannels":
        """Both consumers of one /ws/audio connection.

        The verbatim ASR is a SECOND reader of the same PCM the acoustic
        channel already gets -- not a replacement for it. Keeping both is the
        point: the transcript now carries fillers and cut-off words, and the
        acoustic channel still hears prolongations and blocks that never reach
        text at all. The gain over the browser path is that both channels are
        now stamped by the same sample counter, so a Word's end_ms and an
        AcousticEvent's at_ms are finally comparable.

        Across CONNECTIONS the same argument applies and is why the offset
        exists: the session owns one Timeline, so a socket that joins later
        must not restart the clock at zero.
        """
        offset_ms = int((time.monotonic() - self._epoch) * 1000)

        asr = None
        # Only the first live connection transcribes; a second concurrent ASR
        # would double every word on the shared timeline.
        if self.settings.asr_provider == "crisper" and self._transcript_owner is None:
            try:
                from .stt.verbatim import VerbatimASR

                asr = VerbatimASR(
                    model_name=self.settings.asr_model,
                    device=self.settings.asr_device,
                    compute_type=self.settings.asr_compute_type,
                    mode=self.settings.asr_mode,
                    step_ms=self.settings.asr_step_ms,
                    turn_end_ms=self.settings.asr_turn_end_ms,
                    word_timestamps=self.settings.asr_word_timestamps,
                )
            except Exception as exc:
                # A missing model or a cold GPU must not take the audio socket
                # down: the acoustic channel alone is the pre-swap behaviour,
                # which is degraded but working.
                log.warning("Verbatim ASR unavailable (%s); acoustic channel only.", exc)
        elif self.settings.asr_provider == "crisper":
            log.info("Second /ws/audio connection: acoustic channel only "
                     "(the first connection owns the transcript).")
        channels = AudioChannels(
            acoustic=self.new_audio_stream(clock_offset_ms=offset_ms),
            asr=asr,
            clock_offset_ms=offset_ms,
            on_close=self._release_audio_channels,
        )
        if asr is not None:
            self._transcript_owner = channels
        return channels

    def _release_audio_channels(self, channels: "AudioChannels") -> None:
        """Hand transcript ownership back when the owning socket drops, so the
        next connection (or a page reload) can transcribe again."""
        if self._transcript_owner is channels:
            self._transcript_owner = None

    # A browser timestamp that steps BACKWARDS by more than this is a
    # restarted clock (Start pressed again, page reloaded). Words can arrive a
    # little out of order and a word's end trails the tick stream by the ASR's
    # commit latency (~0.3-0.7 s); a restart steps back by the whole length of
    # the previous session. Arrival time is deliberately NOT used: the
    # Simulate tab drives a synthetic clock, tests feed timestamps at wire
    # speed, and a buffered burst arrives late -- all legitimate, all keeping
    # their own spacing, none a restart.
    UI_RESTART_MS = 2500

    def ui_to_session(self, ui_ms: int) -> int:
        """Lift a browser-clock timestamp onto the session clock.

        One constant offset per browser clock, fixed on first contact so
        "browser now" lands on "session now"; consecutive timestamps keep
        their exact spacing (the pause detector measures now - last word
        end). When the browser clock steps backwards it has restarted, and a
        new offset lands it on the session clock's real elapsed time -- so a
        stall 400 ms into the new session is not "400 ms after" a fire 3.9 s
        into the old one. Called for every /ws word and silence tick."""
        if (self._ui_offset_ms is None
                or ui_ms < self._ui_last_ms - self.UI_RESTART_MS):
            elapsed_ms = int((time.monotonic() - self._epoch) * 1000)
            self._ui_offset_ms = elapsed_ms - ui_ms
            self._ui_last_ms = ui_ms
        self._ui_last_ms = max(self._ui_last_ms, ui_ms)
        return ui_ms + self._ui_offset_ms

    def new_audio_stream(self, clock_offset_ms: int = 0) -> AcousticStream:
        # wearer_gate is passed explicitly here and defaults OFF in the
        # AcousticStream constructor. That split is deliberate: the eval
        # harnesses construct AcousticStream directly and publish their numbers
        # in docs/EVAL.md, so a constructor default of True would confound them
        # silently. The app layer is the only place the gate turns on.
        return AcousticStream(
            model_path=self.settings.acoustic_model or None,
            conf_thresh=self.settings.acoustic_conf,
            wearer_gate=self.settings.wearer_gate,
            wearer_conf_min=self.settings.wearer_conf_min,
            # StutterNet supersedes FillerNet inside AcousticStream when the
            # checkpoint exists. Passed from here rather than defaulted in the
            # constructor for the same reason as the speaker gate: the eval
            # harnesses build AcousticStream directly and their FillerNet
            # numbers are published.
            stutter_model=self.settings.stutter_model or None,
            stutter_backend=self.settings.stutter_backend,
            stutter_scale=self.settings.stutter_scale,
            # A checkpoint the operator NAMED must exist; the built-in default
            # may be absent on a clean clone (models/*.pt is untracked).
            stutter_required=self.settings.stutter_required,
            # One knob for both models. Without it the SSL backend could only
            # ever run on the CPU, where it costs 438 ms per 125 ms hop.
            device=self.settings.acoustic_device,
            clock_offset_ms=clock_offset_ms,
            # Live-path detection gates. These default to values that FIX the
            # "obvious stutter -> nothing" failure (context_lag_ms in
            # particular); the AcousticStream constructor still defaults to the
            # old behaviour so the eval harnesses that build a stream directly
            # are unchanged. See backend/config.py and eval/diagnose_live_stutter.py.
            context_lag_ms=self.settings.acoustic_context_lag_ms,
            min_voiced_ms=self.settings.acoustic_min_voiced_ms,
            refractory_ms=self.settings.acoustic_refractory_ms,
        )

    async def handle_audio_events(self, channels: "AudioChannels | AcousticStream",
                                  pcm16: bytes) -> None:
        """Feed one PCM frame to every channel and route what comes out.

        The ASR goes first: its Words establish the fragment, and
        `observe_acoustic` refuses to fire before at least one content word
        exists, so a filler heard acoustically in the same frame as the first
        word should see that word already on the timeline.
        """
        # Accept a bare AcousticStream so existing callers and tests that
        # construct one directly keep working unchanged.
        acoustic = getattr(channels, "acoustic", channels)
        asr = getattr(channels, "asr", None)
        offset = int(getattr(channels, "clock_offset_ms", 0))

        if asr is not None:
            for item in asr.feed(pcm16):
                # The ASR counts samples from ITS OWN connect time. The
                # AcousticStream is offset in its constructor; the ASR is not
                # ours to change, so its timestamps are lifted onto the session
                # clock here, before anything downstream compares them.
                if offset:
                    if isinstance(item, Word):
                        item.start_ms += offset
                        item.end_ms += offset
                    elif isinstance(item, SilenceTick):
                        item.at_ms += offset
                if isinstance(item, Word):
                    # Wearer confidence now comes from the acoustic gate rather
                    # than from the browser. Both channels are reading the same
                    # samples, so this is a strictly better estimate than the
                    # browser's -- which had to align a transcript on one clock
                    # against levels on another. It is None (unknown, never
                    # suppresses) whenever the gate is off, which is the
                    # default: measured bystander suppression is 0.000 at
                    # 0/3/6 dB, so proximity alone does not solve it.
                    item.wearer_conf = acoustic.wearer_conf
                    await self.broadcast_ui({
                        "type": "transcript_word", "text": item.text,
                        "start_ms": item.start_ms, "end_ms": item.end_ms,
                        "source": "asr",
                    })
                await self.pipeline.handle(item)
            interim = asr.interim_text
            if interim != self._last_interim:
                self._last_interim = interim
                await self.broadcast_ui({"type": "interim", "text": interim})

        # OFF THE EVENT LOOP. acoustic.feed() is torch, and it is synchronous:
        # measured 438 ms per hop for the SSL backend on CPU against a 125 ms
        # hop, during which the awaited handler blocked PCM ingest, UI
        # broadcasts and every in-flight prediction for the whole process.
        # to_thread keeps the ordering guarantee (the /ws/audio loop awaits one
        # frame before reading the next, so a stream is never fed
        # concurrently) while giving the loop back to the other sockets.
        for event in await asyncio.to_thread(acoustic.feed, pcm16):
            await self.broadcast_ui({
                "type": "acoustic_event",
                "kind": event.kind,
                "at_ms": event.at_ms,
                "confidence": event.confidence,
            })
            await self.pipeline.handle(event)

    # --- output fan-out ------------------------------------------------------
    async def _on_prediction(self, p: Prediction) -> None:
        payload = {
            "type": "prediction",
            "fragment": p.fragment,
            "trigger": p.trigger,
            "served": p.served,
            "latency_ms": round(p.latency_ms, 1),
            "candidates": [{"word": c.word, "confidence": c.confidence} for c in p.candidates],
        }
        await self.broadcast_ui(payload)


_session: EchoSession | None = None


def get_session() -> EchoSession:
    global _session
    if _session is None:
        _session = EchoSession()
    return _session


def reset_session() -> None:
    """Test hook."""
    global _session
    _session = None
