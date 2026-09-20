"""FastAPI app — two WebSocket endpoints joined by one EchoSession:

  /ws         UI protocol (JSON): word / silence / turn_end / context / reject
              / ping in; prediction / acoustic_event / pong out.
  /ws/audio   binary PCM16 @ 16 kHz mono in (browser AudioWorklet capturing
              the DJI Mic 2S or the laptop mic) -> acoustic stall channel.

Runs out-of-the-box: missing API key falls back to MockPredictor; missing
FillerNet checkpoint leaves the prolongation rule active.
"""
from __future__ import annotations

import asyncio
import logging
import math
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .audio_sources import AudioSource, profiles_payload
from .config import get_settings
from .schemas import SilenceTick, TurnEnd, Word
from .session import get_session, make_predictor

log = logging.getLogger("echo")


def _prewarm_sync() -> None:
    """Synchronous pre-warm: run in a thread via asyncio.to_thread."""
    from .acoustic.stream import AcousticStream, warm_vad

    warm_vad()
    log.info("Pre-warm: VAD weights loaded")
    s = get_settings()
    if s.acoustic_model and Path(s.acoustic_model).exists():
        # Same device as the live streams, or the cache key differs and every
        # connection pays its own torch.load anyway (see _get_fillernet).
        AcousticStream(model_path=s.acoustic_model, device=s.acoustic_device)
        log.info("Pre-warm: FillerNet checkpoint loaded from %s", s.acoustic_model)
    if s.asr_provider == "crisper":
        # Weights AND one throwaway decode: the first CUDA generate() compiles
        # kernels and allocates workspace, seconds of it, and it would
        # otherwise land on the speaker's first sentence.
        from .stt.verbatim import warm_asr

        warm_asr(s.asr_model, s.asr_device, s.asr_compute_type)
        log.info("Pre-warm: CrisperWhisper '%s' ready", s.asr_model)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """Fire-and-forget background pre-warm so startup is never blocked."""
    async def _prewarm():
        try:
            await asyncio.to_thread(_prewarm_sync)
        except Exception as exc:  # silero absent, network error, etc.
            log.warning("Pre-warm failed (non-fatal, lazy load will apply): %s", exc)

    # keep a strong reference so the task can't be GC'd before it completes
    app.state.prewarm_task = asyncio.create_task(_prewarm())
    yield


app = FastAPI(title="Echo — aphasia word-finding co-pilot", lifespan=_lifespan)

_FRONTEND = Path(__file__).resolve().parent.parent / "frontend"


@app.get("/healthz")
def healthz():
    s = get_settings()
    model = {"gemini": s.gemini_model, "claude": s.claude_model,
             "deepseek": s.deepseek_model, "local": s.local_llm_model}.get(s.predictor_provider, "(mock)")
    active = type(make_predictor(s)).__name__
    sess = get_session()
    # StutterNet supersedes FillerNet when its checkpoint exists, so report
    # which one is actually loaded rather than which one is configured -- the
    # frontend chip says "acoustic: full" off this string, and it must not
    # claim a five-type model when a four-class one is running.
    if s.stutter_model and Path(s.stutter_model).exists():
        acoustic = "stutternet+prolongation"
    elif s.acoustic_model and Path(s.acoustic_model).exists():
        acoustic = "fillernet+prolongation"
    else:
        acoustic = "prolongation-only"
    return {
        "status": "ok",
        "provider": s.predictor_provider,
        "model": model,
        "active_predictor": active,
        # Provider failover chain (PREDICTOR_FALLBACKS), so an operator can see
        # at a glance that a DeepSeek outage will be answered by Gemini.
        "fallbacks": list(getattr(s, "predictor_fallbacks", ())),
        "acoustic": acoustic,
        # Which backend and which device, because "stutternet+prolongation"
        # alone cannot tell an operator whether WavLM is on the GPU or the CNN
        # is on the CPU -- and those differ by 0.07 AP on Block and 400 ms.
        "acoustic_backend": (s.stutter_backend
                             if acoustic.startswith("stutternet") else "n/a"),
        "acoustic_device": s.acoustic_device,
        "asr": (("%s/%s" % (s.asr_model, s.asr_mode))
                if s.asr_provider == "crisper" else "browser"),
        "prefetch": s.prefetch,
        # The microphone the page reported it is capturing through (null until
        # the first POST /api/audio/source). See backend/audio_sources.py.
        "audio_source": sess.audio_source.to_dict() if sess.audio_source else None,
    }


# --------------------------------------------------------------------------
@app.get("/api/audio/sources")
async def api_audio_sources():
    """Known capture-hardware profiles: match patterns, the getUserMedia
    constraints to request, and whether the speaker gate means anything on
    that hardware. The page uses this to auto-prefer the DJI Mic 2S lav
    over the laptop array."""
    return profiles_payload()


@app.post("/api/audio/source")
async def api_audio_source(body: dict):
    """The page reports which device it started capturing from.

    Body: {label, deviceId, profile_id|null, floor_dbfs|null}. A null
    profile_id is resolved from the label by the matcher; an unknown one is a
    400 rather than a silent "unknown hardware". Stored on the session and
    echoed back resolved, so the caller sees the same record /healthz will.
    """
    try:
        source = AudioSource.from_report(body if isinstance(body, dict) else {})
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    sess = get_session()
    sess.audio_source = source
    log.info("audio source: %s (profile=%s, floor=%s dBFS)",
             source.label.encode("ascii", "replace").decode("ascii"),
             source.profile_id, source.floor_dbfs)
    return source.to_dict()


# --------------------------------------------------------------------------
def _opt_conf(raw: object) -> float | None:
    """Parse the optional `wearer_conf` hint off a `word` message.

    Absent, null, unparseable, or non-finite -> None, which means "unknown" and
    never suppresses (backend/acoustic/speaker_gate.py). Deliberately does NOT
    raise: raising here would drop the whole word inside the /ws try/except, so
    a browser bug in a confidence hint would silently swallow the transcript.
    Out-of-range values are clamped rather than trusted -- a stray 5.0 must not
    read as super-confident, and a stray -1.0 must not mute the wearer.
    """
    if raw is None:
        return None
    try:
        val = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not math.isfinite(val):
        return None
    return min(max(val, 0.0), 1.0)


@app.get("/api/config")
async def api_config():
    """Who owns the transcript.

    With server-side ASR the page must NOT also run SpeechRecognition: both
    would feed the same detector and every word would land on the timeline
    twice. Served over HTTP rather than as a websocket greeting on purpose --
    /ws has a documented request/response shape (docs/PROTOCOL.md) and an
    unsolicited first frame silently breaks every client that reads the next
    message as the answer to what it just sent.
    """
    s = get_settings()
    return {
        "asr_provider": s.asr_provider,
        "asr_mode": s.asr_mode,
        "asr_model": s.asr_model,
        "predictor": s.predictor_provider,
        "stall_pause_ms": s.pause_ms,
        "stall_min_gap_ms": s.stall_min_gap_ms,
        # Whether the level-based speaker gate is ON, and through which mic
        # the page said it is listening. Together they tell the page whether
        # the gate recommendation for that hardware is actually in effect.
        "wearer_gate": s.wearer_gate,
        "audio_source": (get_session().audio_source.to_dict()
                         if get_session().audio_source else None),
    }


@app.websocket("/ws")
async def ws_ui(websocket: WebSocket):
    await websocket.accept()
    sess = get_session()
    sess.attach_ui(websocket)
    pipeline = sess.pipeline
    try:
        while True:
            try:
                msg = await websocket.receive_json()
                kind = msg.get("type")
                if kind == "word":
                    # Browser timestamps are lifted onto the session clock
                    # here, before anything downstream compares them with the
                    # acoustic channel or with an earlier fire (see
                    # EchoSession.ui_to_session). end_ms pins the clock; the
                    # start keeps the word's exact duration.
                    end_ms = sess.ui_to_session(int(msg.get("end_ms", 0)))
                    start_ms = end_ms - max(0, int(msg.get("end_ms", 0)) - int(msg.get("start_ms", 0)))
                    await pipeline.handle(Word(
                        text=msg.get("text", ""),
                        start_ms=start_ms,
                        end_ms=end_ms,
                        is_final=bool(msg.get("is_final", True)),
                        wearer_conf=_opt_conf(msg.get("wearer_conf")),
                    ))
                elif kind == "silence":
                    await pipeline.handle(SilenceTick(
                        at_ms=sess.ui_to_session(int(msg.get("at_ms", 0)))))
                elif kind == "turn_end":
                    await pipeline.handle(TurnEnd())
                elif kind == "context":
                    for line in msg.get("lines", []):
                        pipeline.conversation.add_turn(str(line))
                elif kind == "reject":
                    raw = msg.get("rejected")
                    rejected = [str(w) for w in raw if str(w).strip()] \
                        if isinstance(raw, list) else []
                    if rejected:
                        # Result (if any) fans out via on_prediction like a
                        # normal prediction, so every UI client updates.
                        await pipeline.reject(rejected)
                elif kind == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                log.warning("ws message error (ignored): %s", exc)
                # A closed/half-open socket raises a generic error (not
                # WebSocketDisconnect) from receive/send; without this the
                # loop retries it forever and starves the event loop.
                if (websocket.client_state != WebSocketState.CONNECTED
                        or "not connected" in str(exc).lower()
                        or "accept" in str(exc).lower()):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        sess.detach_ui(websocket)


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    sess = get_session()
    channels = sess.new_audio_channels()
    try:
        while True:
            try:
                pcm = await websocket.receive_bytes()
                await sess.handle_audio_events(channels, pcm)
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                log.warning("ws/audio frame error (ignored): %s", exc)
                # A closed/half-open socket raises a generic error (not
                # WebSocketDisconnect) from receive/send; without this the
                # loop retries it forever and starves the event loop.
                if (websocket.client_state != WebSocketState.CONNECTED
                        or "not connected" in str(exc).lower()
                        or "accept" in str(exc).lower()):
                    break
    except WebSocketDisconnect:
        pass
    finally:
        channels.close()   # or every reconnect leaks the ASR worker thread


# Serve the static frontend if present (mounted last so /ws and /healthz win).
if _FRONTEND.is_dir():
    app.mount("/", StaticFiles(directory=str(_FRONTEND), html=True), name="frontend")
else:  # pragma: no cover
    @app.get("/")
    def root():
        return HTMLResponse("<h1>Echo backend running</h1><p>No frontend/ dir found.</p>")
