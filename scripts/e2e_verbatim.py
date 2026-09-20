"""End-to-end check of the v3 stack against a RUNNING server, on real aphasic audio.

Streams APROCSA audio into `/ws/audio` at real-time pace -- the same socket the
browser worklet uses for the DJI Mic 2S, the same 100 ms frames -- and reports
everything that comes back on `/ws`. Nothing is mocked: real CrisperWhisper, real StutterNet, real
Gemini.

Real-time pacing is the point, not a nicety. The ASR worker is asynchronous in
the live path, so feeding faster than real time measures how fast this machine
can drain a queue, not whether the system keeps up with a person talking. Every
previous timing mistake in this repo has had that shape.

    uvicorn backend.app:app --port 8000        # with ASR_PROVIDER=crisper
    python scripts/e2e_verbatim.py --seconds 45
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

AUDIO = ROOT / "data" / "aprocsa" / "audio"
SR = 16_000
FRAME_MS = 100


def load(pid: str, seconds: float, skip_s: float):
    import numpy as np
    import soundfile as sf

    wav = AUDIO / ("%s.wav" % pid)
    if not wav.exists():
        return None
    info = sf.info(str(wav))
    a0 = int(skip_s * info.samplerate)
    a1 = min(info.frames, a0 + int(seconds * info.samplerate))
    audio, sr = sf.read(str(wav), start=a0, stop=a1, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SR / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    return (audio * 32767).astype("<i2")


async def main_async(args) -> int:
    import urllib.request

    import websockets

    base = "http://127.0.0.1:%d" % args.port
    try:
        health = json.loads(urllib.request.urlopen(base + "/healthz", timeout=10).read())
        cfg = json.loads(urllib.request.urlopen(base + "/api/config", timeout=10).read())
    except Exception as exc:
        print("server not reachable on port %d (%s)" % (args.port, exc))
        print("start it with:  ASR_PROVIDER=crisper uvicorn backend.app:app --port %d" % args.port)
        return 1

    print("SERVER")
    print("  predictor : %s (%s), active=%s"
          % (health["provider"], health["model"], health["active_predictor"]))
    print("  acoustic  : %s" % health["acoustic"])
    print("  asr       : %s" % health.get("asr"))
    print("  pause=%s ms  min_gap=%s ms" % (cfg["stall_pause_ms"], cfg["stall_min_gap_ms"]))
    if cfg["asr_provider"] != "crisper":
        print("")
        print("  ASR_PROVIDER is '%s', not 'crisper' -- this run would only exercise"
              % cfg["asr_provider"])
        print("  the acoustic channel. Restart the server with ASR_PROVIDER=crisper.")
        return 2

    pcm = load(args.participant, args.seconds, args.skip_s)
    if pcm is None:
        print("no audio for participant %s" % args.participant)
        return 1
    print("")
    print("STREAMING %.0f s of real aphasic speech (participant %s) at real-time pace"
          % (len(pcm) / SR, args.participant))

    ws_url = "ws://127.0.0.1:%d" % args.port
    got = {"transcript_word": [], "prediction": [], "acoustic_event": [], "interim": 0}

    async with websockets.connect(ws_url + "/ws", max_size=None) as ui, \
            websockets.connect(ws_url + "/ws/audio", max_size=None) as au:

        async def reader():
            try:
                async for raw in ui:
                    m = json.loads(raw)
                    t = m.get("type")
                    if t == "transcript_word":
                        got["transcript_word"].append(m["text"])
                    elif t == "interim":
                        got["interim"] += 1
                    elif t == "acoustic_event":
                        got["acoustic_event"].append(m["kind"])
                    elif t == "prediction":
                        got["prediction"].append(m)
                        words = ", ".join(c["word"] for c in m["candidates"][:3])
                        print("    [%6.1fs] STALL trigger=%-15s served=%-9s %5.0f ms -> %s"
                              % (time.perf_counter() - t0, m["trigger"], m["served"],
                                 m["latency_ms"], words or "(none)"), flush=True)
                        print("             fragment: %r" % m["fragment"][-90:], flush=True)
            except Exception:
                pass

        task = asyncio.create_task(reader())
        t0 = time.perf_counter()
        step = SR * FRAME_MS // 1000
        for i in range(0, len(pcm) - step, step):
            await au.send(pcm[i:i + step].tobytes())
            # Pace against the wall clock rather than sleeping a fixed amount,
            # so accumulated send latency cannot silently make this a
            # faster-than-real-time replay.
            target = t0 + (i + step) / SR
            delay = target - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        await asyncio.sleep(2.5)      # let the tail drain
        task.cancel()

    words = got["transcript_word"]
    print("")
    print("RESULT")
    print("  transcript words : %d" % len(words))
    print("  interim updates  : %d" % got["interim"])
    print("  acoustic events  : %d  %s"
          % (len(got["acoustic_event"]),
             {k: got["acoustic_event"].count(k) for k in sorted(set(got["acoustic_event"]))}))
    print("  stalls served    : %d" % len(got["prediction"]))
    print("")
    print("  transcript: %s" % " ".join(words)[:400])

    # The whole point of the swap: dysfluency must survive into the transcript.
    kept = [w for w in words if w.startswith("[") or w.endswith("-")]
    print("")
    print("  dysfluency tokens preserved: %d  %s" % (len(kept), kept[:12]))
    if not words:
        print("  FAIL -- no transcript words arrived")
        return 3
    if not kept:
        print("  WARNING -- no filler/fragment tokens in this window. Possible on a")
        print("  fluent stretch; re-run with a different --skip-s before concluding.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--participant", default="1554")
    ap.add_argument("--seconds", type=float, default=45.0)
    ap.add_argument("--skip-s", type=float, default=120.0)
    args = ap.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
