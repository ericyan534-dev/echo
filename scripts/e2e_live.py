"""End-to-end live test: full WebSocket round-trip against the real app with the
real Gemini predictor.

Drives the exact same JSON protocol the browser uses (context -> words ->
silence -> prediction), in-process via Starlette's TestClient, and asserts the
LLM returns sensible intended words. Requires GEMINI_API_KEY (via .env).

    python -m scripts.e2e_live
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import app
from backend.config import get_settings

CASES = [
    {
        "name": "toaster (pause stall)",
        "context": ["So what did you have for breakfast?"],
        "words": ["I", "made", "some", "toast", "in", "the"],
        "accept": {"toaster", "oven"},
    },
    {
        "name": "Maria (name from context)",
        "context": ["My sister Maria visited yesterday.", "She wants me to call her."],
        "words": ["I", "need", "to", "call", "um", "the"],
        "accept": {"maria", "sister"},
    },
    {
        "name": "Tokyo (place from fragment)",
        "context": ["Where are you traveling next month?"],
        "words": ["we're", "flying", "to", "the", "big", "city", "in", "Japan", "called"],
        "accept": {"tokyo"},
    },
]


def run_case(client: TestClient, case: dict) -> bool:
    with client.websocket_connect("/ws") as ws:
        ws.send_json({"type": "context", "lines": case["context"]})
        t = 0
        for w in case["words"]:
            ws.send_json({"type": "word", "text": w, "start_ms": t, "end_ms": t + 280, "is_final": True})
            t += 400
        t0 = time.perf_counter()
        ws.send_json({"type": "silence", "at_ms": t + 1500})  # mid-utterance pause
        msg = ws.receive_json()
        wall_ms = (time.perf_counter() - t0) * 1000

    assert msg["type"] == "prediction", f"unexpected message: {msg}"
    words = [c["word"].lower() for c in msg["candidates"]]
    hit = any(any(a in w for a in case["accept"]) for w in words)
    print(f"  [{'PASS' if hit else 'FAIL'}] {case['name']}")
    print(f"         fragment : \"{msg['fragment']}\"  (trigger={msg['trigger']})")
    print(f"         predicted: {[(c['word'], c['confidence']) for c in msg['candidates']]}")
    print(f"         latency  : llm={msg['latency_ms']:.0f} ms / round-trip={wall_ms:.0f} ms")
    return hit


def run_acoustic_replay() -> bool:
    """Stream a REAL 'um' clip into /ws/audio after fluent context words on /ws;
    expect an acoustic-channel trigger to produce a prediction.

    Runs against a REAL uvicorn server (production path: one event loop shared
    by all sockets — avoids TestClient's per-socket loops, which break the
    genai client's aiohttp session binding). Every receive is bounded by
    asyncio.wait_for, so this case can never hang. Requires models/fillernet.pt
    and PFSD clips on disk; skips cleanly otherwise."""
    import asyncio
    import glob
    import json
    import subprocess

    import httpx
    import numpy as np
    import soundfile as sf
    import websockets

    ums = sorted(glob.glob("data/pfsd/clips/test/Um/*.wav"))
    words_clips = sorted(glob.glob("data/pfsd/clips/*/Words/*.wav"))
    if not Path("models/fillernet.pt").exists() or len(ums) < 3 or len(words_clips) < 4:
        print("  [SKIP] acoustic replay (model or clips not ready)")
        return True

    def pcm(x):
        return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()

    speech = np.concatenate([sf.read(w, dtype="float32")[0] for w in words_clips[:4]])

    port = 8765
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app:app",
         "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    async def flow() -> dict | None:
        async with websockets.connect(f"ws://127.0.0.1:{port}/ws") as ui, \
                websockets.connect(f"ws://127.0.0.1:{port}/ws/audio") as au:
            await ui.send(json.dumps({"type": "context",
                                      "lines": ["So what did you have for breakfast?"]}))
            t = 0
            for w in ["I", "made", "some", "toast", "in", "the"]:
                await ui.send(json.dumps({"type": "word", "text": w, "start_ms": t,
                                          "end_ms": t + 280, "is_final": True}))
                t += 400
            await au.send(pcm(speech))  # voiced-time gate accumulates on real speech
            for um_path in ums[:3]:     # a few real ums (stream recall < clip recall)
                um, _ = sf.read(um_path, dtype="float32")
                for i in range(0, len(um), 1600):
                    await au.send(pcm(um[i:i + 1600]))
                try:
                    deadline = asyncio.get_event_loop().time() + 12.0
                    while asyncio.get_event_loop().time() < deadline:
                        raw = await asyncio.wait_for(ui.recv(), timeout=12.0)
                        msg = json.loads(raw)
                        if msg.get("type") == "prediction":
                            return msg
                except asyncio.TimeoutError:
                    continue
        return None

    got = None
    try:
        for _ in range(60):  # wait for server readiness (model load etc.)
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).status_code == 200:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        else:
            print("  [FAIL] acoustic replay (server never became ready)")
            return False
        got = asyncio.run(asyncio.wait_for(flow(), timeout=60.0))
    except Exception as exc:
        print(f"  [FAIL] acoustic replay (error: {type(exc).__name__}: {exc})")
        return False
    finally:
        proc.terminate()

    ok = bool(got) and got["trigger"] in ("filler_acoustic", "prolongation")
    print(f"  [{'PASS' if ok else 'FAIL'}] acoustic replay "
          f"(real 'um' -> {got['trigger'] if got else 'no event'}, "
          f"served={got.get('served') if got else '-'})")
    if got:
        print(f"         predicted: {[(c['word'], c['confidence']) for c in got['candidates']]}")
    return ok


def run_prefetch(client: TestClient) -> bool:
    """Speak fluently (shadow prediction warms), then stall — the word must be
    served from the prefetch cache in <300ms."""
    with client.websocket_connect("/ws") as ui:
        ui.send_json({"type": "context", "lines": ["Where are you traveling next month?"]})
        t = 0
        for w in ["we're", "flying", "to", "the", "big", "city", "in", "Japan", "called"]:
            ui.send_json({"type": "word", "text": w, "start_ms": t, "end_ms": t + 280,
                          "is_final": True})
            t += 400
            time.sleep(0.15)  # natural speech pacing
        time.sleep(3.0)  # let the (chasing) shadow predictions complete
        ui.send_json({"type": "silence", "at_ms": t + 1500})
        msg = ui.receive_json()

    ok = (msg["type"] == "prediction" and msg["served"] == "prefetch"
          and msg["latency_ms"] < 300)
    print(f"  [{'PASS' if ok else 'FAIL'}] prefetch (served={msg.get('served')}, "
          f"{msg.get('latency_ms')} ms)")
    print(f"         predicted: {[(c['word'], c['confidence']) for c in msg['candidates']]}")
    return ok


def main() -> int:
    s = get_settings()
    print(f"provider={s.predictor_provider} model={s.gemini_model}")
    if not s.gemini_api_key:
        print("FATAL: no GEMINI_API_KEY in environment/.env")
        return 2

    client = TestClient(app)

    h = client.get("/healthz").json()
    print(f"/healthz: {h}")
    assert h["status"] == "ok"
    assert h["active_predictor"] == "GeminiPredictor", (
        f"live predictor not active (got {h['active_predictor']}) -- check the API key"
    )

    # static frontend served?
    r = client.get("/")
    assert r.status_code == 200 and "Echo" in r.text, "frontend not served at /"
    print("/        : frontend served OK")

    print("\nLive cases:")
    from backend.session import reset_session

    results = []
    for c in CASES:
        reset_session()  # one shared session per server; isolate cases
        results.append(run_case(client, c))

    print("\nv2 cases:")
    reset_session()
    ok = run_prefetch(client)
    if not ok:  # shadow timing can flake on a slow network round; one retry
        print("  (retrying prefetch once)")
        reset_session()
        ok = run_prefetch(client)
    results.append(ok)
    results.append(run_acoustic_replay())  # real-server case (own subprocess)

    print(f"\n{sum(results)}/{len(results)} live e2e cases passed.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
