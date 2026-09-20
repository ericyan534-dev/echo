"""Detection + serving latency benchmark for Echo. No network calls.

Four numbers, each labelled with exactly how it was obtained:

(a) PAUSE baseline — the transcript-only fallback trigger fires `pause_ms`
    after the last word by construction; we read the configured value from
    backend.config.get_settings() rather than hard-coding it.

(b) ACOUSTIC FILLER latency — real PFSD test clips are composed into a
    synthetic mic stream (real Words-clip preamble to satisfy the VAD voiced
    gate, then a real Um clip at a known sample-accurate onset) and fed to
    backend.acoustic.stream.AcousticStream in 20 ms chunks.  Latency = event
    at_ms - filler onset ms, median over >= 20 runs with the onset phase
    varied against the classifier hop (backend.acoustic.stream.HOP_MS;
    otherwise every run would land on the same hop boundary and the median
    would be a single quantized value).  Requires models/fillernet.pt;
    reported SKIPPED if absent.

(c) PROLONGATION latency — per the approach validated in tests/: take the
    loudest 800-sample (50 ms) frame of a real Uh clip and tile it into a
    sustained vowel, then drive ProlongationTracker frame by frame.  Latency
    = audio elapsed from sustain onset to fire; expected min_ms (700) plus
    one-frame quantization.  Tracker-level: the live stream adds <= 50 ms of
    frame buffering and its own >= 800 ms voiced gate at utterance start.

(d) SERVING — EchoPipeline + MockPredictor, prefetch on vs off, replaying a
    scripted utterance + SilenceTick and recording Prediction.latency_ms.
    The live-LLM round-trip is NOT measured here (no network by design): the
    "live" row injects a 1500 ms artificial predictor delay (midpoint of the
    measured live range) purely to prove the live path actually waits for
    the round-trip, and the real number is reported as an external constant:
    live Gemini measured 1.25-1.8 s in e2e runs (scripts/e2e_live.py).

Usage:  python eval/run_latency_bench.py [--device cuda|cpu] [--runs N]
Output: eval/results/latency_bench.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.prolongation import ENERGY_FLOOR, ProlongationTracker  # noqa: E402
from backend.acoustic.stream import HOP_MS  # noqa: E402
from backend.config import get_settings  # noqa: E402
from backend.pipeline import EchoPipeline  # noqa: E402
from backend.predictor.mock import MockPredictor  # noqa: E402
from backend.schemas import Prediction, SilenceTick, Word  # noqa: E402
from backend.stall_detector import StallDetector  # noqa: E402

SR = 16_000
CLIPS = ROOT / "data" / "pfsd" / "clips" / "test"
CKPT = ROOT / "models" / "fillernet.pt"
RESULTS = ROOT / "eval" / "results" / "latency_bench.json"

# External constant — measured live Gemini stall->word round-trip from e2e
# runs (scripts/e2e_live.py); cited, never re-measured here (no network).
LIVE_GEMINI_MS = (1250, 1800)
LIVE_GEMINI_CITATION = 'scripts/e2e_live.py (live e2e measurement of Gemini stall-to-word round-trip)'


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    return x if x.ndim == 1 else x.mean(axis=1)


def _stats(values: list[float]) -> dict:
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 1),
        "min_ms": round(min(values), 1),
        "max_ms": round(max(values), 1),
    }


# ---------------------------------------------------------------------------
# (b) acoustic filler detection latency
# ---------------------------------------------------------------------------
def bench_acoustic_filler(runs: int, device: str) -> dict:
    if not CKPT.exists():
        return {"status": f"SKIPPED (no checkpoint at {CKPT} -- train first)"}
    um_paths = sorted((CLIPS / "Um").glob("*.wav"))
    word_paths = sorted((CLIPS / "Words").glob("*.wav"))
    if len(um_paths) < 5 or len(word_paths) < 10:
        return {"status": "SKIPPED (test split Um/Words clips not downloaded yet)"}

    from backend.acoustic.stream import AcousticStream

    rng = random.Random(13)
    latencies: list[float] = []
    misses = preamble_false_fires = 0
    for i in range(runs):
        # Composition: phase pad (varies um onset vs the HOP_MS hop grid so
        # `runs` unique phases cover one full hop period)
        # + 2 s of real podcast speech (passes the >= 800 ms voiced gate)
        # + the real um clip + 0.5 s tail so trailing hops still run.
        pad = np.zeros(int(SR * (i * HOP_MS / runs) / 1000), dtype="float32")
        preamble = np.concatenate([_load_wav(rng.choice(word_paths)) for _ in range(2)])
        um = _load_wav(rng.choice(um_paths))
        audio = np.concatenate([pad, preamble, um, np.zeros(SR // 2, dtype="float32")])
        onset_ms = (len(pad) + len(preamble)) * 1000.0 / SR

        stream = AcousticStream(model_path=CKPT, device=device)
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        events = []
        for j in range(0, len(pcm16), 640):  # 20 ms chunks, like the live socket
            events.extend(stream.feed(pcm16[j:j + 640]))

        filler_evs = [e for e in events if e.kind == "filler"]
        preamble_false_fires += sum(1 for e in filler_evs if e.at_ms < onset_ms)
        true_evs = [e for e in filler_evs if e.at_ms >= onset_ms]
        if true_evs:
            latencies.append(true_evs[0].at_ms - onset_ms)
        else:
            misses += 1

    out = {
        "status": "ok",
        "runs": runs,
        "fired": len(latencies),
        "missed": misses,
        "preamble_false_fires": preamble_false_fires,
        "note": "onset = start of the appended Um clip; the voiced filler may "
                "begin some ms into the clip, so this is a conservative "
                "(upper-bound) latency",
    }
    if latencies:
        out.update(_stats(latencies))
    return out


# ---------------------------------------------------------------------------
# (c) prolongation detection latency
# ---------------------------------------------------------------------------
def bench_prolongation(runs: int) -> dict:
    uh_paths = sorted((CLIPS / "Uh").glob("*.wav"))
    if len(uh_paths) < 5:
        return {"status": "SKIPPED (test split Uh clips not downloaded yet)"}

    rng = random.Random(13)
    latencies: list[float] = []
    skipped_quiet = 0
    for path in rng.sample(uh_paths, min(runs, len(uh_paths))):
        x = _load_wav(path)
        n_frames = len(x) // 800
        frames = x[: n_frames * 800].reshape(n_frames, 800)
        rms = np.sqrt((frames ** 2).mean(axis=1))
        frame = torch.from_numpy(frames[int(rms.argmax())].copy())
        if float(rms.max()) < ENERGY_FLOOR:
            skipped_quiet += 1  # clip too quiet to sustain a vowel from
            continue

        tracker = ProlongationTracker()
        for k in range(40):  # up to 2.0 s of tiled sustain
            if tracker.observe_frame(frame.clone(), now_ms=k * 50):
                latencies.append((k + 1) * 50)  # audio elapsed since onset
                break

    out = {
        "status": "ok" if latencies else "SKIPPED (no usable vowel frames)",
        "runs": runs,
        "fired": len(latencies),
        "skipped_quiet_clips": skipped_quiet,
        "note": "tracker-level: live stream adds <= 50 ms frame buffering and a "
                ">= 800 ms voiced gate at utterance start",
    }
    if latencies:
        out.update(_stats(latencies))
    return out


# ---------------------------------------------------------------------------
# (d) prefetch vs live serving
# ---------------------------------------------------------------------------
class _DelayedMock(MockPredictor):
    """MockPredictor with an injected constant delay simulating the measured
    live-LLM round-trip (the mock itself answers in microseconds)."""

    def __init__(self, delay_s: float) -> None:
        super().__init__()
        self.delay_s = delay_s

    async def predict(self, context, fragment, excluded=None):
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        return await super().predict(context, fragment, excluded=excluded)


async def _speak(pipe: EchoPipeline, words: list[str], t: int) -> int:
    for w in words:
        await pipe.handle(Word(text=w, start_ms=t, end_ms=t + 250))
        await asyncio.sleep(0)  # let shadow tasks progress
        t += 400
    return t


async def _one_serving_run(prefetch: bool, delay_s: float) -> "Prediction":
    """Scripted utterance ending in a stall; MockPredictor keys on 'toast'."""
    pipe = EchoPipeline(StallDetector(pause_ms=1300), _DelayedMock(delay_s),
                        prefetch=prefetch, prefetch_every=3)
    t = await _speak(pipe, ["I", "want", "to", "make", "some", "toast"], t=0)
    if prefetch:
        # In conversation the shadow completes while the speaker keeps talking;
        # here we yield long enough for the in-flight shadow call to land.
        await asyncio.sleep(delay_s + 0.05)
    t = await _speak(pipe, ["but", "the"], t=t)
    pred = await pipe.handle(SilenceTick(at_ms=t + 1400))  # > pause_ms
    assert pred is not None, "scripted stall did not fire"
    return pred


def bench_serving(prefetch_runs: int, live_runs: int) -> dict:
    prefetch_lat: list[float] = []
    for _ in range(prefetch_runs):
        pred = asyncio.run(_one_serving_run(prefetch=True, delay_s=0.0))
        assert pred.served == "prefetch", f"expected prefetch, got {pred.served}"
        prefetch_lat.append(pred.latency_ms)

    live_delay_ms = 1500.0  # ~midpoint of the measured live range, injected
    live_lat: list[float] = []
    for _ in range(live_runs):
        pred = asyncio.run(_one_serving_run(prefetch=False, delay_s=live_delay_ms / 1000))
        assert pred.served == "live", f"expected live, got {pred.served}"
        live_lat.append(pred.latency_ms)

    return {
        "prefetch": {**_stats(prefetch_lat), "served": "prefetch",
                     "note": "cache hit measured via Prediction.latency_ms; "
                             "independent of LLM round-trip"},
        "live_simulated": {**_stats(live_lat), "served": "live",
                           "injected_delay_ms": live_delay_ms,
                           "note": "mechanism check only — proves the live path "
                                   "waits for the predictor round-trip"},
        "live_gemini_external_ms": list(LIVE_GEMINI_MS),
        "live_gemini_citation": LIVE_GEMINI_CITATION,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cpu",
                    help="device for FillerNet in the stream bench (cpu = live config)")
    ap.add_argument("--runs", type=int, default=24, help="runs for benches (b) and (c)")
    args = ap.parse_args()
    torch.manual_seed(13)
    t0 = time.time()

    pause_ms = get_settings().pause_ms
    pause = {
        "pause_ms": pause_ms,
        "source": "backend.config.get_settings().pause_ms (env STALL_PAUSE_MS)",
        "note": "the pause trigger fires pause_ms after the last word BY "
                "CONSTRUCTION — its detection latency equals its threshold",
    }
    print(f"[pause]        baseline trigger latency: {pause_ms} ms (by construction)")

    filler = bench_acoustic_filler(args.runs, args.device)
    print(f"[filler]       {filler.get('status')}: "
          f"median {filler.get('median_ms', 'n/a')} ms "
          f"({filler.get('fired', 0)}/{filler.get('runs', 0)} fired, "
          f"{filler.get('preamble_false_fires', 0)} preamble false fires)")

    prolong = bench_prolongation(args.runs)
    print(f"[prolongation] {prolong.get('status')}: "
          f"median {prolong.get('median_ms', 'n/a')} ms "
          f"({prolong.get('fired', 0)}/{prolong.get('runs', 0)} fired)")

    serving = bench_serving(prefetch_runs=20, live_runs=5)
    print(f"[serving]      prefetch median {serving['prefetch']['median_ms']} ms | "
          f"live(simulated {serving['live_simulated']['injected_delay_ms']:.0f} ms delay) "
          f"median {serving['live_simulated']['median_ms']} ms | "
          f"live Gemini external {LIVE_GEMINI_MS[0]}-{LIVE_GEMINI_MS[1]} ms (cited)")

    out = {
        "pause_baseline": pause,
        "acoustic_filler": filler,
        "prolongation": prolong,
        "serving": serving,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
