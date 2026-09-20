"""Does the streaming ASR keep up, and how late is the word the predictor needs?

RTF ON A CLIP IS THE WRONG NUMBER. `eval/bench_asr_models.py` measures one
transcribe of one window. The streaming wrapper calls transcribe repeatedly on
a GROWING window, so its cost is (calls per second) x (cost per call), and the
window length -- not the clip length -- sets the second term. A policy that
looks fine per-call can still sit above real time.

WHY THIS SCRIPT EXISTS -- A DIAGNOSIS THAT WAS WRONG
----------------------------------------------------
The aphasia eval appeared to run at RTF ~8, and the window policy was blamed
and changed on that basis. Then this bench was written to confirm it, and the
OLD policy measured RTF 0.54 on the same audio. The 8x had nothing to do with
the window: the eval had been sharing the GPU with a StutterNet training run
sitting at 98% utilisation, and the number was contention.

The policy change (step 500->700 ms, window 12->8 s, reset 6->3.5 s) was kept
because it does measurably help -- RTF 0.54 -> 0.41, commit lag 300 -> 200 ms
-- but it is a modest improvement, not a fix for an 8x problem, and the
buffer never actually reached the old 12 s cap on this audio. The lesson is
the one this repo keeps relearning: a timing measurement taken while something
else owns the accelerator measures the other thing.

TWO NUMBERS DECIDE IT
---------------------

  RTF          processing seconds per audio second, over a continuous replay.
               Must be < 1 with margin, or the transcript falls further behind
               for as long as the person keeps talking.

  COMMIT LAG   ms from the speaker going silent to the last word of that
               utterance being available to the detector. This is the one that
               matters for stalls: the predictor cannot run on a sentence that
               is missing its final word, and the stall fires ~1.3 s after the
               silence starts. A commit lag above that means the prompt is
               built from a truncated fragment at exactly the wrong moment.

    python eval/bench_asr_stream.py --seconds 90
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from backend.schemas import TurnEnd, Word  # noqa: E402
from backend.stt.verbatim import SR, VerbatimASR  # noqa: E402

AUDIO = ROOT / "data" / "aprocsa" / "audio"
OUT = ROOT / "eval" / "results" / "asr_stream_bench.json"
FRAME_MS = 100


def load(pid: str, seconds: float, skip_s: float) -> np.ndarray | None:
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
    return audio


def run(audio: np.ndarray, **kw) -> dict:
    asr = VerbatimASR(sync=True, **kw)
    step = int(SR * FRAME_MS / 1000)
    t0 = time.perf_counter()
    n_words = n_turns = 0
    lags: list[int] = []
    silent_since: int | None = None
    max_window_seen = 0

    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        items = asr.feed(pcm)
        max_window_seen = max(max_window_seen, asr._window.size)
        # A commit that lands while the audio is silent is the stall-critical
        # one; measure from when the silence began.
        was_silent = silent_since is not None
        if asr.speech_prob >= 0.5:
            silent_since = None
        elif not was_silent:
            silent_since = asr.now_ms
        for it in items:
            if isinstance(it, Word):
                n_words += 1
                if silent_since is not None:
                    lags.append(asr.now_ms - silent_since)
            elif isinstance(it, TurnEnd):
                n_turns += 1
    wall = time.perf_counter() - t0
    asr.close()
    audio_s = len(audio) / SR
    return {
        "audio_s": round(audio_s, 1),
        "wall_s": round(wall, 1),
        "rtf": round(wall / audio_s, 3),
        "words": n_words,
        "turns": n_turns,
        "max_window_s": round(max_window_seen / SR, 2),
        "commit_lag_ms_median": round(statistics.median(lags)) if lags else None,
        "commit_lag_ms_p90": (round(sorted(lags)[int(0.9 * (len(lags) - 1))])
                              if lags else None),
        "n_commit_lag_samples": len(lags),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--skip-s", type=float, default=120.0)
    ap.add_argument("--participants", default="1554,1713")
    ap.add_argument("--configs", default="",
                    help="comma-separated VerbatimASR overrides, e.g. "
                         "'silence_commit_ms=700@reset_window_s=6.0'. Empty "
                         "runs the before/after pair below.")
    args = ap.parse_args()

    pids = [p.strip() for p in args.participants.split(",") if p.strip()]
    if not AUDIO.is_dir():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED", "reason": "no APROCSA audio"},
                                  indent=2), encoding="utf-8")
        print("SKIPPED -- no APROCSA audio")
        return 0

    # The old policy and the shipped one, on identical audio, so the
    # difference is the policy and nothing else -- which is exactly what the
    # first (contended) measurement could not tell.
    configs = {
        "before (step 500ms, window 12s, reset 6s)":
            {"step_ms": 500, "max_window_s": 12.0, "reset_window_s": 6.0},
        "shipped (step 700ms, window 8s, reset 3.5s)":
            {"step_ms": 700, "max_window_s": 8.0, "reset_window_s": 3.5},
    }
    if args.configs:
        # Same "@k=v" spelling as eval/bench_asr_offline_wer.py, so a config
        # that was swept for WER there is timed here under the same name.
        floats = {"max_window_s", "reset_window_s", "max_turn_s"}
        configs = {}
        for name in args.configs.split(","):
            name = name.strip()
            if not name:
                continue
            kw = {}
            for part in name.split("@"):
                k, _, v = part.partition("=")
                if not v:
                    continue
                kw[k] = float(v) if k in floats else (
                    int(v) if v.lstrip("-").isdigit() else v)
            configs[name] = kw

    results: dict[str, list[dict]] = {k: [] for k in configs}
    for pid in pids:
        audio = load(pid, args.seconds, args.skip_s)
        if audio is None:
            print("  %s: no audio" % pid)
            continue
        print("  %s: %.0f s of real aphasic speech" % (pid, len(audio) / SR), flush=True)
        for name, kw in configs.items():
            r = run(audio, **kw)
            r["participant"] = pid
            results[name].append(r)
            print("    %-44s RTF %5.2f  commit lag %s ms (p90 %s)  window<=%.1fs"
                  % (name, r["rtf"], r["commit_lag_ms_median"],
                     r["commit_lag_ms_p90"], r["max_window_s"]), flush=True)

    summary = {}
    for name, rows in results.items():
        if not rows:
            continue
        lags = [r["commit_lag_ms_median"] for r in rows if r["commit_lag_ms_median"]]
        summary[name] = {
            "rtf_mean": round(float(np.mean([r["rtf"] for r in rows])), 3),
            "rtf_max": round(max(r["rtf"] for r in rows), 3),
            "commit_lag_ms_median": round(float(np.median(lags))) if lags else None,
            "words": sum(r["words"] for r in rows),
            "per_participant": rows,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "audio": "APROCSA -- real aphasic connected speech",
        "note": ("RTF here is the STREAM's, not a single transcribe's: the "
                 "wrapper re-transcribes a growing window every step, so cost "
                 "is calls-per-second times cost-per-call, and window length "
                 "sets the second term."),
        "stall_budget_ms": 1300,
        "summary": summary,
    }, indent=2), encoding="utf-8")

    print("")
    print("STREAMING ASR ON REAL APHASIC SPEECH")
    print("  %-44s %8s %8s %11s" % ("config", "RTF", "RTF max", "commit lag"))
    for name, s in summary.items():
        print("  %-44s %8.2f %8.2f %8s ms"
              % (name, s["rtf_mean"], s["rtf_max"], s["commit_lag_ms_median"]))
    print("")
    print("  RTF must stay below 1.0 or the transcript falls behind for as long")
    print("  as the speaker keeps talking. Commit lag must stay below the")
    print("  1300 ms stall threshold, or the predictor is handed a fragment")
    print("  that is missing the word right before the pause.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
