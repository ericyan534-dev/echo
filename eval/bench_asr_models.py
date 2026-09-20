"""Which CrisperWhisper size should Echo ship, and does verbatim mode survive it?

Two axes decide this, and they pull against each other:

  QUALITY  -- does the dysfluency reach the transcript at all? This is the
              entire reason for the swap. Chrome scores 0.000 here (measured,
              n=5044, eval/results/stall_eval.json), so any model that keeps
              the evidence is an improvement; the question is how much is left
              on the table by taking a smaller one.

  LATENCY  -- Echo's stall->word budget is ~1.5-2 s TOTAL, shared with the
              LLM. The ASR's share is not its RTF, it is the delay between the
              speaker stopping and the words being committed, because the
              predictor cannot run on a sentence that is missing its last word.

Reported per dysfluency type, not just in aggregate, because the types are not
interchangeable evidence for a word-finding aid: a Block (silent struggle to
initiate) is the strongest signal that the speaker is stuck, and an Interjection
("um") is the weakest, since fluent speakers produce them constantly.

    python eval/bench_asr_models.py --n 60
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sep28k_audio import episode_path  # noqa: E402

MANIFEST = ROOT / "data" / "sep28k" / "manifest.json"
OUT = ROOT / "eval" / "results" / "asr_model_bench.json"

SR = 16000
WINDOW_S = 6.0
TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
MODELS = ["small", "medium", "turbo", "large"]

# Apple's spec: an Interjection is "um"/"uh" OR a PERSON-SPECIFIC filler the
# speaker uses to cope. No fixed word list can be complete, so every lexical
# rule here is a LOWER BOUND on what a human annotator would accept.
FILLER_RE = re.compile(
    r"\[(uh|um|uhm)\]|(?<![a-z])(uh+|um+|erm|er|mm+|hmm+)(?![a-z])"
    r"|you know|i mean|sort of|kind of", re.I)
# CrisperWhisper renders a cut-off word with a trailing hyphen -- "f- Facebook",
# "R- Re- re- re- Received". That token IS the block/sound-repetition evidence,
# and it is exactly what an "intended" transcript deletes.
FRAGMENT_RE = re.compile(r"(?<![a-z])[a-z]{1,4}-(?=\s|$)", re.I)
_TOK = re.compile(r"[a-z']+")


def has_repetition(text: str) -> bool:
    toks = _TOK.findall((text or "").lower())
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1]:
            return True
    for n in (2, 3):
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i + n] == toks[i + n:i + 2 * n]:
                return True
    return False


def evidence(text: str) -> dict:
    """Which kinds of dysfluency evidence survived into the transcript."""
    return {
        "filler": bool(FILLER_RE.search(text or "")),
        "fragment": bool(FRAGMENT_RE.search(text or "")),
        "repetition": has_repetition(text),
    }


def load_events(n: int) -> list[dict]:
    """Frozen, seeded selection: every model sees the SAME events, and a rerun
    reproduces the set exactly."""
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    pool = [c for c in man["clips"]
            if c["usable"] and any(c["labels"][t] for t in TYPES)]
    pool.sort(key=lambda c: (c["show"], int(c["ep"]), int(c["clip"])))
    # Round-robin by dominant type so no single type dominates the sample.
    buckets: dict[str, list] = {t: [] for t in TYPES}
    for c in pool:
        for t in TYPES:
            if c["labels"][t]:
                buckets[t].append(c)
                break
    # Spread each type's picks EVENLY through its (show, ep, clip)-sorted
    # bucket rather than taking the head. Taking the head drew all 60 events
    # from a single show's first episode, which is not a sample of SEP-28k --
    # it is a sample of one speaker, and it silently made an unreadable episode
    # look like a total decode failure.
    per_type = max(1, n // len(TYPES))
    out = []
    for t in TYPES:
        b = buckets[t]
        if not b:
            continue
        stride = max(1, len(b) // per_type)
        out.extend(b[i * stride] for i in range(min(per_type, len(b) // stride)))
    return out[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--models", default=",".join(MODELS))
    args = ap.parse_args()

    if not MANIFEST.exists():
        print("SKIPPED -- no SEP-28k manifest")
        return 0

    import numpy as np
    import soundfile as sf
    import torch

    from backend.stt.verbatim import get_model

    events = load_events(args.n)
    print("ASR MODEL BENCH -- %d SEP-28k events, %.0f s windows" % (len(events), WINDOW_S))

    # Decode audio ONCE so every model sees byte-identical input.
    clips = []
    for c in events:
        ep = episode_path(c["show"], c["ep"])
        if ep is None:
            continue
        try:
            info = sf.info(str(ep))
            scale = info.samplerate / SR       # SEP-28k offsets are 16 kHz samples
            mid = int(((c["start"] + c["stop"]) // 2) * scale)
            half = int(WINDOW_S * info.samplerate / 2)
            a, sr = sf.read(str(ep), start=max(0, mid - half),
                            stop=min(info.frames, mid + half), dtype="float32")
        except Exception:
            continue
        if a.ndim > 1:
            a = a.mean(axis=1)
        if sr != SR:
            idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
            a = np.interp(idx, np.arange(len(a)), a).astype("float32")
        if len(a) < SR:
            continue
        clips.append((c, a))
    print("  decoded %d/%d events\n" % (len(clips), len(events)))

    results = {}
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        try:
            model = get_model(name)
        except Exception as exc:
            print("  %s: LOAD FAILED (%s)" % (name, exc))
            continue
        model.transcribe(np.zeros(SR, dtype="float32"), sr=SR, language="en")  # warm

        rows, lat_nots, lat_ts = [], [], []
        for c, a in clips:
            t0 = time.perf_counter()
            r = model.transcribe(a, sr=SR, language="en", mode="verbatim")
            lat_nots.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            model.transcribe(a, sr=SR, language="en", mode="verbatim", word_timestamps=True)
            lat_ts.append((time.perf_counter() - t0) * 1000)
            ev = evidence(r.text)
            rows.append({
                "show": c["show"], "ep": c["ep"], "clip": c["clip"],
                "labels": {t: c["labels"][t] for t in TYPES},
                "text": (r.text or "")[:160],
                "evidence": ev,
                "any": any(ev.values()),
                # The metric the repo already published, kept so the new number
                # is comparable to the committed one rather than replacing it
                # with a quietly easier test.
                "legacy": ev["filler"] or ev["repetition"],
            })

        by_type = {}
        for t in TYPES:
            sel = [r for r in rows if r["labels"][t]]
            by_type[t] = {
                "n": len(sel),
                "preserved": round(sum(r["any"] for r in sel) / len(sel), 3) if sel else None,
            }
        results[name] = {
            "n": len(rows),
            "preserved": round(sum(r["any"] for r in rows) / len(rows), 4) if rows else None,
            "preserved_legacy_metric": (round(sum(r["legacy"] for r in rows) / len(rows), 4)
                                        if rows else None),
            "by_type": by_type,
            "latency_ms_median": round(statistics.median(lat_nots), 1),
            "latency_ms_median_word_timestamps": round(statistics.median(lat_ts), 1),
            "word_timestamp_overhead_ms": round(statistics.median(lat_ts)
                                                - statistics.median(lat_nots), 1),
            "samples": rows[:8],
        }
        print("  %-8s preserved %.3f (legacy %.3f)  %5.0f ms  (+%.0f ms word-ts)"
              % (name, results[name]["preserved"], results[name]["preserved_legacy_metric"],
                 results[name]["latency_ms_median"],
                 results[name]["word_timestamp_overhead_ms"]))
        del model
        from backend.stt import verbatim as _v
        _v._MODEL_CACHE.clear()
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "source": "SEP-28k (74%% subset, 5 of 8 shows) -- see docs/DATA_PROVENANCE.md",
        "window_s": WINDOW_S, "n_events": len(clips),
        "metric": ("dysfluency preserved = explicit filler token OR discourse "
                   "filler OR cut-off word fragment ('f-') OR immediate "
                   "word/phrase repetition. LOWER BOUND: Apple's Interjection "
                   "definition includes person-specific fillers no fixed list "
                   "can enumerate."),
        "chrome_baseline": {"preserved": 0.0, "n": 5044,
                            "source": "eval/results/stall_eval.json transcript_baseline"},
        "models": results,
    }, indent=2), encoding="utf-8")

    print("")
    print("DYSFLUENCY PRESERVED BY TYPE (%d events, %.0f s windows)" % (len(clips), WINDOW_S))
    hdr = "  %-8s %7s " % ("model", "all") + " ".join("%12s" % t for t in TYPES)
    print(hdr)
    for name, r in results.items():
        line = "  %-8s %7.3f " % (name, r["preserved"])
        for t in TYPES:
            v = r["by_type"][t]["preserved"]
            line += " %12s" % (("%.3f (%d)" % (v, r["by_type"][t]["n"])) if v is not None else "-")
        print(line)
    print("  %-8s %7.3f  <- current stack, measured n=5044" % ("Chrome", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
