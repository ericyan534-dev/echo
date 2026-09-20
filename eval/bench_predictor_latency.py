"""Live predictor round-trip: the shipped non-streaming Gemini call against
the same call over generate_content_stream (GEMINI_STREAM), interleaved.

WHY THIS EXISTS
---------------
docs/EVAL.md cites the live LLM round-trip as an external constant
(1250-1800 ms, from scripts/e2e_live.py) and never re-measures it. That
number bounds every stall the prefetch cache misses, and nothing in the repo
could say which part of it is the model and which part is the transport.
This bench times `GeminiPredictor.predict` itself -- the exact class the app
constructs, prompt assembly and parsing included -- so the arms differ ONLY in
the `stream` flag (and, optionally, the model id).

METHOD
------
Arms are run INTERLEAVED, one call per arm per fragment per round, so a slow
minute at the API lands on every arm equally rather than on whichever arm ran
first. Four fragments cover a name-from-context recall, two concrete
circumlocutions and a hedge. Per arm: n, median, p90, min, max of wall-clock
ms around `predict`, plus whether every call returned at least one candidate
(a faster arm that returns nothing has not won anything).

WHAT IT DOES NOT MEASURE
------------------------
WebSocket and pipeline overhead (scripts/e2e_live.py), and accuracy -- a model
swap must be checked on the frozen set with eval/run_prediction_eval.py before
it is anything more than a latency number.

Usage:
    python eval/bench_predictor_latency.py                  # shipped model, 2 arms
    python eval/bench_predictor_latency.py --rounds 5
    python eval/bench_predictor_latency.py --models gemini-3.5-flash-lite
Requires GEMINI_API_KEY (via .env). Fails loudly without it; never falls back
to the mock, because a mock latency would be a fabricated number.

Output: eval/results/predictor_latency_bench.json (override with --out).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.predictor.gemini import GeminiPredictor  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "predictor_latency_bench.json"

# Deliberately NOT drawn from the frozen eval set: this bench is re-run freely
# and the freeze protocol forbids reacting to those items. Themes match the
# live demo beats (docs/DEMO_SCRIPT.md).
FRAGMENTS = [
    {"name": "name-from-context",
     "context": ["My sister Maria visited yesterday.", "She wants me to call her."],
     "fragment": "I need to call, um, the"},
    {"name": "concrete-object",
     "context": ["So what did you have for breakfast?"],
     "fragment": "I made some toast in the, um, the thing"},
    {"name": "place-from-fragment",
     "context": ["Where are you traveling next month?"],
     "fragment": "We're flying to, uh, the big city in Japan called"},
    {"name": "hedge",
     "context": ["Did you finish the garden?"],
     "fragment": "I still need to cut the, the, you know, the grass with the"},
]


def _stats(values: list[float]) -> dict:
    xs = sorted(values)
    return {
        "n": len(xs),
        "median_ms": round(statistics.median(xs), 1),
        "p90_ms": round(xs[max(0, int(round(0.9 * len(xs))) - 1)], 1),
        "min_ms": round(xs[0], 1),
        "max_ms": round(xs[-1], 1),
    }


async def _one(pred: GeminiPredictor, frag: dict) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    cands = await pred.predict(frag["context"], frag["fragment"])
    ms = (time.perf_counter() - t0) * 1000.0
    top = cands[0].word if cands else ""
    return ms, len(cands), top


async def run(arms: dict[str, GeminiPredictor], rounds: int, gap_s: float,
              echo) -> dict:
    calls: dict[str, list[dict]] = {k: [] for k in arms}
    for r in range(rounds):
        for frag in FRAGMENTS:
            for name, pred in arms.items():
                try:
                    ms, n_c, top = await _one(pred, frag)
                    calls[name].append({"round": r, "fragment": frag["name"],
                                        "ms": round(ms, 1), "n_candidates": n_c,
                                        "top": top})
                    echo(f"  r{r} {frag['name']:<20s} {name:<34s} {ms:7.0f} ms  "
                         f"{n_c} cands  top={top!r}")
                except Exception as exc:  # recorded, never hidden
                    calls[name].append({"round": r, "fragment": frag["name"],
                                        "error": f"{type(exc).__name__}: {exc}"[:200]})
                    echo(f"  r{r} {frag['name']:<20s} {name:<34s} ERROR {type(exc).__name__}")
                if gap_s:
                    await asyncio.sleep(gap_s)
    summary = {}
    for name, rows in calls.items():
        ok = [x for x in rows if "ms" in x]
        summary[name] = {
            **(_stats([x["ms"] for x in ok]) if ok else {"n": 0}),
            "errors": len(rows) - len(ok),
            "empty_candidate_calls": sum(1 for x in ok if x["n_candidates"] == 0),
        }
    return {"summary": summary, "calls": calls}


def _ascii(s: str) -> str:
    return s.encode("ascii", "backslashreplace").decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--rounds", type=int, default=3,
                    help="rounds over the 4 fragments per arm (default 3 -> n=12 per arm)")
    ap.add_argument("--gap", type=float, default=0.3, help="seconds between calls")
    ap.add_argument("--models", default="",
                    help="comma-separated extra model ids to add as non-streaming arms")
    ap.add_argument("--out", type=Path, default=RESULTS)
    args = ap.parse_args()

    s = get_settings()
    if not s.gemini_api_key:
        print("GEMINI_API_KEY (or GOOGLE_API_KEY) is required; this bench never uses the mock.")
        return 2

    arms: dict[str, GeminiPredictor] = {
        f"{s.gemini_model} nonstream (shipped)": GeminiPredictor(
            s.gemini_api_key, s.gemini_model, s.max_candidates, stream=False),
        f"{s.gemini_model} stream": GeminiPredictor(
            s.gemini_api_key, s.gemini_model, s.max_candidates, stream=True),
    }
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        arms[f"{m} nonstream"] = GeminiPredictor(s.gemini_api_key, m, s.max_candidates, stream=False)
        arms[f"{m} stream"] = GeminiPredictor(s.gemini_api_key, m, s.max_candidates, stream=True)

    print(f"predictor latency bench: {len(arms)} arms x {len(FRAGMENTS)} fragments x "
          f"{args.rounds} rounds, interleaved")
    t0 = time.time()
    res = asyncio.run(run(arms, args.rounds, args.gap, lambda m: print(_ascii(m))))

    print()
    for name, st in res["summary"].items():
        if st.get("n"):
            print(f"{name:<40s} n={st['n']:2d} median {st['median_ms']:6.0f} ms  p90 {st['p90_ms']:6.0f}"
                  f"  min {st['min_ms']:6.0f}  max {st['max_ms']:6.0f}  errors={st['errors']}"
                  f"  empty={st['empty_candidate_calls']}")
        else:
            print(f"{name:<40s} no successful calls (errors={st['errors']})")

    out = {
        "provenance": {
            "script": "eval/bench_predictor_latency.py",
            "model": s.gemini_model,
            "extra_models": args.models,
            "rounds": args.rounds,
            "fragments": [f["name"] for f in FRAGMENTS],
            "gap_s": args.gap,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "measures": "wall-clock ms around GeminiPredictor.predict (prompt assembly + "
                        "API round-trip + parse); excludes WebSocket/pipeline overhead",
            "interleaved": True,
        },
        **res,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
