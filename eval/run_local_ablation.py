"""Local-LLM parameter ablation: what do thinking depth, GBNF grammar, and
temperature actually buy on Echo's task?

WHY THIS EXISTS
---------------
The local provider (llama.cpp serving Qwen3.8-27B Q4_K_M) has knobs the cloud
providers do not, and the shipped defaults were chosen from one measurement
plus a bug fix, not from a sweep. This measures them on the frozen 60-item
circumlocution set -- the same benchmark and the same scorer as
eval/run_prediction_eval.py -- so accuracy and latency are directly comparable
to the cloud numbers in docs/EVAL.md.

The relationship being tested is a TRADE, not a maximum. Echo has a latency
budget: a word offered after the conversation has moved on is worthless however
correct it is. So an arm that buys +0.05 top-1 for +5 s is a loss, and this
script reports both columns side by side rather than ranking on accuracy alone.

METHOD NOTE, stated because it changes how the numbers should be read: the
payload is built by the real `LocalPredictor._build_payload` and parsed by the
real `_text_of` / `_parse`, so this exercises the shipped code path rather than
a reimplementation of it. The HTTP call is made here only so that per-request
server telemetry (token counts, prefill/decode split, cache hits) can be
captured, which `predict()` does not surface.

Run llama-server first (scripts/setup_local_llm.py prints the command), then:

    python eval/run_local_ablation.py
    python eval/run_local_ablation.py --limit 10        # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET = ROOT / "eval" / "data" / "prediction_eval_set.jsonl"
OUT = ROOT / "eval" / "results" / "local_ablation.json"

# (label, kwargs for LocalPredictor)
# Arm 1 is the shipped configuration; every other arm is a single-knob change
# from it, so each delta is attributable to one parameter.
ARMS: list[tuple[str, dict]] = [
    ("shipped (no thinking)",   {}),
    ("thinking=low",            {"reasoning_effort": "low"}),
    ("thinking=medium",         {"reasoning_effort": "medium"}),
    ("thinking=high",           {"reasoning_effort": "high"}),
    ("thinking=xhigh",          {"reasoning_effort": "xhigh"}),
    ("no GBNF grammar",         {"use_grammar": False}),
    ("temperature=0.0",         {"temperature": 0.0}),
]


def load_items(limit: int | None) -> list[dict]:
    with DATASET.open(encoding="utf-8") as fh:
        items = [json.loads(line) for line in fh if line.strip()]
    return items[:limit] if limit else items


async def run_arm(label: str, kwargs: dict, items: list[dict],
                  base_url: str, timeout_s: float) -> dict:
    import aiohttp

    from backend.predictor.local import LocalPredictor, _parse, _text_of
    from eval.run_prediction_eval import score_candidates

    pred = LocalPredictor(base_url=base_url, timeout_s=timeout_s,
                          max_tokens=512, **kwargs)
    rows: list[dict] = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_s)) as session:
        for item in items:
            payload = pred._build_payload(item.get("context") or [], item["fragment"])
            t0 = time.perf_counter()
            text, usage, timings, err = None, {}, {}, None
            try:
                async with session.post(pred.endpoint, json=payload) as resp:
                    data = await resp.json(content_type=None)
                if resp.status == 200:
                    text = _text_of(data)
                    usage = data.get("usage") or {}
                    timings = data.get("timings") or {}
                else:
                    err = "http %s" % resp.status
            except Exception as exc:            # a dead arm must not kill the sweep
                err = type(exc).__name__
            dt = (time.perf_counter() - t0) * 1000.0

            cands = _parse(text, pred.max_candidates) if text else []
            words = [c.word for c in cands]
            top1, top3 = score_candidates(item["gold"], words)
            rows.append({
                "id": item.get("id"), "category": item.get("category"),
                "expected": item["gold"], "got": words,
                "top1": top1, "top3": top3,
                "latency_ms": round(dt, 1),
                "completion_tokens": usage.get("completion_tokens"),
                "decode_tok_s": timings.get("predicted_per_second"),
                "empty": not words, "error": err,
            })

    lat = sorted(r["latency_ms"] for r in rows)
    toks = [r["completion_tokens"] for r in rows if r["completion_tokens"]]
    n = len(rows)
    return {
        "arm": label, "kwargs": kwargs, "n": n,
        "top1": round(sum(r["top1"] for r in rows) / n, 4),
        "top3": round(sum(r["top3"] for r in rows) / n, 4),
        "empty_rate": round(sum(r["empty"] for r in rows) / n, 4),
        "error_rate": round(sum(1 for r in rows if r["error"]) / n, 4),
        "latency_ms_median": round(statistics.median(lat), 1),
        "latency_ms_p90": round(lat[min(len(lat) - 1, int(0.9 * len(lat)))], 1),
        "completion_tokens_median": (round(statistics.median(toks), 1) if toks else None),
        "rows": rows,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N items only (smoke test; NOT publishable)")
    ap.add_argument("--timeout", type=float, default=180.0,
                    help="per-call ceiling; thinking arms are slow")
    args = ap.parse_args()

    import aiohttp
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(args.base_url + "/health",
                             timeout=aiohttp.ClientTimeout(total=5)) as r:
                if r.status != 200:
                    raise RuntimeError("status %s" % r.status)
    except Exception as exc:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED",
                                   "reason": "llama-server unreachable at %s (%s)"
                                             % (args.base_url, exc)}, indent=2),
                       encoding="utf-8")
        print("SKIPPED -- no llama-server at %s" % args.base_url)
        print("  start it with: python scripts/setup_local_llm.py")
        return 0

    items = load_items(args.limit)
    print("LOCAL LLM ABLATION -- %d items x %d arms" % (len(items), len(ARMS)))
    print("  frozen set: %s" % DATASET.name)
    if args.limit:
        print("  WARNING: --limit is set, results are exploratory, not publishable")
    print("")

    summaries = []
    for label, kwargs in ARMS:
        t0 = time.perf_counter()
        s = await run_arm(label, kwargs, items, args.base_url, args.timeout)
        s["wall_s"] = round(time.perf_counter() - t0, 1)
        summaries.append(s)
        print("  %-24s top1=%.3f top3=%.3f  median=%7.0f ms  p90=%7.0f ms  "
              "tok=%-5s empty=%.2f  (%.0fs)"
              % (label, s["top1"], s["top3"], s["latency_ms_median"],
                 s["latency_ms_p90"], s["completion_tokens_median"],
                 s["empty_rate"], s["wall_s"]))

    base = summaries[0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "dataset": DATASET.name, "n_items": len(items),
        "limited": bool(args.limit),
        "baseline_arm": base["arm"], "arms": summaries,
    }, indent=2), encoding="utf-8")

    print("")
    print("DELTAS vs shipped (%s)" % base["arm"])
    for s in summaries[1:]:
        d_acc = s["top1"] - base["top1"]
        d_lat = s["latency_ms_median"] - base["latency_ms_median"]
        verdict = ("WORSE on both" if d_acc <= 0 and d_lat > 0 else
                   "better, and cheaper" if d_acc > 0 and d_lat <= 0 else
                   "cheaper, less accurate" if d_acc <= 0 and d_lat <= 0 else
                   "more accurate, slower")
        print("  %-24s top1 %+.3f   median latency %+7.0f ms   -> %s"
              % (s["arm"], d_acc, d_lat, verdict))
    print("")
    print("  Read as a TRADE, not a ranking: Echo's word must arrive while the")
    print("  sentence is still open, so latency is not a tiebreaker -- it is a")
    print("  constraint. Cloud reference for the same set is in docs/EVAL.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
