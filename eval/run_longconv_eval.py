"""Does accuracy hold up late in a long conversation?

This is the gate for the product claim "the more context it gains, the more
accurate it gets". It reports top-1 accuracy BUCKETED BY TURN DEPTH, and it
runs A/B: with the v3 ContextBuilder (rolling summary + entity layer) against
the previous behaviour (raw recent-turns tail). If the late buckets are not
better with the builder, the claim is unsupported and this script says so.

The gate can fail. That is the point -- a harness that cannot fail proves
nothing.

Writes eval/results/longconv_eval.json. Never hand-edit docs/EVAL.md;
regenerate it with eval/make_report.py.

    python eval/run_longconv_eval.py                # live predictor from .env
    python eval/run_longconv_eval.py --provider mock

Honest limits: the conversations are hand-authored by the team, not
transcripts of people with aphasia. Filler turns are deliberately neutral, so
this measures LONG-RANGE RECALL under clean conditions -- an upper bound, not
field performance. Same caveat class as the frozen circumlocution set.

FREEZE PROTOCOL: eval/data/longconv_eval_set.jsonl is frozen at its first
commit. After the first live run, items may never be edited, added, or removed
in response to results; any change requires a new versioned filename with both
result sets kept.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "eval" / "data" / "longconv_eval_set.jsonl"
OUT = ROOT / "eval" / "results" / "longconv_eval.json"


def load_set() -> list[dict]:
    with DATA.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def score_probe(candidates: list[str], expected: list[str]) -> bool:
    """Top-1 exact match, case-insensitive."""
    if not candidates:
        return False
    return candidates[0].strip().lower() in {e.strip().lower() for e in expected}


def bucket(depth: int) -> str:
    if depth <= 10:
        return "early(<=10)"
    if depth <= 25:
        return "mid(11-25)"
    return "late(>25)"


async def run_one(conv: dict, predictor, use_builder: bool, verbatim: int) -> list[dict]:
    from backend.context import ContextBuilder
    from backend.pipeline import EchoPipeline
    from backend.stall_detector import StallDetector
    from backend.summarizer import ExtractiveSummarizer

    builder = (ContextBuilder(ExtractiveSummarizer(), verbatim_turns=verbatim)
               if use_builder else None)
    pipe = EchoPipeline(StallDetector(), predictor, prefetch=False,
                        context_builder=builder, entity_memory=True)

    rows: list[dict] = []
    probes = {p["depth"]: p for p in conv["probes"]}
    for i, turn in enumerate(conv["turns"], start=1):
        await pipe.handle_text_turn(turn)
        probe = probes.get(i)
        if probe is None:
            continue
        if builder is not None:
            payload = builder.build(
                conversation_turns=pipe.conversation.turns,
                utterance=probe["fragment"],
                summary=pipe.rolling_summary,
            )
            context, entities = payload.recent_turns, payload.entities
            if payload.summary:
                context = [payload.summary] + context
        else:
            context, entities = pipe.conversation.recent(verbatim), []
        cands = await predictor.predict(context, probe["fragment"],
                                        entities=entities or None)
        words = [c.word for c in cands]
        rows.append({
            "conv": conv["id"], "depth": probe["depth"], "bucket": bucket(probe["depth"]),
            "arm": "builder" if use_builder else "baseline",
            "expected": probe["expect"], "got": words,
            "correct": score_probe(words, probe["expect"]),
        })
    return rows


def summarize(rows: list[dict], arm: str) -> dict:
    buckets: dict[str, list[bool]] = {}
    for r in rows:
        if r["arm"] == arm:
            buckets.setdefault(r["bucket"], []).append(r["correct"])
    return {b: {"n": len(v), "top1": round(sum(v) / len(v), 4)}
            for b, v in sorted(buckets.items())}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None,
                    help="override PREDICTOR_PROVIDER (e.g. mock)")
    ap.add_argument("--verbatim", type=int, default=6)
    args = ap.parse_args()

    if args.provider:
        import os
        os.environ["PREDICTOR_PROVIDER"] = args.provider
    from backend.config import get_settings
    from backend.session import make_predictor

    convs = load_set()
    rows: list[dict] = []
    for arm in (False, True):                      # baseline first, then builder
        predictor = make_predictor(get_settings())
        for conv in convs:
            rows.extend(await run_one(conv, predictor, arm, args.verbatim))

    base, built = summarize(rows, "baseline"), summarize(rows, "builder")

    # Provider-tagged output. Both arms of this A/B are meaningless without
    # knowing WHICH model produced them, and a local run silently overwriting a
    # cloud run is how a report ends up quoting numbers from the wrong engine.
    settings = get_settings()
    provider = settings.predictor_provider
    out_path = OUT if not args.provider else OUT.with_name(
        OUT.stem + "_" + provider + OUT.suffix)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "status": "OK", "n": len(rows),
        "provider": provider,
        "model": (settings.local_llm_model if provider == "local"
                  else settings.gemini_model),
        "verbatim_turns": args.verbatim,
        "baseline_by_depth": base, "builder_by_depth": built,
        "rows": rows,
    }, indent=2), encoding="utf-8")
    print("wrote %s (provider=%s)" % (out_path.name, provider))

    print("LONG-CONVERSATION RECALL -- top-1 by turn depth")
    print("  %-14s %18s %18s" % ("bucket", "baseline", "with ContextBuilder"))
    for b in sorted(set(base) | set(built)):
        bs, bl = base.get(b, {}), built.get(b, {})
        print("  %-14s   n=%3d  top1=%.3f   n=%3d  top1=%.3f"
              % (b, bs.get("n", 0), bs.get("top1", 0.0),
                 bl.get("n", 0), bl.get("top1", 0.0)))

    late_base = base.get("late(>25)", {}).get("top1", 0.0)
    late_built = built.get("late(>25)", {}).get("top1", 0.0)
    early_built = built.get("early(<=10)", {}).get("top1", 0.0)
    print("")
    print("  late bucket: baseline %.3f -> builder %.3f  (delta %+.3f)"
          % (late_base, late_built, late_built - late_base))
    print("  builder late-vs-early delta: %+.3f" % (late_built - early_built))
    print("")
    if late_built <= late_base:
        print("  VERDICT: ContextBuilder did NOT improve deep recall -- claim unsupported")
    elif late_built < early_built - 0.05:
        print("  VERDICT: improved, but accuracy still DEGRADES with depth")
    else:
        print("  VERDICT: deep recall improved and holds at depth")
    print("")
    print("  NOTE: hand-authored conversations with neutral filler; an upper")
    print("        bound on long-range recall, not field performance.")


if __name__ == "__main__":
    asyncio.run(main())
