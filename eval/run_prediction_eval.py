"""Prediction-accuracy eval on the frozen hand-authored circumlocution set.

Every other number in docs/EVAL.md measures detection or latency; this script
measures whether the predicted WORD IS RIGHT.  It runs a WordPredictor over
eval/data/prediction_eval_set.jsonl (60 hand-authored items: 20 concrete
object circumlocutions, 20 proper-noun-from-context items, 20 abstract/verb
items) and scores top-1 / top-3 accuracy against per-item gold lists.

Scoring: a candidate matches a gold entry iff normalize(candidate) ==
normalize(gold) where normalize = lowercase, strip one trailing "'s", strip
punctuation, collapse whitespace, strip one trailing plural "s".  The per-item
gold list IS the equivalence rule -- every acceptable surface form is
enumerated explicitly; nothing fuzzier than the normalization above is ever
applied.

FREEZE PROTOCOL (also echoed into the results JSON):
  The dataset eval/data/prediction_eval_set.jsonl is frozen at its first
  commit.  After the first live run, items may never be edited, added, or
  removed in response to results.  One headline live run per condition (full
  and --ablate-context).  Any future change to the set requires a NEW
  versioned filename (e.g. prediction_eval_set_v2.jsonl) and both the old and
  new results files must be kept.

The provider is constructed through backend.predictor.get_predictor -- the
same factory the app uses -- so the eval exercises the exact prompt assets and
parsing the live system ships.  A missing API key fails loudly (no silent mock
fallback).  --ablate-context re-runs with context=[] to prove conversation
context earns its place (proper_context items should collapse); it writes to a
separate results file so both headline runs coexist.

Usage:
    python eval/run_prediction_eval.py --provider mock  --sleep 0     # smoke
    python eval/run_prediction_eval.py --provider gemini              # live
    python eval/run_prediction_eval.py --provider gemini --ablate-context
Options: --limit N (first N items; output gets a _smoke suffix so headline
results are never overwritten), --sleep S (inter-call sleep, default 1.0 s for
rate caps), --out PATH (override the output path).

Output: eval/results/prediction_eval.json
        eval/results/prediction_eval_ablated.json  (with --ablate-context)

--dataset / --entity-memory (context-window extension, ENTITY_MEMORY):
  --dataset PATH selects a different JSONL set with the same schema; default
  is the frozen set above and behavior for that default is unchanged.
  --entity-memory {on,off} (default off) windows each item's context to
  Settings.context_turns (mirroring the live Conversation.recent) and, when
  "on", injects out-of-window salient entities via backend.entities.
  EntityTracker exactly like the live pipeline -- same code, not
  reimplemented. Results filename encodes dataset+condition, e.g.:
    python eval/run_prediction_eval.py --provider mock --sleep 0 \\
        --dataset eval/data/prediction_eval_longctx_v1.jsonl
        # -> eval/results/prediction_eval_longctx_v1.json
    python eval/run_prediction_eval.py --provider mock --sleep 0 \\
        --dataset eval/data/prediction_eval_longctx_v1.jsonl --entity-memory on
        # -> eval/results/prediction_eval_longctx_v1_entity.json
  Same freeze protocol as above, applied to whichever --dataset is passed.
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.entities import EntityTracker  # noqa: E402
from backend.predictor import get_predictor  # noqa: E402
from backend.predictor.base import WordPredictor  # noqa: E402

DATASET = ROOT / "eval" / "data" / "prediction_eval_set.jsonl"
RESULTS_FULL = ROOT / "eval" / "results" / "prediction_eval.json"
RESULTS_ABLATED = ROOT / "eval" / "results" / "prediction_eval_ablated.json"

CATEGORIES = ("concrete", "proper_context", "abstract_verb")
REQUIRED_KEYS = ("id", "category", "context", "fragment", "gold", "notes")

FREEZE_PROTOCOL = (
    "The dataset eval/data/prediction_eval_set.jsonl is frozen at its first "
    "commit. After the first live run, items may never be edited, added, or "
    "removed in response to results. One headline live run per condition "
    "(full and context-ablated). Any future change to the set requires a new "
    "versioned filename and both old and new results files must be kept."
)

NORMALIZATION_RULE = (
    'lowercase; strip one trailing "\'s"; strip punctuation; collapse '
    'whitespace; strip one trailing plural "s"'
)


# ---------------------------------------------------------------------------
# scoring primitives (pure functions -- unit-tested in tests/test_prediction_eval.py)
# ---------------------------------------------------------------------------
def normalize(word: str) -> str:
    """Normalize a word/phrase for gold matching.

    lowercase -> strip one trailing "'s" -> strip punctuation -> collapse
    whitespace -> strip one trailing plural "s".  Applied identically to gold
    entries and candidates, so simple s-plurals cross-match; anything beyond
    that must be enumerated in the gold list.
    """
    w = word.lower().strip()
    if w.endswith("'s"):
        w = w[:-2]
    w = re.sub(r"[^\w\s]", "", w)
    w = re.sub(r"\s+", " ", w).strip()
    if len(w) > 1 and w.endswith("s"):
        w = w[:-1]
    return w


def score_candidates(gold: list[str], words: list[str]) -> tuple[bool, bool]:
    """Return (top1_hit, top3_hit) for ranked candidate words vs a gold list."""
    gold_norm = {normalize(g) for g in gold}
    cand_norm = [normalize(w) for w in words]
    top1 = bool(cand_norm) and cand_norm[0] in gold_norm
    top3 = any(c in gold_norm for c in cand_norm[:3])
    return top1, top3


def load_items(path: Path) -> list[dict]:
    """Load and validate the JSONL eval set; raise ValueError on any schema error."""
    if not path.exists():
        raise ValueError(f"eval set not found: {path}")
    items: list[dict] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{lineno}: invalid JSON ({exc})") from exc
        if not isinstance(item, dict):
            raise ValueError(f"{path.name}:{lineno}: item is not an object")
        missing = [k for k in REQUIRED_KEYS if k not in item]
        if missing:
            raise ValueError(f"{path.name}:{lineno}: missing keys {missing}")
        if item["category"] not in CATEGORIES:
            raise ValueError(
                f"{path.name}:{lineno}: bad category {item['category']!r} "
                f"(expected one of {CATEGORIES})")
        if not isinstance(item["context"], list) or not all(isinstance(c, str) for c in item["context"]):
            raise ValueError(f"{path.name}:{lineno}: context must be a list of strings")
        if not isinstance(item["fragment"], str) or not item["fragment"].strip():
            raise ValueError(f"{path.name}:{lineno}: fragment must be a non-empty string")
        if (not isinstance(item["gold"], list) or not item["gold"]
                or not all(isinstance(g, str) and g.strip() for g in item["gold"])):
            raise ValueError(f"{path.name}:{lineno}: gold must be a non-empty list of strings")
        if item["id"] in seen_ids:
            raise ValueError(f"{path.name}:{lineno}: duplicate id {item['id']!r}")
        seen_ids.add(item["id"])
        items.append(item)
    if not items:
        raise ValueError(f"{path.name}: no items")
    return items


# ---------------------------------------------------------------------------
# eval loop
# ---------------------------------------------------------------------------
def _rate(hits: int, n: int) -> float:
    return round(hits / n, 4) if n else 0.0


def _counts(hits: int, n: int) -> dict:
    """Raw count + rate together -- '41/60' reads more honestly than '68%'."""
    return {"hits": hits, "n": n, "rate": _rate(hits, n)}


def _percentile(values: list[float], pct: float) -> float | None:
    """Nearest-rank percentile in ms (pct in 0..100). None on empty input.

    Nearest-rank (not interpolated) keeps the number honest on a 60-item set:
    every reported figure is an ACTUAL measured call, not a synthesized point
    between two calls."""
    if not values:
        return None
    ordered = sorted(values)
    k = max(0, min(len(ordered) - 1, int(round((pct / 100.0) * len(ordered) + 0.5)) - 1))
    return round(ordered[k], 1)


async def run_eval(
    predictor: WordPredictor,
    items: list[dict],
    *,
    ablate_context: bool = False,
    sleep_s: float = 0.0,
    provider: str = "?",
    model: str = "?",
    dataset_path: Path | None = None,
    entity_memory: bool = False,
    context_turns: int = 6,
    echo=None,
) -> dict:
    """Run the predictor over all items sequentially and score. Returns the
    full results dict (provenance + aggregates + per-item log).

    *context_turns* windows each item's `context` to its last N turns before
    sending it to the predictor -- mirroring the live system's
    Conversation.recent(context_turns) exactly. This is a no-op for the
    frozen set (whose items never exceed a couple of context turns), so the
    default headline command is unaffected; it matters for a long-context
    dataset (e.g. eval/data/prediction_eval_longctx_v1.jsonl) where the whole
    point is to simulate the live 6-turn collapse.

    *entity_memory*: when True, builds a fresh backend.entities.EntityTracker
    over each item's FULL context (not just the windowed slice) and injects
    the out-of-window entities exactly like the live pipeline does (see
    backend.pipeline.EchoPipeline._entities_kwargs) -- same code, not
    reimplemented. When False, the predictor call is the original 2-arg form
    (no `entities` kwarg at all), so any predictor double with the pre-
    entity-memory signature still works unchanged.
    """
    say = echo or (lambda s: None)
    per_item: list[dict] = []
    conf_correct: list[float] = []
    conf_wrong: list[float] = []
    latencies_ms: list[float] = []
    tracker = EntityTracker()

    for i, item in enumerate(items):
        full_context = list(item["context"])
        if ablate_context:
            context = []
        elif context_turns > 0:
            context = full_context[-context_turns:]
        else:
            context = list(full_context)

        kwargs: dict = {}
        if entity_memory and not ablate_context:
            entities = tracker.out_of_window(full_context, context_turns)
            if entities:
                kwargs["entities"] = entities

        t_call = time.perf_counter()
        cands = await predictor.predict(context, item["fragment"], **kwargs)
        latency_ms = (time.perf_counter() - t_call) * 1000.0
        latencies_ms.append(latency_ms)
        words = [c.word for c in cands]
        top1, top3 = score_candidates(item["gold"], words)
        if cands:  # confidence of the top-ranked candidate, split by top-1 hit
            (conf_correct if top1 else conf_wrong).append(float(cands[0].confidence))
        per_item.append({
            "id": item["id"],
            "category": item["category"],
            "gold": item["gold"],
            "candidates": [{"word": c.word, "confidence": round(float(c.confidence), 3)}
                           for c in cands],
            "top1_hit": top1,
            "top3_hit": top3,
            "latency_ms": round(latency_ms, 1),
        })
        mark = "HIT@1" if top1 else ("HIT@3" if top3 else "MISS ")
        say(f"[{i + 1:2d}/{len(items)}] {item['id']:<14s} {mark}  "
            f"gold={item['gold'][0]!r} -> {words if words else '(no candidates)'}")
        if sleep_s and i < len(items) - 1:
            await asyncio.sleep(sleep_s)

    overall = {
        "top1": _counts(sum(r["top1_hit"] for r in per_item), len(per_item)),
        "top3": _counts(sum(r["top3_hit"] for r in per_item), len(per_item)),
    }
    per_category: dict[str, dict] = {}
    for cat in CATEGORIES:
        rows = [r for r in per_item if r["category"] == cat]
        if rows:
            per_category[cat] = {
                "top1": _counts(sum(r["top1_hit"] for r in rows), len(rows)),
                "top3": _counts(sum(r["top3_hit"] for r in rows), len(rows)),
            }
    confidence = {
        "definition": "confidence of the top-ranked candidate, split by top-1 hit/miss; "
                      "items with zero candidates excluded",
        "mean_when_correct": round(sum(conf_correct) / len(conf_correct), 3) if conf_correct else None,
        "mean_when_wrong": round(sum(conf_wrong) / len(conf_wrong), 3) if conf_wrong else None,
        "n_correct": len(conf_correct),
        "n_wrong": len(conf_wrong),
    }
    latency = {
        "definition": "wall-clock per-item predict() latency in ms (includes network "
                      "round-trip); one measurement per item",
        "n": len(latencies_ms),
        "p50_ms": _percentile(latencies_ms, 50),
        "p95_ms": _percentile(latencies_ms, 95),
        "mean_ms": round(sum(latencies_ms) / len(latencies_ms), 1) if latencies_ms else None,
        "max_ms": round(max(latencies_ms), 1) if latencies_ms else None,
    }

    return {
        "provenance": {
            "script": "eval/run_prediction_eval.py",
            "dataset": str(dataset_path.relative_to(ROOT)) if dataset_path and dataset_path.is_relative_to(ROOT)
                       else (str(dataset_path) if dataset_path else "(in-memory)"),
            "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest() if dataset_path else None,
            "n_items": len(items),
            "provider": provider,
            "model": model,
            "ablate_context": ablate_context,
            "entity_memory": entity_memory,
            "context_turns": context_turns,
            "sleep_s": sleep_s,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "normalization": NORMALIZATION_RULE,
            "freeze_protocol": FREEZE_PROTOCOL,
        },
        "overall": overall,
        "per_category": per_category,
        "confidence": confidence,
        "latency": latency,
        "items": per_item,
    }


# ---------------------------------------------------------------------------
def _ascii(s: str) -> str:
    """GBK Windows consoles crash on non-ASCII; escape anything the LLM returns."""
    return s.encode("ascii", "backslashreplace").decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--provider", choices=["mock", "gemini", "claude", "deepseek"], default="mock",
                    help="predictor provider (constructed via backend.predictor.get_predictor, "
                         "the same factory the app uses); default mock -- live runs are explicit")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="inter-call sleep in seconds for provider rate caps (default 1.0; "
                         "use 0 for mock runs)")
    ap.add_argument("--limit", type=int, default=None,
                    help="run only the first N items (smoke run; output gets a _smoke suffix)")
    ap.add_argument("--ablate-context", action="store_true",
                    help="run every item with context=[] (writes prediction_eval_ablated.json)")
    ap.add_argument("--dataset", type=Path, default=DATASET,
                    help="path to a JSONL eval set with the same schema (default: the frozen "
                         "prediction_eval_set.jsonl; behavior for the default is unchanged). "
                         "A non-default path (e.g. prediction_eval_longctx_v1.jsonl) gets its "
                         "own results filename derived from the dataset stem so it never "
                         "clobbers the frozen headline files.")
    ap.add_argument("--entity-memory", choices=["on", "off"], default="off",
                    help="'on' builds a backend.entities.EntityTracker over each item's context "
                         "and injects out-of-window entities exactly like the live pipeline "
                         "(see backend.pipeline.EchoPipeline). Both settings window each item's "
                         "context to Settings.context_turns (a no-op for the frozen set, whose "
                         "items are always shorter than that window) so the run simulates what "
                         "the live system actually sends.")
    ap.add_argument("--out", type=Path, default=None, help="override the output path")
    args = ap.parse_args()
    t0 = time.time()

    items = load_items(args.dataset)
    if args.limit is not None:
        items = items[: args.limit]

    settings = dataclasses.replace(get_settings(), predictor_provider=args.provider)
    # Fail loudly on a missing key/SDK: scoring a silent mock fallback as if it
    # were the live provider would be dishonest, so no session.make_predictor here.
    predictor = get_predictor(settings)
    model = getattr(predictor, "model", "(mock)")
    entity_memory = args.entity_memory == "on"
    is_frozen_dataset = args.dataset == DATASET

    out = args.out
    if out is None:
        if is_frozen_dataset:
            # Unchanged behavior for the default headline command.
            out = RESULTS_ABLATED if args.ablate_context else RESULTS_FULL
            if args.limit is not None:  # never clobber a headline file with a smoke run
                out = out.with_name(out.stem + "_smoke.json")
        else:
            suffix = ""
            if args.ablate_context:
                suffix += "_ablated"
            if entity_memory:
                suffix += "_entity"
            if args.limit is not None:
                suffix += "_smoke"
            out = ROOT / "eval" / "results" / f"{args.dataset.stem}{suffix}.json"

    print(f"provider={args.provider} model={_ascii(str(model))} items={len(items)} "
          f"dataset={args.dataset.name} ablate_context={args.ablate_context} "
          f"entity_memory={args.entity_memory} sleep={args.sleep}s")
    results = asyncio.run(run_eval(
        predictor, items,
        ablate_context=args.ablate_context, sleep_s=args.sleep,
        provider=args.provider, model=str(model), dataset_path=args.dataset,
        entity_memory=entity_memory, context_turns=settings.context_turns,
        echo=lambda s: print(_ascii(s)),
    ))

    ov = results["overall"]
    print(f"\noverall: top-1 {ov['top1']['hits']}/{ov['top1']['n']} "
          f"({100 * ov['top1']['rate']:.1f}%) | top-3 {ov['top3']['hits']}/{ov['top3']['n']} "
          f"({100 * ov['top3']['rate']:.1f}%)")
    for cat, m in results["per_category"].items():
        print(f"  {cat:<14s} top-1 {m['top1']['hits']}/{m['top1']['n']} "
              f"({100 * m['top1']['rate']:.1f}%) | top-3 {m['top3']['hits']}/{m['top3']['n']} "
              f"({100 * m['top3']['rate']:.1f}%)")
    conf = results["confidence"]
    print(f"confidence: mean when correct {conf['mean_when_correct']} (n={conf['n_correct']}) "
          f"| when wrong {conf['mean_when_wrong']} (n={conf['n_wrong']})")
    lat = results["latency"]
    print(f"latency: p50 {lat['p50_ms']} ms | p95 {lat['p95_ms']} ms "
          f"| mean {lat['mean_ms']} ms | max {lat['max_ms']} ms (n={lat['n']})")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
