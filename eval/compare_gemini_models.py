"""Compare Gemini model ids for Echo's word prediction: accuracy AND speed.

Echo's stall->word loop has a ~1-2 s end-to-end budget, so a candidate model
has to be judged on latency percentiles as much as on top-1. This script runs
the FROZEN prediction eval set (eval/data/prediction_eval_set.jsonl, see the
freeze protocol in eval/run_prediction_eval.py -- the set is never edited)
through the SAME predictor factory the app uses (backend.predictor.
get_predictor -> GeminiPredictor), once per (model id, thinking setting), and
repeats every item --reps times so the latency percentiles are stable.

What is measured per (model, setting):
  * top-1 / top-3 accuracy, scored with run_prediction_eval.score_candidates
    (identical normalization to docs/EVAL.md), reported per rep and as the
    mean over reps
  * per-call wall-clock latency of predictor.predict() (p50 / p95 / max)
  * token usage from usage_metadata (prompt / output / thoughts / total)
  * failures: exceptions (by class), empty-candidate responses (no text,
    MAX_TOKENS, unparseable JSON), finish-reason histogram, 429/503 retries

Thinking settings. The google-genai SDK exposes ThinkingConfig(thinking_budget,
include_thoughts) for these models -- no other reasoning knob. The app ships
thinking_budget=0 (backend/predictor/gemini.py). Settings compared:
  thinking_off         the shipped config, unchanged (budget 0, max_output 256)
  thinking_default     shipped config MINUS thinking_config, i.e. the model's
                       default dynamic thinking under the app's 256-token cap
                       (what the app would do if that one line were deleted)
  thinking_default_4k  default thinking with max_output_tokens=4096 so the
                       thoughts cannot starve the answer -- the fair
                       "does thinking help accuracy" condition
The settings are applied by wrapping the predictor's SDK call and editing the
GenerateContentConfig the app built; the prompt, schema, temperature and the
parser are exactly the app's.

Model ids are resolved against models.list BEFORE any run: an exact match is
used as-is; otherwise the obvious variants (-preview, -latest, and any listed
id that starts with the requested id) are tried, and whatever resolved is
recorded in the JSON. A model that cannot be resolved or run is reported with
the exact exception class -- never silently dropped.

Output (never any other results file):
  eval/results/gemini_model_compare.json   keyed by model id and setting
  docs/MODEL_COMPARISON.md                 rendered from that JSON

Usage:
    python eval/compare_gemini_models.py                       # default 3 ids
    python eval/compare_gemini_models.py gemini-3.5-flash gemini-3.7-flash
    python eval/compare_gemini_models.py --reps 3 --sleep 0.25
    python eval/compare_gemini_models.py --limit 3 --reps 1    # smoke (_smoke suffix)
    python eval/compare_gemini_models.py --report-only         # re-render the .md
    python eval/compare_gemini_models.py --resume              # finish an interrupted run
Options: --settings (subset of the three above), --out PATH, --doc PATH.
Console output is ASCII-only (GBK Windows consoles).
"""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import logging
import statistics
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402
from backend.predictor import get_predictor  # noqa: E402
from eval.run_prediction_eval import (  # noqa: E402  (reused, not copied)
    CATEGORIES,
    DATASET,
    FREEZE_PROTOCOL,
    NORMALIZATION_RULE,
    load_items,
    score_candidates,
)

SCRIPT = "eval/compare_gemini_models.py"
RESULTS = ROOT / "eval" / "results" / "gemini_model_compare.json"
DOC = ROOT / "docs" / "MODEL_COMPARISON.md"

DEFAULT_MODELS = ["gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.8-flash"]
INCUMBENT = "gemini-3.5-flash"          # backend/config.py default; NOT changed here
BUDGET_MS = (1000, 2000)                # Echo's end-to-end stall->word budget

# name -> (thinking_budget or None for SDK/model default, max_output_tokens, description)
SETTINGS: dict[str, tuple[int | None, int, str]] = {
    "thinking_off": (0, 256, "the shipped app config, unchanged"),
    "thinking_default": (None, 256, "shipped config minus thinking_config (model default "
                                    "thinking) under the app's 256-token output cap"),
    "thinking_default_4k": (None, 4096, "model default thinking with max_output_tokens=4096 "
                                        "so thoughts cannot starve the answer"),
}

RETRY_STATUS = {429, 503}
MAX_RETRIES = 3


def _ascii(s: str) -> str:
    return str(s).encode("ascii", "backslashreplace").decode("ascii")


def _pct(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default) on an already-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _latency_stats(ms: list[float]) -> dict:
    s = sorted(ms)
    return {
        "n": len(s),
        "p50": round(_pct(s, 0.50), 1) if s else None,
        "p95": round(_pct(s, 0.95), 1) if s else None,
        "max": round(s[-1], 1) if s else None,
        "min": round(s[0], 1) if s else None,
        "mean": round(statistics.fmean(s), 1) if s else None,
    }


# ---------------------------------------------------------------------------
# model id resolution
# ---------------------------------------------------------------------------
def resolve_models(client, requested: list[str]) -> dict[str, dict]:
    """Map each requested id to what the API actually exposes.

    Returns {requested_id: {"resolved": id_or_None, "how": ..., "actions": [...],
    "error": ...}}. Listing failures are recorded per-model, never raised.
    """
    try:
        listed = {m.name.removeprefix("models/"): list(getattr(m, "supported_actions", None) or [])
                  for m in client.models.list()}
        list_error = None
    except Exception as exc:  # noqa: BLE001 -- recorded, not hidden
        listed, list_error = {}, f"{type(exc).__module__}.{type(exc).__name__}: {_ascii(exc)[:200]}"

    out: dict[str, dict] = {}
    for req in requested:
        entry: dict = {"resolved": None, "how": None, "actions": [], "error": list_error, "tried": []}
        if list_error:
            out[req] = entry
            continue
        if req in listed:
            entry.update(resolved=req, how="exact match in models.list", actions=listed[req])
        else:
            candidates = [f"{req}-preview", f"{req}-latest"]
            candidates += sorted(n for n in listed if n.startswith(req + "-") and n not in candidates)
            entry["tried"] = candidates
            for c in candidates:
                if c in listed and "generateContent" in listed[c]:
                    entry.update(resolved=c, how=f"variant of requested id {req!r}", actions=listed[c])
                    break
            if entry["resolved"] is None:
                entry["error"] = f"ModelNotFound: {req!r} not in models.list and no variant of it resolved"
        if entry["resolved"] and "generateContent" not in entry["actions"]:
            entry["error"] = f"ModelUnsupported: {entry['resolved']!r} does not support generateContent"
            entry["resolved"] = None
        out[req] = entry
    return out


# ---------------------------------------------------------------------------
# one (model, setting) run
# ---------------------------------------------------------------------------
class _Capture:
    """Wraps the predictor's SDK call: applies the thinking setting to the config
    the app built, retries 429/503, and records usage/finish_reason per call."""

    def __init__(self, orig, thinking_budget: int | None, max_output_tokens: int) -> None:
        from google.genai import types  # lazy, same as the predictor
        self._orig = orig
        self._types = types
        self.thinking_budget = thinking_budget
        self.max_output_tokens = max_output_tokens
        self.last: dict = {}
        self.rate_limit_retries = 0

    async def __call__(self, *args, config=None, **kwargs):
        if self.thinking_budget is None:
            config.thinking_config = None
        else:
            config.thinking_config = self._types.ThinkingConfig(thinking_budget=self.thinking_budget)
        config.max_output_tokens = self.max_output_tokens
        self.last = {}
        attempt = 0
        while True:
            try:
                resp = await self._orig(*args, config=config, **kwargs)
                break
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if code in RETRY_STATUS and attempt < MAX_RETRIES:
                    attempt += 1
                    self.rate_limit_retries += 1
                    await asyncio.sleep(2.0 ** attempt)
                    continue
                raise
        u = getattr(resp, "usage_metadata", None)
        cand = (resp.candidates or [None])[0] if getattr(resp, "candidates", None) else None
        fr = getattr(cand, "finish_reason", None) if cand is not None else None
        self.last = {
            "finish_reason": getattr(fr, "name", str(fr)) if fr is not None else "NONE",
            "prompt_tokens": getattr(u, "prompt_token_count", None) if u else None,
            "output_tokens": getattr(u, "candidates_token_count", None) if u else None,
            "thoughts_tokens": getattr(u, "thoughts_token_count", None) if u else None,
            "total_tokens": getattr(u, "total_token_count", None) if u else None,
            "retries": attempt,
        }
        return resp


async def run_condition(model_id: str, setting: str, items: list[dict], *, reps: int,
                        sleep_s: float, context_turns: int, say) -> dict:
    budget, max_out, _ = SETTINGS[setting]
    settings = dataclasses.replace(get_settings(), predictor_provider="gemini", gemini_model=model_id)
    predictor = get_predictor(settings)           # the app's factory, the app's class
    cap = _Capture(predictor._client.aio.models.generate_content, budget, max_out)
    predictor._client.aio.models.generate_content = cap

    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    calls: list[dict] = []
    exc_classes: Counter = Counter()
    finish: Counter = Counter()
    try:
        for rep in range(1, reps + 1):
            for i, item in enumerate(items):
                context = list(item["context"])[-context_turns:] if context_turns > 0 else list(item["context"])
                rec: dict = {"id": item["id"], "category": item["category"], "rep": rep}
                t0 = time.perf_counter()
                try:
                    cands = await predictor.predict(context, item["fragment"])
                    rec["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
                    words = [c.word for c in cands]
                    top1, top3 = score_candidates(item["gold"], words)
                    rec.update(words=words, top1_hit=top1, top3_hit=top3, error=None,
                               empty=not words, **cap.last)
                    finish[rec.get("finish_reason", "NONE")] += 1
                    mark = "HIT@1" if top1 else ("HIT@3" if top3 else ("EMPTY" if not words else "MISS "))
                except Exception as exc:  # noqa: BLE001 -- counted, never hidden
                    rec["latency_ms"] = round(1000 * (time.perf_counter() - t0), 1)
                    cls = f"{type(exc).__module__}.{type(exc).__name__}"
                    exc_classes[cls] += 1
                    rec.update(words=[], top1_hit=False, top3_hit=False, empty=True,
                               error=f"{cls}: {_ascii(exc)[:200]}", finish_reason="EXCEPTION")
                    finish["EXCEPTION"] += 1
                    mark = "ERROR"
                calls.append(rec)
                say(f"  [{model_id} {setting} r{rep} {i + 1:2d}/{len(items)}] {item['id']:<14s} "
                    f"{mark} {rec['latency_ms']:7.0f} ms  gold={item['gold'][0]!r} -> "
                    f"{rec['words'] or rec['error'] or '(no candidates)'}")
                if sleep_s:
                    await asyncio.sleep(sleep_s)
    finally:
        aclose = getattr(predictor._client.aio, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 -- best-effort cleanup
                pass

    n = len(items)
    per_rep_top1 = [sum(c["top1_hit"] for c in calls if c["rep"] == r) for r in range(1, reps + 1)]
    per_rep_top3 = [sum(c["top3_hit"] for c in calls if c["rep"] == r) for r in range(1, reps + 1)]
    per_category = {}
    for cat in CATEGORIES:
        rows = [c for c in calls if c["category"] == cat]
        if rows:
            per_category[cat] = {
                "n_calls": len(rows),
                "top1_rate": round(sum(r["top1_hit"] for r in rows) / len(rows), 4),
                "top3_rate": round(sum(r["top3_hit"] for r in rows) / len(rows), 4),
            }
    ok_lat = [c["latency_ms"] for c in calls if c["error"] is None]
    tok = lambda k: [c[k] for c in calls if c.get(k) is not None]  # noqa: E731
    return {
        "model_resolved": model_id,
        "setting": setting,
        "thinking_budget": budget,
        "max_output_tokens": max_out,
        "n_items": n,
        "reps": reps,
        "n_calls": len(calls),
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "accuracy": {
            "top1": {"per_rep_hits": per_rep_top1, "n": n,
                     "mean_hits": round(statistics.fmean(per_rep_top1), 2),
                     "mean_rate": round(statistics.fmean(per_rep_top1) / n, 4)},
            "top3": {"per_rep_hits": per_rep_top3, "n": n,
                     "mean_hits": round(statistics.fmean(per_rep_top3), 2),
                     "mean_rate": round(statistics.fmean(per_rep_top3) / n, 4)},
            "per_category": per_category,
            "note": "empty responses and exceptions score as misses",
        },
        "latency_ms": {"definition": "wall clock of predictor.predict() per call, successful "
                                     "(non-exception) calls only, all reps pooled",
                       **_latency_stats(ok_lat)},
        "tokens": {
            "calls_with_usage": len(tok("total_tokens")),
            "prompt_total": sum(tok("prompt_tokens")),
            "output_total": sum(tok("output_tokens")),
            "thoughts_total": sum(tok("thoughts_tokens")),
            "total": sum(tok("total_tokens")),
            "thoughts_mean_per_call": round(statistics.fmean(tok("thoughts_tokens")), 1) if tok("thoughts_tokens") else 0.0,
        },
        "failures": {
            "exceptions": sum(exc_classes.values()),
            "exception_classes": dict(exc_classes),
            "empty_output": sum(1 for c in calls if c["error"] is None and c["empty"]),
            "finish_reasons": dict(finish),
            "rate_limit_retries": cap.rate_limit_retries,
        },
        "calls": calls,
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt_ms(v) -> str:
    return "-" if v is None else f"{v:.0f}"


def _acc_cell(a: dict) -> str:
    return f"{100 * a['mean_rate']:.1f}% ({a['mean_hits']:g}/{a['n']})"


def render_doc(res: dict) -> str:
    prov = res["provenance"]
    runs = res["runs"]
    lines: list[str] = []
    L = lines.append
    L("# Gemini model comparison for Echo's word predictor")
    L("")
    L("<!-- GENERATED by eval/compare_gemini_models.py from eval/results/gemini_model_compare.json. "
      "Do not edit by hand; re-run the command at the bottom. -->")
    L("")
    L(f"Run started {prov['timestamp_utc']} (UTC); rendered from `{prov['results_file']}`. "
      f"Dataset: `{prov['dataset']}` (sha256 `{prov['dataset_sha256'][:12]}...`, "
      f"{prov['n_items']} items, frozen -- see eval/run_prediction_eval.py). "
      f"Each item was sent {prov['reps']}x per condition; accuracy is the mean over reps, "
      f"latency percentiles pool every successful call ({prov['n_items'] * prov['reps']} per row). "
      f"Predictor: `backend.predictor.get_predictor` (the app's factory and prompt); "
      f"google-genai {prov['sdk_version']}. Scoring: run_prediction_eval.score_candidates "
      f"({prov['normalization']}). Incumbent default: `{INCUMBENT}` (unchanged in backend/config.py).")
    L("")
    L("## Model ids as resolved")
    L("")
    L("| requested | resolved | how | error |")
    L("|---|---|---|---|")
    for req, r in prov["resolved_models"].items():
        L(f"| `{req}` | {('`' + r['resolved'] + '`') if r['resolved'] else '(none)'} | "
          f"{r['how'] or '-'} | {r['error'] or '-'} |")
    L("")
    L("## Settings")
    L("")
    for name, (b, m, d) in SETTINGS.items():
        if name in prov["settings"]:
            L(f"- `{name}`: {d} -- thinking_budget={b if b is not None else 'unset (model default)'}, "
              f"max_output_tokens={m}")
    L("")
    L("## Results")
    L("")
    L("| model | setting | top-1 | top-3 | p50 ms | p95 ms | max ms | errors |")
    L("|---|---|---|---|---|---|---|---|")
    rows: list[tuple[str, str, dict]] = []
    for mid, by_setting in runs.items():
        for setting, r in by_setting.items():
            rows.append((mid, setting, r))
            if "error" in r:
                L(f"| `{mid}` | `{setting}` | - | - | - | - | - | NOT RUN: {r['error']} |")
                continue
            f = r["failures"]
            err_parts = []
            if f["exceptions"]:
                err_parts.append(f"{f['exceptions']} exc ({', '.join(f['exception_classes'])})")
            if f["empty_output"]:
                err_parts.append(f"{f['empty_output']} empty")
            if f["rate_limit_retries"]:
                err_parts.append(f"{f['rate_limit_retries']} retried")
            lat = r["latency_ms"]
            L(f"| `{mid}` | `{setting}` | {_acc_cell(r['accuracy']['top1'])} | "
              f"{_acc_cell(r['accuracy']['top3'])} | {_fmt_ms(lat['p50'])} | {_fmt_ms(lat['p95'])} | "
              f"{_fmt_ms(lat['max'])} | {'; '.join(err_parts) if err_parts else '0'} |")
    L("")
    L("`errors` = exceptions raised by the predictor + responses with zero parseable candidates "
      "(empty text, MAX_TOKENS, bad JSON); both score as misses. `retried` = 429/503 responses "
      "that were retried (latency of the retry attempt only is recorded).")
    L("")
    L("### Detail: tokens, finish reasons, per-rep accuracy")
    L("")
    L("| model | setting | thoughts tok/call | total tokens | finish reasons | top-1 per rep | n calls |")
    L("|---|---|---|---|---|---|---|")
    for mid, setting, r in rows:
        if "error" in r:
            continue
        fr = ", ".join(f"{k}={v}" for k, v in sorted(r["failures"]["finish_reasons"].items()))
        L(f"| `{mid}` | `{setting}` | {r['tokens']['thoughts_mean_per_call']:g} | {r['tokens']['total']} | "
          f"{fr} | {'/'.join(str(h) for h in r['accuracy']['top1']['per_rep_hits'])} | {r['n_calls']} |")
    L("")
    L("### Per-category top-1 (all reps pooled)")
    L("")
    L("| model | setting | " + " | ".join(CATEGORIES) + " |")
    L("|---|---|" + "---|" * len(CATEGORIES))
    for mid, setting, r in rows:
        if "error" in r:
            continue
        pc = r["accuracy"]["per_category"]
        L(f"| `{mid}` | `{setting}` | " + " | ".join(
            f"{100 * pc[c]['top1_rate']:.1f}%" if c in pc else "-" for c in CATEGORIES) + " |")
    L("")

    # ---- interpretation (computed from the numbers, so it cannot drift from them)
    ok = [(m, s, r) for m, s, r in rows if "error" not in r]
    L("## Interpretation")
    L("")
    if not ok:
        L("No condition completed; see the errors column.")
    else:
        inc = next((r for m, s, r in ok if m == INCUMBENT and s == "thinking_off"), None)
        best_acc = max(ok, key=lambda t: (t[2]["accuracy"]["top1"]["mean_rate"], -t[2]["latency_ms"]["p95"]))
        fastest = min(ok, key=lambda t: t[2]["latency_ms"]["p95"])
        L(f"- Highest top-1: `{best_acc[0]}` / `{best_acc[1]}` at {_acc_cell(best_acc[2]['accuracy']['top1'])}. "
          f"Lowest p95: `{fastest[0]}` / `{fastest[1]}` at {_fmt_ms(fastest[2]['latency_ms']['p95'])} ms.")
        if inc:
            L(f"- Incumbent (`{INCUMBENT}` / `thinking_off`): top-1 {_acc_cell(inc['accuracy']['top1'])}, "
              f"p50 {_fmt_ms(inc['latency_ms']['p50'])} ms, p95 {_fmt_ms(inc['latency_ms']['p95'])} ms. "
              f"The predictor call is one leg of Echo's {BUDGET_MS[0]}-{BUDGET_MS[1]} ms end-to-end budget "
              f"(stall detection + transport sit in front of it), so p95 matters more than p50 here.")
        over = [(m, s, r) for m, s, r in ok if r["latency_ms"]["p95"] and r["latency_ms"]["p95"] > BUDGET_MS[1]]
        if over:
            L("- Conditions whose p95 alone exceeds the whole 2000 ms budget: " +
              ", ".join(f"`{m}`/`{s}` ({_fmt_ms(r['latency_ms']['p95'])} ms)" for m, s, r in over) + ".")
        empties = [(m, s, r) for m, s, r in ok if r["failures"]["empty_output"] or r["failures"]["exceptions"]]
        if empties:
            L("- Conditions with empty or failed responses: " + ", ".join(
                f"`{m}`/`{s}` ({r['failures']['empty_output']} empty, {r['failures']['exceptions']} exceptions, "
                f"finish reasons {r['failures']['finish_reasons']})" for m, s, r in empties) +
              ". An empty response under `thinking_default` means the model spent the 256-token output cap on "
              "thoughts (MAX_TOKENS) -- the failure backend/predictor/gemini.py sets thinking_budget=0 to avoid.")
        leaky = [(m, r) for m, s, r in ok if s == "thinking_off" and r["tokens"]["thoughts_total"] > 0]
        if leaky:
            L("- thinking_budget=0 is NOT fully honoured by: " + ", ".join(
                f"`{m}` (mean {r['tokens']['thoughts_mean_per_call']:g} thought tokens/call)" for m, r in leaky) +
              ". Those models still bill and wait for some reasoning even when the app asks for none.")
            for m, r in leaky:
                cut = [c for c in r["calls"] if c.get("finish_reason") == "MAX_TOKENS"]
                if cut:
                    th = statistics.fmean(c["thoughts_tokens"] or 0 for c in cut)
                    L(f"  - On `{m}` the {len(cut)} MAX_TOKENS responses under `thinking_off` carried a mean of "
                      f"{th:.0f} thought tokens: the leaked reasoning alone can exhaust the app's 256-token "
                      f"output cap, so adopting this model would also require raising max_output_tokens in "
                      f"backend/predictor/gemini.py (not done here).")
        for m in runs:
            offr = runs[m].get("thinking_off", {})
            d4 = runs[m].get("thinking_default_4k", {})
            if "error" in offr or "error" in d4 or not offr or not d4:
                continue
            da = d4["accuracy"]["top1"]["mean_hits"] - offr["accuracy"]["top1"]["mean_hits"]
            dp = d4["latency_ms"]["p95"] - offr["latency_ms"]["p95"]
            L(f"- `{m}`: unconstrained thinking (4k cap) vs shipped thinking-off changes top-1 by "
              f"{da:+.2g} items/rep and p95 by {dp:+.0f} ms.")
        L(f"- n={prov['n_items']} items x {prov['reps']} reps: a one-item difference in top-1 is "
          f"{100 / prov['n_items']:.1f} points, so differences smaller than about two items are within noise "
          f"on this set (the per-rep column shows the spread).")
    L("")

    # ---- recommendation (rule stated, then applied)
    L("## Recommendation")
    L("")
    L(f"Rule applied: switch the default away from `{INCUMBENT}` only if a candidate, under the "
      f"shipped `thinking_off` setting, (a) has mean top-1 at least 2 items/rep higher than the incumbent "
      f"OR equal-or-better top-1 with a p95 at least 15% lower, AND (b) has no empty/failed responses "
      f"the incumbent does not have, AND (c) does not raise p95 by more than 10%. Thinking settings are "
      f"judged separately: enable thinking only if it raises top-1 by 2+ items/rep without pushing p95 "
      f"past the incumbent's p95 by more than 10%.")
    L("")
    inc = runs.get(INCUMBENT, {}).get("thinking_off")
    if not inc or "error" in inc:
        L(f"**No recommendation possible**: the incumbent condition did not complete "
          f"({inc.get('error') if inc else 'not run'}).")
    else:
        inc_t1 = inc["accuracy"]["top1"]["mean_hits"]
        inc_p95 = inc["latency_ms"]["p95"]
        inc_fail = inc["failures"]["empty_output"] + inc["failures"]["exceptions"]
        verdicts = []
        for m in runs:
            if m == INCUMBENT:
                continue
            r = runs[m].get("thinking_off")
            if not r or "error" in r:
                verdicts.append(f"- `{m}`: cannot be recommended -- not run ({r.get('error') if r else 'no thinking_off run'}).")
                continue
            t1, p95 = r["accuracy"]["top1"]["mean_hits"], r["latency_ms"]["p95"]
            fail = r["failures"]["empty_output"] + r["failures"]["exceptions"]
            a = (t1 - inc_t1 >= 2) or (t1 >= inc_t1 and p95 <= 0.85 * inc_p95)
            b = fail <= inc_fail
            c = p95 <= 1.10 * inc_p95
            why = (f"top-1 {t1:g} vs {inc_t1:g} items/rep ({t1 - inc_t1:+.2g}), p95 {p95:.0f} vs {inc_p95:.0f} ms "
                   f"({100 * (p95 / inc_p95 - 1):+.0f}%), failures {fail} vs {inc_fail}")
            verdicts.append(f"- `{m}`: {'SWITCH' if a and b and c else 'DO NOT SWITCH'} -- {why}; "
                            f"gain test {'met' if a else 'not met'}, failure test {'met' if b else 'not met'}, "
                            f"latency test {'met' if c else 'not met'}.")
        lines.extend(verdicts)
        think = []
        for m in runs:
            r0, r4 = runs[m].get("thinking_off"), runs[m].get("thinking_default_4k")
            if r0 and r4 and "error" not in r0 and "error" not in r4:
                gain = r4["accuracy"]["top1"]["mean_hits"] - r0["accuracy"]["top1"]["mean_hits"]
                okp = r4["latency_ms"]["p95"] <= 1.10 * inc_p95
                think.append(f"- thinking on `{m}`: {'worth enabling' if gain >= 2 and okp else 'keep off'} "
                             f"(top-1 {gain:+.2g} items/rep, p95 {r4['latency_ms']['p95']:.0f} ms).")
        lines.extend(think)
        any_switch = any(v.split(":")[1].strip().startswith("SWITCH") for v in verdicts)
        L("")
        L(f"**Bottom line: {'a switch is justified by the rule above -- see the SWITCH row' if any_switch else f'keep `{INCUMBENT}` with `thinking_off` as the default'}.** "
          f"Change it only via the `GEMINI_MODEL` env var; backend/config.py is deliberately untouched by this script.")
    L("")
    L("## Regenerate")
    L("")
    L("```")
    L(f"python eval/compare_gemini_models.py {' '.join(prov['requested_models'])} "
      f"--reps {prov['reps']} --sleep {prov['sleep_s']}")
    L("python eval/compare_gemini_models.py --report-only    # re-render this file from the JSON")
    L("```")
    L("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("models", nargs="*", default=DEFAULT_MODELS, help="Gemini model ids to compare")
    ap.add_argument("--settings", nargs="+", choices=list(SETTINGS), default=list(SETTINGS))
    ap.add_argument("--reps", type=int, default=3, help="passes over the item set per condition (default 3)")
    ap.add_argument("--sleep", type=float, default=0.25, help="inter-call sleep seconds (default 0.25)")
    ap.add_argument("--limit", type=int, default=None, help="first N items only (smoke; _smoke suffix)")
    ap.add_argument("--out", type=Path, default=None, help="results JSON path (default eval/results/gemini_model_compare.json)")
    ap.add_argument("--doc", type=Path, default=None, help="markdown path (default docs/MODEL_COMPARISON.md)")
    ap.add_argument("--report-only", action="store_true", help="render the markdown from the existing JSON; no API calls")
    ap.add_argument("--resume", action="store_true",
                    help="skip (model, setting) conditions already completed in the output JSON "
                         "(same dataset sha256 / n_items / reps); re-runs only missing or failed ones")
    args = ap.parse_args()

    out = args.out or (RESULTS.with_name(RESULTS.stem + "_smoke.json") if args.limit is not None else RESULTS)
    doc = args.doc or (DOC.with_name(DOC.stem + "_smoke.md") if args.limit is not None else DOC)
    # Never write into any results file this script did not itself produce.
    if out.exists():
        try:
            prev = json.loads(out.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            prev = {}
        if prev.get("provenance", {}).get("script") != SCRIPT:
            print(f"refusing to overwrite {out}: not produced by {SCRIPT}")
            return 2

    if args.report_only:
        res = json.loads(out.read_text(encoding="utf-8"))
        doc.write_text(render_doc(res), encoding="utf-8")
        print(f"wrote {doc}")
        return 0

    logging.getLogger("google_genai").setLevel(logging.ERROR)   # SDK 'non-text parts' warning is per call
    t0 = time.time()
    items = load_items(DATASET)
    if args.limit is not None:
        items = items[: args.limit]
    base = get_settings()
    if not base.gemini_api_key:
        print("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set; nothing to compare.")
        return 2

    from google import genai
    client = genai.Client(api_key=base.gemini_api_key)
    resolved = resolve_models(client, args.models)
    print("resolved model ids:")
    for req, r in resolved.items():
        print(f"  {req:<22s} -> {r['resolved'] or '(none)'}  [{r['how'] or r['error']}]")

    res = {
        "provenance": {
            "script": SCRIPT,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results_file": (str(out.relative_to(ROOT)) if out.is_relative_to(ROOT) else str(out)).replace("\\", "/"),
            "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
            "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
            "n_items": len(items),
            "reps": args.reps,
            "sleep_s": args.sleep,
            "context_turns": base.context_turns,
            "sdk_version": getattr(genai, "__version__", "?"),
            "requested_models": list(args.models),
            "resolved_models": resolved,
            "settings": {s: {"thinking_budget": SETTINGS[s][0], "max_output_tokens": SETTINGS[s][1],
                             "description": SETTINGS[s][2]} for s in args.settings},
            "incumbent": INCUMBENT,
            "normalization": NORMALIZATION_RULE,
            "freeze_protocol": FREEZE_PROTOCOL,
        },
        "runs": {},
    }
    say = lambda s: print(_ascii(s))  # noqa: E731

    for req in args.models:                 # key order = requested order
        res["runs"][req] = {}
    if args.resume and out.exists():
        prev = json.loads(out.read_text(encoding="utf-8"))
        pp = prev.get("provenance", {})
        same = (pp.get("script") == SCRIPT and pp.get("dataset_sha256") == res["provenance"]["dataset_sha256"]
                and pp.get("n_items") == len(items) and pp.get("reps") == args.reps)
        if same:
            res["provenance"]["resumed_from_utc"] = pp.get("timestamp_utc")
            for req in args.models:
                for setting in args.settings:
                    done = prev.get("runs", {}).get(req, {}).get(setting)
                    if done and "error" not in done:
                        res["runs"][req][setting] = done
                        say(f"[{req} {setting}] resumed from {out.name} (finished {done['finished_utc']})")
        else:
            say(f"--resume ignored: {out.name} has different dataset/n_items/reps")
    # settings-outer so every model's shipped-config run (the one the
    # recommendation rests on) completes before any thinking variant starts.
    for setting in args.settings:
        for req in args.models:
            r = resolved[req]
            if setting in res["runs"][req] and "error" not in res["runs"][req][setting]:
                continue                    # resumed
            if not r["resolved"]:
                res["runs"][req][setting] = {"error": r["error"], "model_resolved": None}
                say(f"[{req} {setting}] SKIPPED: {r['error']}")
                continue
            say(f"[{req} -> {r['resolved']}] {setting}: {len(items)} items x {args.reps} reps")
            try:
                res["runs"][req][setting] = asyncio.run(run_condition(
                    r["resolved"], setting, items, reps=args.reps, sleep_s=args.sleep,
                    context_turns=base.context_turns, say=say))
            except Exception as exc:  # noqa: BLE001 -- e.g. factory/SDK failure before any call
                res["runs"][req][setting] = {
                    "error": f"{type(exc).__module__}.{type(exc).__name__}: {_ascii(exc)[:300]}",
                    "model_resolved": r["resolved"]}
                say(f"[{req} {setting}] FAILED: {res['runs'][req][setting]['error']}")
                continue
            c = res["runs"][req][setting]
            say(f"  => top-1 {_acc_cell(c['accuracy']['top1'])}  top-3 {_acc_cell(c['accuracy']['top3'])}  "
                f"p50 {_fmt_ms(c['latency_ms']['p50'])} p95 {_fmt_ms(c['latency_ms']['p95'])} "
                f"max {_fmt_ms(c['latency_ms']['max'])} ms  empty {c['failures']['empty_output']} "
                f"exc {c['failures']['exceptions']}  thoughts/call {c['tokens']['thoughts_mean_per_call']:g}")
            out.parent.mkdir(parents=True, exist_ok=True)   # checkpoint after every condition
            out.write_text(json.dumps(res, indent=2), encoding="utf-8")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    doc.write_text(render_doc(res), encoding="utf-8")
    print(f"\nwrote {out}\nwrote {doc}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
