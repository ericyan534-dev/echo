"""Search detector logic against cached streams -- on a TUNE split, never on all six.

WHY A SPLIT
-----------
Everything in this file is a search over decision rules, and a search that is
scored on every speaker produces a number that describes the search, not the
system. Six speakers is not many, so the split is fixed, declared here, and
never touched:

    TUNE     1554, 1731, 1833      (search runs only on these)
    HELDOUT  1713, 1738, 1944      (looked at once, at the end)

The held-out three are speaker-disjoint from the tune three by construction --
different people, not different minutes of the same person.

WHY IT IS FAST
--------------
The ASR and acoustic streams are already on disk from run_aphasia_eval.py and
do not depend on any detector setting, so a whole configuration is scored in
milliseconds instead of the ~30 minutes a full re-transcription costs. That is
the difference between searching this space and guessing at it.

    python eval/tune_aphasia_detector.py --list
    python eval/tune_aphasia_detector.py --sweep gap
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.schemas import AcousticEvent, SilenceTick, TurnEnd, Word  # noqa: E402
from backend.stall_detector import StallDetector  # noqa: E402
from backend.timeline import Timeline  # noqa: E402
from eval.run_aphasia_eval import (CACHE, POST_MS, PRE_MS, TRANSCRIPTS,  # noqa: E402
                                   _decode, run_detector)
from scripts.aprocsa_chat import load_all  # noqa: E402

TUNE = ["1554", "1731", "1833"]
HELDOUT = ["1713", "1738", "1944"]
OUT = ROOT / "eval" / "results" / "aphasia_tuning.json"


def load_stream(key: str):
    p = CACHE / (key + ".json")
    if not p.exists():
        return None
    return [_decode(d) for d in json.loads(p.read_text(encoding="utf-8"))]


def streams(pid: str, skip: int, region: int, mode: str, acoustic: str):
    tag = "%s_%d_%d" % (pid, skip, region)
    a = load_stream("asr_%s_%s" % (tag, mode))
    if a is None:
        return None
    if acoustic == "none":
        return a
    matches = sorted(CACHE.glob("ac_%s_%s*.json" % (tag, acoustic)))
    if not matches:
        return None
    b = [_decode(d) for d in json.loads(matches[-1].read_text(encoding="utf-8"))]
    return a + b


def speaker_map(pid: str, skip: int, region: int):
    """{t0,t1,conf} spans from eval/make_speaker_map.py, or None if absent."""
    p = CACHE / ("spk_%s_%d_%d.json" % (pid, skip, region))
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))["segments"]


def conf_at(spans, t_ms):
    """Wearer confidence covering an instant, or None for unknown.

    None is returned for any instant no segment covers -- between utterances,
    or in a segment too short to embed. Unknown must never suppress, so the
    caller treats None as "let it through".
    """
    if not spans:
        return None
    for s in spans:
        if s["t0"] <= t_ms <= s["t1"]:
            return s["conf"]
    return None


def apply_speaker(items, spans, offset_ms, conf_min):
    """Stamp wearer confidence onto words and drop non-wearer acoustic events.

    Words get the confidence and the Timeline decides what to do with it.
    Acoustic events are filtered here instead, because AcousticEvent carries no
    speaker field -- in the live system that gating happens inside
    AcousticStream, which owns the audio the event came from.
    """
    if not spans:
        return items
    out = []
    for t, it in items:
        if isinstance(it, Word):
            c = conf_at(spans, it.end_ms + offset_ms)
            it = Word(text=it.text, start_ms=it.start_ms, end_ms=it.end_ms,
                      is_final=True, wearer_conf=c)
        elif isinstance(it, AcousticEvent):
            c = conf_at(spans, it.at_ms + offset_ms)
            if c is not None and c < conf_min:
                continue
        out.append((t, it))
    return out


def score_one(fires, utts, offset_ms, region_ms):
    lo, hi = offset_ms, offset_ms + region_ms
    par = [u for u in utts if u["is_participant"] and u["start_ms"] is not None
           and u["start_ms"] >= lo and u["end_ms"] <= hi]
    inv = [u for u in utts if not u["is_participant"] and u["start_ms"] is not None
           and u["start_ms"] >= lo and u["end_ms"] <= hi]
    times = sorted(f["at_ms"] + offset_ms for f in fires)

    def hit(u, pre=PRE_MS, post=POST_MS):
        a, b = u["start_ms"] - pre, u["end_ms"] + post
        i = np.searchsorted(times, a)
        return i < len(times) and times[i] <= b

    pos = [u for u in par if u["word_search"]]
    neg = [u for u in par if not u["word_search"]]
    return {
        "hits": sum(hit(u) for u in pos), "n_pos": len(pos),
        "fa": sum(hit(u) for u in neg), "n_neg": len(neg),
        "partner": sum(hit(u, 0, 0) for u in inv), "n_par": len(inv),
        "fires": len(fires), "region_ms": region_ms,
    }


def run_detector_gated(items, pause_ms, min_gap_ms, conf_min):
    det = StallDetector(pause_ms=pause_ms, min_gap_ms=min_gap_ms,
                        timeline=Timeline(wearer_conf_min=conf_min))
    fires = []
    for _, item in sorted(items, key=lambda kv: kv[0]):
        if isinstance(item, Word):
            ev = det.observe_word(item)
        elif isinstance(item, SilenceTick):
            ev = det.observe_silence(item.at_ms)
        elif isinstance(item, TurnEnd):
            det.reset()
            continue
        else:
            ev = det.observe_acoustic(item)
        if ev is not None:
            fires.append({"at_ms": ev.at_ms, "trigger": ev.trigger,
                          "fragment": ev.fragment[-90:]})
    return fires


def evaluate(pids, parsed, cfg, skip, region, mode, acoustic):
    tot = {"hits": 0, "n_pos": 0, "fa": 0, "n_neg": 0, "partner": 0,
           "n_par": 0, "fires": 0, "region_ms": 0}
    for pid in pids:
        items = streams(pid, skip, region, mode, acoustic)
        if items is None:
            continue
        cmin = cfg.get("wearer_conf_min", 0.0)
        if cmin > 0:
            spans = speaker_map(pid, skip, region)
            items = apply_speaker(items, spans, skip * 1000, cmin)
            fires = run_detector_gated(items, cfg["pause_ms"], cfg["min_gap_ms"], cmin)
        else:
            fires = run_detector(items, cfg["pause_ms"], cfg["min_gap_ms"])
        s = score_one(fires, parsed[pid]["utterances"], skip * 1000, region * 1000)
        for k in tot:
            tot[k] += s[k]
    return summarize(tot)


def summarize(t):
    return {
        "recall": round(t["hits"] / t["n_pos"], 4) if t["n_pos"] else None,
        "false_alarm_rate": round(t["fa"] / t["n_neg"], 4) if t["n_neg"] else None,
        "partner_fire_rate": round(t["partner"] / t["n_par"], 4) if t["n_par"] else None,
        "fires_per_min": round(t["fires"] / (t["region_ms"] / 60000.0), 2)
        if t["region_ms"] else None,
        "n_pos": t["n_pos"], "n_neg": t["n_neg"], "n_par": t["n_par"],
    }


def utility(s):
    """One number to rank configurations by.

    An assistive aid is asymmetric: a missed word-search is a lost chance to
    help, a false suggestion actively interrupts someone who is mid-sentence,
    and a suggestion fired at the conversation partner is simply wrong. The
    weights say so explicitly rather than leaving the trade-off implicit in
    whichever metric happened to get quoted.
    """
    if s["recall"] is None:
        return -9e9
    return (s["recall"]
            - 0.7 * (s["false_alarm_rate"] or 0)
            - 1.0 * (s["partner_fire_rate"] or 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--acoustic", default="stutter")
    ap.add_argument("--list", action="store_true", help="show cached streams")
    args = ap.parse_args()

    if args.list:
        for p in sorted(CACHE.glob("*.json")):
            print("  %-42s %6.0f KB" % (p.name, p.stat().st_size / 1024))
        return 0

    parsed = load_all(TRANSCRIPTS)
    grid = list(itertools.product([900, 1100, 1300, 1600, 2000],
                                  [0, 1500, 2500, 3500, 5000],
                                  [0.0, 0.35, 0.5, 0.65]))
    rows = []
    for pause, gap, cmin in grid:
        cfg = {"pause_ms": pause, "min_gap_ms": gap, "wearer_conf_min": cmin}
        s = evaluate(TUNE, parsed, cfg, args.skip, args.region, args.mode, args.acoustic)
        s.update(cfg)
        s["utility"] = round(utility(s), 4)
        rows.append(s)
    rows.sort(key=lambda r: -r["utility"])

    print("TUNE SPLIT (%s) -- %s ASR + %s acoustic" % (",".join(TUNE), args.mode, args.acoustic))
    print("  %6s %6s %6s %8s %8s %8s %9s %8s"
          % ("pause", "gap", "spk", "recall", "FA", "partner", "fires/min", "utility"))
    for r in rows[:14]:
        print("  %6d %6d %6.2f %8s %8s %8s %9s %8s"
              % (r["pause_ms"], r["min_gap_ms"], r["wearer_conf_min"], r["recall"],
                 r["false_alarm_rate"], r["partner_fire_rate"], r["fires_per_min"],
                 r["utility"]))

    best = rows[0]
    held = evaluate(HELDOUT, parsed, best, args.skip, args.region, args.mode, args.acoustic)
    print("")
    print("BEST ON TUNE -> HELD OUT (%s)" % ",".join(HELDOUT))
    print("  pause=%d gap=%d wearer_conf_min=%.2f"
          % (best["pause_ms"], best["min_gap_ms"], best["wearer_conf_min"]))
    print("  tune    recall %s  FA %s  partner %s"
          % (best["recall"], best["false_alarm_rate"], best["partner_fire_rate"]))
    print("  heldout recall %s  FA %s  partner %s"
          % (held["recall"], held["false_alarm_rate"], held["partner_fire_rate"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "tune": TUNE, "heldout": HELDOUT,
        "mode": args.mode, "acoustic": args.acoustic,
        "utility": "recall - 0.7*false_alarm - 1.0*partner_fire",
        "grid": rows, "best_on_tune": best, "best_on_heldout": held,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
