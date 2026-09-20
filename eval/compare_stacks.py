"""Old stack vs new stack on the HELD-OUT speakers, as curves rather than points.

A single operating point cannot answer "is this better" -- a detector that
fires more will always show more recall and more false alarms, and quoting
either number alone decides the answer in advance. So both stacks are swept
across their own control, and the comparison is made where it is meaningful:
at a MATCHED interruption rate, and at a MATCHED false-alarm rate.

    OLD   browser-equivalent transcript (CrisperWhisper "intended" mode, a
          measured stand-in: same model, same audio, dysfluency stripped to
          0.060 preserved) + FillerNet + the any-of trigger rule. Swept by
          min_gap_ms, which is the only control it has.

    NEW   verbatim transcript + StutterNet + ECAPA speaker attribution + the
          fitted stall scorer. Swept by score threshold.

Speakers 1713, 1738 and 1944 only. The scorer's weights and threshold were
fitted on the other three and these were not looked at during that fit.

    python eval/compare_stacks.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.schemas import SilenceTick, TurnEnd, Word  # noqa: E402
from backend.stall_detector import HEDGES, StallDetector  # noqa: E402
from backend.stall_scorer import DEFAULT_BIAS, DEFAULT_WEIGHTS, StallScorer  # noqa: E402
from backend.timeline import Timeline  # noqa: E402
from eval.run_aphasia_eval import TRANSCRIPTS  # noqa: E402
from eval.tune_aphasia_detector import (HELDOUT, TUNE, apply_speaker,  # noqa: E402
                                        score_one, speaker_map, streams,
                                        summarize)
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "stack_comparison.json"


def run(pids, parsed, args, *, mode, acoustic, conf_min, gap,
        scorer_thr=None, rearm=1, pause_ms=None):
    tot = {"hits": 0, "n_pos": 0, "fa": 0, "n_neg": 0, "partner": 0,
           "n_par": 0, "fires": 0, "region_ms": 0}
    for pid in pids:
        items = streams(pid, args.skip, args.region, mode, acoustic)
        if items is None:
            continue
        if conf_min > 0:
            items = apply_speaker(items, speaker_map(pid, args.skip, args.region),
                                  args.skip * 1000, conf_min)
        sc = None
        if scorer_thr is not None:
            sc = StallScorer(weights=DEFAULT_WEIGHTS, bias=DEFAULT_BIAS,
                             threshold=scorer_thr, hedges=tuple(HEDGES))
        det = StallDetector(pause_ms=(pause_ms or args.pause_ms), min_gap_ms=gap,
                            timeline=Timeline(wearer_conf_min=conf_min),
                            scorer=sc, rearm_content_words=rearm)
        fires = []
        for _, it in sorted(items, key=lambda kv: kv[0]):
            if isinstance(it, Word):
                ev = det.observe_word(it)
            elif isinstance(it, SilenceTick):
                ev = det.observe_silence(it.at_ms)
            elif isinstance(it, TurnEnd):
                det.reset()
                continue
            else:
                ev = det.observe_acoustic(it)
            if ev is not None:
                fires.append({"at_ms": ev.at_ms, "trigger": ev.trigger})
        s = score_one(fires, parsed[pid]["utterances"],
                      args.skip * 1000, args.region * 1000)
        for k in tot:
            tot[k] += s[k]
    return summarize(tot)


def interp_recall(curve, key, target):
    """Recall of a curve at a target value of `key`, linearly interpolated.

    Comparing two systems at whichever points each happened to be sampled is
    not a comparison, so both are read at the same x.
    """
    pts = sorted(((r[key], r["recall"]) for r in curve if r[key] is not None
                  and r["recall"] is not None), key=lambda p: p[0])
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if target <= xs[0] or target >= xs[-1]:
        return None
    return float(np.interp(target, xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--pause-ms", type=int, default=1300)
    ap.add_argument("--conf-min", type=float, default=0.35)
    ap.add_argument("--gap", type=int, default=2500)
    ap.add_argument("--rearm", type=int, default=1)
    ap.add_argument("--pids", default="heldout", choices=["heldout", "tune", "all"])
    args = ap.parse_args()

    parsed = load_all(TRANSCRIPTS)
    pids = {"heldout": HELDOUT, "tune": TUNE, "all": TUNE + HELDOUT}[args.pids]

    old = []
    for gap in (0, 1500, 2500, 3500, 5000, 7000, 10000):
        s = run(pids, parsed, args, mode="intended", acoustic="filler",
                conf_min=0.0, gap=gap)
        s["control"] = gap
        old.append(s)

    # Two controls now (score threshold and pause timeout), so the curve is
    # the Pareto frontier over both rather than a single sweep: for each
    # achievable interruption rate, the best recall any setting reached.
    raw = []
    for thr in (0.20, 0.30, 0.40, 0.50, 0.60, 0.99):
        for pause in (1100, 1300, 1800, 2500, 3500, 6000):
            s = run(pids, parsed, args, mode="verbatim", acoustic="stutter",
                    conf_min=args.conf_min, gap=args.gap, scorer_thr=thr,
                    rearm=args.rearm, pause_ms=pause)
            s["control"] = "thr=%.2f/pause=%d" % (thr, pause)
            raw.append(s)
    new = []
    for r in sorted(raw, key=lambda r: r["fires_per_min"] or 0):
        if not new or (r["recall"] or 0) > (new[-1]["recall"] or 0):
            new.append(r)

    print("SPEAKERS: %s  (%s)" % (",".join(pids), args.pids))
    for name, curve, label in (("OLD  (intended ASR + FillerNet, any-of)", old, "min_gap"),
                               ("NEW  (verbatim + StutterNet + speaker + scorer)", new, "thresh")):
        print("")
        print(name)
        print("  %8s %8s %8s %8s %9s" % (label, "recall", "FA", "partner", "fires/min"))
        for r in curve:
            print("  %8s %8s %8s %8s %9s"
                  % (r["control"], r["recall"], r["false_alarm_rate"],
                     r["partner_fire_rate"], r["fires_per_min"]))

    print("")
    print("MATCHED COMPARISON (recall, interpolated onto the same x)")
    print("  %-22s %10s %10s %10s" % ("matched at", "old", "new", "delta"))
    rows = []
    for target in (6.0, 8.0, 10.0, 12.0, 15.0):
        a = interp_recall(old, "fires_per_min", target)
        b = interp_recall(new, "fires_per_min", target)
        if a is not None and b is not None:
            rows.append({"axis": "fires_per_min", "at": target,
                         "old": round(a, 4), "new": round(b, 4),
                         "delta": round(b - a, 4)})
            print("  %-22s %10.3f %10.3f %+10.3f" % ("%.0f fires/min" % target, a, b, b - a))
    for target in (0.20, 0.30, 0.40, 0.50):
        a = interp_recall(old, "false_alarm_rate", target)
        b = interp_recall(new, "false_alarm_rate", target)
        if a is not None and b is not None:
            rows.append({"axis": "false_alarm_rate", "at": target,
                         "old": round(a, 4), "new": round(b, 4),
                         "delta": round(b - a, 4)})
            print("  %-22s %10.3f %10.3f %+10.3f" % ("FA %.2f" % target, a, b, b - a))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "speakers": pids, "split": args.pids,
        "old_arm": "intended ASR (browser-equivalent) + FillerNet + any-of triggers",
        "new_arm": ("verbatim ASR + StutterNet + ECAPA speaker attribution + "
                    "fitted stall scorer"),
        "note": ("Scorer weights and threshold were fitted on %s and these "
                 "speakers were not used for that fit." % ",".join(TUNE)),
        "old_curve": old, "new_curve": new, "matched": rows,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
