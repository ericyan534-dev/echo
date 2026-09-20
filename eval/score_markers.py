"""Marker-level scoring: does the detector fire ON the event, or merely often?

WHY THE OLD METRIC COULD NOT ANSWER THIS
----------------------------------------
`run_aphasia_eval.py` credits a fire anywhere inside a clinician-coded
utterance plus a 1.75 s margin. Aphasic utterances are long, so that window is
a median of 5.7 s wide. A detector that fires on a timer lands inside it
routinely without detecting anything, and scores the same as one that fired
because it saw a cut-off word. Measured directly: the old stack recovers the
coded evidence into its transcript 0.025 of the time, yet scores 0.84 recall.
Those two numbers cannot both describe detection.

`eval/align_aprocsa.py` fixed the missing half. APROCSA carries per-word media
bullets on its `%wor` tiers, so most markers have corpus-grade timestamps and
the rest are aligned inside windows those bullets pin down. This scores against
those instants instead of the utterance box.

THE GOLD SET
------------
STRONG markers only -- retracings, phonological fragments, abandoned
utterances, long pauses, clinician error codes -- with a trustworthy timing
source and confidence >= 0.8. That is 722 events across six speakers. Filled
pauses and short pauses are deliberately excluded: they are the WEAK markers,
they are the ones the aligner had to place rather than read, and they occur in
fluent speech too.

THE CONTROL THAT MAKES THIS DECISIVE
------------------------------------
A TIMER arm fires at fixed intervals at the same rate as the system under test,
using no audio at all. If a detector cannot beat a metronome at its own firing
rate on this metric, it is not detecting anything, and the metric says so
instead of hiding it. Every comparison here is reported against that floor.

    python eval/score_markers.py --tol 750
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
from eval.run_aphasia_eval import CACHE, TRANSCRIPTS  # noqa: E402
from eval.tune_aphasia_detector import (HELDOUT, TUNE, apply_speaker,  # noqa: E402
                                        speaker_map, streams)
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "marker_scoring.json"

TRUSTED_SOURCES = ("wor_exact", "fa_gap")
MIN_CONF = 0.8


def gold_markers(pid: str, lo_ms: int, hi_ms: int, strict: bool = False):
    """Timed STRONG markers inside the scored region.

    `strict` keeps only `wor_exact` -- the corpus's own single-word bullets,
    which need no aligner at all. That subset is small but its timing is as
    good as this corpus gets, so it is the right set for a tight tolerance.
    """
    p = CACHE / ("align_%s.json" % pid)
    if not p.exists():
        return [], []
    ev = json.loads(p.read_text(encoding="utf-8"))["events"]
    srcs = ("wor_exact",) if strict else TRUSTED_SOURCES
    gold, accept = [], []
    for e in ev:
        t = e.get("t_ms")
        if t is None or not (lo_ms <= t <= hi_ms):
            continue
        # Anything timed counts as an ACCEPTABLE place to fire -- firing next
        # to a filled pause is not a mistake, it just is not evidence of
        # detection. Only strong, well-timed markers count as targets.
        accept.append(t)
        if (e["weight"] == "strong" and e.get("source") in srcs
                and (e.get("confidence") or 0) >= MIN_CONF):
            gold.append(t)
    return sorted(gold), sorted(accept)


def score(fires_ms, gold, accept, region_ms, tol):
    """Recall over gold markers, precision over any timed marker, rate.

    `tol` is a (pre, post) pair and it is ASYMMETRIC on purpose. A detector
    cannot fire before the evidence exists: the pause trigger fires 1300 ms
    after the speaker goes quiet, and a committed word reaches the detector
    ~200-500 ms after it was spoken. Scoring a symmetric window punishes the
    system for causality, not for being wrong -- at +/-250 ms it scored below a
    metronome, which is a statement about the window, not the detector. The
    timer control is scored through the identical window, so widening it
    cannot flatter the system relative to its floor.
    """
    f = np.array(sorted(fires_ms), dtype="float64")
    g = np.array(gold, dtype="float64")
    a = np.array(accept, dtype="float64")

    pre, post = tol

    def hit_marker(ms, fs):
        """A marker is detected if some fire lands in [m - pre, m + post]."""
        if len(ms) == 0 or len(fs) == 0:
            return np.zeros(len(ms), dtype=bool)
        lo = np.searchsorted(fs, ms - pre, side="left")
        hi = np.searchsorted(fs, ms + post, side="right")
        return hi > lo

    def fire_on(fs, ms):
        """A fire is on-evidence if some marker sits in [f - post, f + pre]."""
        if len(ms) == 0 or len(fs) == 0:
            return np.zeros(len(fs), dtype=bool)
        lo = np.searchsorted(ms, fs - post, side="left")
        hi = np.searchsorted(ms, fs + pre, side="right")
        return hi > lo

    hit = hit_marker(g, f).sum() if len(g) else 0
    on = fire_on(f, a).sum() if len(f) else 0
    return {
        "n_gold": int(len(g)), "n_fires": int(len(f)),
        "recall": round(float(hit) / len(g), 4) if len(g) else None,
        "precision": round(float(on) / len(f), 4) if len(f) else None,
        "fires_per_min": round(len(f) / (region_ms / 60000.0), 2),
    }


def collect_fires(pid, args, *, mode, acoustic, conf_min, gap, thr=None,
                  pause_ms=1300, rearm=1):
    items = streams(pid, args.skip, args.region, mode, acoustic)
    if items is None:
        return None
    if conf_min > 0:
        items = apply_speaker(items, speaker_map(pid, args.skip, args.region),
                              args.skip * 1000, conf_min)
    sc = None
    if thr is not None:
        sc = StallScorer(weights=DEFAULT_WEIGHTS, bias=DEFAULT_BIAS,
                         threshold=thr, hedges=tuple(HEDGES))
    det = StallDetector(pause_ms=pause_ms, min_gap_ms=gap,
                        timeline=Timeline(wearer_conf_min=conf_min),
                        scorer=sc, rearm_content_words=rearm)
    out = []
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
            out.append(ev.at_ms + args.skip * 1000)
    return out


def timer_fires(rate_per_min, args):
    """The control: fire on a clock, ignoring the audio entirely."""
    lo = args.skip * 1000
    n = max(1, int(round(rate_per_min * args.region / 60.0)))
    step = (args.region * 1000.0) / n
    return [lo + step * (i + 0.5) for i in range(n)]


def run_arm(pids, args, tol, strict, **kw):
    tot = {"n_gold": 0, "n_fires": 0, "hit": 0, "on": 0, "ms": 0}
    for pid in pids:
        fires = collect_fires(pid, args, **kw)
        if fires is None:
            continue
        lo, hi = args.skip * 1000, (args.skip + args.region) * 1000
        gold, accept = gold_markers(pid, lo, hi, strict)
        s = score(fires, gold, accept, args.region * 1000, tol)
        tot["n_gold"] += s["n_gold"]
        tot["n_fires"] += s["n_fires"]
        tot["hit"] += round((s["recall"] or 0) * s["n_gold"])
        tot["on"] += round((s["precision"] or 0) * s["n_fires"])
        tot["ms"] += args.region * 1000
    return _fin(tot)


def run_timer(pids, args, tol, strict, rate):
    tot = {"n_gold": 0, "n_fires": 0, "hit": 0, "on": 0, "ms": 0}
    for pid in pids:
        lo, hi = args.skip * 1000, (args.skip + args.region) * 1000
        gold, accept = gold_markers(pid, lo, hi, strict)
        if not gold:
            continue
        s = score(timer_fires(rate, args), gold, accept, args.region * 1000, tol)
        tot["n_gold"] += s["n_gold"]
        tot["n_fires"] += s["n_fires"]
        tot["hit"] += round((s["recall"] or 0) * s["n_gold"])
        tot["on"] += round((s["precision"] or 0) * s["n_fires"])
        tot["ms"] += args.region * 1000
    return _fin(tot)


def _fin(t):
    return {
        "n_gold": t["n_gold"], "n_fires": t["n_fires"],
        "recall": round(t["hit"] / t["n_gold"], 4) if t["n_gold"] else None,
        "precision": round(t["on"] / t["n_fires"], 4) if t["n_fires"] else None,
        "fires_per_min": round(t["n_fires"] / (t["ms"] / 60000.0), 2) if t["ms"] else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--pre", type=int, default=250,
                    help="ms a fire may PRECEDE the marker (small: causality)")
    ap.add_argument("--post", type=int, default=2000,
                    help="ms a fire may FOLLOW the marker (the detection budget)")
    ap.add_argument("--strict", action="store_true",
                    help="gold = corpus bullets only (wor_exact)")
    ap.add_argument("--pids", default="all", choices=["all", "tune", "heldout"])
    args = ap.parse_args()

    args.tol = (args.pre, args.post)
    pids = {"all": TUNE + HELDOUT, "tune": TUNE, "heldout": HELDOUT}[args.pids]
    load_all(TRANSCRIPTS)

    arms = []
    for gap in (0, 2500, 5000):
        s = run_arm(pids, args, args.tol, args.strict, mode="intended",
                    acoustic="filler", conf_min=0.0, gap=gap)
        s["arm"] = "OLD  gap=%d" % gap
        arms.append(s)
    for thr in (0.20, 0.35, 0.50):
        s = run_arm(pids, args, args.tol, args.strict, mode="verbatim",
                    acoustic="stutter", conf_min=0.35, gap=1500, thr=thr,
                    pause_ms=1300)
        s["arm"] = "v3   thr=%.2f" % thr
        arms.append(s)
    # v4: same everything, WavLM acoustic channel instead of the log-mel CNN.
    for thr in (0.20, 0.35, 0.50):
        s = run_arm(pids, args, args.tol, args.strict, mode="verbatim",
                    acoustic="sslstutter", conf_min=0.35, gap=1500, thr=thr,
                    pause_ms=1300)
        s["arm"] = "v4   thr=%.2f" % thr
        arms.append(s)

    print("MARKER-LEVEL SCORING  (window -%d/+%d ms, %s gold, speakers=%s)"
          % (args.pre, args.post,
             "wor_exact only" if args.strict else "strong+trusted", args.pids))
    print("  %-16s %7s %8s %10s %10s %8s"
          % ("arm", "n_gold", "n_fires", "recall", "precision", "f/min"))
    rows = []
    for s in arms:
        timer = run_timer(pids, args, args.tol, args.strict, s["fires_per_min"])
        s["timer_recall"] = timer["recall"]
        s["timer_precision"] = timer["precision"]
        s["lift_recall"] = (round(s["recall"] - timer["recall"], 4)
                            if s["recall"] is not None and timer["recall"] is not None
                            else None)
        rows.append(s)
        print("  %-16s %7d %8d %10s %10s %8s"
              % (s["arm"], s["n_gold"], s["n_fires"], s["recall"],
                 s["precision"], s["fires_per_min"]))
        print("  %-16s %7s %8s %10s %10s %8s"
              % ("  timer control", "", "", timer["recall"], timer["precision"], ""))

    print("")
    print("LIFT OVER A METRONOME AT THE SAME FIRING RATE (recall)")
    print("  %-16s %10s %10s %10s" % ("arm", "system", "timer", "lift"))
    for s in rows:
        print("  %-16s %10s %10s %+10s"
              % (s["arm"], s["recall"], s["timer_recall"], s["lift_recall"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "window_pre_ms": args.pre, "window_post_ms": args.post,
        "speakers": pids,
        "gold": ("wor_exact only" if args.strict
                 else "strong markers, source in %s, confidence >= %.1f"
                      % (str(TRUSTED_SOURCES), MIN_CONF)),
        "control": ("a timer firing at fixed intervals at the same rate, using "
                    "no audio; recall above it is the only evidence of detection"),
        "arms": rows,
    }, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
