"""Fit the stall scorer's weights on the TUNE speakers, report on HELD OUT.

The detector's any-of trigger rule cannot trade recall against interruptions --
it can only be made uniformly quieter. This fits one logistic model over the
same signals so the trade-off becomes a threshold on a calibrated score.

DISCIPLINE
----------
Weights are fitted on 1554/1731/1833 and never on 1713/1738/1944. The threshold
is chosen on TUNE as well. The held-out three are looked at once, at the end,
and whatever they say is what gets reported -- including if it is worse.

Labels come from the CHAT coding: an instant is positive when it falls inside a
clinician-coded word-search utterance (with the same pre/post window the
end-to-end scorer uses), negative inside a fluent participant utterance, and
negative during the conversation partner's speech -- firing there is always
wrong. Instants covered by no utterance at all are dropped rather than guessed.

    python eval/fit_stall_scorer.py
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

from backend.schemas import AcousticEvent, SilenceTick, TurnEnd, Word  # noqa: E402
from backend.stall_detector import HEDGES, StallDetector  # noqa: E402
from backend.stall_scorer import FEATURES, StallScorer  # noqa: E402
from backend.timeline import Timeline  # noqa: E402
from eval.run_aphasia_eval import POST_MS, PRE_MS, TRANSCRIPTS  # noqa: E402
from eval.tune_aphasia_detector import (HELDOUT, TUNE, apply_speaker,  # noqa: E402
                                        score_one, speaker_map, streams,
                                        summarize, utility)
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "stall_scorer_fit.json"


def label_of(utts, t_ms):
    """1 word-search, 0 fluent-or-partner, None if no utterance covers it."""
    for u in utts:
        if u["start_ms"] is None:
            continue
        if not u["is_participant"]:
            if u["start_ms"] <= t_ms <= u["end_ms"]:
                return 0
            continue
        if u["start_ms"] - PRE_MS <= t_ms <= u["end_ms"] + POST_MS:
            return 1 if u["word_search"] else 0
    return None


def collect(pids, parsed, skip, region, mode, acoustic, conf_min):
    X, y = [], []
    for pid in pids:
        items = streams(pid, skip, region, mode, acoustic)
        if items is None:
            continue
        spans = speaker_map(pid, skip, region)
        items = apply_speaker(items, spans, skip * 1000, conf_min)
        tl = Timeline(wearer_conf_min=conf_min)
        sc = StallScorer(hedges=tuple(HEDGES))
        utts = parsed[pid]["utterances"]
        off = skip * 1000
        for _, it in sorted(items, key=lambda kv: kv[0]):
            if isinstance(it, Word):
                tl.add_word(it)
                continue
            if isinstance(it, TurnEnd):
                tl.mark_turn_boundary()
                sc.reset()
                continue
            if isinstance(it, AcousticEvent):
                sc.observe_acoustic(it.kind, it.at_ms, it.confidence)
            now = it.at_ms
            if tl.content_count() < 1 or not tl.current_utterance():
                continue
            lab = label_of(utts, now + off)
            if lab is None:
                continue
            f = sc.features(tl, now)
            X.append([f[k] for k in FEATURES])
            y.append(lab)
    return np.array(X, dtype="float64"), np.array(y, dtype="float64")


# Features where MORE of the thing can never mean LESS likely a word search.
# This is domain knowledge, not a fitting trick, and it is load-bearing: the
# unconstrained fit put pause_log at -0.59, i.e. "the longer they are stuck,
# the less stuck they are". That happened because the labels are utterance
# level and 75% positive, so a long pause inside an already-positive utterance
# adds no discriminative signal -- and removing the pause fallback cost recall
# on exactly the utterances where no lexical evidence survives.
MONOTONE = ("pause_log", "fillers_recent", "hedge", "fragments",
            "acou_block", "acou_prolong", "acou_rep", "acou_filler")


def fit_logistic(X, y, l2=1.0, iters=4000, lr=0.5, nonneg=None):
    """Projected gradient descent, clipping the monotone features at zero.

    Not sklearn: this environment's sklearn import chain pulls a pandas built
    against NumPy 1.x and raises under NumPy 2.4."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    # Class-balanced: word-search instants are the minority, and an unweighted
    # fit is minimised by predicting "no search" everywhere.
    pos = max(1.0, y.sum())
    neg = max(1.0, n - y.sum())
    sw = np.where(y > 0, n / (2 * pos), n / (2 * neg))
    for _ in range(iters):
        z = X @ w + b
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        g = (p - y) * sw
        gw = X.T @ g / n + l2 * w / n
        gb = g.mean()
        w -= lr * gw
        b -= lr * gb
        if nonneg is not None:
            w[nonneg] = np.maximum(w[nonneg], 0.0)
    return w, b


def build_runner(args, parsed, weights, bias):
    def run(pids, thr, gap):
        tot = {"hits": 0, "n_pos": 0, "fa": 0, "n_neg": 0, "partner": 0,
               "n_par": 0, "fires": 0, "region_ms": 0}
        for pid in pids:
            items = streams(pid, args.skip, args.region, args.mode, args.acoustic)
            if items is None:
                continue
            spans = speaker_map(pid, args.skip, args.region)
            items = apply_speaker(items, spans, args.skip * 1000, args.conf_min)
            det = StallDetector(
                pause_ms=1300, min_gap_ms=gap,
                timeline=Timeline(wearer_conf_min=args.conf_min),
                scorer=StallScorer(weights=weights, bias=bias, threshold=thr,
                                   hedges=tuple(HEDGES)))
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
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--acoustic", default="stutter")
    ap.add_argument("--conf-min", type=float, default=0.35)
    ap.add_argument("--min-gap", type=int, default=2500)
    args = ap.parse_args()

    parsed = load_all(TRANSCRIPTS)
    Xt, yt = collect(TUNE, parsed, args.skip, args.region, args.mode,
                     args.acoustic, args.conf_min)
    print("tune instants: %d (%.1f%% positive)" % (len(yt), 100 * yt.mean()))
    mask = np.array([k in MONOTONE for k in FEATURES])
    w, b = fit_logistic(Xt, yt, nonneg=mask)
    weights = {k: round(float(v), 4) for k, v in zip(FEATURES, w)}
    print("")
    print("FITTED WEIGHTS (bias %.4f)" % b)
    for k in FEATURES:
        print("  %-16s %+8.4f" % (k, weights[k]))

    run = build_runner(args, parsed, weights, float(b))

    print("")
    print("THRESHOLD SWEEP ON TUNE (min_gap=%d)" % args.min_gap)
    print("  %6s %8s %8s %8s %9s %8s"
          % ("thr", "recall", "FA", "partner", "fires/min", "utility"))
    rows = []
    for thr in (0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80):
        s = run(TUNE, thr, args.min_gap)
        s["threshold"] = thr
        s["utility"] = round(utility(s), 4)
        rows.append(s)
        print("  %6.2f %8s %8s %8s %9s %8s"
              % (thr, s["recall"], s["false_alarm_rate"], s["partner_fire_rate"],
                 s["fires_per_min"], s["utility"]))

    best = max(rows, key=lambda r: r["utility"])
    held = run(HELDOUT, best["threshold"], args.min_gap)
    print("")
    print("HELD OUT (%s) at threshold %.2f" % (",".join(HELDOUT), best["threshold"]))
    print("  recall %s   FA %s   partner %s   fires/min %s"
          % (held["recall"], held["false_alarm_rate"],
             held["partner_fire_rate"], held["fires_per_min"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "tune": TUNE, "heldout_speakers": HELDOUT,
        "n_tune_instants": int(len(yt)), "positive_rate": round(float(yt.mean()), 4),
        "weights": weights, "bias": round(float(b), 4),
        "conf_min": args.conf_min, "min_gap_ms": args.min_gap,
        "threshold_sweep_tune": rows, "best_on_tune": best, "heldout": held,
    }, indent=2), encoding="utf-8")

    print("")
    print("Paste into backend/stall_scorer.py:")
    print("DEFAULT_WEIGHTS = %s" % json.dumps(weights))
    print("DEFAULT_BIAS = %.4f" % b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
