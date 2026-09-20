"""Pick the per-FRAME operating point. The clip thresholds were the wrong scale.

THE BUG THIS FIXES
------------------
`scripts/train_stutter.py` fits one threshold per type on the CLIP-level
probability -- the linear-softmax pool over ~75 frames. `AcousticStream` then
compared those thresholds against a PER-FRAME probability, because the live
stream reads only the frames covering the last hop.

Those are different quantities. Pooling averages a short event down, so a
clip threshold is systematically LOWER than the frame probability that event
produces, and comparing one against the other fires far too readily. Measured
consequence on real aphasic speech: 25 fires per minute, and a false-alarm
rate that rose from 0.704 to 0.889 when StutterNet replaced FillerNet. The
model was not the problem; the scale was.

HOW THE THRESHOLD IS CHOSEN
---------------------------
By an interruption budget, on SEP-28k -- never on APROCSA. For each type, take
the highest frame probability in each clip, and pick the threshold at which
clips containing NO dysfluency at all exceed it only `--target-fpr` of the
time. That is a property of the model against its own training distribution,
so the aphasia evaluation stays an independent test rather than becoming a
tuning set.

A 3 s negative clip firing at rate p means roughly 20p false events per minute
per type, so 0.02 lands near 2 false events/minute across all five -- before
the detector's own refractory and its requirement of a content word.

    python eval/calibrate_stutter_frames.py --target-fpr 0.02
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
import torch  # noqa: E402

from backend.acoustic.features import logmel  # noqa: E402
from backend.acoustic.stutter import TYPES, load_checkpoint  # noqa: E402
from scripts.train_stutter import CLIPS_IDX, CLIPS_NPY, load_index, make_splits  # noqa: E402

CKPT = ROOT / "models" / "stutternet.pt"
OUT = ROOT / "eval" / "results" / "stutter_frame_calibration.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="max share of dysfluency-free clips allowed to fire")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--write", action="store_true",
                    help="store the thresholds in the checkpoint")
    args = ap.parse_args()

    if not CKPT.exists() or not CLIPS_IDX.exists():
        print("SKIPPED -- no checkpoint or clip index")
        return 0

    rows = load_index()
    splits = make_splits(rows, holdout_show=None)
    idx = splits[args.split]
    arr = np.load(CLIPS_NPY, mmap_mode="r")
    model = load_checkpoint(CKPT, args.device)

    print("Frame-level calibration on the %s split (n=%d)" % (args.split, len(idx)))
    peaks = np.zeros((len(idx), len(TYPES)), dtype="float32")
    labels = np.zeros((len(idx), len(TYPES)), dtype="float32")
    B = 64
    for lo in range(0, len(idx), B):
        chunk = idx[lo:lo + B]
        pcm = torch.from_numpy(
            np.stack([arr[j] for j in chunk]).astype("float32") / 32768.0)
        feats = torch.stack([logmel(p) for p in pcm]).unsqueeze(1).to(args.device)
        with torch.no_grad():
            frames = torch.sigmoid(model(feats))          # (B, types, T)
        # The live stream reads the max over the frames covering the last hop,
        # so the clip-level analogue of that decision is the max over all
        # frames -- "would this clip have fired at any moment".
        peaks[lo:lo + len(chunk)] = frames.max(dim=2).values.cpu().numpy()
        labels[lo:lo + len(chunk)] = np.array([rows[j]["labels"] for j in chunk],
                                              dtype="float32")
        if lo and lo % (B * 20) == 0:
            print("  %d/%d" % (lo, len(idx)), flush=True)

    # "Clean" = no dysfluency of any type. Using per-type negatives instead
    # would count a Block clip as a negative for WordRep, and those clips are
    # dysfluent -- firing on them is not the false alarm being budgeted.
    clean = labels.max(axis=1) == 0
    print("  clean (no dysfluency) clips: %d / %d" % (clean.sum(), len(idx)))

    result = {}
    for k, t in enumerate(TYPES):
        neg = peaks[clean, k]
        pos = peaks[labels[:, k] > 0, k]
        # Threshold at the (1 - fpr) quantile of the clean clips: by
        # construction exactly `target_fpr` of them exceed it.
        thr = float(np.quantile(neg, 1.0 - args.target_fpr)) if len(neg) else 0.9
        result[t] = {
            "threshold": round(thr, 4),
            "n_clean": int(len(neg)), "n_pos": int(len(pos)),
            "recall_at_threshold": round(float((pos >= thr).mean()), 4) if len(pos) else None,
            "clean_fire_rate": round(float((neg >= thr).mean()), 4) if len(neg) else None,
            "pos_peak_median": round(float(np.median(pos)), 4) if len(pos) else None,
            "clean_peak_median": round(float(np.median(neg)), 4) if len(neg) else None,
        }

    # The budget is a choice, so show what each one costs. Peaks are already
    # computed, so the sweep is free.
    sweep = []
    for f in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20):
        row = {"target_fpr": f}
        for k, t in enumerate(TYPES):
            neg = peaks[clean, k]
            pos = peaks[labels[:, k] > 0, k]
            thr = float(np.quantile(neg, 1.0 - f))
            row[t] = {"threshold": round(thr, 4),
                      "recall": round(float((pos >= thr).mean()), 4)}
        sweep.append(row)

    prev = torch.load(CKPT, map_location="cpu", weights_only=False)
    old = prev.get("thresholds", {})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK", "split": args.split, "target_fpr": args.target_fpr,
        "n_clips": len(idx), "n_clean": int(clean.sum()),
        "clip_thresholds_previously_used": old,
        "frame_thresholds": {t: v["threshold"] for t, v in result.items()},
        "detail": result,
        "sweep": sweep,
        "why": ("Clip thresholds were fitted on the linear-softmax POOL and were "
                "being compared against PER-FRAME probabilities. Pooling averages a "
                "short event down, so the clip threshold is systematically lower "
                "than the frame probability the same event produces."),
        "not_fitted_on": ("APROCSA. The budget is set against SEP-28k so the aphasia "
                          "evaluation stays an independent test."),
    }, indent=2), encoding="utf-8")

    print("")
    print("PER-FRAME OPERATING POINT (target clean-clip fire rate %.3f)" % args.target_fpr)
    print("  %-14s %10s %10s %10s %12s" % ("type", "clip thr", "frame thr", "recall",
                                           "clean fires"))
    for t, v in result.items():
        print("  %-14s %10s %10.3f %10s %12s"
              % (t, old.get(t, "--"), v["threshold"], v["recall_at_threshold"],
                 v["clean_fire_rate"]))

    print("")
    print("WHAT EACH INTERRUPTION BUDGET COSTS (recall per type)")
    print("  %-10s %s" % ("clean FPR", " ".join("%13s" % t for t in TYPES)))
    for row in sweep:
        print("  %-10.3f %s" % (row["target_fpr"],
                                " ".join("%13.3f" % row[t]["recall"] for t in TYPES)))

    if args.write:
        prev["frame_thresholds"] = {t: v["threshold"] for t, v in result.items()}
        prev["frame_calibration"] = {"target_fpr": args.target_fpr, "split": args.split}
        torch.save(prev, CKPT)
        print("")
        print("  wrote frame_thresholds into %s" % CKPT.name)
    else:
        print("")
        print("  (dry run -- pass --write to store them in the checkpoint)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
