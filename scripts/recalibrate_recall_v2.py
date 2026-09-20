"""Recall-targeted per-frame operating point -> models/stutternet_recall_v2.pt.

This does NOT retrain. It copies the trained WavLM weights from
models/stutternet_ssl_v2.pt unchanged and only moves the per-frame operating
point (temperature + threshold) that backend/acoustic/stream.py compares
against. AP is therefore identical to the source checkpoint; only recall/
precision at the operating point change, and that is the whole point.

WHY A NEW OPERATING POINT
-------------------------
The shipped ssl_v2 point was fitted at target_fpr=0.02 (a 2% clean-clip fire
budget). Measured on VAL that gives Block recall 0.158, Prolongation 0.373 --
the model finds obvious blocks (Block AP 0.35) but the threshold is set so
tight that it almost never fires. See eval/diagnose_live_stutter.py.

HOW THE THRESHOLD IS CHOSEN (research: recall-targeted with an FPR cap)
----------------------------------------------------------------------
For each type, pick the frame logit threshold that is the MORE CONSERVATIVE of:
  * the point that recalls `target_recall` of that type's positive clips, and
  * the point whose clean-clip fire rate is `fpr_cap` (the anti-nag budget).
So recall is raised toward the target but never past the FPR cap -- a type that
cannot hit the target without nagging stays at the cap. Positives and clean
clips are the VAL split's peak frame logits, exactly as train_stutter_ssl and
eval/calibrate_stutter_frames do it; TEST is never touched. The temperature is
then set so the threshold maps to sigmoid(2.0)=0.881, the representable prob the
live stream compares against (same scheme as the source checkpoint).

    python scripts/recalibrate_recall_v2.py --target-recall 0.85 --fpr-cap 0.15
    python scripts/recalibrate_recall_v2.py --interjection-recall 0.75

Never overwrites the three protected checkpoints. Writes only
models/stutternet_recall_v2.pt (+ its _metrics.json). ASCII only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.utils.data as tud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.stutter_ssl import TYPES, load_checkpoint  # noqa: E402
from scripts.train_stutter import load_sources, make_splits  # noqa: E402
from scripts.train_stutter_ssl import WaveDataset, run_eval  # noqa: E402

SRC = ROOT / "models" / "stutternet_ssl_v2.pt"
OUT = ROOT / "models" / "stutternet_recall_v2.pt"
PROTECTED = {"stutternet.pt", "fillernet.pt", "stutternet_ssl_v2.pt"}
TARGET_LOGIT = 2.0


def recall_targeted(peak_logits, labels, target_recall, fpr_cap,
                    per_type_recall) -> dict:
    clean = labels.max(axis=1) == 0
    out = {}
    for k, t in enumerate(TYPES):
        neg = peak_logits[clean, k].astype("float64")
        pos = peak_logits[labels[:, k] > 0, k].astype("float64")
        tr = per_type_recall.get(t, target_recall)
        # threshold that recalls `tr` of positives
        thr_recall = float(np.quantile(pos, 1.0 - tr)) if len(pos) else TARGET_LOGIT
        # threshold whose clean-clip fire rate is fpr_cap
        thr_fpr = float(np.quantile(neg, 1.0 - fpr_cap)) if len(neg) else TARGET_LOGIT
        # the anti-nag budget wins ties: never fire more than fpr_cap on clean
        thr = max(thr_recall, thr_fpr)
        temp = max(1.0, thr / TARGET_LOGIT)
        prob_thr = float(1.0 / (1.0 + np.exp(-thr / temp)))
        out[t] = {
            "temperature": round(temp, 4),
            "threshold": round(prob_thr, 6),
            "logit_threshold": round(thr, 4),
            "target_recall": tr,
            "fpr_cap": fpr_cap,
            "capped_by_fpr": bool(thr_fpr > thr_recall),
            "recall_at_threshold": round(float((pos >= thr).mean()), 4) if len(pos) else None,
            "clean_fire_rate": round(float((neg >= thr).mean()), 4) if len(neg) else None,
            "n_pos": int(len(pos)), "n_clean": int(len(neg)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--target-recall", type=float, default=0.85)
    ap.add_argument("--fpr-cap", type=float, default=0.15)
    ap.add_argument("--interjection-recall", type=float, default=0.75,
                    help="fillers nag most; keep their recall target lower")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.name in PROTECTED:
        raise SystemExit("refusing to write protected checkpoint %s" % out_path.name)
    if not Path(args.src).exists():
        raise SystemExit("source checkpoint %s not found" % args.src)

    model = load_checkpoint(args.src, args.device)
    rows, npy = load_sources([])
    sp = make_splits(rows, holdout_show=None)
    val = tud.DataLoader(WaveDataset(npy, rows, sp["val"], train=False),
                         batch_size=32, num_workers=0)
    yv, _, peaks = run_eval(model, val, args.device, amp=True)
    print("VAL n=%d  (recall-targeted operating point, TEST untouched)" % len(yv))

    per_type = {"Interjection": args.interjection_recall}
    cal = recall_targeted(peaks, yv, args.target_recall, args.fpr_cap, per_type)

    print("\n  %-13s %6s %7s %10s %8s %10s %8s"
          % ("type", "temp", "prob", "logit_thr", "recall", "clean_ff", "cap?"))
    for t in TYPES:
        v = cal[t]
        print("  %-13s %6.2f %7.4f %10.3f %8s %10s %8s"
              % (t, v["temperature"], v["threshold"], v["logit_threshold"],
                 v["recall_at_threshold"], v["clean_fire_rate"],
                 "fpr" if v["capped_by_fpr"] else "rec"))

    # Overlay the new operating point onto the SOURCE weights, unchanged.
    apply = torch.tensor([cal[t]["temperature"] for t in TYPES], dtype=torch.float32)
    model.frame_temperature.copy_(apply)
    state = torch.load(args.src, map_location="cpu", weights_only=False)
    state["model"] = model.trainable_state_dict()
    state["frame_thresholds"] = {t: cal[t]["threshold"] for t in TYPES}
    state["frame_temperature"] = {t: cal[t]["temperature"] for t in TYPES}
    state["frame_calibration"] = {
        "method": "recall-targeted with fpr cap",
        "target_recall": args.target_recall, "fpr_cap": args.fpr_cap,
        "interjection_recall": args.interjection_recall,
        "split": "val", "space": "logit", "detail": cal,
        "source_checkpoint": Path(args.src).name}
    state["recall_recipe"] = state["frame_calibration"]
    torch.save(state, out_path)

    metrics = {
        "checkpoint": out_path.name,
        "derived_from": Path(args.src).name,
        "note": ("recall-targeted operating point ONLY; weights are byte-identical "
                 "to the source, so AP is unchanged -- only recall/precision at "
                 "the operating point move. Fitted on VAL; TEST untouched."),
        "frame_thresholds": state["frame_thresholds"],
        "frame_temperature": state["frame_temperature"],
        "frame_calibration_detail": cal,
        "shipped_ssl_v2_recall_at_fpr_0.02": {
            "Block": 0.158, "Prolongation": 0.373, "SoundRep": 0.521,
            "WordRep": 0.816, "Interjection": 0.725},
    }
    Path(str(out_path.with_suffix("")) + "_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8")
    print("\n  wrote %s" % out_path)
    print("  metrics -> %s" % (out_path.with_suffix("").name + "_metrics.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
