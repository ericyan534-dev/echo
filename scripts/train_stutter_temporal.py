"""Train StutterTemporal -- a BiLSTM sequence head on frozen WavLM features.

    python scripts/train_stutter_temporal.py --out models/stutternet_temporal_v1.pt

WHAT THIS IS
------------
A controlled swap of ONLY the head. Everything that would confound the
comparison with the per-frame conv head (StutterSSL, frozen) is imported rather
than reimplemented:

  * load_sources / make_splits from scripts/train_stutter.py -- the SAME
    episode-disjoint-stratified-by-show assignment down to the RNG seed. No clip
    from a test episode is ever trained on.
  * WaveDataset / run_eval / frame_calibration / apply_calibration from
    scripts/train_stutter_ssl.py -- the SAME raw-waveform pipeline, the SAME
    peak-logit VAL eval, and the SAME per-frame operating point fitted to the
    SAME clean-fire budget (target_fpr, default 0.02). So the frame threshold is
    placed at an equal clean-speech false-fire rate for both models, which is
    the whole point: recall is then read at an equal-false-fire operating point.
  * average_precision / report from scripts/train_stutter.py -- the SAME
    metrics table.

The only differences from train_stutter_ssl.py are that the head is a BiLSTM and
the backbone is ALWAYS frozen (no encoder parameter group, no --unfreeze). A
larger fine-tuned backbone is explicitly a GX10 job.

DISCIPLINE: nothing is chosen on TEST. Epoch selection, thresholds and the
operating point come off VAL; TEST is read once at the end.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.stutter_temporal import (  # noqa: E402
    DEFAULT_ENCODER, FRAME_MS, TYPES, StutterTemporal, load_checkpoint)
# Imported, never reimplemented -- identical data, splits, eval and operating
# point are the entire basis of the head-vs-head comparison.
from scripts.train_stutter import (  # noqa: E402
    MODELS, average_precision, load_sources, make_splits, report)
from scripts.train_stutter_ssl import (  # noqa: E402
    WaveDataset, apply_calibration, frame_calibration, print_calibration,
    run_eval)

SR = 16_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--hidden", type=int, default=128, help="BiLSTM hidden per direction")
    ap.add_argument("--lstm-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--holdout-show", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="per-frame interruption budget, fitted on VAL")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--extra", action="append", default=[])
    ap.add_argument("--split-version", default="legacy", choices=["legacy", "stable"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    amp = not args.no_amp
    if not args.out:
        raise SystemExit("pass an explicit --out (e.g. models/stutternet_temporal_v1.pt); "
                         "this script never claims a default checkpoint name")
    ckpt_path = Path(args.out)
    if ckpt_path.exists():
        raise SystemExit("refusing to overwrite existing checkpoint %s "
                         "(new filenames only)" % ckpt_path)
    metrics_path = ckpt_path.with_name(ckpt_path.stem + "_metrics.json")

    print("StutterTemporal training (BiLSTM head, FROZEN WavLM backbone)")
    print("  encoder: %s" % args.encoder)

    rows, npy_paths = load_sources(args.extra)
    splits = make_splits(rows, args.holdout_show, split_version=args.split_version)
    if args.limit:
        for k in splits:
            splits[k] = splits[k][:args.limit]
    print("  clips: %d   shows: %s" % (len(rows), sorted({r["show"] for r in rows})))
    for k, v in splits.items():
        print("  %-6s %6d clips  %s"
              % (k, len(v), dict(Counter(rows[i]["show"] for i in v))))

    loaders = {
        k: torch.utils.data.DataLoader(
            WaveDataset(npy_paths, rows, splits[k], train=(k == "train")),
            batch_size=args.batch, shuffle=(k == "train"),
            num_workers=args.workers, persistent_workers=bool(args.workers),
            pin_memory=(device == "cuda"), drop_last=(k == "train"))
        for k in ("train", "val", "test")
    }

    model = StutterTemporal(args.encoder, hidden=args.hidden,
                            lstm_layers=args.lstm_layers, dropout=args.dropout).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("  params: %.1fM total, %.3fM trainable  (frozen conv head: ~1.3M trainable)"
          % (n_total / 1e6, n_train_params / 1e6))
    print("  frame rate: %d ms/frame" % FRAME_MS)

    # Class-balanced BCE on the pooled probability -- identical to the baselines.
    train_y = np.array([rows[i]["labels"] for i in splits["train"]], dtype="float32")
    prev = train_y.mean(axis=0).clip(1e-3, 1 - 1e-3)
    pos_weight = torch.tensor((1 - prev) / prev, dtype=torch.float32, device=device)
    print("  pos_weight: %s" % dict(zip(TYPES, np.round(pos_weight.cpu().numpy(), 2))))

    # The layer weights get their own group at 10x the head rate and no weight
    # decay (13 numbers behind a softmax), exactly as train_stutter_ssl does.
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n != "layer_logits"]
    groups = [{"params": [model.layer_logits], "lr": args.lr * 10, "weight_decay": 0.0},
              {"params": head_params, "lr": args.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=1e-2)
    steps = args.epochs * max(1, len(loaders["train"]))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.15)

    eps = 1e-6
    best_ap, best_state, history = -1.0, None, []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16,
                              enabled=amp and device == "cuda")
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in loaders["train"]:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            with autocast:
                clip, _ = model.clip_logits(xb)
            clip = clip.float().clamp(eps, 1 - eps)
            loss = -(pos_weight * yb * clip.log()
                     + (1 - yb) * (1 - clip).log()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(xb)
            n += len(xb)
        y, p, _ = run_eval(model, loaders["val"], device, amp)
        val_ap = float(np.nanmean([average_precision(y[:, k], p[:, k])
                                   for k in range(len(TYPES))]))
        val_block = average_precision(y[:, 0], p[:, 0])
        history.append({"epoch": epoch, "train_loss": round(tot / max(1, n), 4),
                        "val_mAP": round(val_ap, 4),
                        "val_Block_AP": round(float(val_block), 4)})
        print("  epoch %2d  loss %.4f  val mAP %.4f  val Block AP %.4f  (%.0fs)"
              % (epoch, tot / max(1, n), val_ap, val_block, time.time() - t0),
              flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    weights = model.layer_weights()
    print("\n  LEARNED LAYER WEIGHTS (0 = conv/embedding output, 12 = top)")
    print("    " + " ".join("%5.3f" % w for w in weights))
    print("    argmax layer %d" % int(np.argmax(weights)))

    yv, pv, peaks_v = run_eval(model, loaders["val"], device, amp)
    val_report = report(yv, pv, "VAL (thresholds fitted here)")
    yt, pt, _ = run_eval(model, loaders["test"], device, amp)
    split_kind = ("speaker-disjoint (holdout show %s)" % args.holdout_show
                  if args.holdout_show else "episode-disjoint")
    test_report = report(yt, pt, "TEST -- %s" % split_kind)

    thresholds = {t: val_report[t]["threshold"] for t in TYPES}
    thresholds["ANY"] = val_report["ANY"]["threshold"]
    frame_cal = frame_calibration(peaks_v, yv, args.target_fpr)
    frame_thresholds = {t: v["threshold"] for t, v in frame_cal.items()}
    frame_temperature = {t: v["temperature"] for t, v in frame_cal.items()}
    apply_calibration(model, frame_cal)
    print_calibration(frame_cal, args.target_fpr)

    shows_seen = sorted({r["show"] for r in rows})
    trained_on = ("SEP-28k reconstruction, %d clips, %d speaker pools (%s) -- "
                  "partial reconstruction, NOT comparable to published SEP-28k"
                  % (len(rows), len(shows_seen), ", ".join(shows_seen)))

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.trainable_state_dict(),
        "types": TYPES,
        "thresholds": thresholds,
        "frame_thresholds": frame_thresholds,
        "frame_temperature": frame_temperature,
        "frame_calibration": {"target_fpr": args.target_fpr, "split": "val",
                              "space": "logit", "detail": frame_cal},
        "metrics": {"val": val_report, "test": test_report},
        "trained_on": trained_on,
        "split": split_kind,
        "ssl": model.config(),
        "layer_weights": weights,
    }, ckpt_path)

    metrics_path.write_text(json.dumps({
        "checkpoint": ckpt_path.name,
        "encoder": args.encoder,
        "head": "bilstm",
        "params_total": n_total,
        "params_trainable": n_train_params,
        "frame_ms": FRAME_MS,
        "split": split_kind,
        "holdout_show": args.holdout_show,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "hparams": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                    "hidden": args.hidden, "lstm_layers": args.lstm_layers,
                    "dropout": args.dropout, "amp_bf16": amp,
                    "target_fpr": args.target_fpr},
        "history": history,
        "layer_weights": [round(w, 4) for w in weights],
        "val": val_report,
        "test": test_report,
        "thresholds": thresholds,
        "frame_thresholds": frame_thresholds,
        "frame_temperature": frame_temperature,
        "frame_calibration_detail": frame_cal,
        "baseline": "models/stutternet_ssl_frozen_metrics.json (frozen conv head, same splits)",
        "trained_on": trained_on,
        "caveat": ("Stuttered speech, not aphasic speech. Thresholds and epoch "
                   "selection fitted on VAL only."),
    }, indent=2), encoding="utf-8")
    print("\n  saved %s" % ckpt_path)
    print("  metrics -> %s" % metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
