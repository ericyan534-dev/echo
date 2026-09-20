"""Is the episode-disjoint number inflated by HOST leakage? A controlled test.

    python eval/eval_stutter_ssl_hostleak.py

THE CONFOUND THIS REMOVES
-------------------------
Comparing the episode-disjoint TEST number (a mixture of nine shows) against a
holdout-show number (one show) compares two different things at once: whether
the host was in training, AND how hard that particular show is. Shows differ a
lot -- so the naive comparison cannot tell "our estimate is inflated" apart
from "that show is hard".

So hold the test clips FIXED. For each show S with a holdout checkpoint:

  * the episode-disjoint model (trained on OTHER episodes of S, so it has heard
    S's recurring host) is scored on the clips of S inside the episode-disjoint
    TEST split;
  * the holdout-S model (trained with every clip of S removed, so it has never
    heard that host) is scored on THOSE SAME CLIPS.

Same audio, same labels, same architecture, same hyper-parameters. The only
difference is host exposure, so the gap IS the host-leakage effect.

Neither model trained on these clips: they are test clips for the first by
construction of the split, and the whole show is test for the second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.stutter_ssl import TYPES, load_checkpoint  # noqa: E402
from scripts.train_stutter import (  # noqa: E402
    average_precision, load_sources, make_splits)
from scripts.train_stutter_ssl import WaveDataset, run_eval  # noqa: E402


def score(model, rows, paths, idx, device, batch, workers, amp):
    loader = torch.utils.data.DataLoader(
        WaveDataset(paths, rows, idx, train=False), batch_size=batch,
        shuffle=False, num_workers=workers, pin_memory=(device == "cuda"))
    y, p, _ = run_eval(model, loader, device, amp)
    out = {t: {"ap": round(average_precision(y[:, k], p[:, k]), 4),
               "n_pos": int(y[:, k].sum())} for k, t in enumerate(TYPES)}
    y_any = (y.max(axis=1) > 0).astype("float32")
    out["ANY"] = {"ap": round(average_precision(y_any, p.max(axis=1)), 4),
                  "n_pos": int(y_any.sum())}
    out["n"] = len(idx)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode-ckpt", default="models/stutternet_ssl_v2.pt")
    ap.add_argument("--extra", default="data/sep28k_hf")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--out", default="eval/results/stutter_ssl_hostleak.json")
    args = ap.parse_args()
    device, amp = args.device, not args.no_amp

    rows, paths = load_sources([args.extra])
    sp = make_splits(rows, None)

    # Discover which holdout checkpoints actually exist.
    shows = sorted({r["show"] for r in rows})
    pairs = []
    for s in shows:
        p = Path("models/stutternet_ssl_v2_ho_%s.pt" % s.lower())
        if p.exists():
            pairs.append((s, p))
    if not pairs:
        print("no holdout checkpoints found")
        return 1

    epi = load_checkpoint(args.episode_ckpt, device)
    results = {}
    hdr = ("  %-18s %5s %-14s %s %7s"
           % ("show", "n", "host in train?", " ".join("%6s" % t[:6] for t in TYPES), "ANY"))
    print("CONTROLLED HOST-LEAKAGE TEST (identical clips, only host exposure differs)")
    print(hdr)
    for show, ckpt in pairs:
        idx = [i for i in sp["test"] if rows[i]["show"] == show]
        if len(idx) < 50:
            print("  %-18s skipped (only %d test clips)" % (show, len(idx)))
            continue
        seen = score(epi, rows, paths, idx, device, args.batch, args.workers, amp)
        ho = load_checkpoint(str(ckpt), device)
        unseen = score(ho, rows, paths, idx, device, args.batch, args.workers, amp)
        del ho
        if device == "cuda":
            torch.cuda.empty_cache()
        for lbl, m in (("SEEN (episode)", seen), ("UNSEEN (holdout)", unseen)):
            print("  %-18s %5d %-14s %s %7.3f"
                  % (show, m["n"], lbl,
                     " ".join("%6.3f" % m[t]["ap"] for t in TYPES), m["ANY"]["ap"]))
        delta = {t: round(unseen[t]["ap"] - seen[t]["ap"], 4)
                 for t in list(TYPES) + ["ANY"]}
        print("  %-18s %5s %-14s %s %+7.3f"
              % ("", "", "gap (unseen-seen)",
                 " ".join("%+6.3f" % delta[t] for t in TYPES), delta["ANY"]))
        results[show] = {"n": len(idx), "host_seen": seen,
                         "host_unseen": unseen, "gap": delta,
                         "holdout_ckpt": str(ckpt)}

    if results:
        print("\n  MEAN GAP across %d shows (negative = episode-disjoint was optimistic)"
              % len(results))
        for t in list(TYPES) + ["ANY"]:
            vals = [results[s]["gap"][t] for s in results]
            print("    %-14s %+.4f   (per show: %s)"
                  % (t, float(np.mean(vals)), ", ".join("%+.3f" % v for v in vals)))
        results["_mean_gap"] = {t: round(float(np.mean(
            [results[s]["gap"][t] for s in results if not s.startswith("_")])), 4)
            for t in list(TYPES) + ["ANY"]}
    results["_caveat"] = (
        "SEP-28k here is a partial reconstruction plus an unlicensed HF mirror; NOT "
        "comparable to published SEP-28k figures. Stuttering is not aphasia.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
