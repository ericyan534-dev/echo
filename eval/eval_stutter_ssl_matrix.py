"""Cross-evaluate StutterSSL checkpoints across the OLD and EXPANDED corpora.

    python eval/eval_stutter_ssl_matrix.py

WHY THIS EXISTS
---------------
When the corpus grows, `make_splits` does NOT keep episodes in the split they
were in before. It draws `rng.permutation(len(eps))` per show from one shared
RandomState(13), so adding the mirror's episodes changes both the permutation
length for a grown show and the position in the RNG stream of every show after
it. Measured here: 1380 of the 4411 clips in the NEW test split were in the
OLD model's TRAINING set.

So "old checkpoint on new test split" is NOT a fair number -- 31% of it is
memorisation. The only honest head-to-head is the CLEAN INTERSECTION: clips in
the new test split that neither model was trained or validated on. Both models
are scored on exactly those clips, and that is the row that answers "did more
data help".

Nothing here is fitted; every threshold travels with its checkpoint.
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
    average_precision, load_sources, make_splits, speaker_group)
from scripts.train_stutter_ssl import WaveDataset, run_eval  # noqa: E402


def clip_key(r: dict) -> tuple:
    return (r["show"], str(r["ep"]), str(r["clip"]))


def ep_key(r: dict) -> tuple:
    return (r["show"], speaker_group(r["show"], r["ep"]))


def score(model, rows, npy_paths, idx, device, batch, workers, amp):
    """AP per type plus ANY, on exactly the rows in `idx`."""
    loader = torch.utils.data.DataLoader(
        WaveDataset(npy_paths, rows, idx, train=False),
        batch_size=batch, shuffle=False, num_workers=workers,
        pin_memory=(device == "cuda"))
    y, p, _ = run_eval(model, loader, device, amp)
    out = {}
    for k, t in enumerate(TYPES):
        out[t] = {"ap": round(average_precision(y[:, k], p[:, k]), 4),
                  "n_pos": int(y[:, k].sum())}
    y_any = (y.max(axis=1) > 0).astype("float32")
    out["ANY"] = {"ap": round(average_precision(y_any, p.max(axis=1)), 4),
                  "n_pos": int(y_any.sum())}
    out["n"] = len(idx)
    return out, y, p


def paired_bootstrap(y_old, p_old, y_new, p_new, n_boot, seed=13):
    """Paired bootstrap over CLIPS of the AP difference (new - old).

    Paired: the same resampled clip indices are used for both models, so the
    clip-sampling noise that both share cancels and the interval is about the
    difference rather than about the two APs separately. Both models were
    scored on the same rows in the same order, so row i is the same clip.
    """
    assert y_old.shape == y_new.shape
    rng = np.random.RandomState(seed)
    n = len(y_old)
    cols = list(TYPES) + ["ANY"]
    draws = {c: [] for c in cols}
    for _ in range(n_boot):
        b = rng.randint(0, n, n)
        for k, t in enumerate(TYPES):
            yo = y_old[b, k]
            if yo.sum() == 0:
                continue
            draws[t].append(average_precision(yo, p_new[b, k])
                            - average_precision(yo, p_old[b, k]))
        ya = (y_old[b].max(axis=1) > 0).astype("float32")
        if ya.sum():
            draws["ANY"].append(average_precision(ya, p_new[b].max(axis=1))
                                - average_precision(ya, p_old[b].max(axis=1)))
    out = {}
    for c in cols:
        d = np.array(draws[c])
        out[c] = {"lo": round(float(np.percentile(d, 2.5)), 4),
                  "hi": round(float(np.percentile(d, 97.5)), 4),
                  "n_boot": len(d)}
    return out


def row_str(name, m):
    cells = " ".join("%6.3f" % m[t]["ap"] for t in TYPES)
    return "  %-34s %5d  %s  %6.3f" % (name, m["n"], cells, m["ANY"]["ap"])


def header():
    return ("  %-34s %5s  %s  %6s"
            % ("evaluation", "n", " ".join("%6s" % t[:6] for t in TYPES), "ANY"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="models/stutternet_ssl_v1repro.pt")
    ap.add_argument("--new", default="models/stutternet_ssl_v2.pt")
    ap.add_argument("--extra", default="data/sep28k_hf")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=2000,
                    help="paired-bootstrap resamples for the clean head-to-head")
    ap.add_argument("--clean-only", action="store_true",
                    help="skip the leaky/memorised rows already measured")
    ap.add_argument("--out", default="eval/results/stutter_ssl_corpus_matrix.json")
    args = ap.parse_args()
    device, amp = args.device, not args.no_amp

    old_rows, old_paths = load_sources([])
    new_rows, new_paths = load_sources([args.extra])
    old_sp = make_splits(old_rows, None)
    new_sp = make_splits(new_rows, None)

    old_train_clips = {clip_key(old_rows[i]) for i in old_sp["train"]}
    old_val_clips = {clip_key(old_rows[i]) for i in old_sp["val"]}
    old_seen = old_train_clips | old_val_clips

    # Clips in the NEW test split that the OLD model never trained or
    # validated on. The NEW model has by construction not trained on any of
    # the new test split, so this subset is clean for BOTH.
    clean = [i for i in new_sp["test"] if clip_key(new_rows[i]) not in old_seen]
    dirty = [i for i in new_sp["test"] if clip_key(new_rows[i]) in old_train_clips]

    print("SPLIT DRIFT (why old-on-new is not a fair number)")
    print("  old corpus %d clips  -> train %d / val %d / test %d"
          % (len(old_rows), len(old_sp["train"]), len(old_sp["val"]), len(old_sp["test"])))
    print("  new corpus %d clips  -> train %d / val %d / test %d"
          % (len(new_rows), len(new_sp["train"]), len(new_sp["val"]), len(new_sp["test"])))
    print("  new TEST clips that were in old TRAIN : %d (%.1f%%)"
          % (len(dirty), 100.0 * len(dirty) / len(new_sp["test"])))
    print("  new TEST clips clean for BOTH models  : %d" % len(clean))

    # --clean-only re-runs just the head-to-head rows and merges them into an
    # existing results file, so adding the bootstrap does not pay for the
    # leaky/memorised rows a second time. They are deterministic.
    outp = Path(args.out)
    results = (json.loads(outp.read_text(encoding="utf-8"))
               if args.clean_only and outp.exists() else {})
    print("\n" + header())

    old_model = load_checkpoint(args.old, device)
    if not args.clean_only:
        results["old_on_old"], _y, _p = score(
            old_model, old_rows, old_paths, old_sp["test"],
            device, args.batch, args.workers, amp)
        print(row_str("old ckpt / OLD test (n=3209)", results["old_on_old"]))
        results["old_on_new_all"], _y, _p = score(
            old_model, new_rows, new_paths, new_sp["test"],
            device, args.batch, args.workers, amp)
        print(row_str("old ckpt / NEW test  [LEAKY]", results["old_on_new_all"]))
    results["old_on_new_clean"], y_clean_old, p_clean_old = score(
        old_model, new_rows, new_paths, clean, device, args.batch, args.workers, amp)
    print(row_str("old ckpt / NEW test clean", results["old_on_new_clean"]))
    if dirty and not args.clean_only:
        results["old_on_new_dirty"], _y, _p = score(old_model, new_rows, new_paths, dirty,
                                            device, args.batch, args.workers, amp)
        print(row_str("old ckpt / NEW test memorised", results["old_on_new_dirty"]))
    del old_model
    if device == "cuda":
        torch.cuda.empty_cache()

    new_model = load_checkpoint(args.new, device)
    if not args.clean_only:
        results["new_on_new_all"], _y, _p = score(
            new_model, new_rows, new_paths, new_sp["test"],
            device, args.batch, args.workers, amp)
        print(row_str("new ckpt / NEW test", results["new_on_new_all"]))
    results["new_on_new_clean"], y_clean_new, p_clean_new = score(
        new_model, new_rows, new_paths, clean, device, args.batch, args.workers, amp)
    print(row_str("new ckpt / NEW test clean", results["new_on_new_clean"]))

    print("\n  HEAD TO HEAD on the %d clips clean for both (new minus old):" % len(clean))
    d = {t: round(results["new_on_new_clean"][t]["ap"]
                  - results["old_on_new_clean"][t]["ap"], 4)
         for t in list(TYPES) + ["ANY"]}
    print("    " + "  ".join("%s %+.3f" % (t[:6], v) for t, v in d.items()))
    results["delta_clean"] = d
    if args.bootstrap:
        ci = paired_bootstrap(y_clean_old, p_clean_old, y_clean_new, p_clean_new,
                              args.bootstrap)
        results["delta_clean_ci95"] = ci
        print("\n  PAIRED BOOTSTRAP 95%% CI on that difference (%d resamples)"
              % args.bootstrap)
        for t in list(TYPES) + ["ANY"]:
            sig = "" if ci[t]["lo"] <= 0 <= ci[t]["hi"] else "  *"
            print("    %-14s %+.4f  [%+.4f, %+.4f]%s"
                  % (t, d[t], ci[t]["lo"], ci[t]["hi"], sig))
        print("    (* = interval excludes zero; no star = consistent with no change)")

    results["split_drift"] = {
        "old_n": len(old_rows), "new_n": len(new_rows),
        "new_test_n": len(new_sp["test"]),
        "new_test_in_old_train": len(dirty),
        "clean_intersection_n": len(clean),
    }
    results["checkpoints"] = {"old": args.old, "new": args.new}
    results["caveat"] = (
        "SEP-28k here is a partial reconstruction plus an unlicensed HF mirror; "
        "NOT comparable to published SEP-28k figures. Stuttering is not aphasia.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
