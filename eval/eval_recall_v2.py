"""Before/after measurement for the recall-focused acoustic operating point.

Three numbers, each computed for a BEFORE checkpoint/settings and an AFTER one,
so a change is never claimed without its cost:

  1. OBVIOUS-CLIP FIRING (the headline). Unambiguous SEP-28k clips (3/3
     annotator agreement, one type, clean audio) fed through the REAL
     AcousticStream. Fraction that fire, per type, in the mid-utterance
     ("leadin") mode that isolates the model+threshold from the min_voiced gate.

  2. AGGREGATE per-type recall / clean-fire-rate at the checkpoint's own
     per-frame operating point, on the held-out episode-disjoint TEST split
     (the clip-level analogue of the live decision: peak frame prob >= frame
     threshold). This is the honest recall/precision trade the operating point
     buys.

  3. FLUENT FALSE FIRES PER MINUTE. Real fluent SEP-28k clips (NoStutteredWords,
     clean) concatenated into a continuous stream and fed through the REAL
     AcousticStream; events counted and divided by minutes. This is the "does
     it nag" budget.

Run (after the recall checkpoint exists):
  python eval/eval_recall_v2.py --device cuda \
      --before models/stutternet_ssl_v2.pt --before-min-voiced 800 \
      --after  models/stutternet_recall_v2.pt --after-min-voiced 200 \
      --out eval/results/recall_v2_eval.json

ASCII only. Never overwrites weights. Writes JSON if --out given.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.stream import AcousticStream, SR  # noqa: E402
from backend.acoustic.stutter_ssl import (  # noqa: E402
    TYPES, linear_softmax_pool, load_checkpoint as ssl_load)
from scripts.train_stutter import load_sources, make_splits  # noqa: E402

DATA = ROOT / "data" / "sep28k"
CSV = DATA / "SEP-28k_labels.csv"
NPY = DATA / "clips_16k.npy"
IDX = DATA / "clips_index.json"

STUTTER_COLS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
COL_TO_KIND = {"Block": "block", "Prolongation": "prolongation",
               "SoundRep": "sound_rep", "WordRep": "word_rep",
               "Interjection": "filler"}
FRAME_BYTES = 640


def _key(show, ep, clip):
    return (str(show).strip(), str(ep).strip(), str(clip).strip())


def load_pool():
    idx = json.loads(IDX.read_text(encoding="utf-8"))
    key_to_row = {_key(r["show"], r["ep"], r["clip"]): r["row"] for r in idx["rows"]}
    counts = {}
    with open(CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            counts[_key(r["Show"], r["EpId"], r["ClipId"])] = r
    return key_to_row, counts


def select_obvious(key_to_row, counts, per_type, agree=3):
    out = {}
    for target in STUTTER_COLS:
        picks = []
        for k, row in key_to_row.items():
            c = counts.get(k)
            if c is None or int(c[target]) < agree:
                continue
            if any(int(c[o]) > 0 for o in STUTTER_COLS if o != target):
                continue
            if int(c["PoorAudioQuality"]) or int(c["DifficultToUnderstand"]):
                continue
            if int(c["Unsure"]) or int(c["Music"]) or int(c["NoSpeech"]):
                continue
            picks.append((k, row))
            if len(picks) >= per_type:
                break
        out[target] = picks
    return out


def select_fluent(key_to_row, counts, n):
    out = []
    for k, row in key_to_row.items():
        c = counts.get(k)
        if c is None:
            continue
        if any(int(c[o]) > 0 for o in STUTTER_COLS):
            continue
        if int(c["NoStutteredWords"]) < 2:
            continue
        if int(c["PoorAudioQuality"]) or int(c["NoSpeech"]) or int(c["Music"]):
            continue
        out.append((k, row))
        if len(out) >= n:
            break
    return out


def pcm_of(arr, row):
    return torch.from_numpy(arr[row].astype("float32") / 32768.0)


def f2b(x):
    return (x.clamp(-1, 1) * 32768.0).to(torch.int16).numpy().tobytes()


def make_stream(ckpt, device, min_voiced_ms, refractory_ms, scale, backend="ssl",
                context_lag_ms=0):
    return AcousticStream(
        model_path=str(ROOT / "models" / "fillernet.pt"),
        stutter_model=str(ckpt), stutter_backend=backend, device=device,
        min_voiced_ms=min_voiced_ms, refractory_ms=refractory_ms,
        stutter_scale=scale, context_lag_ms=context_lag_ms)


# --- 1. obvious-clip firing ------------------------------------------------
def obvious_firing(ckpt, device, min_voiced_ms, refractory_ms, scale,
                   obvious, lead, arr, context_lag_ms=0):
    res = {}
    for target, clips in obvious.items():
        kind = COL_TO_KIND[target]
        for mode in ("bare", "leadin"):
            fired = 0
            for (k, row) in clips:
                wav = pcm_of(arr, row)
                w = wav if mode == "bare" else torch.cat([lead, wav])
                st = make_stream(ckpt, device, min_voiced_ms, refractory_ms, scale,
                                 context_lag_ms=context_lag_ms)
                pcm = f2b(w)
                hit = False
                for off in range(0, len(pcm), FRAME_BYTES):
                    for e in st.feed(pcm[off:off + FRAME_BYTES]):
                        if e.kind == kind:
                            hit = True
                fired += int(hit)
            res["%s/%s" % (target, mode)] = {"fired": fired, "n": len(clips)}
    return res


# --- 2. aggregate recall at the operating point on TEST --------------------
def aggregate_recall(ckpt, device):
    """Per-type recall and clean-fire-rate at the checkpoint's frame operating
    point, on the episode-disjoint TEST split. Uses peak frame prob per clip vs
    the stored frame_threshold -- the clip-level analogue of the live decision.
    """
    model = ssl_load(ckpt, device)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    fthr = state.get("frame_thresholds", {})
    rows, npy_paths = load_sources([])
    splits = make_splits(rows, holdout_show=None)
    idx = splits["test"]
    arr = np.load(npy_paths[0], mmap_mode="r")
    peaks = np.zeros((len(idx), len(TYPES)), dtype="float32")
    labels = np.zeros((len(idx), len(TYPES)), dtype="float32")
    B = 64
    for lo in range(0, len(idx), B):
        chunk = idx[lo:lo + B]
        pcm = torch.from_numpy(
            np.stack([arr[rows[j]["row"]] for j in chunk]).astype("float32") / 32768.0
        ).to(device)
        with torch.no_grad():
            fp = torch.sigmoid(model(pcm))          # tempered serving probs
        peaks[lo:lo + len(chunk)] = fp.max(dim=2).values.cpu().numpy()
        labels[lo:lo + len(chunk)] = np.array(
            [rows[j]["labels"] for j in chunk], dtype="float32")
    clean = labels.max(axis=1) == 0
    out = {}
    for k, t in enumerate(TYPES):
        thr = fthr.get(t, 0.8808)
        pos = peaks[labels[:, k] > 0, k]
        neg = peaks[clean, k]
        out[t] = {
            "frame_thresh": round(float(thr), 4),
            "recall": round(float((pos >= thr).mean()), 4) if len(pos) else None,
            "n_pos": int(len(pos)),
            "clean_fire_rate": round(float((neg >= thr).mean()), 4) if len(neg) else None,
        }
    # ANY = clip fires on at least one type
    fires_any = np.zeros(len(idx), dtype=bool)
    for k, t in enumerate(TYPES):
        fires_any |= peaks[:, k] >= fthr.get(t, 0.8808)
    ypos = labels.max(axis=1) > 0
    out["ANY"] = {
        "recall": round(float(fires_any[ypos].mean()), 4),
        "n_pos": int(ypos.sum()),
        "clean_fire_rate": round(float(fires_any[clean].mean()), 4),
    }
    return out


# --- 3. fluent false fires per minute --------------------------------------
def fluent_false_fires(ckpt, device, min_voiced_ms, refractory_ms, scale,
                       fluent, arr, context_lag_ms=0):
    st = make_stream(ckpt, device, min_voiced_ms, refractory_ms, scale,
                     context_lag_ms=context_lag_ms)
    total_s = 0.0
    by_kind = {}
    total = 0
    for (k, row) in fluent:
        wav = pcm_of(arr, row)
        total_s += wav.numel() / SR
        pcm = f2b(wav)
        for off in range(0, len(pcm), FRAME_BYTES):
            for e in st.feed(pcm[off:off + FRAME_BYTES]):
                by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
                total += 1
    mins = total_s / 60.0
    return {
        "duration_s": round(total_s, 1),
        "events": total,
        "per_min": round(total / mins, 3) if mins else None,
        "by_kind": by_kind,
        "by_kind_per_min": {k: round(v / mins, 3) for k, v in by_kind.items()} if mins else {},
    }


def run_condition(tag, ckpt, device, min_voiced_ms, refractory_ms, scale,
                  obvious, fluent, lead, arr, context_lag_ms=0):
    print("\n==== %s ====" % tag)
    print("  ckpt=%s min_voiced_ms=%d refractory_ms=%d scale=%.2f context_lag_ms=%d"
          % (Path(ckpt).name, min_voiced_ms, refractory_ms, scale, context_lag_ms))
    fire = obvious_firing(ckpt, device, min_voiced_ms, refractory_ms, scale,
                          obvious, lead, arr, context_lag_ms=context_lag_ms)
    print("  OBVIOUS-CLIP FIRING (fired/n):")
    for target in STUTTER_COLS:
        b = fire["%s/bare" % target]
        l = fire["%s/leadin" % target]
        print("    %-12s bare %d/%d   leadin %d/%d"
              % (target, b["fired"], b["n"], l["fired"], l["n"]))
    agg = aggregate_recall(ckpt, device)
    print("  AGGREGATE per-type recall @ operating point (TEST):")
    for t in TYPES + ["ANY"]:
        v = agg[t]
        print("    %-12s recall %s  clean_fire %s (thr %s)"
              % (t, v["recall"], v.get("clean_fire_rate"), v.get("frame_thresh", "-")))
    ff = fluent_false_fires(ckpt, device, min_voiced_ms, refractory_ms, scale,
                            fluent, arr, context_lag_ms=context_lag_ms)
    print("  FLUENT FALSE FIRES: %d events in %.1fs = %s/min  %s"
          % (ff["events"], ff["duration_s"], ff["per_min"], ff["by_kind_per_min"]))
    return {"config": {"ckpt": Path(ckpt).name, "min_voiced_ms": min_voiced_ms,
                       "refractory_ms": refractory_ms, "scale": scale,
                       "context_lag_ms": context_lag_ms},
            "obvious_firing": fire, "aggregate_recall": agg,
            "fluent_false_fires": ff}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--before", default=str(ROOT / "models" / "stutternet_ssl_v2.pt"))
    ap.add_argument("--before-min-voiced", type=int, default=800)
    ap.add_argument("--before-refractory", type=int, default=1200)
    ap.add_argument("--before-scale", type=float, default=1.0)
    ap.add_argument("--before-lag", type=int, default=0)
    ap.add_argument("--after", default=str(ROOT / "models" / "stutternet_recall_v2.pt"))
    ap.add_argument("--after-min-voiced", type=int, default=200)
    ap.add_argument("--after-refractory", type=int, default=1000)
    ap.add_argument("--after-scale", type=float, default=1.0)
    ap.add_argument("--after-lag", type=int, default=400)
    ap.add_argument("--per-type", type=int, default=12)
    ap.add_argument("--fluent", type=int, default=40)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key_to_row, counts = load_pool()
    obvious = select_obvious(key_to_row, counts, args.per_type)
    fluent = select_fluent(key_to_row, counts, args.fluent)
    arr = np.load(NPY, mmap_mode="r")
    lead = pcm_of(arr, fluent[0][1])[-int(1.5 * SR):]
    for t, c in obvious.items():
        print("  obvious %-12s %d clips" % (t, len(c)))
    print("  fluent stream clips: %d" % len(fluent))

    out = {"before": None, "after": None}
    out["before"] = run_condition(
        "BEFORE", args.before, args.device, args.before_min_voiced,
        args.before_refractory, args.before_scale, obvious, fluent, lead, arr,
        context_lag_ms=args.before_lag)
    if Path(args.after).exists():
        out["after"] = run_condition(
            "AFTER", args.after, args.device, args.after_min_voiced,
            args.after_refractory, args.after_scale, obvious, fluent, lead, arr,
            context_lag_ms=args.after_lag)
    else:
        print("\n  AFTER checkpoint %s not found -- run the recalibration first."
              % args.after)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
