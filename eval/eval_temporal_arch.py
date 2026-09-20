"""Head-to-head: BiLSTM temporal head vs the per-frame conv head, both on a
FROZEN WavLM backbone.

This isolates the ONE thing that changed -- the head architecture -- by pairing
StutterTemporal against models/stutternet_ssl_frozen.pt (same frozen backbone,
same target_fpr=0.02 operating point, same splits). It reuses the clip
selection and methodology of eval/eval_recall_v2.py, but compares at the
OPERATING-POINT level (peak frame prob per clip vs the checkpoint's own per-type
frame threshold) rather than through AcousticStream, because the stream's
backend='ssl' path is hard-wired to StutterSSL and this eval must not modify it.
The VAD / refractory / min-voiced machinery is identical for both heads and
orthogonal to the architecture question, so removing it makes the comparison
cleaner, not weaker.

Three numbers per model, each at the checkpoint's own frame operating point
(so the clean-fire budget is held equal across models by construction):

  1. OBVIOUS-CLIP RECALL (headline). Unambiguous SEP-28k clips (3/3 agreement,
     single type, clean audio). Fraction whose peak frame prob for the target
     type clears that type's frame threshold. Especially Block.
  2. AGGREGATE per-type recall / clean-fire-rate on the episode-disjoint TEST
     split (identical to eval_recall_v2.aggregate_recall).
  3. FLUENT CLEAN FIRING. Real fluent clips (NoStutteredWords, clean); fraction
     that fire on ANY type -- the "does it nag" budget.

Plus per-window GPU inference latency (batch 1, 3 s window) for both heads.

    python eval/eval_temporal_arch.py --device cuda \
        --temporal models/stutternet_temporal_v1.pt \
        --baseline models/stutternet_ssl_frozen.pt \
        --out eval/results/temporal_arch_eval.json

ASCII only. Never overwrites weights. Refuses to overwrite --out if it exists.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic.stutter import TYPES  # noqa: E402
from backend.acoustic.stutter_ssl import load_checkpoint as ssl_load  # noqa: E402
from backend.acoustic.stutter_temporal import load_checkpoint as temporal_load  # noqa: E402
from scripts.train_stutter import load_sources, make_splits  # noqa: E402
from eval.eval_recall_v2 import (  # noqa: E402
    COL_TO_KIND, NPY, STUTTER_COLS, load_pool, pcm_of, select_fluent,
    select_obvious)

SR = 16_000


def load_model(kind, ckpt, device):
    if kind == "temporal":
        return temporal_load(ckpt, device)
    return ssl_load(ckpt, device)


def frame_thresholds_of(ckpt):
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    return state.get("frame_thresholds", {})


def peak_probs(model, pcms, device, batch=32):
    """List of (T,)-waveforms -> (n, n_types) peak serving-prob per clip.

    Uses model.forward (tempered serving logits) then sigmoid, exactly the
    quantity the live stream compares against its frame threshold.
    """
    # pad to a common length per batch (clips are all ~3 s here)
    out = np.zeros((len(pcms), len(TYPES)), dtype="float32")
    for lo in range(0, len(pcms), batch):
        chunk = pcms[lo:lo + batch]
        L = max(w.numel() for w in chunk)
        xb = torch.zeros(len(chunk), L, dtype=torch.float32)
        for i, w in enumerate(chunk):
            xb[i, :w.numel()] = w
        xb = xb.to(device)
        with torch.no_grad():
            fp = torch.sigmoid(model(xb).float())      # (B, n_types, T)
        out[lo:lo + len(chunk)] = fp.max(dim=2).values.cpu().numpy()
    return out


# --- 1. obvious-clip recall at operating point -----------------------------
def obvious_recall(model, fthr, obvious, arr, device):
    res = {}
    for target, clips in obvious.items():
        thr = fthr.get(target, 0.8808)
        if not clips:
            res[target] = {"recall": None, "n": 0, "thr": round(float(thr), 4)}
            continue
        pcms = [pcm_of(arr, row) for (_, row) in clips]
        peaks = peak_probs(model, pcms, device)
        k = TYPES.index(target)
        fired = int((peaks[:, k] >= thr).sum())
        res[target] = {"recall": round(fired / len(clips), 4), "fired": fired,
                       "n": len(clips), "thr": round(float(thr), 4)}
    return res


# --- 2. aggregate recall on TEST (mirrors eval_recall_v2.aggregate_recall) --
def aggregate_recall(model, fthr, device):
    rows, npy_paths = load_sources([])
    splits = make_splits(rows, holdout_show=None)
    idx = splits["test"]
    arr = np.load(npy_paths[0], mmap_mode="r")
    pcms = [torch.from_numpy(arr[rows[j]["row"]].astype("float32") / 32768.0)
            for j in idx]
    peaks = peak_probs(model, pcms, device)
    labels = np.array([rows[j]["labels"] for j in idx], dtype="float32")
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


# --- 3. fluent clean firing ------------------------------------------------
def fluent_firing(model, fthr, fluent, arr, device):
    pcms = [pcm_of(arr, row) for (_, row) in fluent]
    peaks = peak_probs(model, pcms, device)
    total_s = sum(w.numel() for w in pcms) / SR
    thr = np.array([fthr.get(t, 0.8808) for t in TYPES], dtype="float32")
    fires = peaks >= thr[None, :]
    any_fire = fires.any(axis=1)
    by_type = {t: int(fires[:, k].sum()) for k, t in enumerate(TYPES)}
    mins = total_s / 60.0
    return {
        "n_clips": len(fluent),
        "duration_s": round(total_s, 1),
        "clips_firing_any": int(any_fire.sum()),
        "clip_fire_rate": round(float(any_fire.mean()), 4),
        "events_any": int(fires.any(axis=1).sum()),
        "per_min_clip_fires": round(float(any_fire.sum()) / mins, 3) if mins else None,
        "by_type_fires": by_type,
    }


# --- 4. per-window GPU latency ---------------------------------------------
def latency(model, device, n=30):
    if not str(device).startswith("cuda"):
        return {"device": str(device), "note": "cuda not used"}
    x = torch.randn(1, 3 * SR, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n):
            model(x)
        torch.cuda.synchronize()
    return {"cuda_batch1_ms_per_3s_window": round((time.time() - t0) / n * 1000, 2)}


def run_model(tag, kind, ckpt, device, obvious, fluent, arr):
    print("\n==== %s (%s) ====" % (tag, Path(ckpt).name))
    model = load_model(kind, ckpt, device)
    fthr = frame_thresholds_of(ckpt)
    obv = obvious_recall(model, fthr, obvious, arr, device)
    print("  OBVIOUS-CLIP RECALL (fired/n @ frame thr):")
    for t in STUTTER_COLS:
        v = obv[t]
        print("    %-12s recall %s  (%s/%s, thr %s)"
              % (t, v["recall"], v.get("fired", "-"), v["n"], v["thr"]))
    agg = aggregate_recall(model, fthr, device)
    print("  AGGREGATE per-type recall @ operating point (TEST):")
    for t in TYPES + ["ANY"]:
        v = agg[t]
        print("    %-12s recall %s  clean_fire %s (thr %s)"
              % (t, v["recall"], v.get("clean_fire_rate"), v.get("frame_thresh", "-")))
    ff = fluent_firing(model, fthr, fluent, arr, device)
    print("  FLUENT CLEAN FIRING: %d/%d clips fire (rate %s), %s clip-fires/min"
          % (ff["clips_firing_any"], ff["n_clips"], ff["clip_fire_rate"],
             ff["per_min_clip_fires"]))
    lat = latency(model, device)
    print("  LATENCY: %s" % lat)
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return {"ckpt": Path(ckpt).name, "obvious_recall": obv,
            "aggregate_recall": agg, "fluent_firing": ff, "latency": lat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--temporal", default=str(ROOT / "models" / "stutternet_temporal_v1.pt"))
    ap.add_argument("--baseline", default=str(ROOT / "models" / "stutternet_ssl_frozen.pt"))
    ap.add_argument("--per-type", type=int, default=40)
    ap.add_argument("--fluent", type=int, default=120)
    ap.add_argument("--out", default=str(ROOT / "eval" / "results" / "temporal_arch_eval.json"))
    args = ap.parse_args()

    if args.out and Path(args.out).exists():
        raise SystemExit("refusing to overwrite existing results %s" % args.out)

    key_to_row, counts = load_pool()
    obvious = select_obvious(key_to_row, counts, args.per_type)
    fluent = select_fluent(key_to_row, counts, args.fluent)
    arr = np.load(NPY, mmap_mode="r")
    for t, c in obvious.items():
        print("  obvious %-12s %d clips" % (t, len(c)))
    print("  fluent clips: %d" % len(fluent))

    out = {"baseline": None, "temporal": None,
           "notes": ("Both at each checkpoint's own target_fpr=0.02 frame "
                     "operating point; equal clean-fire budget by construction. "
                     "Baseline = frozen WavLM + per-frame conv head; temporal = "
                     "frozen WavLM + BiLSTM head. Operating-point (peak-prob vs "
                     "frame threshold) comparison, not through AcousticStream.")}
    if Path(args.baseline).exists():
        out["baseline"] = run_model("BASELINE conv head", "ssl", args.baseline,
                                    args.device, obvious, fluent, arr)
    else:
        print("  baseline %s not found" % args.baseline)
    if Path(args.temporal).exists():
        out["temporal"] = run_model("TEMPORAL BiLSTM head", "temporal", args.temporal,
                                    args.device, obvious, fluent, arr)
    else:
        print("  temporal %s not found -- train it first" % args.temporal)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
