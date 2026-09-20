"""Train StutterSSL -- the StutterNet task on a pretrained speech encoder.

    python scripts/train_stutter_ssl.py                        # frozen WavLM Base+
    python scripts/train_stutter_ssl.py --unfreeze 4 --batch 16 --enc-lr 1e-5 \
        --out models/stutternet_ssl_ft.pt
    python scripts/train_stutter_ssl.py --encoder facebook/wav2vec2-base
    python scripts/train_stutter_ssl.py --bench-only          # inference cost only

WHY THIS SCRIPT EXISTS SEPARATELY
---------------------------------
It is a controlled swap, so everything that is not the representation is
imported rather than rewritten: `load_index` and `make_splits` come straight
from scripts/train_stutter.py, so the episode-disjoint-stratified-by-show
assignment is IDENTICAL down to the RNG seed, and `average_precision`,
`best_f1` and `report` come from there too, so the metrics table is the same
table. Reimplementing any of those would make the comparison a comparison of
two experiments instead of two models.

The only differences from train_stutter.py are the ones being tested:
  * the dataset yields raw 16 kHz waveform instead of log-mel;
  * SpecAugment is gone (there is no spectrogram to mask) and is replaced by
    waveform-domain augmentation;
  * the optimiser has two parameter groups, because a pretrained encoder that
    is being fine-tuned needs a learning rate ~100x below the head's.

DISCIPLINE
----------
Nothing is chosen on TEST. Epoch selection, thresholds and every
hyper-parameter come off VAL; TEST is read exactly once per configuration, at
the end, and is reported whatever it says.

WHAT IT MEASURED (episode-disjoint TEST, n=3209, AP)
----------------------------------------------------
                 StutterNet   frozen   unfreeze-4
    Block             0.256    0.347    0.385
    Prolongation      0.485    0.560    0.540
    SoundRep          0.306    0.520    0.572
    WordRep           0.254    0.721    0.826
    Interjection      0.734    0.860    0.857
    ANY               0.786    0.884    0.894

Every gain has a paired-bootstrap 95% CI clear of zero. Block, the head that
matters most and previously carried a downstream weight of exactly 0.0, gains
+0.129 AP [+0.089, +0.169]. Selection between the two SSL configurations was
made on VAL mAP (0.596 unfrozen-top-4 vs 0.582 frozen), not on TEST.

CAUTION: the unfreeze-4 column above cannot be reproduced from disk. The
checkpoint it describes (models/stutternet_ssl.pt) was overwritten by a
`--limit 200` smoke run -- the file now embeds n_train=200, epochs=1,
n_unfreeze=0 -- and `*.pt` is gitignored, so nothing could be restored. A
clean rerun of the same configuration (stutternet_ssl_v1repro.pt) gives Block
0.371 / ANY 0.893, so 0.385 was within run-to-run noise, but it is a
reconstruction and not the original.

THE EXPANDED CORPUS (--extra data/sep28k_hf): 20,124 -> 30,962 clips,
5 -> 9 speaker pools. Two things it is easy to get wrong:

1. make_splits does NOT preserve split membership when the corpus grows. It
   draws rng.permutation(len(eps)) per show from one shared RandomState(13),
   so a grown show reshuffles and every show after it in the stream shifts
   too. Measured: 1380 of the 4411 new TEST clips were in the OLD model's
   TRAIN. Scoring the old checkpoint on the new test split therefore reads
   ~31% memorisation (0.410 Block on that subset vs 0.304 on the clean rest).
   Old and new numbers are only comparable on the intersection that neither
   model trained on -- see eval/eval_stutter_ssl_matrix.py.

2. On that clean intersection (n=2833) more data did NOT help the head that
   matters. Paired bootstrap, new minus old:
       Block        +0.001  [-0.027, +0.028]     no effect
       Prolongation +0.010  [-0.022, +0.042]     no effect
       SoundRep     +0.012  [-0.019, +0.044]     no effect
       WordRep      +0.038  [-0.018, +0.095]     no effect
       Interjection +0.030  [+0.015, +0.046]     real
       ANY          +0.011  [+0.001, +0.022]     real
   10,838 extra clips and 4 extra speaker pools bought Interjection and a
   sliver of ANY, and nothing at all on Block.

IS THE EPISODE-DISJOINT NUMBER INFLATED BY HOST LEAKAGE? No -- measured, not
assumed. eval/eval_stutter_ssl_hostleak.py scores the episode-disjoint model
and a holdout-show model on the SAME clips of that show, so only host exposure
differs. Mean gap over three shows (HeStutters, StutterTalk, StutteringIsCool):
Block +0.003, ANY -0.004; no per-show gap exceeds 0.08. What DOES move the
number is which show you test on: the same model scores Block 0.430 on
StutterTalk and 0.222 on StutteringIsCool. Show difficulty dominates host
identity by roughly an order of magnitude, and raw AP across shows is further
confounded by prevalence, which ranges 0.09-0.15 for Block and 0.35-0.65 for
ANY. Compare lift over chance, not AP, when comparing shows.

AND WHAT IT COSTS
-----------------
156 ms per 3 s clip on CPU with 24 threads, 963 ms on one thread, against
StutterNet's 11 ms / 46 ms. The live acoustic channel classifies one window
per 125 ms hop on CPU while the GPU serves the ASR, so this model does not fit
there -- see the note in backend/acoustic/stutter_ssl.py. The accuracy result
stands; the deployment claim does not follow from it.

PROVENANCE CAVEAT
-----------------
This SEP-28k is reconstructed from Apple's labels and covers ~74% of the clips
across 5 of 8 shows (link rot took the rest). With --extra it is extended by a
HuggingFace mirror which DECLARES NO LICENCE and which restores the
link-rotted shows plus FluencyBank. Numbers here are NOT comparable to
published SEP-28k results -- it is a different corpus. They ARE comparable to
models/stutternet_metrics.json, which is the only comparison this script
claims to make.

And none of it is aphasia. SEP-28k is stuttered speech: a motor-speech
disorder in which the word is known and will not come out. Echo's users have a
language disorder in which the word is not retrievable. The surface evidence
overlaps, which is why this transfers at all; nothing here is an aphasia
measurement.
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

from backend.acoustic.stutter_ssl import (  # noqa: E402
    DEFAULT_ENCODER, FRAME_MS, TYPES, StutterSSL, linear_softmax_pool,
    load_checkpoint)
# Imported, never reimplemented: identical splits and identical metrics are the
# entire basis of the comparison with StutterNet.
from scripts.train_stutter import (  # noqa: E402
    CLIPS_NPY, MODELS, average_precision, load_sources, make_splits, report)

SR = 16_000
CKPT = MODELS / "stutternet_ssl.pt"


# --- data ----------------------------------------------------------------
class WaveDataset(torch.utils.data.Dataset):
    """Raw float PCM in [-1, 1]. No normalisation, deliberately.

    WavLM Base+ / wav2vec2-base have feat_extract_norm="group" and ship
    do_normalize=false: they were pretrained on unnormalised waveform and their
    first GroupNorm expects that scale. Applying the zero-mean/unit-variance
    normalisation that the LARGE variants want is a silent accuracy loss here,
    not an error, which is exactly the kind of bug that survives a run.
    """

    def __init__(self, path, rows, idx, train: bool) -> None:
        # A list of .npy paths, one per corpus. Each row carries `src` (which
        # array) and `row` (offset within it), so the HF-mirror recovery of the
        # link-rotted shows can train alongside the locally cut clips.
        self.paths = list(path) if isinstance(path, (list, tuple)) else [path]
        self.rows = rows
        self.idx = idx
        self.train = train
        self._arr = None      # opened lazily, per worker (see train_stutter.py)

    @property
    def arr(self):
        if self._arr is None:
            self._arr = [np.load(p, mmap_mode="r") for p in self.paths]
        return self._arr

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        r = self.rows[j]
        pcm = torch.from_numpy(
            self.arr[r.get("src", 0)][r["row"]].astype("float32") / 32768.0)
        y = torch.tensor(self.rows[j]["labels"], dtype=torch.float32)
        if self.train:
            # Gain jitter DOES train something here, unlike in the log-mel
            # model where per-example normalisation made it a no-op: the
            # encoder sees absolute waveform amplitude. The lav mic (DJI Mic
            # 2S) is not the podcast mic these recordings were made on, and
            # level is the first thing that differs.
            if np.random.rand() < 0.5:
                pcm = pcm * float(np.random.uniform(0.7, 1.4))
            if np.random.rand() < 0.5:
                pcm = pcm + torch.randn_like(pcm) * float(np.random.uniform(0, 0.005))
            pcm = pcm.clamp(-1.0, 1.0)
        return pcm, y


# --- eval ----------------------------------------------------------------
def run_eval(model, loader, device, amp: bool):
    """Returns (labels, clip_probs, peak_frame_LOGITS).

    Peaks are collected here rather than in a second pass because the per-frame
    operating point has to be calibrated on VAL and a second pass over the
    encoder costs as much as an epoch.

    Logits, not probabilities, and this is the whole point: the trained head's
    peak frame logits reach 28 on clips with no dysfluency in them at all, and
    sigmoid(28) is exactly 1.0 in float32. Calibrating in probability space
    produces a threshold of 1.0 that nothing can be above and everything
    saturated is equal to -- a broken operating point that looks like a number.
    """
    model.eval()
    ys, ps, peaks = [], [], []
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda")
    with torch.no_grad(), autocast:
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model.raw_frame_logits(xb).float()
            probs = torch.sigmoid(logits)
            ys.append(yb.numpy())
            ps.append(linear_softmax_pool(probs, dim=-1).cpu().numpy())
            peaks.append(logits.max(dim=2).values.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(peaks))


# Where the calibrated operating point is placed in logit space. 2.0 -> a
# frame threshold of sigmoid(2.0) = 0.881, comfortably inside float32 and
# leaving room on both sides for stream.py's `stutter_scale` multiplier.
TARGET_LOGIT = 2.0


def frame_calibration(peak_logits: np.ndarray, labels: np.ndarray,
                      target_fpr: float) -> dict:
    """Per-FRAME operating point by interruption budget, in logit space.

    Same procedure as eval/calibrate_stutter_frames.py -- for each type, take
    the highest frame score in each clip and put the threshold at the
    (1 - target_fpr) quantile of the clips containing NO dysfluency, so exactly
    that share of clean clips would fire. Repeated here rather than imported
    because that script is hard-wired to the log-mel checkpoint, and run on VAL
    rather than TEST because an operating point is a hyper-parameter.

    "Clean" means no dysfluency of ANY type. Using per-type negatives would
    count a Block clip as a negative for WordRep, and firing on a dysfluent
    clip is not the false alarm being budgeted.

    Returns per type: the fitted temperature, the probability threshold to use
    after dividing by it, and what that costs in recall.
    """
    clean = labels.max(axis=1) == 0
    out = {}
    for k, t in enumerate(TYPES):
        neg = peak_logits[clean, k].astype("float64")
        pos = peak_logits[labels[:, k] > 0, k].astype("float64")
        q = float(np.quantile(neg, 1.0 - target_fpr)) if len(neg) else TARGET_LOGIT
        # Never below 1.0: a temperature under 1 would sharpen an already
        # saturating head, and there is no reason to.
        temp = max(1.0, q / TARGET_LOGIT)
        thr = float(1.0 / (1.0 + np.exp(-q / temp)))
        out[t] = {
            "temperature": round(temp, 4),
            "threshold": round(thr, 6),
            "logit_threshold": round(q, 4),
            "n_clean": int(len(neg)), "n_pos": int(len(pos)),
            # Computed on the logits, which is identical to computing it on the
            # tempered probabilities -- dividing by a positive constant cannot
            # reorder them -- and does not lose the saturated ones to rounding.
            "recall_at_threshold": round(float((pos >= q).mean()), 4) if len(pos) else None,
            "clean_fire_rate": round(float((neg >= q).mean()), 4) if len(neg) else None,
            "clean_peak_logit_median": round(float(np.median(neg)), 2) if len(neg) else None,
            "pos_peak_logit_median": round(float(np.median(pos)), 2) if len(pos) else None,
        }
    return out


def apply_calibration(model, frame_cal: dict) -> None:
    model.frame_temperature.copy_(torch.tensor(
        [frame_cal[t]["temperature"] for t in TYPES], dtype=torch.float32))


def print_calibration(frame_cal: dict, target_fpr: float) -> None:
    print("\n  PER-FRAME OPERATING POINT (VAL, clean-clip fire budget %.3f)"
          % target_fpr)
    print("    %-14s %6s %10s %9s %8s %12s"
          % ("type", "temp", "logit thr", "prob thr", "recall", "clean fires"))
    for t, v in frame_cal.items():
        print("    %-14s %6.2f %10.2f %9.4f %8s %12s"
              % (t, v["temperature"], v["logit_threshold"], v["threshold"],
                 v["recall_at_threshold"], v["clean_fire_rate"]))


# --- inference cost ------------------------------------------------------
def bench(model, device_list=("cuda", "cpu"), n: int = 20) -> dict:
    """ms per 3 s clip, batch 1, plus the batched GPU figure.

    Batch 1 is the number that decides whether this can serve the live stream:
    the stream classifies one 3 s window every hop, it cannot batch across
    time it has not heard yet. The batched figure is only for offline eval.
    """
    out = {}
    was = next(model.parameters()).device
    x1 = torch.randn(1, 3 * SR)
    for dev in device_list:
        if dev == "cuda" and not torch.cuda.is_available():
            continue
        model.to(dev).eval()
        xb = x1.to(dev)
        with torch.no_grad():
            for _ in range(3):
                model(xb)
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(n):
                model(xb)
            if dev == "cuda":
                torch.cuda.synchronize()
        out["%s_batch1_ms" % dev] = round((time.time() - t0) / n * 1000, 2)
        if dev == "cuda":
            x32 = torch.randn(32, 3 * SR, device=dev)
            with torch.no_grad():
                model(x32)
                torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(5):
                    model(x32)
                torch.cuda.synchronize()
            out["cuda_batch32_ms_per_clip"] = round((time.time() - t0) / 5 / 32 * 1000, 2)
    # One thread is the honest shared-host figure: on the demo laptop the
    # acoustic channel does not get all 24 cores, it gets a slice next to the
    # VAD while the ASR is running.
    prev = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        model.to("cpu").eval()
        with torch.no_grad():
            model(x1)
            t0 = time.time()
            for _ in range(max(3, n // 4)):
                model(x1)
        out["cpu_1thread_batch1_ms"] = round(
            (time.time() - t0) / max(3, n // 4) * 1000, 2)
    finally:
        torch.set_num_threads(prev)
    out["cpu_threads_default"] = prev
    model.to(was)
    return out


def bench_baseline() -> dict:
    """Same measurement on the shipped log-mel StutterNet, for the ratio.

    An absolute millisecond count means nothing without the thing it replaces.
    """
    from backend.acoustic.features import logmel
    from backend.acoustic.stutter import load_checkpoint

    path = MODELS / "stutternet.pt"
    if not path.exists():
        return {}

    class _Wrapped(nn.Module):
        """Wraps log-mel + CNN so bench() measures the same end-to-end unit
        (waveform in, frames out) for both models."""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, wav):
            feats = torch.stack([logmel(w) for w in wav]).unsqueeze(1)
            return self.m(feats)

    # CPU only. backend/acoustic/features.py holds its MelSpectrogram in a
    # module-level CPU transform, so .to("cuda") does not move it -- and CPU is
    # the only device this comparison is about: StutterNet exists because the
    # acoustic channel has to run there while the GPU serves the ASR.
    return bench(_Wrapped(load_checkpoint(path, "cpu")), device_list=("cpu",))


# --- main ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default=DEFAULT_ENCODER)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    ap.add_argument("--enc-lr", type=float, default=1e-5,
                    help="learning rate for unfrozen encoder layers")
    ap.add_argument("--unfreeze", type=int, default=0,
                    help="thaw the top N transformer layers (0 = fully frozen)")
    ap.add_argument("--mask-time-prob", type=float, default=0.0,
                    help="encoder-internal SpecAugment; only sensible when unfreezing")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--holdout-show", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--target-fpr", type=float, default=0.02,
                    help="per-frame interruption budget, fitted on VAL")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true",
                    help="refit the per-frame operating point on VAL and rewrite "
                         "it into an existing checkpoint; touches no weights")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional clip directory, e.g. data/sep28k_hf. The "
                         "HF mirror recovers the three link-rotted shows and "
                         "FluencyBank: +54%% clips, +50%% Block positives, and "
                         "4 speaker pools we did not have.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-version", default="legacy",
                    choices=["legacy", "stable"],
                    help="legacy reproduces every published number but "
                         "reshuffles when the corpus grows; stable keeps each "
                         "episode where it was. See make_splits.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    amp = not args.no_amp
    ckpt_path = Path(args.out) if args.out else CKPT
    # A `--limit 200` smoke run once wrote to the default path and destroyed
    # the shipped checkpoint. `*.pt` is gitignored, so there was nothing to
    # restore and the published 0.384/0.894 became unreproducible. A truncated
    # or single-epoch run may never claim a name that something else ships.
    if (args.limit or args.epochs < 2) and not args.out and not args.bench_only:
        raise SystemExit(
            "refusing to overwrite %s from a truncated run "
            "(limit=%d, epochs=%d). Pass an explicit --out; the "
            "default checkpoint name is for full runs only."
            % (ckpt_path, args.limit, args.epochs))
    metrics_path = ckpt_path.with_name(ckpt_path.stem + "_metrics.json")

    print("StutterSSL training")
    print("  encoder: %s   unfreeze_top: %d" % (args.encoder, args.unfreeze))

    if args.bench_only:
        model = StutterSSL(args.encoder, hidden=args.hidden, dropout=args.dropout)
        b = bench(model)
        print("  INFERENCE COST (3.0 s clip)")
        for k, v in b.items():
            print("    %-28s %s" % (k, v))
        base = bench_baseline()
        if base:
            print("  BASELINE StutterNet (log-mel CNN, waveform in)")
            for k, v in base.items():
                print("    %-28s %s" % (k, v))
        return 0

    rows, npy_paths = load_sources(args.extra)
    splits = make_splits(rows, args.holdout_show,
                         split_version=args.split_version)
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

    model = StutterSSL(args.encoder, hidden=args.hidden, dropout=args.dropout,
                       n_unfreeze=args.unfreeze,
                       mask_time_prob=args.mask_time_prob).to(device)
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print("  params: %.1fM total, %.2fM trainable  (StutterNet: 0.58M)"
          % (n_total / 1e6, n_train_params / 1e6))
    print("  frame rate: %d ms/frame" % FRAME_MS)

    if args.eval_only:
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"], strict=False)
        model.to(device)
        y, p, _ = run_eval(model, loaders["test"], device, amp)
        report(y, p, "TEST")
        return 0

    if args.calibrate_only:
        # Post-hoc, weight-free: the operating point is not part of training,
        # so refitting it must not require a retrain. Mirrors
        # eval/calibrate_stutter_frames.py --write for the log-mel model.
        model = load_checkpoint(ckpt_path, device)
        yv, _, peaks_v = run_eval(model, loaders["val"], device, amp)
        frame_cal = frame_calibration(peaks_v, yv, args.target_fpr)
        apply_calibration(model, frame_cal)
        print_calibration(frame_cal, args.target_fpr)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state["model"] = model.trainable_state_dict()
        state["frame_thresholds"] = {t: v["threshold"] for t, v in frame_cal.items()}
        state["frame_temperature"] = {t: v["temperature"] for t, v in frame_cal.items()}
        state["frame_calibration"] = {"target_fpr": args.target_fpr, "split": "val",
                                      "space": "logit", "detail": frame_cal}
        torch.save(state, ckpt_path)
        if metrics_path.exists():
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            m["frame_thresholds"] = state["frame_thresholds"]
            m["frame_temperature"] = state["frame_temperature"]
            m["frame_calibration_detail"] = frame_cal
            metrics_path.write_text(json.dumps(m, indent=2), encoding="utf-8")
        print("\n  rewrote operating point in %s" % ckpt_path)
        return 0

    # Positives run 7-29% per type; unweighted BCE is minimised by predicting
    # "no dysfluency" everywhere. Same weighting as the baseline, so the loss
    # is not a confound.
    train_y = np.array([rows[i]["labels"] for i in splits["train"]], dtype="float32")
    prev = train_y.mean(axis=0).clip(1e-3, 1 - 1e-3)
    pos_weight = torch.tensor((1 - prev) / prev, dtype=torch.float32, device=device)
    print("  pos_weight: %s" % dict(zip(TYPES, np.round(pos_weight.cpu().numpy(), 2))))

    # Two groups: a pretrained encoder fine-tuned at the head's learning rate
    # is destroyed in one epoch. The layer weights ride with the head.
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith("encoder.")
                   and n != "layer_logits"]
    enc_params = [p for n, p in model.named_parameters()
                  if p.requires_grad and n.startswith("encoder.")]
    # The layer weights get their own group at 10x the head rate and no weight
    # decay. They are 13 numbers behind a softmax: at the head's rate they
    # barely leave the uniform initialisation inside one training run, and
    # decaying them is decaying TOWARD uniform, which is a prior we do not want
    # to impose on the answer we are trying to read off.
    groups = [{"params": [model.layer_logits], "lr": args.lr * 10, "weight_decay": 0.0},
              {"params": head_params, "lr": args.lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": args.enc_lr})
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
            # Class-balanced BCE on the POOLED probability. The sigmoid and the
            # pool are both already inside clip_logits, so this cannot use
            # BCEWithLogits -- identical to the baseline's loss.
            loss = -(pos_weight * yb * clip.log()
                     + (1 - yb) * (1 - clip).log()).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
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
        # Model selection on VAL mAP, exactly as the baseline does it.
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
                  if args.holdout_show else "episode-disjoint (host leakage measured negligible)")
    test_report = report(yt, pt, "TEST -- %s" % split_kind)

    thresholds = {t: val_report[t]["threshold"] for t in TYPES}
    thresholds["ANY"] = val_report["ANY"]["threshold"]
    frame_cal = frame_calibration(peaks_v, yv, args.target_fpr)
    frame_thresholds = {t: v["threshold"] for t, v in frame_cal.items()}
    frame_temperature = {t: v["temperature"] for t, v in frame_cal.items()}
    apply_calibration(model, frame_cal)
    print_calibration(frame_cal, args.target_fpr)

    cost = {} if args.no_bench else bench(model)
    if cost:
        model.to(device)
        print("\n  INFERENCE COST (3.0 s clip)")
        for k, v in cost.items():
            print("    %-28s %s" % (k, v))

    # Provenance is derived from what was actually loaded, not hard-coded: with
    # --extra the corpus is no longer "5 of 8 shows", and a checkpoint that
    # misdescribes its own training data is worse than one that says nothing.
    shows_seen = sorted({r["show"] for r in rows})
    trained_on = ("SEP-28k reconstruction, %d clips, %d speaker pools (%s)"
                  % (len(rows), len(shows_seen), ", ".join(shows_seen)))
    if args.extra:
        trained_on += ("; includes HF mirror %s, which declares NO LICENCE"
                       % ", ".join(str(e) for e in args.extra))
    trained_on += (" -- partial reconstruction, NOT comparable to published "
                   "SEP-28k figures; see docs/DATA_PROVENANCE.md")

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        # Trainable tensors only; the frozen backbone is rebuilt from the
        # HuggingFace cache by backend/acoustic/stutter_ssl.load_checkpoint.
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
        "unfreeze_top_n": args.unfreeze,
        "params_total": n_total,
        "params_trainable": n_train_params,
        "frame_ms": FRAME_MS,
        "split": split_kind,
        "holdout_show": args.holdout_show,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "hparams": {"epochs": args.epochs, "batch": args.batch, "lr": args.lr,
                    "enc_lr": args.enc_lr if enc_params else None,
                    "dropout": args.dropout, "hidden": args.hidden,
                    "mask_time_prob": args.mask_time_prob, "amp_bf16": amp},
        "history": history,
        "layer_weights": [round(w, 4) for w in weights],
        "val": val_report,
        "test": test_report,
        "thresholds": thresholds,
        "frame_thresholds": frame_thresholds,
        "frame_temperature": frame_temperature,
        "frame_calibration_detail": frame_cal,
        "inference_cost_ms": cost,
        "baseline": "models/stutternet_metrics.json (log-mel CNN, same splits)",
        "trained_on": trained_on,
        "extra_sources": [str(e) for e in args.extra],
        "provenance": ("SEP-28k reconstructed from Apple's official labels; 258/385 "
                       "episodes recovered, 3 shows lost to link rot. NOT comparable "
                       "to published SEP-28k numbers (different corpus)."
                       + ("  Expanded with the HuggingFace mirror, which declares no "
                          "licence; it restores the link-rotted shows and FluencyBank."
                          if args.extra else "")),
        "caveat": ("Stuttered speech, not aphasic speech. Transfers because the "
                   "surface evidence overlaps; it is not an aphasia measurement. "
                   "Thresholds and epoch selection are fitted on VAL only."),
    }, indent=2), encoding="utf-8")
    print("\n  saved %s" % ckpt_path)
    print("  metrics -> %s" % metrics_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
