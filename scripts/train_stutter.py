"""Train StutterNet on the reconstructed SEP-28k.

    python scripts/train_stutter.py                     # train + eval + save
    python scripts/train_stutter.py --holdout-show HeStutters   # cross-show
    python scripts/train_stutter.py --eval-only

SPLITS ARE THE WHOLE EXPERIMENT
-------------------------------
A dysfluency detector trained and tested on the same speaker measures almost
nothing: stuttering is highly speaker-specific, and a model can score well by
recognising a voice. SEP-28k gives Show (podcast) and EpId, so:

  * The default split is EPISODE-disjoint, stratified by show. No clip from a
    test episode is ever trained on. It DOES leak the recurring HOST of each
    podcast across the split, and that was assumed for a long time to make it
    an optimistic estimate.

    That assumption was tested and is wrong. Training the same model with a
    show's host seen versus unseen and scoring the same test clips
    (eval/eval_stutter_ssl_hostleak.py) gives a mean gap of Block +0.009 and
    ANY -0.003 across two held-out shows -- negligible, and ANY is slightly
    negative. Show difficulty dominates instead: per-show scores vary far more
    than the seen/unseen gap ever does. Do not re-add the "optimistic" caveat
    without re-measuring it.
  * --holdout-show trains with one entire show removed and tests on it. That
    is speaker-disjoint by construction and is the honest generalisation
    number. It is lower. Both get reported; neither is quietly dropped.

WHAT THIS CANNOT MEASURE
------------------------
Aphasia. SEP-28k is stuttered speech -- a motor-speech disorder in which the
word is known and will not come out. Echo's users have a language disorder in
which the word is not retrievable. The surface evidence overlaps (silent
blocks, filled pauses, repetitions) which is why this transfers at all, but
nothing here is an aphasia measurement. See eval/run_aphasia_eval.py for that,
on APROCSA, and docs/DATA_PROVENANCE.md for what each corpus forbids claiming.
"""
from __future__ import annotations

import argparse
import hashlib
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

from backend.acoustic.features import logmel  # noqa: E402
from backend.acoustic.stutter import TYPES, StutterNet, linear_softmax_pool  # noqa: E402

DATA = ROOT / "data" / "sep28k"
CLIPS_NPY = DATA / "clips_16k.npy"
CLIPS_IDX = DATA / "clips_index.json"
MODELS = ROOT / "models"
CKPT = MODELS / "stutternet.pt"
METRICS = MODELS / "stutternet_metrics.json"

SR = 16_000


# --- data ----------------------------------------------------------------
# FluencyBank episode ids map to media files named age+gender+letter (24fa,
# 24fb, 24fc ...). The letter almost certainly disambiguates distinct subjects,
# but the TalkBank metadata that would confirm it is gated -- so six age+gender
# groups are treated as ONE speaker each. That is the conservative direction:
# if the letters are different people we lose a little split granularity; if
# they are the same person and we split them, every FluencyBank number is
# inflated by testing on a speaker we trained on.
_FB_GROUPS = None


def fluencybank_group(ep: str) -> str:
    global _FB_GROUPS
    if _FB_GROUPS is None:
        import csv
        import io
        import re
        import urllib.request
        _FB_GROUPS = {}
        try:
            url = ("https://raw.githubusercontent.com/apple/"
                   "ml-stuttering-events-dataset/main/fluencybank_episodes.csv")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=30).read().decode()
            for r in csv.reader(io.StringIO(raw)):
                if len(r) < 4:
                    continue
                stem = r[2].strip().rsplit("/", 1)[-1].split(".")[0]
                m = re.match(r"(\d+[fm])", stem)
                _FB_GROUPS[r[1].strip()] = m.group(1) if m else stem
        except Exception:
            _FB_GROUPS = {}
    return _FB_GROUPS.get(str(ep), str(ep))


def speaker_group(show: str, ep: str) -> str:
    """The unit a split may not cut through."""
    if show == "FluencyBank":
        return fluencybank_group(ep)
    return str(ep)


def load_sources(extra_dirs: list | None = None) -> tuple[list[dict], list]:
    """Rows plus the .npy path each row's audio lives in.

    Multiple sources are supported so the HuggingFace-mirror recovery of the
    three link-rotted shows and of FluencyBank can be trained on alongside the
    locally cut clips. Each row carries `src`, an index into the returned path
    list, and `row`, its offset within that array.
    """
    sources = [(CLIPS_IDX, CLIPS_NPY)]
    for d in (extra_dirs or []):
        d = Path(d)
        sources.append((d / "clips_index.json", d / "clips_16k.npy"))
    rows, paths, seen = [], [], set()
    for si, (idx_p, npy_p) in enumerate(sources):
        if not idx_p.exists() or not npy_p.exists():
            if si == 0:
                raise SystemExit("missing %s -- run scripts/cut_sep28k_clips.py" % idx_p)
            print("  skipping missing source %s" % idx_p)
            continue
        paths.append(npy_p)
        for r in json.loads(idx_p.read_text(encoding="utf-8"))["rows"]:
            key = (r["show"], str(r["ep"]), str(r["clip"]))
            # The mirror overlaps what we already hold. Concatenating naively
            # would put byte-identical clips in both train and test.
            if key in seen:
                continue
            seen.add(key)
            r = dict(r)
            r["src"] = len(paths) - 1
            rows.append(r)
    return rows, paths


def load_index(extra_dirs: list | None = None) -> list[dict]:
    """Rows only. Kept with its original signature and return type because
    eval/calibrate_stutter_frames.py and other consumers import it; use
    load_sources() when you also need the .npy paths."""
    return load_sources(extra_dirs)[0]


def _stable_unit(seed: int, show: str, ep: str) -> float:
    """Deterministic float in [0,1) from (seed, show, episode) alone.

    blake2b rather than hash() because CPython salts str hashing per process.
    """
    h = hashlib.blake2b(("%d:%s:%s" % (seed, show, ep)).encode("utf-8"),
                        digest_size=8).digest()
    return int.from_bytes(h, "big") / float(1 << 64)


def make_splits(rows: list[dict], holdout_show: str | None,
                seed: int = 13,
                split_version: str = "legacy") -> dict[str, list[int]]:
    """Episode-disjoint splits, stratified by show.

    Assignment is by EPISODE, never by clip: two clips from one episode are
    the same speaker seconds apart, and splitting them is self-testing.

    TWO VERSIONS, AND WHY BOTH EXIST
    --------------------------------
    "legacy" draws one permutation per show from a SHARED RandomState. That
    makes a show's split depend on every show processed before it, so adding
    episodes anywhere reshuffles everything downstream. It cost real work:
    when the corpus grew 20,124 -> 30,962 clips, 1,380 of the 4,411 new-test
    clips turned out to sit in the old checkpoint's TRAIN set, and scoring the
    old model there read Block 0.410 against 0.304 on clips it had never seen.
    Every published number was produced under "legacy", so it stays the
    default and they all still reproduce.

    "stable" assigns each episode by hashing (seed, show, episode) and
    thresholding. Membership depends on the episode's own identity and
    nothing else, so growing the corpus can only ADD episodes to a split,
    never move one across. Quotas become approximate rather than exact --
    that is the price, and it is worth paying for a corpus that is still
    growing. Use it for any new training line; do not mix the two.

    ONE HONEST LIMIT ON "stable". A show with few episodes can hash entirely
    into the train band (0.75^n for n episodes -- 32% at n=4), which would
    drop it out of val and test altogether. Those shows get episodes forced
    back off the bottom of the hash order, and THAT choice is not growth-
    invariant: if such a show gains an episode that hashes lower, the forced
    one returns to train. On the current corpus this affects exactly two of
    nine shows, HVSA (4 episodes) and IStutterSoWhat (5). Everything else is
    invariant by construction. Stratifying a four-episode show and pinning it
    against growth are not simultaneously achievable; per-show test coverage
    was judged the more valuable of the two.
    """
    if split_version not in ("legacy", "stable"):
        raise ValueError("split_version must be 'legacy' or 'stable'")
    rng = np.random.RandomState(seed)
    by_show: dict[str, set] = {}
    for r in rows:
        by_show.setdefault(r["show"], set()).add(speaker_group(r["show"], r["ep"]))

    ep_split: dict[tuple[str, str], str] = {}
    for show, eps in by_show.items():
        eps = sorted(eps, key=lambda e: (len(str(e)), str(e)))
        if holdout_show and show == holdout_show:
            for e in eps:
                ep_split[(show, e)] = "test"
            continue
        if split_version == "stable":
            u = {e: _stable_unit(seed, show, str(e)) for e in eps}
            for e in eps:
                ep_split[(show, e)] = ("test" if u[e] < 0.15
                                       else "val" if u[e] < 0.25
                                       else "train")
            # A small show can hash entirely into one band and silently drop
            # out of val or test. Force it back in by taking episodes off the
            # bottom of the hash order -- deterministic, and still a function
            # of this show's episodes alone. A show with fewer than three
            # episodes cannot fill three splits and is left as hashed.
            order = sorted(eps, key=lambda e: u[e])
            if len(eps) >= 3:
                forced: set = set()
                for want in ("test", "val"):
                    have = [e for e in eps
                            if ep_split[(show, e)] == want and e not in forced]
                    if have:
                        forced.update(have)
                        continue
                    pick = next(e for e in order if e not in forced)
                    ep_split[(show, pick)] = want
                    forced.add(pick)
            continue
        idx = rng.permutation(len(eps))
        n_test = max(1, int(0.15 * len(eps)))
        n_val = max(1, int(0.10 * len(eps)))
        for j, i in enumerate(idx):
            ep_split[(show, eps[i])] = ("test" if j < n_test
                                        else "val" if j < n_test + n_val
                                        else "train")
    if holdout_show:
        # With a show held out, the remaining shows contribute train/val only:
        # mixing them into test would blend a speaker-disjoint measurement
        # with an episode-disjoint one and report the average as if it were
        # the former.
        for k, v in list(ep_split.items()):
            if v == "test" and k[0] != holdout_show:
                ep_split[k] = "train"

    out: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        out[ep_split[(r["show"], speaker_group(r["show"], r["ep"]))]].append(i)
    return out


def _synthetic_rir(sr: int = SR, rt60_ms: float = 200.0,
                   direct_frac: float = 0.5) -> torch.Tensor:
    """A cheap exponentially-decaying random room impulse response.

    Reverberation is the one mic-robustness augmentation that logmel's
    per-example mean/std normalization does NOT wash out: it reshapes the
    temporal envelope and (through comb-filtering) the spectral envelope
    non-uniformly, so unlike a flat gain it survives normalization and teaches
    the model that the same dysfluency arrives smeared in a real room. A full
    measured-RIR corpus would be better; this is a zero-dependency stand-in.
    """
    n = max(1, int(sr * rt60_ms / 1000.0))
    t = torch.arange(n, dtype=torch.float32)
    decay = torch.exp(-6.9078 * t / n)            # -60 dB over rt60
    rir = torch.randn(n) * decay
    rir[0] = 1.0 / max(direct_frac, 1e-3)         # direct path
    rir = rir / rir.abs().max().clamp_min(1e-6)
    return rir


def _augment_waveform(pcm: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    """Stronger real-mic augmentation, applied on the WAVEFORM before logmel.

    Order matters: gain jitter FIRST, then SNR-referenced additive noise. A
    flat gain alone is a no-op here (logmel normalizes per example), but by
    changing the signal level BEFORE a noise floor referenced to that level is
    added, gain jitter widens the effective-SNR distribution the model sees.
    Reverb is applied last because a real mic hears the room, then its own
    noise floor -- both survive normalization.
    """
    # 1) gain jitter (meaningful only in combination with the noise below)
    pcm = pcm * float(rng.uniform(0.3, 1.5))
    # 2) reverberation (50% of the time)
    if rng.random() < 0.5:
        rir = _synthetic_rir(rt60_ms=float(rng.uniform(80.0, 400.0)))
        pcm = torch.nn.functional.conv1d(
            pcm.view(1, 1, -1),
            rir.flip(0).view(1, 1, -1),
            padding=rir.numel() - 1)[0, 0, : pcm.numel()]
    # 3) additive noise at a wide, SNR-referenced range (was uniform(0, 0.01),
    #    an absolute floor with no SNR meaning). 5-30 dB spans a noisy hall to
    #    a quiet room; 20% of the time no noise so clean speech stays in-dist.
    if rng.random() < 0.8:
        snr_db = float(rng.uniform(5.0, 30.0))
        sig_rms = pcm.pow(2).mean().sqrt().clamp_min(1e-6)
        noise = torch.randn_like(pcm)
        noise_rms = sig_rms / (10.0 ** (snr_db / 20.0))
        pcm = pcm + noise * noise_rms
    return pcm.clamp(-1.0, 1.0)


class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, path, rows, idx, train: bool, aug_strong: bool = False) -> None:
        self.paths = list(path) if isinstance(path, (list, tuple)) else [path]
        self.rows = rows
        self.idx = idx
        self.train = train
        self.aug_strong = aug_strong
        self._rng = None      # per-worker numpy Generator, seeded lazily
        self._arr = None      # opened lazily, per worker

    @property
    def arr(self):
        # Opened on first access rather than in __init__ so each DataLoader
        # worker gets its own map. A memmap created in the parent would be
        # pickled to the workers, which serialises the whole 1.9 GB array.
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
        if self.train and self.aug_strong:
            # Wider real-mic augmentation: gain jitter + SNR-referenced noise +
            # reverberation, on the waveform. Opt-in via --aug-strong so the
            # published default recipe is untouched. See _augment_waveform.
            if self._rng is None:
                info = torch.utils.data.get_worker_info()
                self._rng = np.random.default_rng(13 + (info.id if info else 0))
            pcm = _augment_waveform(pcm, self._rng)
        elif self.train and np.random.rand() < 0.5:
            # DEFAULT (published) recipe: additive noise only. Gain jitter would
            # be a NO-OP here: logmel() normalizes per example, so scaling the
            # waveform just shifts the log-mel by a constant that the mean
            # subtraction removes. It read like mic-robustness augmentation and
            # trained nothing.
            pcm = pcm + torch.randn_like(pcm) * float(np.random.uniform(0, 0.01))
        feats = logmel(pcm).unsqueeze(0)
        if self.train:
            feats = spec_augment(feats)
        return feats, y


def spec_augment(x: torch.Tensor, n_freq: int = 2, n_time: int = 2,
                 max_f: int = 8, max_t: int = 20) -> torch.Tensor:
    """SpecAugment (Park et al. 2019). Masking teaches the model not to rely
    on any single band or instant, which matters here because the lav mic
    (DJI Mic 2S) is not the podcast mic these recordings were made on."""
    x = x.clone()
    _, n_mels, n_frames = x.shape
    for _ in range(n_freq):
        f = np.random.randint(0, max_f + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        x[:, f0:f0 + f, :] = 0
    for _ in range(n_time):
        t = np.random.randint(0, max_t + 1)
        t0 = np.random.randint(0, max(1, n_frames - t))
        x[:, :, t0:t0 + t] = 0
    return x


# --- metrics -------------------------------------------------------------
def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Area under the precision-recall curve, computed directly.

    Not sklearn: this environment's sklearn import chain pulls a pandas built
    against NumPy 1.x and raises under NumPy 2.4.
    """
    order = np.argsort(-y_score)
    y = y_true[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    denom = tp + fp
    precision = tp / np.maximum(denom, 1)
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    return float((precision * y).sum() / n_pos)


def best_f1(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float, float, float]:
    """(f1, threshold, precision, recall) at the F1-optimal operating point."""
    order = np.argsort(-y_score)
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(1.0, float(y.sum()))
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(np.argmax(f1))
    return float(f1[k]), float(s[k]), float(prec[k]), float(rec[k])


def best_at_recall(y_true: np.ndarray, y_score: np.ndarray,
                   target_recall: float) -> tuple[float, float, float, float]:
    """(f1, threshold, precision, recall) at the highest-precision point whose
    recall is >= target_recall.

    "Miss > nag" is the shipped design, but the acoustic channel had drifted so
    far toward miss that it recalls ~0.15 on blocks. This picks the operating
    point that guarantees the recall the product needs and then buys back as
    much precision as that recall allows -- the opposite selection pressure to
    best_f1, which trades recall away whenever it lifts F1. Falls back to the
    F1-optimal point if no threshold reaches the target (the type simply cannot
    hit it on this model).
    """
    order = np.argsort(-y_score)
    y = y_true[order]
    s = y_score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    n_pos = max(1.0, float(y.sum()))
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / n_pos
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    ok = np.where(rec >= target_recall)[0]
    if ok.size == 0:
        return best_f1(y_true, y_score)
    k = int(ok[np.argmax(prec[ok])])
    return float(f1[k]), float(s[k]), float(prec[k]), float(rec[k])


# --- train / eval ---------------------------------------------------------
def run_eval(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in loader:
            clip, _ = model.clip_logits(xb.to(device))
            ys.append(yb.numpy())
            ps.append(clip.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def _select(y_true, y_score, recall_target: float):
    """F1-optimal by default; recall-targeted when recall_target > 0."""
    if recall_target and recall_target > 0:
        return best_at_recall(y_true, y_score, recall_target)
    return best_f1(y_true, y_score)


def report(y, p, title: str, recall_target: float = 0.0) -> dict:
    out = {}
    print("\n  %s%s" % (title, ("  [thresholds @ recall>=%.2f]" % recall_target)
                        if recall_target else ""))
    print("    %-14s %7s %7s %7s %7s %7s %7s"
          % ("type", "n_pos", "AP", "F1", "prec", "rec", "thresh"))
    for k, t in enumerate(TYPES):
        ap = average_precision(y[:, k], p[:, k])
        f1, th, pr, rc = _select(y[:, k], p[:, k], recall_target)
        base = float(y[:, k].mean())
        out[t] = {"n_pos": int(y[:, k].sum()), "prevalence": round(base, 4),
                  "ap": round(ap, 4), "f1": round(f1, 4),
                  "precision": round(pr, 4), "recall": round(rc, 4),
                  "threshold": round(th, 4),
                  "ap_lift_over_chance": round(ap / base, 2) if base else None}
        print("    %-14s %7d %7.3f %7.3f %7.3f %7.3f %7.3f"
              % (t, y[:, k].sum(), ap, f1, pr, rc, th))
    # "any dysfluency" is the decision Echo actually makes at the stall layer.
    y_any = (y.max(axis=1) > 0).astype("float32")
    p_any = p.max(axis=1)
    ap = average_precision(y_any, p_any)
    f1, th, pr, rc = _select(y_any, p_any, recall_target)
    out["ANY"] = {"n_pos": int(y_any.sum()), "prevalence": round(float(y_any.mean()), 4),
                  "ap": round(ap, 4), "f1": round(f1, 4), "precision": round(pr, 4),
                  "recall": round(rc, 4), "threshold": round(th, 4)}
    print("    %-14s %7d %7.3f %7.3f %7.3f %7.3f %7.3f"
          % ("ANY", y_any.sum(), ap, f1, pr, rc, th))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout-show", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--extra", action="append", default=[],
                    help="additional clip directory (e.g. data/sep28k_hf)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--split-version", default="legacy",
                    choices=["legacy", "stable"],
                    help="legacy reproduces every published number but "
                         "reshuffles when the corpus grows; stable keeps each "
                         "episode where it was. See make_splits.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aug-strong", action="store_true",
                    help="stronger real-mic augmentation (gain jitter + "
                         "SNR-referenced additive noise 5-30 dB + synthetic "
                         "reverb). Default off -> published recipe unchanged.")
    ap.add_argument("--recall-target", type=float, default=0.0,
                    help="if >0, fit each clip threshold at the highest-precision "
                         "point with recall >= this (e.g. 0.90) instead of the "
                         "F1-optimal point. Default 0 -> F1-optimal (published).")
    args = ap.parse_args()

    ckpt_path = Path(args.out) if args.out else CKPT
    rows, npy_paths = load_sources(args.extra)
    print("StutterNet training")
    print("  clips: %d   shows: %s" % (len(rows), sorted({r["show"] for r in rows})))

    splits = make_splits(rows, args.holdout_show,
                         split_version=args.split_version)
    if args.limit:
        for k in splits:
            splits[k] = splits[k][:args.limit]
    for k, v in splits.items():
        shows = Counter(rows[i]["show"] for i in v)
        print("  %-6s %6d clips  %s" % (k, len(v), dict(shows)))
    if args.holdout_show:
        print("  HOLDOUT SHOW: %s -- test is speaker-disjoint" % args.holdout_show)

    device = args.device
    loaders = {
        k: torch.utils.data.DataLoader(
            ClipDataset(npy_paths, rows, splits[k], train=(k == "train"),
                        aug_strong=(args.aug_strong and k == "train")),
            batch_size=args.batch, shuffle=(k == "train"),
            num_workers=args.workers, persistent_workers=bool(args.workers),
            drop_last=(k == "train"))
        for k in ("train", "val", "test")
    }
    if args.aug_strong:
        print("  augmentation: STRONG (gain jitter + SNR noise 5-30 dB + reverb)")
    if args.recall_target:
        print("  threshold selection: recall-target >= %.2f (not F1-optimal)"
              % args.recall_target)

    model = StutterNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("  params: %.0fk" % (n_params / 1e3))

    if args.eval_only:
        if not ckpt_path.exists():
            raise SystemExit("no checkpoint at %s" % ckpt_path)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        y, p = run_eval(model, loaders["test"], device)
        report(y, p, "TEST", recall_target=args.recall_target)
        return 0

    # Positives run 9-23% per type, so an unweighted BCE is minimised by
    # predicting "no dysfluency" everywhere -- which is exactly the failure the
    # old model had in the other direction.
    train_y = np.array([rows[i]["labels"] for i in splits["train"]], dtype="float32")
    prev = train_y.mean(axis=0).clip(1e-3, 1 - 1e-3)
    pos_weight = torch.tensor((1 - prev) / prev, dtype=torch.float32, device=device)
    print("  pos_weight: %s" % dict(zip(TYPES, np.round(pos_weight.cpu().numpy(), 2))))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 3, total_steps=args.epochs * max(1, len(loaders["train"])))
    eps = 1e-6
    best_ap, best_state, history = -1.0, None, []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for xb, yb in loaders["train"]:
            xb, yb = xb.to(device), yb.to(device)
            clip, _ = model.clip_logits(xb)
            clip = clip.clamp(eps, 1 - eps)
            # Weighted BCE on the POOLED probability. The pooling is already
            # inside clip_logits, so this cannot use BCEWithLogits.
            loss = -(pos_weight * yb * clip.log()
                     + (1 - yb) * (1 - clip).log()).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            tot += float(loss) * len(xb)
            n += len(xb)
        y, p = run_eval(model, loaders["val"], device)
        val_ap = float(np.nanmean([average_precision(y[:, k], p[:, k])
                                   for k in range(len(TYPES))]))
        history.append({"epoch": epoch, "train_loss": round(tot / max(1, n), 4),
                        "val_mAP": round(val_ap, 4)})
        print("  epoch %2d  loss %.4f  val mAP %.4f  (%.0fs)"
              % (epoch, tot / max(1, n), val_ap, time.time() - t0), flush=True)
        if val_ap > best_ap:
            best_ap = val_ap
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    yv, pv = run_eval(model, loaders["val"], device)
    val_report = report(yv, pv, "VAL (thresholds fitted here)",
                        recall_target=args.recall_target)
    yt, pt = run_eval(model, loaders["test"], device)
    split_kind = ("speaker-disjoint (holdout show %s)" % args.holdout_show
                  if args.holdout_show else "episode-disjoint (host leakage possible)")
    test_report = report(yt, pt, "TEST -- %s" % split_kind,
                         recall_target=args.recall_target)

    # Thresholds come from VAL, never TEST. Fitting them on the test set would
    # make every number here a training-set number.
    thresholds = {t: val_report[t]["threshold"] for t in TYPES}
    thresholds["ANY"] = val_report["ANY"]["threshold"]

    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "types": TYPES,
        "thresholds": thresholds,
        "metrics": {"val": val_report, "test": test_report},
        "trained_on": ("SEP-28k (74%% episode subset, 5 of 8 shows) -- "
                       "see docs/DATA_PROVENANCE.md"),
        "split": split_kind,
        "recipe": {"aug_strong": bool(args.aug_strong),
                   "recall_target": float(args.recall_target),
                   "epochs": args.epochs, "split_version": args.split_version},
    }, ckpt_path)

    METRICS.write_text(json.dumps({
        "checkpoint": ckpt_path.name,
        "params": n_params,
        "split": split_kind,
        "holdout_show": args.holdout_show,
        "n_train": len(splits["train"]), "n_val": len(splits["val"]),
        "n_test": len(splits["test"]),
        "history": history,
        "val": val_report,
        "test": test_report,
        "thresholds": thresholds,
        "provenance": ("SEP-28k reconstructed from Apple's official labels; 258/385 "
                       "episodes recovered, 3 shows lost to link rot. NOT comparable "
                       "to published SEP-28k numbers (different corpus)."),
        "caveat": ("Stuttered speech, not aphasic speech. Transfers because the "
                   "surface evidence overlaps; it is not an aphasia measurement."),
    }, indent=2), encoding="utf-8")
    print("\n  saved %s" % ckpt_path)
    print("  metrics -> %s" % METRICS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
