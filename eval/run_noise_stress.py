"""Noise-robustness stress test for FillerNet -- does the classifier survive
a noisy demo hall? EVAL-ONLY: no retraining, no threshold changes. The
shipped checkpoint (models/fillernet.pt) and the shipped conf=0.75 operating
point (backend/acoustic/stream.py AcousticStream default) are used exactly
as they ship.

SAMPLE: a fixed-seed sample of n PFSD TEST-split clips, drawn with
scripts/train_filler.py's own scan_split(["test"], limit=n) -- the identical
code path and shuffle (random.Random(13)) used at training/eval time, just
sliced to n for runtime. Never touches train/validation.

INTERFERER: PFSD TEST-split "Music" clips (real recorded music -- an ambient
demo-hall-noise proxy, zero new downloads, fully reproducible). One Music
clip is pre-assigned per eval item via a seeded RNG (--seed, default 13) and
reused UNCHANGED across every SNR level, so only the mix level differs
between conditions -- this isolates the SNR variable from interferer-instance
variance. (A degenerate self-mix -- an item assigned itself as interferer --
is redrawn; astronomically rare but checked.)

SNR DEFINITION (RMS over the 1.0 s clip): target_noise_rms =
rms(signal) / 10**(snr_db / 20); the interferer is scaled to that RMS and
added; the mix is clipped to [-1, 1] (clipping incidence is reported per
condition -- "clean" is the unmixed signal, never clipped).

METRICS, reported per SNR level (clean, 15, 10, 5 dB), both against ground
truth filler=uh|um, both computed on the exact same mixed features:
  (a) "standard" -- plain argmax over the 4-class model output, binary-
      collapsed. Identical methodology to scripts/train_filler.py's
      evaluate(); this is the number that should reproduce ~0.933 on the
      clean condition (a sample-size caveat applies: n here is a few hundred,
      not the full ~9.4k-clip test split evaluate() runs on).
  (b) "operating_point" -- the live AcousticStream firing rule: argmax in
      {uh, um} AND filler_p (p_uh+p_um) >= conf_thresh (0.75). Same rule as
      eval/tune_stall_thresholds.py's fire_mask; this is what actually
      governs whether a filler event reaches the demo screen.

DISCLOSURE (repeated in the JSON): this measures the CLASSIFIER IN
ISOLATION on fixed 1.0 s clips. The live pipeline additionally applies a
Silero VAD gate, a >=800 ms accumulated-voiced-time gate, and a 1200 ms
per-kind refractory (backend/acoustic/stream.py) that may mitigate
noise-induced misses in practice -- that mitigation is NOT measured here and
is not claimed, only named as an open question.

Usage:  python eval/run_noise_stress.py [--n N] [--seed S] [--device cuda|cpu]
Output: eval/results/noise_stress.json
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.features import logmel  # noqa: E402
from backend.acoustic.model import CLASSES, load_checkpoint  # noqa: E402
from scripts.train_filler import CLIPS, scan_split  # noqa: E402

SR = 16_000
CKPT = ROOT / "models" / "fillernet.pt"
RESULTS = ROOT / "eval" / "results" / "noise_stress.json"
CONF_THRESH = 0.75  # shipped operating point; backend/acoustic/stream.py AcousticStream default
SNR_LEVELS_DB: list[float | None] = [None, 15.0, 10.0, 5.0]  # None = clean (unmixed)


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    x = x if x.ndim == 1 else x.mean(axis=1)
    if len(x) < SR:
        x = np.pad(x, (0, SR - len(x)))
    return x[:SR].astype("float32")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)) + 1e-12))


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, bool]:
    """Scale `noise` so rms(signal)/rms(scaled_noise) == 10**(snr_db/20),
    add, clip to [-1, 1]. Returns (mixed, was_clipped)."""
    s_rms = rms(signal)
    n_rms = rms(noise)
    if n_rms < 1e-8:  # near-silent interferer clip: nothing to mix
        return signal.copy(), False
    target_n_rms = s_rms / (10.0 ** (snr_db / 20.0))
    scale = target_n_rms / n_rms
    mixed = signal + scale * noise
    clipped = bool(np.any(np.abs(mixed) > 1.0))
    return np.clip(mixed, -1.0, 1.0).astype("float32"), clipped


def assign_interferers(items: list[tuple[Path, int]], pool: list[Path], seed: int) -> list[Path]:
    rng = np.random.default_rng(seed)
    out = []
    for path, _ in items:
        idx = int(rng.integers(0, len(pool)))
        tries = 0
        while pool[idx] == path and tries < 5:  # guard the degenerate self-mix
            idx = int(rng.integers(0, len(pool)))
            tries += 1
        out.append(pool[idx])
    return out


def featurize(items: list[tuple[Path, int]], interferers: list[Path],
             snr_db: float | None) -> tuple[torch.Tensor, torch.Tensor, int]:
    feats, labels, clipped_count = [], [], 0
    for (path, y), inter_path in zip(items, interferers):
        x = _load_wav(path)
        if snr_db is not None:
            noise = _load_wav(inter_path)
            x, was_clipped = mix_at_snr(x, noise, snr_db)
            clipped_count += int(was_clipped)
        feats.append(logmel(torch.from_numpy(x)))
        labels.append(y)
    X = torch.stack(feats).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.long)
    return X, y, clipped_count


@torch.no_grad()
def classify(model, X: torch.Tensor, device: str) -> torch.Tensor:
    probs = []
    for i in range(0, len(X), 512):
        probs.append(torch.softmax(model(X[i:i + 512].to(device)), dim=1).cpu())
    return torch.cat(probs)


def binary_prf(true_mask: torch.Tensor, pred_mask: torch.Tensor) -> dict:
    tp = int((pred_mask & true_mask).sum())
    fp = int((pred_mask & ~true_mask).sum())
    fn = int((~pred_mask & true_mask).sum())
    tn = int((~pred_mask & ~true_mask).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score_condition(probs: torch.Tensor, y: torch.Tensor) -> dict:
    uh, um = CLASSES.index("uh"), CLASSES.index("um")
    top = probs.argmax(dim=1)
    true_filler = (y == uh) | (y == um)

    # (a) standard -- plain argmax, identical to scripts/train_filler.py evaluate()
    pred_argmax = (top == uh) | (top == um)
    standard = binary_prf(true_filler, pred_argmax)

    # (b) operating point -- the live AcousticStream firing rule
    filler_p = probs[:, uh] + probs[:, um]
    pred_op = pred_argmax & (filler_p >= CONF_THRESH)
    operating_point = binary_prf(true_filler, pred_op)

    return {"standard": standard, "operating_point": operating_point,
            "n": len(y), "n_filler_true": int(true_filler.sum())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--n", type=int, default=300, help="fixed-seed sample size from the test split")
    ap.add_argument("--seed", type=int, default=13, help="RNG seed for interferer assignment")
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()

    if not CKPT.exists():
        out = {"status": f"SKIPPED (no FillerNet checkpoint at {CKPT} -- train first)"}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(out["status"])
        return 0

    items = scan_split(["test"], limit=args.n)
    music_pool = sorted((CLIPS / "test" / "Music").glob("*.wav"))
    if len(items) < 20 or len(music_pool) < 5:
        out = {"status": "SKIPPED (test split clips not downloaded yet)"}
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(out["status"])
        return 0

    interferers = assign_interferers(items, music_pool, args.seed)
    model = load_checkpoint(CKPT, args.device)
    class_balance = {CLASSES[i]: n for i, n in sorted(Counter(y for _, y in items).items())}
    print(f"sample: n={len(items)} clips {class_balance}, "
          f"interferer pool: {len(music_pool)} Music clips", flush=True)

    rows: dict[str, dict] = {}
    clip_counts: dict[str, int] = {}
    for snr in SNR_LEVELS_DB:
        label = "clean" if snr is None else f"{snr:.0f}dB"
        X, y, clipped = featurize(items, interferers, snr)
        probs = classify(model, X, args.device)
        rows[label] = score_condition(probs, y)
        clip_counts[label] = clipped
        s, o = rows[label]["standard"], rows[label]["operating_point"]
        print(f"[{label:>6s}] standard F1={s['f1']} (P={s['precision']} R={s['recall']})  |  "
              f"op-point(conf>={CONF_THRESH}) F1={o['f1']} (P={o['precision']} R={o['recall']})  "
              f"[{clipped}/{len(items)} mixes clipped]", flush=True)

    out = {
        "status": "ok",
        "sample": {
            "n": len(items),
            "class_balance": class_balance,
            "note": "items drawn via scripts/train_filler.py:scan_split(['test'], limit=n) -- "
                    "identical code path and shuffle (random.Random(13)) as the training-time test "
                    "scan, sliced to n for runtime. class_balance (uh/um/speech/other counts) is "
                    "reported so the clean-condition F1's comparability to the full-test-split "
                    "0.933 anchor is auditable, not asserted.",
            "interferer_assignment_seed": args.seed,
        },
        "interferer": {
            "source": "PFSD TEST-split 'Music' clips (real recorded music, an ambient-noise proxy "
                      "for a demo hall; zero new downloads, fully reproducible)",
            "pool_size": len(music_pool),
            "assignment": "one Music clip pre-assigned per eval item via numpy default_rng(seed), "
                          "reused unchanged across all SNR levels so only the mix level differs "
                          "between conditions -- isolates the SNR variable from interferer-instance "
                          "variance",
        },
        "snr_definition": "target_noise_rms = rms(signal) / 10**(snr_db/20); interferer scaled to "
                          "that RMS then added; mix clipped to [-1, 1] (clipping incidence reported "
                          "per condition below; 'clean' is the unmixed signal, never clipped)",
        "operating_point_conf_thresh": CONF_THRESH,
        "clipped_mixes": clip_counts,
        "conditions": rows,
        "clean_sanity_anchor": "docs/EVAL.md Table 1 reports binary filler F1=0.933 on the FULL "
                               "~9.4k-clip test split (scripts/train_filler.py evaluate()); this "
                               "script's 'clean'/'standard' row uses the identical decision rule on "
                               f"a random n={len(items)} subset of that same split, so it should land "
                               "close to 0.933 within sampling noise, not reproduce it exactly.",
        "caveat": "measures the FillerNet CLASSIFIER IN ISOLATION on fixed 1.0 s clips. The live "
                 "AcousticStream pipeline additionally applies a Silero VAD gate, a >=800 ms "
                 "accumulated-voiced-time gate, and a 1200 ms per-kind refractory "
                 "(backend/acoustic/stream.py) that may mitigate noise-induced misses in practice -- "
                 "that mitigation is NOT measured here, only named as an open question.",
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
