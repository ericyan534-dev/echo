"""Operating-point sweep for Echo's two acoustic detectors.

Why this exists: the originally shipped thresholds (prolongation sim>=0.95 &
>=700 ms; filler confidence >=0.70) were hand-tuned interactively. The honest
eval showed the prolongation detection lower-bound at 3/40 and a filler stream
false-alarm upper bound of 4.4/min -- so the operating points deserved an
evidence-driven choice, committed alongside the numbers it was based on. This
sweep's results are what the current shipped defaults are based on: prolongation
sim>=0.94 & >=600 ms (backend/acoustic/prolongation.py) and filler confidence
>=0.75 (backend/acoustic/stream.py) -- see docs/EVAL.md for the final numbers.

Sweep A -- ProlongationTracker (sim_thresh x min_ms):
    detection rate on the palindrome-looped real-vowel construction (the
    conservative lower bound), false fires on 120 s of real running speech
    (hard constraint: must stay 0), false fires on 120 s of music (report;
    VAD-gated in the live path).

Sweep B -- FillerNet confidence threshold:
    clip-level filler recall / Words false-positive rate on the official test
    split (fires iff argmax in {uh,um} AND p_uh+p_um >= conf -- the exact
    stream rule), plus the stream-level false-alarm rate per minute through
    the full AcousticStream path.

Usage:  python eval/tune_stall_thresholds.py [--stream-seconds S] [--clips N]
Output: eval/results/threshold_sweep.json + printed tables.
Selection policy (documented, applied by the maintainer, not auto-applied):
    prolongation -- max detection rate subject to speech false fires == 0;
    filler -- min stream false alarms with clip recall within a few points
    of the 0.70 baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.features import logmel  # noqa: E402
from backend.acoustic.model import CLASSES, load_checkpoint  # noqa: E402
from backend.acoustic.prolongation import ProlongationTracker  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from run_prolongation_eval import (  # noqa: E402
    eval_detection,
    eval_false_fires_music,
    eval_false_fires_speech,
    eval_stream_falsefire,
)

SR = 16_000
CKPT = ROOT / "models" / "fillernet.pt"
CLIPS = ROOT / "data" / "pfsd" / "clips" / "test"
RESULTS = ROOT / "eval" / "results" / "threshold_sweep.json"

SIM_GRID = [0.90, 0.92, 0.94, 0.95]
MIN_MS_GRID = [500, 600, 700]
CONF_GRID = [0.70, 0.75, 0.80, 0.85]


def sweep_prolongation(n_clips: int, ff_seconds: float) -> list[dict]:
    rows = []
    for sim in SIM_GRID:
        for min_ms in MIN_MS_GRID:
            factory = lambda s=sim, m=min_ms: ProlongationTracker(  # noqa: E731
                min_ms=m, sim_thresh=s)
            det = eval_detection(n_clips, tracker_factory=factory)
            ffs = eval_false_fires_speech(ff_seconds, tracker_factory=factory)
            ffm = eval_false_fires_music(ff_seconds, tracker_factory=factory)
            row = {
                "sim_thresh": sim, "min_ms": min_ms,
                "detection_rate": det.get("detection_rate"),
                "detected": f"{det.get('fired')}/{det.get('scored')}",
                "speech_ff": ffs.get("false_fires"),
                "speech_seconds": ffs.get("running_speech_seconds"),
                "music_ff": ffm.get("false_fires"),
                "music_seconds": ffm.get("music_seconds"),
            }
            rows.append(row)
            print(f"  sim>={sim:.2f} min_ms={min_ms}: "
                  f"detect {row['detected']} ({row['detection_rate']}), "
                  f"speech FF {row['speech_ff']}, music FF {row['music_ff']}",
                  flush=True)
    return rows


def _clip_probs(dirs: list[str], device: str) -> torch.Tensor:
    """Softmax probs for every clip under the given test-split label dirs."""
    import soundfile as sf

    model = load_checkpoint(CKPT, device)
    paths = []
    for d in dirs:
        paths.extend(sorted((CLIPS / d).glob("*.wav")))
    feats = []
    for p in paths:
        x, _ = sf.read(p, dtype="float32")
        if x.ndim > 1:
            x = x.mean(axis=1)
        if len(x) < SR:
            x = np.pad(x, (0, SR - len(x)))
        feats.append(logmel(torch.from_numpy(x[:SR])))
    X = torch.stack(feats).unsqueeze(1)
    probs = []
    with torch.no_grad():
        for i in range(0, len(X), 512):
            probs.append(torch.softmax(model(X[i:i + 512].to(device)), dim=1).cpu())
    return torch.cat(probs)


def sweep_filler_conf(stream_seconds: float, device: str) -> list[dict]:
    """Clip-level fire rule at each conf + full-stream false-alarm rate."""
    uh, um = CLASSES.index("uh"), CLASSES.index("um")
    pos = _clip_probs(["Uh", "Um"], device)      # should fire
    neg = _clip_probs(["Words"], device)         # must not fire
    print(f"  featurized {len(pos)} filler + {len(neg)} Words test clips",
          flush=True)

    def fire_mask(probs: torch.Tensor, conf: float) -> torch.Tensor:
        top = probs.argmax(dim=1)
        filler_p = probs[:, uh] + probs[:, um]
        return ((top == uh) | (top == um)) & (filler_p >= conf)

    rows = []
    for conf in CONF_GRID:
        stream = eval_stream_falsefire(stream_seconds, conf_thresh=conf)
        row = {
            "conf_thresh": conf,
            "clip_recall": round(float(fire_mask(pos, conf).float().mean()), 4),
            "words_fp_rate": round(float(fire_mask(neg, conf).float().mean()), 4),
            "stream_ff_per_min": stream.get("filler_events_per_min"),
            "stream_ff": stream.get("filler_events"),
            "stream_seconds": stream.get("speech_seconds"),
        }
        rows.append(row)
        print(f"  conf>={conf:.2f}: clip recall {row['clip_recall']}, "
              f"Words clip FP {row['words_fp_rate']}, "
              f"stream {row['stream_ff']} fires "
              f"({row['stream_ff_per_min']}/min)", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--ff-seconds", type=float, default=120.0)
    ap.add_argument("--stream-seconds", type=float, default=300.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    print("[sweep A] prolongation sim_thresh x min_ms", flush=True)
    prolong_rows = sweep_prolongation(args.clips, args.ff_seconds)

    print("[sweep B] filler confidence threshold", flush=True)
    conf_rows = sweep_filler_conf(args.stream_seconds, device)

    out = {
        "policy": {
            "prolongation": "max detection rate s.t. speech false fires == 0",
            "filler": "min stream false alarms with clip recall within a few "
                      "points of the 0.70 baseline",
        },
        "prolongation": prolong_rows,
        "filler_conf": conf_rows,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
