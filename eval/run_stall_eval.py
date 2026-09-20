"""Evaluate Echo's stall-detection channels on the PodcastFillers test split.

Two questions, answered honestly and side by side:

(a) Can the ACOUSTIC channel hear fillers?  Clip-level uh/um classification
    with the trained FillerNet.  Metrics are computed with the *same*
    evaluate()/scan_split()/load_features() imported from
    scripts/train_filler.py, so the numbers here and in
    models/fillernet_metrics.json can never disagree.

(b) Can a TRANSCRIPT-ONLY detector?  Under the documented live-Chrome
    condition, consumer ASR (Chrome Web Speech and friends) strips fillers
    from transcripts.  For every filler clip in the test split we therefore
    construct the transcript such an ASR would actually emit for that clip —
    no filler token — and feed it through the real backend StallDetector,
    counting filler-trigger fires.  This is 0 by construction; the point of
    running it anyway is to pin the number to the real detector code (and to
    the real clip count n) instead of asserting it.  A verbatim-ASR ablation
    (token present) is included to show the bottleneck is the ASR, not the
    detector logic: given the token, the detector fires.

HONEST FRAMING: this is *trigger-level* recall.  The transcript-only system
still catches the stall eventually via the 1300 ms pause timeout; the
acoustic channel's value is firing earlier and on direct filler evidence.

The script runs on a partial dataset (prints n per class for whatever clips
exist) and degrades gracefully when models/fillernet.pt is missing
(acoustic metrics reported as PENDING, baseline still runs).

Usage:
    python eval/run_stall_eval.py [--device cuda|cpu] [--limit N]
    python eval/run_stall_eval.py --wav-dir recordings/   # self-recorded set
                                                          # (eval/record_protocol.md)
Output: eval/results/stall_eval.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.model import load_checkpoint  # noqa: E402
from backend.schemas import Word  # noqa: E402
from backend.stall_detector import StallDetector  # noqa: E402
from scripts.train_filler import evaluate, load_features, scan_split  # noqa: E402

CKPT = ROOT / "models" / "fillernet.pt"
RESULTS = ROOT / "eval" / "results" / "stall_eval.json"
SR = 16_000
FILLER_LABELS = ("Uh", "Um")  # PFSD directory names for filler clips

# Filename convention for the self-recorded set — see eval/record_protocol.md.
_WAV_NAME_RE = re.compile(
    r"^(?P<speaker>[a-z0-9]+)_(?P<category>pause|filler|prolongation|hedge|fluent)_(?P<idx>\d+)\.wav$"
)


# ---------------------------------------------------------------------------
# (a) acoustic clip-level classification
# ---------------------------------------------------------------------------
def eval_acoustic(items: list[tuple[Path, int]], device: str) -> dict | None:
    """FillerNet metrics on the test clips, in the exact shape of
    models/fillernet_metrics.json. None if no checkpoint exists yet."""
    if not CKPT.exists():
        return None
    model = load_checkpoint(CKPT, device)
    X, y = load_features(items, "test")
    return evaluate(model, X, y, device)


# ---------------------------------------------------------------------------
# (b) transcript-only baseline under the live-Chrome condition
# ---------------------------------------------------------------------------
def chrome_baseline(items: list[tuple[Path, int]], pause_ms: int) -> dict:
    """Trigger-level filler recall of the transcript-only StallDetector when
    the ASR strips fillers (the documented live-Chrome behaviour).

    Per filler clip we simulate the most FAVOURABLE case for the baseline:
    the speaker has already said >= 1 content word (the filler trigger's own
    precondition), then produces the uh/um at 1200-1700 ms.  The Chrome-
    condition transcript contains no token for it, so nothing reaches
    observe_word; a silence tick right after the filler ends (1800 ms) is
    still inside the pause window, confirming nothing else masks the result.
    """
    preamble = [Word("I", 0, 200), Word("want", 300, 550), Word("the", 700, 900)]

    n = fires_chrome = fires_verbatim = 0
    for path, _ in items:
        label = path.parent.name
        if label not in FILLER_LABELS:
            continue
        n += 1
        token = label.lower()  # the word actually spoken in the clip: "uh" / "um"

        # --- Chrome condition: filler-stripping ASR emits NO token ---------
        det = StallDetector(pause_ms=pause_ms)
        for w in preamble:
            det.observe_word(w)
        # (no Word arrives for the filler — that is the condition)
        tick = det.observe_silence(1800)  # 900 ms since last word < pause_ms
        if tick is not None and tick.trigger == "filler":
            fires_chrome += 1  # unreachable; counted honestly anyway

        # --- verbatim ablation: an ASR that DID emit the token -------------
        det = StallDetector(pause_ms=pause_ms)
        for w in preamble:
            det.observe_word(w)
        ev = det.observe_word(Word(token, 1200, 1700))
        if ev is not None and ev.trigger == "filler":
            fires_verbatim += 1

    return {
        "condition": "live Chrome / consumer ASR (filler tokens stripped from transcript)",
        "n_filler_clips": n,
        "filler_trigger_fires": fires_chrome,
        "recall": round(fires_chrome / n, 4) if n else None,
        "verbatim_asr_ablation": {
            "fires": fires_verbatim,
            "recall": round(fires_verbatim / n, 4) if n else None,
            "note": "if the ASR emitted the token, the detector catches it — "
                    "the bottleneck is the ASR, not the detector logic",
        },
        "note": "trigger-level recall only; the transcript-only system still catches "
                f"the stall later via the {pause_ms} ms pause timeout",
    }


# ---------------------------------------------------------------------------
# --wav-dir extension: self-recorded protocol set (eval/record_protocol.md)
# ---------------------------------------------------------------------------
def eval_wav_dir(wav_dir: Path, device: str) -> dict:
    """Run self-recorded utterances through the acoustic channel.

    Stub scope (honest): filler/prolongation files are scored on whether the
    expected AcousticEvent fires anywhere in the file; pause/hedge/fluent
    need a transcript replay through the full pipeline (STT), so they are
    reported UNSCORED here until the replay path is wired up.
    """
    import numpy as np
    import soundfile as sf

    from backend.acoustic.stream import AcousticStream

    per_file: list[dict] = []
    hits: Counter = Counter()
    totals: Counter = Counter()
    for wav in sorted(wav_dir.glob("*.wav")):
        m = _WAV_NAME_RE.match(wav.name)
        if not m:
            print(f"  SKIP {wav.name}: does not match <speaker>_<category>_<idx>.wav")
            continue
        x, sr = sf.read(wav, dtype="float32")
        if sr != SR:
            print(f"  SKIP {wav.name}: {sr} Hz != 16000 Hz (re-record per protocol)")
            continue
        if x.ndim > 1:
            x = x.mean(axis=1)

        stream = AcousticStream(model_path=str(CKPT) if CKPT.exists() else None,
                                device=device)
        pcm16 = (np.clip(x, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        events = []
        for i in range(0, len(pcm16), 640):  # 20 ms chunks, like the live socket
            events.extend(stream.feed(pcm16[i:i + 640]))

        cat = m["category"]
        kinds = sorted({e.kind for e in events})
        if cat == "filler":
            scored = ("PENDING (no checkpoint)" if not CKPT.exists()
                      else "filler" in kinds)
        elif cat == "prolongation":
            scored = "prolongation" in kinds
        else:
            scored = "UNSCORED (requires STT transcript replay)"
        if isinstance(scored, bool):
            totals[cat] += 1
            hits[cat] += int(scored)
        per_file.append({"file": wav.name, "category": cat,
                         "acoustic_events": kinds, "expected_event_fired": scored})
        print(f"  {wav.name:32s} {cat:13s} events={kinds} -> {scored}")

    return {
        "n_files": len(per_file),
        "acoustic_hit_rate": {c: round(hits[c] / totals[c], 4) for c in totals},
        "files": per_file,
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--device", choices=["cuda", "cpu"],
                    default="cuda" if torch.cuda.is_available() else "cpu",
                    help="device for FillerNet inference (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap evaluated test clips (smoke run)")
    ap.add_argument("--wav-dir", type=Path, default=None,
                    help="also evaluate the self-recorded set (record_protocol.md)")
    args = ap.parse_args()
    torch.manual_seed(13)  # determinism (eval path has no sampling, belt+braces)

    t0 = time.time()
    items = scan_split(["test"], limit=args.limit)
    counts = Counter(p.parent.name for p, _ in items)
    print(f"test split clips found: {sum(counts.values())} "
          f"({'partial dataset OK' if counts else 'NONE -- dataset still downloading?'})")
    for label in sorted(counts):
        print(f"  {label:10s} n={counts[label]}")

    out: dict = {
        "split": "test",
        "limit": args.limit or None,
        "device": args.device,
        "test_split_counts": dict(sorted(counts.items())),
    }

    # (a) acoustic clip-level metrics
    acoustic = eval_acoustic(items, args.device) if items else None
    if acoustic is None:
        why = "no test clips" if not items else f"no checkpoint at {CKPT}"
        print(f"\n[acoustic] PENDING -- {why}")
        out["acoustic_clip_metrics"] = f"PENDING ({why})"
    else:
        out["acoustic_clip_metrics"] = acoustic
        fb = acoustic["filler_binary"]
        print(f"\n[acoustic] binary filler (uh-union-um): "
              f"P={fb['precision']:.3f} R={fb['recall']:.3f} F1={fb['f1']:.3f}")
        for cls, m in acoustic["per_class"].items():
            print(f"  {cls:7s} F1={m['f1']:.3f} P={m['precision']:.3f} "
                  f"R={m['recall']:.3f} n={m['n']}")

    # (b) transcript-only baseline, Chrome condition
    from backend.config import get_settings

    pause_ms = get_settings().pause_ms
    baseline = chrome_baseline(items, pause_ms=pause_ms)
    out["transcript_baseline"] = baseline
    print(f"\n[baseline] transcript-only filler recall (Chrome condition): "
          f"{baseline['recall']} on n={baseline['n_filler_clips']} filler clips "
          f"(verbatim-ASR ablation recall: {baseline['verbatim_asr_ablation']['recall']})")

    # optional: self-recorded set
    if args.wav_dir is not None:
        if args.wav_dir.is_dir():
            print(f"\n[wav-dir] {args.wav_dir}")
            out["self_recorded"] = eval_wav_dir(args.wav_dir, args.device)
        else:
            print(f"\n[wav-dir] {args.wav_dir} not found -- skipped")
            out["self_recorded"] = f"PENDING (dir {args.wav_dir} not found)"

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
