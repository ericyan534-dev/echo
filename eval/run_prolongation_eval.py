"""Prolongation-rule validation on real PFSD audio. No network, no training.

The prolongation detector (backend/acoustic/prolongation.py) is rule-based, so
it is not part of the FillerNet classification report. This script measures the
numbers that matter for it, all grounded in REAL PodcastFillers audio:

(a) DETECTION rate -- real filler-vowel material sustained into a prolongation.
    PFSD has no labelled prolongations, so we synthesize them the way a real
    prolongation is produced: cycle through the VOICED FRAMES of a real Uh/Um
    clip in order (looped to >=1.5 s), feeding consecutive distinct 50 ms frames
    to ProlongationTracker. Consecutive frames have natural frame-to-frame
    acoustic variation (~0.95-0.98 cos-sim per the code comments), NOT sim=1.0
    from tiling a single identical frame. Reported as fired/total over N distinct
    source clips. NOTE: still synthetic (constructed from real vowel frames) --
    a labelled prolongation corpus would give a ground-truth rate.

(b) FALSE-FIRE rate (Words) -- real running speech must NOT trigger the rule.
    Concatenated real PFSD "Words" (lexical speech) clips fed through the tracker
    frame by frame. Running speech changes phones every ~100-150 ms, breaking
    the similarity streak, so the expectation is ~0 fires.

(c) FALSE-FIRE rate (Music) -- music has near-static spectral envelopes, the
    known false-fire hazard for the mel-envelope cosine rule. Concatenated real
    PFSD "Music" clips fed through the tracker. A non-zero rate here is expected
    and should be disclosed.

(d) STREAM-LEVEL FillerNet FALSE-ALARM -- feed concatenated Words clips through
    the FULL AcousticStream path (VAD gate + FillerNet + confidence gate, 20 ms
    chunks like the live socket) and report filler-event fires per minute of
    fluent speech. Uses whatever checkpoint is at models/fillernet.pt.

Usage:  python eval/run_prolongation_eval.py [--clips N] [--speech-seconds S]
                                              [--stream-seconds S]
Output: eval/results/prolongation_eval.json
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

from backend.acoustic.prolongation import ENERGY_FLOOR, ProlongationTracker  # noqa: E402

SR = 16_000
FRAME = 800       # 50 ms @ 16 kHz -- the tracker's frame size
N_SUSTAIN = 30    # 30 * 50 ms = 1.5 s minimum sustained vowel for detection test
CKPT = ROOT / "models" / "fillernet.pt"
CLIPS = ROOT / "data" / "pfsd" / "clips" / "test"
RESULTS = ROOT / "eval" / "results" / "prolongation_eval.json"


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"{path} is {sr} Hz, expected {SR}"
    return x if x.ndim == 1 else x.mean(axis=1)


# ---------------------------------------------------------------------------
# (a) detection -- looped voiced frames (natural jitter, not tautological tile)
# ---------------------------------------------------------------------------
def eval_detection(n_clips: int, tracker_factory=ProlongationTracker) -> dict:
    """Score detection rate with HONEST construction: cycle voiced frames of
    real Uh/Um clips in order so consecutive frames have natural frame-to-frame
    variation, not sim=1.0 from tiling a single identical frame."""
    src = sorted((CLIPS / "Uh").glob("*.wav")) + sorted((CLIPS / "Um").glob("*.wav"))
    if len(src) < 5:
        return {"status": "SKIPPED (test split Uh/Um clips not downloaded yet)"}

    rng = np.random.default_rng(13)
    idx = rng.permutation(len(src))[: min(n_clips, len(src))]
    fired = quiet = 0
    for i in idx:
        x = _load_wav(src[int(i)])
        n_frames = len(x) // FRAME
        if n_frames == 0:
            quiet += 1
            continue
        frames = x[: n_frames * FRAME].reshape(n_frames, FRAME)
        rms = np.sqrt((frames ** 2).mean(axis=1))

        # Isolate voiced frames (above energy floor) so we loop only the
        # energetic vowel content and not silence at clip boundaries.
        voiced_mask = rms >= ENERGY_FLOOR
        voiced_frames = frames[voiced_mask]
        if len(voiced_frames) == 0:
            quiet += 1
            continue

        # Build a >=1.5 s sustained vowel by PALINDROME-cycling the voiced
        # frames (0,1,..,N-1,N-2,..,1,0,1,..). Every consecutive pair in this
        # sequence is a pair of frames that were adjacent in the real clip, so
        # every transition carries the real frame-to-frame jitter
        # (~0.95-0.98 cos-sim). A plain in-order loop would insert a spectral
        # discontinuity at every wrap (last frame -> first frame); with 1 s
        # source clips (< 14 voiced frames) that discontinuity lands inside
        # every possible 700 ms window, structurally guaranteeing failure --
        # as rigged-to-fail as tiling one identical frame is rigged-to-pass.
        n_voiced = len(voiced_frames)
        if n_voiced < 2:
            # palindrome of one frame degenerates to tiling (sim=1.0 tautology)
            quiet += 1
            continue
        cycle = list(range(n_voiced)) + list(range(n_voiced - 2, 0, -1))
        sustained = np.stack(
            [voiced_frames[cycle[k % len(cycle)]] for k in range(N_SUSTAIN)])

        tracker = tracker_factory()
        hit = False
        for k in range(N_SUSTAIN):
            frame_t = torch.from_numpy(sustained[k].copy())
            if tracker.observe_frame(frame_t, now_ms=k * 50):
                hit = True
                break
        fired += int(hit)

    scored = len(idx) - quiet
    return {
        "status": "ok" if scored else "SKIPPED (no usable vowel frames)",
        "construction": "palindrome-looped real vowel frames -- natural jitter, no wrap discontinuity",
        "n_source_clips": int(len(idx)),
        "scored": int(scored),
        "skipped_quiet": int(quiet),
        "fired": int(fired),
        "detection_rate": round(fired / scored, 4) if scored else None,
        "note": (
            "Voiced 50 ms frames of real Uh/Um clips palindrome-cycled to "
            f"{N_SUSTAIN * 50} ms, so EVERY transition is between frames that "
            "were adjacent in the real clip (natural jitter, cos-sim "
            "~0.95-0.98), with no wrap discontinuity and no sim=1.0 tiling. "
            "Still synthetic: PFSD has no labelled prolongations, and a "
            "conversational um/uh is not a deliberately held vowel; the "
            "self-recorded set (eval/record_protocol.md) is the ground-truth "
            "path. Tracker-level."
        ),
    }


# ---------------------------------------------------------------------------
# (b) false fires on real running speech (Words)
# ---------------------------------------------------------------------------
def eval_false_fires_speech(target_seconds: float,
                            tracker_factory=ProlongationTracker) -> dict:
    words = sorted((CLIPS / "Words").glob("*.wav"))
    if len(words) < 20:
        return {"status": "SKIPPED (test split Words clips not downloaded yet)"}

    rng = np.random.default_rng(13)
    order = rng.permutation(len(words))
    chunks: list[np.ndarray] = []
    total = 0
    for i in order:
        x = _load_wav(words[int(i)])
        chunks.append(x)
        total += len(x)
        if total >= target_seconds * SR:
            break
    stream = np.concatenate(chunks)
    n_frames = len(stream) // FRAME
    frames = stream[: n_frames * FRAME].reshape(n_frames, FRAME)

    tracker = tracker_factory()
    fires = 0
    for k in range(n_frames):
        if tracker.observe_frame(torch.from_numpy(frames[k].copy()), now_ms=k * 50):
            fires += 1
    seconds = round(n_frames * FRAME / SR, 1)

    return {
        "status": "ok",
        "running_speech_seconds": seconds,
        "n_clips_concatenated": len(chunks),
        "false_fires": int(fires),
        "false_fire_rate_per_min": round(fires / (seconds / 60.0), 3) if seconds else None,
        "note": (
            "real PFSD 'Words' (lexical speech) concatenated into a "
            "continuous stream; running speech changes phones every "
            "~100-150 ms, breaking the envelope-similarity streak, so "
            "expectation is ~0 fires"
        ),
    }


# ---------------------------------------------------------------------------
# (c) false fires on music (near-static envelopes -- known false-fire hazard)
# ---------------------------------------------------------------------------
def eval_false_fires_music(target_seconds: float,
                           tracker_factory=ProlongationTracker) -> dict:
    music_clips = sorted((CLIPS / "Music").glob("*.wav"))
    if len(music_clips) < 5:
        return {"status": "SKIPPED (test split Music clips not downloaded yet)"}

    rng = np.random.default_rng(13)
    order = rng.permutation(len(music_clips))
    chunks: list[np.ndarray] = []
    total = 0
    for i in order:
        x = _load_wav(music_clips[int(i)])
        chunks.append(x)
        total += len(x)
        if total >= target_seconds * SR:
            break
    stream = np.concatenate(chunks)
    n_frames = len(stream) // FRAME
    frames = stream[: n_frames * FRAME].reshape(n_frames, FRAME)

    tracker = tracker_factory()
    fires = 0
    for k in range(n_frames):
        if tracker.observe_frame(torch.from_numpy(frames[k].copy()), now_ms=k * 50):
            fires += 1
    seconds = round(n_frames * FRAME / SR, 1)

    return {
        "status": "ok",
        "music_seconds": seconds,
        "n_clips_concatenated": len(chunks),
        "false_fires": int(fires),
        "false_fire_rate_per_min": round(fires / (seconds / 60.0), 3) if seconds else None,
        "note": (
            "real PFSD 'Music' clips concatenated; music has near-static "
            "spectral envelopes -- the known false-fire hazard for the "
            "mel-envelope cosine rule. A non-zero false-fire rate here is "
            "expected and should be disclosed. The live AcousticStream applies "
            "a VAD gate (min_voiced_ms=800) which Music clips will not pass "
            "in the real path, limiting real-world exposure."
        ),
    }


# ---------------------------------------------------------------------------
# (d) stream-level FillerNet false-alarm (full AcousticStream path)
# ---------------------------------------------------------------------------
def eval_stream_falsefire(target_seconds: float,
                          conf_thresh: float | None = None) -> dict:
    """Feed real Words clips through the full AcousticStream pipeline and count
    filler events per minute of fluent speech."""
    words = sorted((CLIPS / "Words").glob("*.wav"))
    if len(words) < 20:
        return {"status": "SKIPPED (test split Words clips not downloaded yet)"}
    if not CKPT.exists():
        return {"status": f"SKIPPED (no FillerNet checkpoint at {CKPT} -- train first)"}

    try:
        from backend.acoustic.stream import AcousticStream  # noqa: E402
    except Exception as exc:  # noqa: BLE001
        return {"status": f"SKIPPED (import error: {exc})"}

    rng = np.random.default_rng(42)
    order = rng.permutation(len(words))
    chunks: list[np.ndarray] = []
    total = 0
    for i in order:
        x = _load_wav(words[int(i)])
        chunks.append(x)
        total += len(x)
        if total >= target_seconds * SR:
            break

    audio = np.concatenate(chunks)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()

    kwargs = {} if conf_thresh is None else {"conf_thresh": conf_thresh}
    stream = AcousticStream(model_path=str(CKPT), device="cpu", **kwargs)
    conf_used = stream.conf_thresh
    filler_events = 0
    for j in range(0, len(pcm16), 640):   # 20 ms chunks like the live socket
        evs = stream.feed(pcm16[j:j + 640])
        filler_events += sum(1 for e in evs if e.kind == "filler")

    seconds = round(len(audio) / SR, 1)
    return {
        "status": "ok",
        "speech_seconds": seconds,
        "n_clips_concatenated": len(chunks),
        "filler_events": int(filler_events),
        "filler_events_per_min": round(filler_events / (seconds / 60.0), 3) if seconds else None,
        "note": (
            "Full AcousticStream path: VAD gate + FillerNet + confidence gate "
            f"(min_voiced_ms=800, conf_thresh={conf_used}), fed 20 ms chunks. Source: "
            "real PFSD 'Words' (lexical speech). Caveat: concatenating 1 s "
            "clips from many speakers inserts a speaker/segment boundary "
            "every second, which likely inflates the rate vs one continuous "
            "speaker; treat as a conservative upper bound."
        ),
    }


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--clips", type=int, default=40,
                    help="source Uh/Um clips for the detection test")
    ap.add_argument("--speech-seconds", type=float, default=120.0,
                    help="target seconds for the Words and Music false-fire tests")
    ap.add_argument("--stream-seconds", type=float, default=60.0,
                    help="target seconds for the stream-level FillerNet false-alarm bench")
    args = ap.parse_args()
    t0 = time.time()

    detection = eval_detection(args.clips)
    print(f"[detection]         {detection.get('status')}: "
          f"{detection.get('fired', 0)}/{detection.get('scored', 0)} fired "
          f"(rate {detection.get('detection_rate')}) "
          f"[{detection.get('construction', 'unknown construction')}]")

    ff_speech = eval_false_fires_speech(args.speech_seconds)
    print(f"[false-fire/speech] {ff_speech.get('status')}: "
          f"{ff_speech.get('false_fires', 0)} fires in "
          f"{ff_speech.get('running_speech_seconds', 0)} s "
          f"({ff_speech.get('false_fire_rate_per_min', 'n/a')}/min)")

    ff_music = eval_false_fires_music(args.speech_seconds)
    print(f"[false-fire/music]  {ff_music.get('status')}: "
          f"{ff_music.get('false_fires', 0)} fires in "
          f"{ff_music.get('music_seconds', 0)} s "
          f"({ff_music.get('false_fire_rate_per_min', 'n/a')}/min) "
          f"[known hazard: near-static envelopes]")

    stream_ff = eval_stream_falsefire(args.stream_seconds)
    print(f"[stream-falsefire]  {stream_ff.get('status')}: "
          f"{stream_ff.get('filler_events', 0)} filler events in "
          f"{stream_ff.get('speech_seconds', 0)} s "
          f"({stream_ff.get('filler_events_per_min', 'n/a')}/min)")

    _t = ProlongationTracker()  # report the shipped defaults, not stale copies
    out = {
        "thresholds": {"sim_thresh": _t.sim_thresh, "min_ms": _t.min_ms,
                       "energy_floor": ENERGY_FLOOR},
        "detection": detection,
        "false_fires_speech": ff_speech,
        "false_fires_music": ff_music,
        "stream_falsefire": stream_ff,
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {RESULTS}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
