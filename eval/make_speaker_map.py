"""Per-segment wearer confidence for each APROCSA region, cached to disk.

Echo fires during the clinician's speech on up to 0.465 of their utterances and
puts their words into the fragment it sends to the predictor. This builds the
evidence needed to stop that: a time-indexed wearer confidence, from a speaker
embedding rather than from loudness.

TWO RULES THAT KEEP THIS FROM BEING CHEATING
--------------------------------------------
1. ENROLMENT COMES FROM BEFORE THE SCORED WINDOW. The wearer's centroid is
   built from participant speech in the first `skip_s` seconds, which no metric
   is computed on. That mirrors what a real deployment has -- one enrolment at
   setup -- rather than borrowing labels from the region under test.

2. SEGMENTATION IS THE VAD's, NOT THE TRANSCRIPT's. It would be easier to cut
   the region on CHAT utterance boundaries, but those are a clinician's
   judgement about who spoke when, which is half the problem being solved.
   Silero decides where speech starts and stops, exactly as it would live.

The CHAT transcript is used for one thing only: locating the wearer's speech
during ENROLMENT, before the window. Inside the window it is never consulted
until scoring.

    python eval/make_speaker_map.py --region-s 300 --skip-s 120
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from backend.acoustic.speaker_id import SR, SpeakerID  # noqa: E402
from backend.acoustic.stream import VAD_CHUNK, _get_vad_instance  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

AUDIO = ROOT / "data" / "aprocsa" / "audio"
TRANSCRIPTS = ROOT / "data" / "aprocsa" / "transcripts"
CACHE = ROOT / "eval" / "results" / "cache" / "aprocsa"

MIN_SEG_MS = 700
MAX_SEG_MS = 8000
GAP_MS = 320             # silence this long ends a segment


def read(pid: str, t0_ms: int, t1_ms: int):
    wav = AUDIO / ("%s.wav" % pid)
    if not wav.exists():
        return None
    info = sf.info(str(wav))
    a0 = max(0, int(t0_ms * info.samplerate / 1000))
    a1 = min(info.frames, int(t1_ms * info.samplerate / 1000))
    if a1 <= a0:
        return None
    a, sr = sf.read(str(wav), start=a0, stop=a1, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
        a = np.interp(idx, np.arange(len(a)), a).astype("float32")
    return a


def vad_segments(audio: np.ndarray, offset_ms: int) -> list[tuple[int, int]]:
    vad = _get_vad_instance()
    chunk_ms = VAD_CHUNK * 1000 // SR
    voiced = []
    for i in range(0, len(audio) - VAD_CHUNK, VAD_CHUNK):
        with torch.no_grad():
            p = float(vad(torch.from_numpy(audio[i:i + VAD_CHUNK].copy()), SR).item())
        voiced.append(p >= 0.5)
    segs, start, gap = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += chunk_ms
            if gap >= GAP_MS:
                a, b = start * chunk_ms, (i * chunk_ms) - gap
                if b - a >= MIN_SEG_MS:
                    segs.append((offset_ms + a, offset_ms + min(b, a + MAX_SEG_MS)))
                start, gap = None, 0
    if start is not None:
        a, b = start * chunk_ms, len(voiced) * chunk_ms
        if b - a >= MIN_SEG_MS:
            segs.append((offset_ms + a, offset_ms + min(b, a + MAX_SEG_MS)))
    return segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-s", type=int, default=300)
    ap.add_argument("--skip-s", type=int, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-enrol", type=int, default=10)
    args = ap.parse_args()

    if not AUDIO.is_dir():
        print("SKIPPED -- no APROCSA audio")
        return 0
    parsed = load_all(TRANSCRIPTS)

    for pid in sorted(parsed):
        if not (AUDIO / ("%s.wav" % pid)).exists():
            continue
        sid = SpeakerID(device=args.device)

        # --- enrolment, strictly before the scored window ---
        enrol_clips = []
        for u in parsed[pid]["utterances"]:
            if not u["is_participant"] or u["start_ms"] is None:
                continue
            if u["end_ms"] > args.skip_s * 1000:
                break
            if u["end_ms"] - u["start_ms"] < 900:
                continue
            a = read(pid, u["start_ms"], min(u["end_ms"], u["start_ms"] + 6000))
            if a is not None:
                enrol_clips.append(a)
            if len(enrol_clips) >= args.max_enrol:
                break
        if not sid.enroll(enrol_clips):
            print("  %s: enrolment FAILED (%d clips before %ds) -- map withheld"
                  % (pid, len(enrol_clips), args.skip_s))
            continue

        # --- the scored region, segmented by VAD alone ---
        off = args.skip_s * 1000
        audio = read(pid, off, off + args.region_s * 1000)
        if audio is None:
            continue
        segs = vad_segments(audio, off)
        clips = [audio[int((a - off) * SR / 1000):int((b - off) * SR / 1000)]
                 for a, b in segs]
        conf = sid.attribute_session(clips)

        rows = [{"t0": a, "t1": b, "conf": (None if c is None else round(c, 4))}
                for (a, b), c in zip(segs, conf)]
        known = [r for r in rows if r["conf"] is not None]
        CACHE.mkdir(parents=True, exist_ok=True)
        out = CACHE / ("spk_%s_%d_%d.json" % (pid, args.skip_s, args.region_s))
        out.write_text(json.dumps({
            "participant": pid, "n_enrol_clips": len(enrol_clips),
            "enrol_window_s": [0, args.skip_s],
            "n_segments": len(rows), "n_attributed": len(known),
            "method": ("ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb), spherical "
                       "k-means over the session, cluster nearest the enrolment "
                       "centroid = wearer"),
            "segments": rows}, indent=2), encoding="utf-8")
        hi = sum(1 for r in known if r["conf"] >= 0.5)
        print("  %s: %d enrol clips, %d segments, %d attributed, %d look like the wearer"
              % (pid, len(enrol_clips), len(rows), len(known), hi), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
