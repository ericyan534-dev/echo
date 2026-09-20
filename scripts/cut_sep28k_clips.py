"""Cut SEP-28k clips out of the recovered episodes into one memmap-able array.

WHY NOT 20,000 LITTLE WAV FILES
-------------------------------
That is what Apple's extract_clips.py does, and on Windows it is the slowest
possible layout: 20k opens per epoch, each a few hundred KB, all of them
seeking. Training reads every clip every epoch, so the decode cost gets paid
over and over for data that never changes. One int16 array of shape
(N, 48000) is 1.9 GB, memory-maps in constant time, and slices without a
decoder in the loop.

THE SAMPLE-RATE TRAP
--------------------
SEP-28k Start/Stop are offsets in 16 kHz SAMPLES (Stop-Start == 48000 == 3.0 s
for every row). The episodes are whatever the podcast shipped -- 32 kHz and
16 kHz both occur in this corpus. Seeking with the raw offsets lands in the
wrong part of the episode at 32 kHz, silently, producing a plausible-looking
clip of the wrong audio. This measured a 0-event result in an earlier eval
before it was caught. Every seek here is rescaled by the file's own rate.

    python scripts/cut_sep28k_clips.py
    python scripts/cut_sep28k_clips.py --limit 2000   # quick smoke run
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sep28k_audio import episode_path  # noqa: E402

DATA = ROOT / "data" / "sep28k"
EP_DIR = DATA / "episodes"
MANIFEST = DATA / "manifest.json"
OUT_NPY = DATA / "clips_16k.npy"
OUT_IDX = DATA / "clips_index.json"

SR = 16_000
CLIP_SAMPLES = 3 * SR          # every SEP-28k clip is exactly 3.00 s
TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    n = int(round(len(x) * SR / sr))
    idx = np.linspace(0, len(x) - 1, n)
    return np.interp(idx, np.arange(len(x)), x).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="cut at most N clips (smoke run)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print("MISSING %s -- run scripts/fetch_sep28k.py first" % MANIFEST)
        return 1
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    clips = [c for c in man["clips"] if c["usable"]]
    if args.limit:
        clips = clips[:args.limit]
    print("SEP-28k clip cutting")
    print("  usable clips in manifest: %d" % len(clips))

    by_ep: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in clips:
        by_ep[(c["show"], c["ep"])].append(c)
    print("  episodes to open: %d" % len(by_ep))

    # Pre-allocate on disk; write in place so peak RAM stays at one episode.
    # Written to a .part file because the final row count is only known after
    # decoding (link rot, out-of-range offsets), and on Windows a memory-mapped
    # file cannot be reopened for writing while the map is alive -- so the
    # truncation has to be a copy into a second file, not an overwrite.
    tmp_npy = OUT_NPY.with_suffix(".part.npy")
    arr = np.lib.format.open_memmap(
        tmp_npy, mode="w+", dtype="int16", shape=(len(clips), CLIP_SAMPLES))

    index: list[dict] = []
    written = 0
    skipped_missing = 0
    skipped_short = 0
    for i, ((show, ep), group) in enumerate(sorted(by_ep.items()), 1):
        # Not `EP_DIR/show/ep.mp3`: 27 of the 258 recovered episodes are M4A
        # served under an .mp3 URL and were being skipped here in silence,
        # taking out all of HVSA and most of MyStutteringLife. See
        # scripts/sep28k_audio.py.
        path = episode_path(show, ep)
        if path is None:
            skipped_missing += len(group)
            continue
        try:
            info = sf.info(str(path))
            audio, file_sr = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            skipped_missing += len(group)
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        scale = file_sr / SR            # THE trap -- see module docstring

        for c in group:
            a0 = int(round(c["start"] * scale))
            a1 = a0 + int(round(CLIP_SAMPLES * scale))
            if a0 < 0 or a1 > len(audio):
                skipped_short += 1
                continue
            seg = resample_to_16k(audio[a0:a1], file_sr)
            if len(seg) < CLIP_SAMPLES:
                seg = np.pad(seg, (0, CLIP_SAMPLES - len(seg)))
            seg = seg[:CLIP_SAMPLES]
            peak = float(np.abs(seg).max())
            arr[written] = np.clip(seg * 32767.0, -32768, 32767).astype("int16")
            index.append({
                "row": written, "show": show, "ep": ep, "clip": c["clip"],
                "labels": [int(c["labels"][t]) for t in TYPES],
                "any": int(c["any_dysfluency"]),
                "no_stuttered_words": int(c.get("no_stuttered_words", 0)),
                "peak": round(peak, 4),
            })
            written += 1

        if i % 20 == 0:
            print("    %d/%d episodes, %d clips written"
                  % (i, len(by_ep), written), flush=True)

    arr.flush()
    del arr
    # Truncate to what was actually written, copying in chunks so peak RAM
    # stays at one chunk rather than the whole 1.9 GB array.
    src = np.load(tmp_npy, mmap_mode="r")
    dst = np.lib.format.open_memmap(
        OUT_NPY, mode="w+", dtype="int16", shape=(written, CLIP_SAMPLES))
    for lo in range(0, written, 2000):
        hi = min(lo + 2000, written)     # src is the full pre-allocation; dst is not
        dst[lo:hi] = src[lo:hi]
    dst.flush()
    del dst
    src._mmap.close()          # Windows: the map must close before the unlink
    del src
    tmp_npy.unlink(missing_ok=True)

    per_type = {t: sum(r["labels"][k] for r in index) for k, t in enumerate(TYPES)}
    OUT_IDX.write_text(json.dumps({
        "n": written, "sample_rate": SR, "clip_samples": CLIP_SAMPLES,
        "types": TYPES, "per_type": per_type,
        "min_agree": man.get("min_agree"),
        "skipped_missing_episode": skipped_missing,
        "skipped_out_of_range": skipped_short,
        "provenance": ("SEP-28k, Apple official labels, 74% episode recovery. "
                       "See docs/DATA_PROVENANCE.md -- three shows are entirely "
                       "absent, so this is a BIASED subset and must be described "
                       "as 'SEP-28k (74% subset, 5 of 8 shows)'."),
        "rows": index,
    }, indent=2), encoding="utf-8")

    print("")
    print("  wrote %d clips -> %s (%.1f GB)"
          % (written, OUT_NPY.name, OUT_NPY.stat().st_size / 1e9))
    print("  skipped: %d (episode missing), %d (offset past end of file)"
          % (skipped_missing, skipped_short))
    print("  positives per type at >=%s/3 agreement:" % man.get("min_agree"))
    for t in TYPES:
        print("    %-14s %6d  (%.1f%%)" % (t, per_type[t], 100 * per_type[t] / max(1, written)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
