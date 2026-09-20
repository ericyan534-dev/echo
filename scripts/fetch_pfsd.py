"""Fetch the PodcastFillers clip dataset by streaming full episodes from the
HuggingFace mirror and cutting the official 85,803 one-second clips locally
using the official Zenodo annotation CSV.

Why this route: the Zenodo release is a 25GB multi-part zip; the HF mirror
(ylacombe/podcast_fillers_by_license, ungated) holds the same 199 episodes as
~8.3GB parquet. Clips are deterministic cuts (clip_start/end_inepisode), so the
result reproduces the official clip set, labels, and train/val/test splits.

Output layout:
    data/pfsd/clips/<split>/<label>/<clip_name>.wav   (16 kHz mono PCM16, 1.0 s)
    data/pfsd/done_episodes.txt                        (resume manifest)

Usage:
    python scripts/fetch_pfsd.py             # full run (background it)
    python scripts/fetch_pfsd.py --limit 2   # smoke test on 2 episodes

License note: PodcastFillers annotations are non-commercial (research/education
use OK); episode audio is CC BY / CC BY-SA / CC BY-ND 3.0.
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "data" / "pfsd" / "PodcastFillers.csv"
CLIPS_DIR = ROOT / "data" / "pfsd" / "clips"
DONE_PATH = ROOT / "data" / "pfsd" / "done_episodes.txt"

SR = 16_000
CLIP_SAMPLES = SR  # 1.0 s

HF_DATASET = "ylacombe/podcast_fillers_by_license"
HF_SPLITS = ["CC_BY_3.0", "CC_BY_SA_3.0", "CC_BY_ND_3.0"]


def norm_name(name: str) -> str:
    """Normalize episode names for joining CSV <-> mirror (case/punct-tolerant)."""
    name = re.sub(r"\.(mp3|wav|flac|ogg|m4a)$", "", name, flags=re.I)
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def load_clip_index() -> dict[str, list[dict]]:
    by_episode: dict[str, list[dict]] = defaultdict(list)
    with open(CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_episode[norm_name(row["podcast_filename"])].append(row)
    return by_episode


def decode_audio(b: bytes) -> tuple[np.ndarray, int]:
    data, sr = sf.read(io.BytesIO(b), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr  # mono


def resample_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SR:
        return x
    import torch
    import torchaudio.functional as AF

    t = torch.from_numpy(x).unsqueeze(0)
    return AF.resample(t, sr, SR).squeeze(0).numpy()


def cut_episode(audio16k: np.ndarray, clips: list[dict]) -> int:
    written = 0
    n = len(audio16k)
    for row in clips:
        split = row["clip_split_subset"]  # train / validation / test / extra
        label = row["label_consolidated_vocab"]  # Uh/Um/Words/None/Breath/Laughter/Music
        out = CLIPS_DIR / split / label / row["clip_name"]
        if out.exists():
            continue
        start = int(float(row["clip_start_inepisode"]) * SR)
        end = start + CLIP_SAMPLES
        if start < 0 or start >= n:
            continue
        seg = audio16k[start:min(end, n)]
        if len(seg) < CLIP_SAMPLES:  # pad tail clips to exactly 1.0 s
            seg = np.pad(seg, (0, CLIP_SAMPLES - len(seg)))
        out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, seg, SR, subtype="PCM_16")
        written += 1
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="stop after N episodes (smoke test)")
    args = ap.parse_args()

    if not CSV_PATH.exists():
        print(f"FATAL: {CSV_PATH} missing -- download PodcastFillers.csv from Zenodo first.")
        return 2

    from datasets import Audio, load_dataset

    index = load_clip_index()
    done: set[str] = set()
    if DONE_PATH.exists():
        done = set(DONE_PATH.read_text(encoding="utf-8").splitlines())
    print(f"clip index: {sum(len(v) for v in index.values())} clips across "
          f"{len(index)} episodes; {len(done)} episodes already done", flush=True)

    processed = matched = total_written = 0
    unmatched: list[str] = []

    for hf_split in HF_SPLITS:
        ds = load_dataset(HF_DATASET, split=hf_split, streaming=True)
        ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes; we decode
        for ex in ds:
            ep_raw = ex.get("episode_name") or ex.get("file_name") or ""
            key = norm_name(ep_raw)
            if key in done:
                continue
            clips = index.get(key)
            if clips is None:
                unmatched.append(ep_raw)
                print(f"  [unmatched] {ep_raw.encode('ascii', 'replace').decode()!r}", flush=True)
                continue
            matched += 1
            audio_field = ex["audio"]
            b = audio_field["bytes"] if isinstance(audio_field, dict) else audio_field.read()
            x, sr = decode_audio(b)
            x16 = resample_16k(x, sr)
            w = cut_episode(x16, clips)
            total_written += w
            processed += 1
            with open(DONE_PATH, "a", encoding="utf-8") as f:
                f.write(key + "\n")
            safe = ep_raw.encode("ascii", "replace").decode()  # GBK console safety
            print(f"[{processed:3d}] {safe[:60]:60s} clips+{w:5d} (total {total_written})",
                  flush=True)
            if args.limit and processed >= args.limit:
                print("limit reached -- smoke test OK")
                return 0

    print(f"\nDONE: episodes processed={processed} matched={matched} "
          f"unmatched={len(unmatched)} clips written={total_written}")
    if unmatched:
        print("unmatched episodes:", *unmatched[:20], sep="\n  ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
