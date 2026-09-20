"""Recover the SEP-28k clips that link rot took, plus FluencyBank, from a
HuggingFace mirror that was checked against Apple's labels before it was used.

READ THIS BEFORE YOU TRUST IT
-----------------------------
docs/DATA_PROVENANCE.md records that the first version of this project used a
third-party HuggingFace repackaging of SEP-28k and reported ITS schema as the
dataset's. That was wrong, and the rule that came out of it is
"convenience mirrors are not sources". This script does not ask you to relax
that rule. It asks you to apply it: the mirror is used only after every row it
contains has been joined to Apple's official CSVs and found identical.

What is being mirrored:

    saeedzou/sep28k-fluencybank-stutter-dataset   31,908 rows, 2.87 GB parquet

WHAT WAS VERIFIED, AND HOW (2026-08-18)
---------------------------------------
1. LABELS. All 31,908 rows joined to the union of Apple's SEP-28k_labels.csv
   (28,177) and fluencybank_labels.csv (4,144) on (Show, EpId, ClipId).
   31,908 / 31,908 keys found. All 12 annotator-count columns equal on every
   row. Start and Stop equal on every row. Zero mismatches, zero extra keys.
   413 official rows are absent from the mirror, spread over 7 episodes that
   are otherwise present -- it is a subset of Apple's labels, never a
   contradiction of them. This script re-runs that join and REFUSES to write
   anything if the match rate is not 100%.

2. AUDIO IS REALLY THERE, and is really 16 kHz. The `audio` column is
   struct<bytes: binary, path: string> carrying HF's Audio(sampling_rate=16000)
   feature metadata -- but feature metadata is a decode directive, not a fact
   about the bytes. The bytes were decoded: every clip is RIFF/PCM, 16000 Hz,
   1 channel, 16-bit, exactly 48000 frames = 3.000 s, 96044 bytes. This script
   asserts all five of those per clip and counts any that fail.

3. THE AUDIO IS THE AUDIO WE ALREADY HAVE, where we can check. 98 clips from
   HeStutters episodes 0 and 1 were cross-correlated against the same clips cut
   locally from our own downloaded episodes: mean best-lag correlation 0.996,
   median 0.998, all 98 above 0.94, best lag 0 samples for every one. The
   mirror is cut from the same episodes at the same offsets. (It is not
   bit-identical because the two copies went through different MP3 decoders.)

4. THE SHOWS WE COULD NOT CHECK LOCALLY WERE CHECKED AGAINST THEMSELVES.
   StutteringIsCool, StrongVoices, IStutterSoWhat and FluencyBank cannot be
   compared to a local copy, because not having them is the entire point. But
   SEP-28k contains clip windows that OVERLAP in time within an episode, and
   an overlap is a self-consistency test no repackaging can fake: if the audio
   were misaligned, wrong, or reused, the overlapping samples of two different
   clips would not agree. See `--verify` for the check.

WHAT THIS DOES AND DOES NOT FIX
-------------------------------
Fixes: the three shows lost to link rot (StutteringIsCool 4,013 clips,
StrongVoices 2,272, IStutterSoWhat 870) and FluencyBank (3,986), whose audio is
auth-gated at TalkBank and which Apple ships labels for but not media. That is
four speaker pools this project has never had.

Does not fix: SEP-28k is still stuttered speech, not aphasic speech. Nothing
here is an aphasia measurement, and the caution in scripts/train_stutter.py
still stands unchanged.

    python scripts/fetch_sep28k_hf.py --check          # join labels only, no download
    python scripts/fetch_sep28k_hf.py --verify         # + audio self-consistency probe
    python scripts/fetch_sep28k_hf.py                  # download 2.87 GB + decode
    python scripts/fetch_sep28k_hf.py --only-new       # skip clips we already hold
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import urllib.request
import wave
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    print("ERROR: pyarrow is required (pip install pyarrow)")
    raise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sep28k_hf"
PARQUET_DIR = DATA / "parquet"
OUT_NPY = DATA / "clips_16k.npy"
OUT_IDX = DATA / "clips_index.json"
MANIFEST = DATA / "manifest.json"
LOCAL_IDX = ROOT / "data" / "sep28k" / "clips_index.json"

REPO = "saeedzou/sep28k-fluencybank-stutter-dataset"
# Pinned to the commit the verification above was run against. `main` on a
# third-party repo can be force-pushed under you, and then the label check that
# passed yesterday is a check of different bytes. Re-pin deliberately, and
# re-run --verify when you do.
REVISION = "e8f1b699b9c05feedbdd617a1bceb27ee5ed1eab"
SHARDS = ["data/train-%05d-of-00007.parquet" % i for i in range(7)]

APPLE = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/"
APPLE_CSVS = ["SEP-28k_labels.csv", "fluencybank_labels.csv"]

SR = 16_000
CLIP_SAMPLES = 3 * SR
CLIP_WAV_BYTES = 96_044          # 44-byte RIFF header + 48000 int16 frames
TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
EXCLUDE = ["Unsure", "PoorAudioQuality", "Music", "NoSpeech"]
# Every annotator-count column Apple ships. The join compares ALL of them, not
# just the five Echo trains on -- a mirror that got Music right and Block wrong
# would still be a mirror that cannot be trusted.
LABEL_COLS = ["Unsure", "PoorAudioQuality", "Prolongation", "Block", "SoundRep",
              "WordRep", "DifficultToUnderstand", "Interjection",
              "NoStutteredWords", "NaturalPause", "Music", "NoSpeech"]
KEY_COLS = ["Show", "EpId", "ClipId", "Start", "Stop"]
META_COLS = KEY_COLS + LABEL_COLS

UA = {"User-Agent": "Mozilla/5.0"}


# --- HTTP ----------------------------------------------------------------
def resolve_url(path: str) -> str:
    return "https://huggingface.co/datasets/%s/resolve/%s/%s" % (REPO, REVISION, path)


def http_get(url: str, timeout: int = 120, headers: dict | None = None) -> bytes:
    h = dict(UA)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


class HttpRangeFile(io.RawIOBase):
    """Seekable file over HTTP Range requests, with coalesced prefetch.

    Parquet is columnar, so the 17 metadata columns of a 2.87 GB corpus are a
    few hundred KB. Reading them remotely means we can run the whole label
    verification BEFORE deciding whether the download is worth 2.87 GB. Without
    the prefetch this is ~800 tiny requests per shard and takes minutes; with
    it, one request per row group.
    """

    def __init__(self, url: str) -> None:
        req = urllib.request.Request(url, headers=UA, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.url = r.url          # follow the CDN redirect once, then reuse
        self._pos = 0
        self._chunks: list[tuple[int, int, bytes]] = []
        self.nbytes = 0

    def readable(self) -> bool: return True
    def seekable(self) -> bool: return True
    def tell(self) -> int: return self._pos

    def seek(self, off: int, whence: int = 0) -> int:
        self._pos = (off if whence == 0 else
                     self._pos + off if whence == 1 else self.size + off)
        return self._pos

    def _raw(self, lo: int, hi: int) -> bytes:
        return http_get(self.url, timeout=180,
                        headers={"Range": "bytes=%d-%d" % (lo, hi - 1)})

    def prefetch(self, intervals: list[tuple[int, int]], gap: int = 1 << 20,
                 workers: int = 8) -> None:
        merged: list[list[int]] = []
        for lo, hi in sorted(intervals):
            if merged and lo - merged[-1][1] <= gap:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, min(hi, self.size)])
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [(lo, ex.submit(self._raw, lo, min(hi, self.size)))
                    for lo, hi in merged]
            for lo, fut in futs:
                b = fut.result()
                self._chunks.append((lo, lo + len(b), b))
                self.nbytes += len(b)
        self._chunks.sort()

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self._pos
        n = min(n, self.size - self._pos)
        if n <= 0:
            return b""
        lo, hi = self._pos, self._pos + n
        data = None
        for clo, chi, b in self._chunks:
            if clo <= lo and hi <= chi:
                data = b[lo - clo:hi - clo]
                break
        if data is None:
            data = self._raw(lo, hi)
            self.nbytes += len(data)
        self._pos += len(data)
        return data

    def readinto(self, b) -> int:   # type: ignore[override]
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def column_intervals(md, wanted: set[str]) -> list[tuple[int, int]]:
    """Byte ranges of the named columns across every row group."""
    schema = md.schema
    idx = [i for i in range(len(schema))
           if schema.column(i).path.split(".")[0] in wanted]
    out = []
    for rg in range(md.num_row_groups):
        row = md.row_group(rg)
        for i in idx:
            c = row.column(i)
            lo = c.data_page_offset
            if c.has_dictionary_page:
                lo = min(lo, c.dictionary_page_offset)
            out.append((lo, lo + c.total_compressed_size))
    return out


# --- Apple's labels, the only source of truth here ------------------------
def load_official() -> dict[tuple[str, int, int], dict]:
    DATA.mkdir(parents=True, exist_ok=True)
    out: dict[tuple[str, int, int], dict] = {}
    for name in APPLE_CSVS:
        dest = DATA / name
        if not dest.exists():
            dest.write_bytes(http_get(APPLE + name))
        rows = list(csv.reader(dest.open(newline="", encoding="utf-8")))
        header = [h.strip() for h in rows[0]]
        missing = [c for c in META_COLS if c not in header]
        if missing:
            raise SystemExit("official CSV %s is missing columns %s -- Apple "
                             "changed the schema, stop and re-read it" % (name, missing))
        for r in rows[1:]:
            if len(r) != len(header):
                continue
            d = {h: v.strip() for h, v in zip(header, r)}
            out[(d["Show"], int(d["EpId"]), int(d["ClipId"]))] = d
    return out


def read_shard_meta(shard: str) -> tuple[dict[str, list], int]:
    """Metadata columns of one shard, over the network, without the audio."""
    f = HttpRangeFile(resolve_url(shard))
    pf = pq.ParquetFile(f)
    f.prefetch(column_intervals(pf.metadata, set(META_COLS)))
    t = pf.read(columns=META_COLS)
    return {n: t.column(n).to_pylist() for n in META_COLS}, f.nbytes


def verify_labels(cols: dict[str, list], official: dict) -> dict:
    """Join on (Show, EpId, ClipId) and compare every label column."""
    n = len(cols["Show"])
    matched = mismatched = absent = offset_bad = 0
    examples: list[str] = []
    for i in range(n):
        k = (cols["Show"][i], cols["EpId"][i], cols["ClipId"][i])
        r = official.get(k)
        if r is None:
            absent += 1
            if len(examples) < 5:
                examples.append("key not in Apple CSVs: %s" % (k,))
            continue
        bad = [c for c in LABEL_COLS if int(r[c]) != cols[c][i]]
        if int(r["Start"]) != cols["Start"][i] or int(r["Stop"]) != cols["Stop"][i]:
            offset_bad += 1
            bad.append("Start/Stop")
        if bad:
            mismatched += 1
            if len(examples) < 5:
                examples.append("%s disagrees on %s" % (k, bad))
        else:
            matched += 1
    return {"n": n, "matched": matched, "mismatched": mismatched,
            "absent_from_official": absent, "start_stop_mismatched": offset_bad,
            "match_rate": matched / n if n else 0.0, "examples": examples}


# --- audio ---------------------------------------------------------------
def decode_clip(raw: bytes) -> np.ndarray:
    """Decode one clip and assert the format instead of assuming it."""
    w = wave.open(io.BytesIO(raw))
    try:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SR, 1, 2):
            raise ValueError("not 16 kHz mono 16-bit: %d Hz, %d ch, %d bytes/sample"
                             % (w.getframerate(), w.getnchannels(), w.getsampwidth()))
        if w.getnframes() != CLIP_SAMPLES:
            raise ValueError("expected %d frames, got %d" % (CLIP_SAMPLES, w.getnframes()))
        return np.frombuffer(w.readframes(CLIP_SAMPLES), dtype="<i2")
    finally:
        w.close()


def overlap_probe(shard_meta: list[dict], shows: list[str], per_show: int) -> list[dict]:
    """Self-consistency: two clips whose windows overlap must agree in the
    overlap, sample for sample. Nothing about a repackaging can fake that --
    it is the only check available for shows we have no local copy of."""
    results = []
    for si, cols in enumerate(shard_meta):
        rows_per_group = 100
        by_ep: dict[tuple, list] = defaultdict(list)
        for i in range(len(cols["Show"])):
            by_ep[(cols["Show"][i], cols["EpId"][i])].append(
                (cols["Start"][i], cols["Stop"][i], cols["ClipId"][i], i))
        # Only pairs inside one row group: a row group is one range request.
        want: dict[int, list] = defaultdict(list)
        done = Counter()
        for k, v in sorted(by_ep.items()):
            if k[0] not in shows or done[k[0]] >= per_show:
                continue
            v.sort()
            for a, b in zip(v, v[1:]):
                ov = min(a[1], b[1]) - b[0]
                if ov < 8000:
                    continue
                ga, gb = a[3] // rows_per_group, b[3] // rows_per_group
                if ga != gb:
                    continue
                want[ga].append((k, a, b, ov))
                done[k[0]] += 1
                break
        if not want:
            continue
        f = HttpRangeFile(resolve_url(SHARDS[si]))
        pf = pq.ParquetFile(f)
        for rg, pairs in sorted(want.items()):
            md = pf.metadata.row_group(rg)
            lo = min(min(md.column(i).data_page_offset,
                         md.column(i).dictionary_page_offset
                         if md.column(i).has_dictionary_page
                         else md.column(i).data_page_offset)
                     for i in range(md.num_columns))
            hi = max(md.column(i).data_page_offset + md.column(i).total_compressed_size
                     for i in range(md.num_columns))
            f.prefetch([(lo, hi + 4096)], gap=1 << 30)
            au = pf.read_row_group(rg).column("audio").to_pylist()
            for k, a, b, ov in pairs:
                wa = decode_clip(au[a[3] % rows_per_group]["bytes"]).astype(np.float64)
                wb = decode_clip(au[b[3] % rows_per_group]["bytes"]).astype(np.float64)
                off = b[0] - a[0]
                sa, sb = wa[off:off + ov], wb[:ov]
                exact = bool(np.array_equal(sa, sb))
                corr = (float(np.corrcoef(sa, sb)[0, 1]) if sa.std() > 0 and sb.std() > 0
                        else float("nan"))
                results.append({"show": k[0], "ep": k[1], "clips": [a[2], b[2]],
                                "overlap_samples": int(ov),
                                "sample_exact": exact, "corr": round(corr, 6)})
    return results


# --- download ------------------------------------------------------------
def repo_files() -> dict[str, dict]:
    """File sizes and LFS sha256 from the Hub, so a truncated download is an
    error rather than a silently short array. docs/DATA_PROVENANCE.md records
    what happened last time a size check passed for the wrong reason."""
    url = "https://huggingface.co/api/datasets/%s/tree/%s/data?recursive=true&expand=true" % (REPO, REVISION)
    out = {}
    for e in json.loads(http_get(url)):
        if e.get("type") != "file":
            continue
        out[e["path"]] = {"size": e.get("size"),
                          "sha256": (e.get("lfs") or {}).get("oid")}
    return out


def download_shard(path: str, dest: Path, expect: dict) -> str:
    if dest.exists() and expect.get("size") and dest.stat().st_size == expect["size"]:
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    req = urllib.request.Request(resolve_url(path), headers=UA)
    with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    if expect.get("size") and tmp.stat().st_size != expect["size"]:
        tmp.unlink(missing_ok=True)
        raise SystemExit("%s: got %d bytes, Hub says %d -- truncated download"
                         % (path, tmp.stat().st_size, expect["size"]))
    if expect.get("sha256"):
        h = hashlib.sha256()
        with tmp.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                h.update(chunk)
        if h.hexdigest() != expect["sha256"]:
            tmp.unlink(missing_ok=True)
            raise SystemExit("%s: sha256 %s != Hub %s" % (path, h.hexdigest(),
                                                          expect["sha256"]))
    tmp.replace(dest)
    return "ok"


def load_local_keys() -> set[tuple[str, int, int]]:
    if not LOCAL_IDX.exists():
        return set()
    rows = json.loads(LOCAL_IDX.read_text(encoding="utf-8"))["rows"]
    return {(r["show"], int(r["ep"]), int(r["clip"])) for r in rows}


# --- main ----------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="verify labels against Apple's CSVs over the network; download nothing")
    ap.add_argument("--verify", action="store_true",
                    help="--check plus the audio overlap self-consistency probe (~250 MB)")
    ap.add_argument("--only-new", action="store_true",
                    help="keep only clips absent from data/sep28k/clips_index.json")
    ap.add_argument("--min-agree", type=int, default=2,
                    help="annotator votes required to count a type as present")
    ap.add_argument("--probe-per-show", type=int, default=3)
    args = ap.parse_args()

    print("SEP-28k + FluencyBank via %s" % REPO)
    print("  a mirror is used only after it agrees with the source. Checking.")

    official = load_official()
    print("  Apple official labels: %d unique (Show, EpId, ClipId) keys" % len(official))

    print("  reading mirror metadata columns over HTTP (no audio) ...")
    shard_meta = []
    pulled = 0
    for i, s in enumerate(SHARDS):
        cols, nb = read_shard_meta(s)
        shard_meta.append(cols)
        pulled += nb
        print("    shard %d: %d rows, %.2f MB of metadata" % (i, len(cols["Show"]), nb / 1e6))
    merged = {c: [v for cols in shard_meta for v in cols[c]] for c in META_COLS}
    print("  mirror rows: %d  (metadata cost %.1f MB, not %.1f GB)"
          % (len(merged["Show"]), pulled / 1e6, 2.87))

    v = verify_labels(merged, official)
    print("")
    print("  LABEL JOIN on (Show, EpId, ClipId), all %d annotator columns + Start/Stop:"
          % len(LABEL_COLS))
    print("     rows matched exactly    %6d / %d  (%.4f%%)"
          % (v["matched"], v["n"], 100 * v["match_rate"]))
    print("     rows disagreeing        %6d" % v["mismatched"])
    print("     keys not in Apple CSVs  %6d" % v["absent_from_official"])
    for e in v["examples"]:
        print("       %s" % e)
    if v["match_rate"] < 1.0:
        print("")
        print("  ABORT: the mirror does not reproduce Apple's labels. A mirror that")
        print("  disagrees with the source is not usable, whatever else it offers.")
        return 2
    print("     -> the mirror reproduces Apple's labels exactly. Usable.")

    mirror_keys = {(merged["Show"][i], merged["EpId"][i], merged["ClipId"][i])
                   for i in range(v["n"])}
    absent = [k for k in official if k not in mirror_keys]
    print("     Apple rows NOT in the mirror: %d (a subset, not a contradiction)"
          % len(absent))

    per_show = Counter(merged["Show"])
    local = load_local_keys()
    print("")
    print("  per show, and what is new to this project:")
    new_by_show = Counter()
    for i in range(v["n"]):
        k = (merged["Show"][i], merged["EpId"][i], merged["ClipId"][i])
        if k not in local:
            new_by_show[k[0]] += 1
    for show in sorted(per_show):
        print("     %-18s %5d clips   %5d not already held" % (show, per_show[show],
                                                               new_by_show[show]))
    print("     %-18s %5d clips   %5d not already held"
          % ("TOTAL", v["n"], sum(new_by_show.values())))

    probe = []
    if args.verify:
        shows = ["StutteringIsCool", "StrongVoices", "IStutterSoWhat", "FluencyBank"]
        print("")
        print("  audio self-consistency probe on the shows we hold no local copy of.")
        print("  Overlapping clip windows must agree sample-for-sample in the overlap:")
        probe = overlap_probe(shard_meta, shows, args.probe_per_show)
        by = defaultdict(list)
        for r in probe:
            by[r["show"]].append(r)
        for s in shows:
            rs = by.get(s, [])
            if not rs:
                print("     %-18s no testable overlapping pair found" % s)
                continue
            print("     %-18s %d pairs, %d sample-exact, mean corr %.6f"
                  % (s, len(rs), sum(r["sample_exact"] for r in rs),
                     float(np.mean([r["corr"] for r in rs]))))

    DATA.mkdir(parents=True, exist_ok=True)
    if args.check or args.verify:
        MANIFEST.write_text(json.dumps({
            "status": "VERIFIED_NO_AUDIO", "repo": REPO, "revision": REVISION,
            "verification": v, "per_show": dict(per_show),
            "new_vs_local_sep28k": dict(new_by_show),
            "apple_rows_absent_from_mirror": len(absent),
            "audio_overlap_probe": probe,
        }, indent=2), encoding="utf-8")
        print("")
        print("  wrote %s (verification only, no audio)" % MANIFEST.name)
        return 0

    files = repo_files()
    total = sum(files[s]["size"] or 0 for s in SHARDS)
    print("")
    print("  downloading %d parquet shards, %.2f GB ..." % (len(SHARDS), total / 1e9))
    for i, s in enumerate(SHARDS):
        dest = PARQUET_DIR / Path(s).name
        state = download_shard(s, dest, files.get(s, {}))
        print("    %d/%d %s  %s (%.0f MB)" % (i + 1, len(SHARDS), Path(s).name,
                                              state, dest.stat().st_size / 1e6))

    keep_new_only = args.only_new
    n_max = sum(new_by_show.values()) if keep_new_only else v["n"]
    print("")
    print("  decoding %d clips -> %s" % (n_max, OUT_NPY.name))
    tmp = OUT_NPY.with_suffix(".part.npy")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype="int16",
                                    shape=(n_max, CLIP_SAMPLES))
    index: list[dict] = []
    written = 0
    bad_audio = 0
    bad_examples: list[str] = []
    for si, s in enumerate(SHARDS):
        pf = pq.ParquetFile(PARQUET_DIR / Path(s).name)
        base = 0
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            au = t.column("audio").to_pylist()
            cols = shard_meta[si]
            for j, a in enumerate(au):
                i = base + j
                key = (cols["Show"][i], cols["EpId"][i], cols["ClipId"][i])
                if keep_new_only and key in local:
                    continue
                if len(a["bytes"]) != CLIP_WAV_BYTES:
                    bad_audio += 1
                    if len(bad_examples) < 5:
                        bad_examples.append("%s: %d bytes" % (key, len(a["bytes"])))
                    continue
                try:
                    pcm = decode_clip(a["bytes"])
                except Exception as exc:
                    bad_audio += 1
                    if len(bad_examples) < 5:
                        bad_examples.append("%s: %s" % (key, exc))
                    continue
                votes = {c: cols[c][i] for c in LABEL_COLS}
                if any(votes[c] >= 2 for c in EXCLUDE):
                    continue          # Unsure / PoorAudioQuality / Music / NoSpeech
                labels = [int(votes[t] >= args.min_agree) for t in TYPES]
                arr[written] = pcm
                index.append({
                    "row": written, "show": key[0], "ep": str(key[1]),
                    "clip": str(key[2]), "labels": labels, "any": int(any(labels)),
                    "no_stuttered_words": int(votes["NoStutteredWords"] >= args.min_agree),
                    "peak": round(float(np.abs(pcm).max()) / 32768.0, 4),
                    "src": "hf:" + REPO,
                })
                written += 1
            base += t.num_rows
        print("    shard %d done, %d clips written" % (si, written), flush=True)

    arr.flush()
    del arr
    src = np.load(tmp, mmap_mode="r")
    dst = np.lib.format.open_memmap(OUT_NPY, mode="w+", dtype="int16",
                                    shape=(written, CLIP_SAMPLES))
    for lo in range(0, written, 2000):
        hi = min(lo + 2000, written)
        dst[lo:hi] = src[lo:hi]
    dst.flush()
    del dst
    src._mmap.close()           # Windows: close the map before the unlink
    del src
    tmp.unlink(missing_ok=True)

    per_type = {t: sum(r["labels"][k] for r in index) for k, t in enumerate(TYPES)}
    shows_written = Counter(r["show"] for r in index)
    OUT_IDX.write_text(json.dumps({
        "n": written, "sample_rate": SR, "clip_samples": CLIP_SAMPLES,
        "types": TYPES, "per_type": per_type, "min_agree": args.min_agree,
        "per_show": dict(shows_written),
        "only_new": bool(keep_new_only),
        "bad_audio_clips": bad_audio,
        "provenance": (
            "SEP-28k + FluencyBank clips from the HuggingFace mirror %s, "
            "verified row-for-row against Apple's SEP-28k_labels.csv and "
            "fluencybank_labels.csv: %d/%d keys matched on all %d annotator "
            "columns and on Start/Stop. Audio decoded and asserted 16 kHz mono "
            "16-bit 3.000 s per clip. Adds StutteringIsCool, StrongVoices, "
            "IStutterSoWhat and FluencyBank, which data/sep28k does not "
            "contain. STILL stuttered speech, not aphasia -- see "
            "docs/DATA_PROVENANCE.md." % (REPO, v["matched"], v["n"], len(LABEL_COLS))),
        "rows": index,
    }, indent=2), encoding="utf-8")

    MANIFEST.write_text(json.dumps({
        "status": "OK", "repo": REPO, "revision": REVISION,
        "shard_sha256": {s: files.get(s, {}).get("sha256") for s in SHARDS},
        "verification": v, "per_show_available": dict(per_show),
        "per_show_written": dict(shows_written),
        "new_vs_local_sep28k": dict(new_by_show),
        "apple_rows_absent_from_mirror": len(absent),
        "bad_audio_clips": bad_audio, "bad_audio_examples": bad_examples,
        "leakage_warning": (
            "This corpus OVERLAPS data/sep28k on five shows. Concatenating both "
            "without --only-new duplicates ~20k clips and will put the same clip "
            "in train and test. scripts/train_stutter.py splits by (show, ep), so "
            "duplicates land on the same side only if the key is identical -- it "
            "is, so dedupe by (show, ep, clip) before merging."),
    }, indent=2), encoding="utf-8")

    print("")
    print("  wrote %d clips -> %s (%.2f GB)"
          % (written, OUT_NPY.name, OUT_NPY.stat().st_size / 1e9))
    if bad_audio:
        print("  clips whose audio failed the format assertions: %d" % bad_audio)
        for e in bad_examples:
            print("     %s" % e)
    print("  positives per type at >=%d/3 agreement:" % args.min_agree)
    for t in TYPES:
        print("    %-14s %6d  (%.1f%%)" % (t, per_type[t], 100 * per_type[t] / max(1, written)))
    print("  per show:")
    for s in sorted(shows_written):
        print("    %-18s %6d" % (s, shows_written[s]))
    print("")
    print("  NEXT: train with data/sep28k_hf as an ADDITIONAL corpus. Dedupe by")
    print("  (show, ep, clip) against data/sep28k first, or pass --only-new here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
