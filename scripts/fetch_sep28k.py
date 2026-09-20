"""Reconstruct SEP-28k audio from Apple's official release.

WHY NOT A HUGGINGFACE MIRROR
----------------------------
Because I used one and it was wrong. `isabelarvelo/sep28k-*-4-second-clips`
bundles audio conveniently, but it is a third-party derivation: 8,142 clips with
a single int64 label collapsed to 0/1. The official release is 28,177 clips with
MULTI-LABEL annotator counts (0-3 of 3 raters) across five dysfluency types.
3,009 clips carry two or more types at >=2 agreement, so any single-label
repackaging is wrong for ~21% of the labelled data -- and it erases exactly the
distinction Echo needs, between a block (silent struggle) and an interjection
("um"), which are opposite signals for a word-finding aid.

Labels come from the CSVs in apple/ml-stuttering-events-dataset. Audio does not:
the release ships URLs to 385 podcast episodes and a recipe for cutting clips at
sample offsets. This script does that.

LINK ROT IS REAL AND NOT RANDOM. The dataset is from 2021 and podcast hosting
moves. A 40-episode probe found ~70% still resolving, with failures concentrated
in specific shows (StutteringIsCool, IStutterSoWhat, StrongVoices) rather than
spread evenly. That means the reconstructed corpus is a BIASED subset of
SEP-28k, not a random sample of it, and any metric computed on it must say so.
This script writes the per-show recovery rate into the manifest for exactly that
reason.

    python scripts/fetch_sep28k.py                 # download + cut
    python scripts/fetch_sep28k.py --labels-only   # manifest only, no audio
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sep28k"
EP_DIR = DATA / "episodes"
CLIP_DIR = DATA / "clips"
MANIFEST = DATA / "manifest.json"

BASE = "https://raw.githubusercontent.com/apple/ml-stuttering-events-dataset/main/"
LABELS_URL = BASE + "SEP-28k_labels.csv"
EPISODES_URL = BASE + "SEP-28k_episodes.csv"

SR = 16000
# The five dysfluency types. Ordered by how much they matter to a word-finding
# aid: a Block is a silent struggle to initiate -- the strongest evidence the
# speaker is stuck -- while an Interjection ("um") is the weakest, because
# fluent speakers produce them constantly. Echo currently detects only the
# weakest one, which is why it false-fires.
TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection"]
# Columns that mean "this clip is not usable evidence".
EXCLUDE = ["Unsure", "PoorAudioQuality", "Music", "NoSpeech"]
UA = {"User-Agent": "Mozilla/5.0"}


def fetch_csv(url: str, dest: Path) -> list[list[str]]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=60) as r:
            dest.write_bytes(r.read())
    with dest.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def load_labels() -> list[dict]:
    rows = fetch_csv(LABELS_URL, DATA / "SEP-28k_labels.csv")
    header = [h.strip() for h in rows[0]]
    out = []
    for r in rows[1:]:
        if len(r) != len(header):
            continue
        d = {h: v.strip() for h, v in zip(header, r)}
        out.append(d)
    return out


def download_episode(args: tuple[str, str, str, Path]) -> tuple[str, str, bool, str]:
    show, epid, url, dest = args
    if dest.exists() and dest.stat().st_size > 10_000:
        return show, epid, True, "cached"
    try:
        req = urllib.request.Request(url, headers=UA)
        with urllib.request.urlopen(req, timeout=120) as r:
            data = r.read()
        if len(data) < 10_000:
            return show, epid, False, "too small"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return show, epid, True, "ok"
    except Exception as exc:
        return show, epid, False, type(exc).__name__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-only", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-agree", type=int, default=2,
                    help="annotator votes required to count a type as present")
    args = ap.parse_args()

    print("SEP-28k reconstruction (official Apple release)")
    labels = load_labels()
    print("  labels: %d clips" % len(labels))

    def votes(row: dict, col: str) -> int:
        try:
            return int(row.get(col, 0) or 0)
        except ValueError:
            return 0

    # Multi-label targets at the chosen agreement level, plus a usable flag.
    manifest_clips = []
    per_type = Counter()
    for r in labels:
        usable = all(votes(r, c) < 2 for c in EXCLUDE)
        y = {t: int(votes(r, t) >= args.min_agree) for t in TYPES}
        for t, v in y.items():
            per_type[t] += v
        manifest_clips.append({
            "show": r["Show"], "ep": r["EpId"], "clip": r["ClipId"],
            "start": int(r["Start"]), "stop": int(r["Stop"]),
            "labels": y,
            "any_dysfluency": int(any(y.values())),
            "no_stuttered_words": int(votes(r, "NoStutteredWords") >= args.min_agree),
            "usable": int(usable),
        })
    print("  at >=%d/3 annotator agreement:" % args.min_agree)
    for t in TYPES:
        print("     %-14s %6d" % (t, per_type[t]))
    print("     %-14s %6d" % ("any", sum(c["any_dysfluency"] for c in manifest_clips)))
    print("     %-14s %6d" % ("unusable", sum(1 for c in manifest_clips if not c["usable"])))

    eps_rows = fetch_csv(EPISODES_URL, DATA / "SEP-28k_episodes.csv")
    episodes = []
    for r in eps_rows:
        if len(r) >= 4 and r[2].strip().startswith("http"):
            show, url = r[3].strip(), r[2].strip()
            episodes.append((show, url))
    # EpId is the row index WITHIN a show, in file order (Apple's convention).
    by_show: dict[str, list[str]] = defaultdict(list)
    for show, url in episodes:
        by_show[show].append(url)
    print("  episodes listed: %d across %d shows" % (len(episodes), len(by_show)))

    if args.labels_only:
        DATA.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({
            "status": "LABELS_ONLY", "min_agree": args.min_agree,
            "n_clips": len(manifest_clips), "per_type": dict(per_type),
            "clips": manifest_clips,
        }, indent=2), encoding="utf-8")
        print("  wrote %s (labels only, no audio)" % MANIFEST.name)
        return 0

    jobs = []
    for show, urls in by_show.items():
        for i, url in enumerate(urls):
            jobs.append((show, str(i), url, EP_DIR / show / ("%s.mp3" % i)))
    print("  downloading %d episodes (~10-18 GB, link rot expected) ..." % len(jobs))

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(download_episode, jobs), 1):
            results.append(res)
            if i % 25 == 0:
                ok = sum(1 for r in results if r[2])
                print("    %d/%d  (%d ok)" % (i, len(jobs), ok))

    ok_eps = {(s, e) for s, e, good, _ in results if good}
    fail_by_show = Counter(s for s, _, good, _ in results if not good)
    tot_by_show = Counter(s for s, _, _, _ in results)
    print("  episodes recovered: %d / %d" % (len(ok_eps), len(results)))
    print("  per-show recovery (this is the BIAS, record it):")
    recovery = {}
    for show in sorted(tot_by_show):
        got = tot_by_show[show] - fail_by_show[show]
        recovery[show] = {"got": got, "total": tot_by_show[show],
                          "rate": round(got / tot_by_show[show], 3)}
        print("     %-20s %3d/%3d  %.0f%%" % (show, got, tot_by_show[show],
                                              100 * got / tot_by_show[show]))

    reachable = [c for c in manifest_clips if (c["show"], c["ep"]) in ok_eps]
    print("  clips whose audio is reachable: %d / %d (%.0f%%)"
          % (len(reachable), len(manifest_clips),
             100 * len(reachable) / len(manifest_clips)))

    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "status": "OK", "min_agree": args.min_agree,
        "sample_rate": SR, "types": TYPES,
        "n_clips_labelled": len(manifest_clips),
        "n_clips_reachable": len(reachable),
        "per_type_all": dict(per_type),
        "per_show_episode_recovery": recovery,
        "bias_warning": ("Episode recovery is NOT uniform across shows; the "
                         "reconstructed corpus is a biased subset of SEP-28k. "
                         "Any metric computed on it must disclose this."),
        "clips": reachable,
    }, indent=2), encoding="utf-8")
    print("  wrote %s" % MANIFEST)
    print("")
    print("  NEXT: cut clips with scripts/cut_sep28k_clips.py (uses start/stop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
