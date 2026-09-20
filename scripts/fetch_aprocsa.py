"""Fetch APROCSA -- the only REAL aphasic speech Echo has ever been tested on.

WHY THIS DATASET EXISTS IN THIS REPO
------------------------------------
Every detection number Echo has published so far comes from stuttered podcast
speech (SEP-28k) or fluent podcast speech (PodcastFillers). Neither is aphasia.
Aphasia is a LANGUAGE disorder -- the word is not retrievable -- while
stuttering is a MOTOR-SPEECH disorder -- the word is known and will not come
out. They produce overlapping surface evidence (silent blocks, filled pauses,
repetitions) for completely different reasons, and an aid built for one is not
automatically right for the other. Shipping without a single aphasic recording
in the loop is what makes a demo a toy.

AphasiaBank is the standard corpus and is membership-gated: every media URL
under media.talkbank.org returns an auth modal, verified 2026-08-18 (see
docs/DATA_PROVENANCE.md). APROCSA is the exception -- six people with chronic
post-stroke aphasia, released unrestricted "for research, education, and
clinical uses":

    Casilio, M., Rising, K., Beeson, P. M., Bunton, K., & Wilson, S. M. (2022).
    An Open Dataset of Connected Speech in Aphasia with Consensus Ratings of
    Auditory-Perceptual Features. Data, 7(11), 148.
    https://doi.org/10.3390/data7110148
    Dataset: https://langneurosci.org/aprocsa-dataset

WHAT IT IS AND IS NOT
---------------------
IS:  6 speakers, full elicitation protocol (free speech, three picture
     descriptions, Cinderella retell, procedural discourse), audiovisual, with
     CHAT (.cha) transcripts and consensus ratings on 27 auditory-perceptual
     features (0=absent .. 4=severe).
NOT: a training set. Six speakers cannot train a detector, and must never be
     split into train/test -- one speaker's idiosyncrasies would leak straight
     across the split. This is an EVALUATION and THRESHOLD-FITTING set only.
     That restriction is enforced in code, not left to memory: every consumer
     goes through eval/, none through scripts/train_*.

    python scripts/fetch_aprocsa.py            # video + audio + transcripts
    python scripts/fetch_aprocsa.py --no-video # transcripts only (fast)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "aprocsa"
VIDEO_DIR = DATA / "video"
AUDIO_DIR = DATA / "audio"
CHA_DIR = DATA / "transcripts"
MANIFEST = DATA / "manifest.json"

# Participant ids as published. The .cha filename carries an 'a' suffix; the
# media file does not.
PARTICIPANTS = ["1554", "1713", "1731", "1738", "1833", "1944"]
VIDEO_URL = "https://lnl.app.vumc.org/aprocsa-dataset/%s.mp4"
CHA_URL = "https://langneurosci.org/files/aprocsa/dataset/aprocsa%sa.cha"

SR = 16_000
UA = {"User-Agent": "Mozilla/5.0"}


def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def download(url: str, dest: Path, min_bytes: int = 1000, tries: int = 4) -> tuple[bool, str]:
    """Resumable GET. The media files are 0.5-1.2 GB each over a link that
    measured ~400 KB/s, so a dropped connection an hour in must not restart
    from zero -- the .part file is kept and continued with a Range request.
    """
    if dest.exists() and dest.stat().st_size >= min_bytes:
        return True, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last = "unknown"
    for _ in range(tries):
        have = tmp.stat().st_size if tmp.exists() else 0
        headers = dict(UA)
        if have:
            headers["Range"] = "bytes=%d-" % have
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=180) as r:
                if have and r.status != 206:
                    have = 0          # server ignored Range: start over
                mode = "ab" if have else "wb"
                total = int(r.headers.get("Content-Length") or 0) + have
                with tmp.open(mode) as fh:
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        fh.write(chunk)
            if total and tmp.stat().st_size < total:
                last = "short read"
                continue              # resume from where it stopped
            if tmp.stat().st_size < min_bytes:
                tmp.unlink(missing_ok=True)
                return False, "too small"
            tmp.replace(dest)
            return True, "ok"
        except Exception as exc:
            last = type(exc).__name__
    return False, last


def extract_audio(mp4: Path, wav: Path) -> bool:
    """mp4 -> 16 kHz mono wav. Echo's whole stack is 16 kHz mono; resampling
    once here means no eval script has to remember to do it (the SEP-28k
    sample-rate bug came from exactly that kind of implicit assumption)."""
    if wav.exists() and wav.stat().st_size > 1000:
        return True
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp4),
           "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(wav)]
    return subprocess.run(cmd, capture_output=True).returncode == 0


# --- CHAT parsing --------------------------------------------------------
# CHAT marks media alignment with a NAK-delimited bullet: \x15<start>_<end>\x15
# where the numbers are MILLISECONDS into the media file. That is what makes
# this dataset usable as a timed evaluation set rather than just text.
_BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")


def parse_cha(path: Path) -> dict:
    """Extract participant (*PAR) utterances with media-aligned times.

    Only *PAR tiers are kept: *INV is the investigator, and counting the
    clinician's fluent speech as the wearer's would invert every metric.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    lines, cur = [], None
    for raw in text.splitlines():
        if raw.startswith("*") or raw.startswith("@"):
            if cur:
                lines.append(cur)
            cur = raw
        elif raw.startswith("\t") and cur is not None:
            cur += " " + raw.strip()
        else:
            if cur:
                lines.append(cur)
            cur = None
    if cur:
        lines.append(cur)

    utts = []
    for ln in lines:
        if not ln.startswith("*PAR:"):
            continue
        m = _BULLET.search(ln)
        body = _BULLET.sub("", ln[5:]).strip()
        utts.append({
            "start_ms": int(m.group(1)) if m else None,
            "end_ms": int(m.group(2)) if m else None,
            "chat": body,
        })
    header = {}
    for ln in lines:
        if ln.startswith("@ID:") and "PAR" in ln:
            header["id"] = ln[4:].strip()
        elif ln.startswith("@Media:"):
            header["media"] = ln[7:].strip()
    return {"utterances": utts, "header": header}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true",
                    help="transcripts only (skip the ~4.5 GB media download)")
    args = ap.parse_args()

    print("APROCSA -- open aphasic connected speech (Casilio et al. 2022)")
    print("  license: unrestricted for research, education, clinical use")
    print("  NOTE: evaluation + threshold fitting ONLY. Six speakers cannot")
    print("        train a detector and must never be train/test split.")
    print("")

    entries = []
    for pid in PARTICIPANTS:
        print("  %s" % pid, flush=True)
        cha = CHA_DIR / ("aprocsa%sa.cha" % pid)
        ok_cha, why = download(CHA_URL % pid, cha)
        print("    transcript: %s" % ("ok" if ok_cha else "FAILED " + why), flush=True)

        entry = {"participant": pid, "transcript": str(cha.relative_to(ROOT)) if ok_cha else None}
        if ok_cha:
            parsed = parse_cha(cha)
            timed = [u for u in parsed["utterances"] if u["start_ms"] is not None]
            entry["n_utterances"] = len(parsed["utterances"])
            entry["n_timed"] = len(timed)
            entry["header"] = parsed["header"]
            print("    utterances: %d (%d media-aligned)"
                  % (len(parsed["utterances"]), len(timed)), flush=True)

        entries.append(entry)

    if not args.no_video:
        # 4.5 GB over a ~400 KB/s link is over two hours serially. The six
        # files are independent, so fetch them concurrently.
        from concurrent.futures import ThreadPoolExecutor

        print("")
        print("  downloading 6 recordings (~4.5 GB total, resumable) ...", flush=True)

        def fetch(pid: str):
            mp4 = VIDEO_DIR / ("%s.mp4" % pid)
            ok, why = download(VIDEO_URL % pid, mp4, min_bytes=1 << 20)
            print("    %s video: %s (%.0f MB)"
                  % (pid, "ok" if ok else "FAILED " + why,
                     mp4.stat().st_size / 1e6 if mp4.exists() else 0), flush=True)
            return pid, ok, mp4

        with ThreadPoolExecutor(max_workers=6) as ex:
            fetched = list(ex.map(fetch, PARTICIPANTS))

        import soundfile as sf
        by_pid = {e["participant"]: e for e in entries}
        for pid, ok, mp4 in fetched:
            if not ok:
                continue
            wav = AUDIO_DIR / ("%s.wav" % pid)
            if not extract_audio(mp4, wav):
                print("    %s audio: ffmpeg FAILED" % pid, flush=True)
                continue
            info = sf.info(str(wav))
            by_pid[pid]["audio"] = str(wav.relative_to(ROOT))
            by_pid[pid]["duration_s"] = round(info.duration, 1)
            by_pid[pid]["samplerate"] = info.samplerate
            print("    %s audio: %.1f min @ %d Hz mono"
                  % (pid, info.duration / 60, info.samplerate), flush=True)

    DATA.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({
        "dataset": "APROCSA",
        "citation": ("Casilio M, Rising K, Beeson PM, Bunton K, Wilson SM (2022). "
                     "An Open Dataset of Connected Speech in Aphasia with Consensus "
                     "Ratings of Auditory-Perceptual Features. Data 7(11):148. "
                     "doi:10.3390/data7110148"),
        "source": "https://langneurosci.org/aprocsa-dataset",
        "license": "unrestricted for research, education, and clinical uses",
        "sample_rate": SR,
        "usage_restriction": ("EVALUATION AND THRESHOLD FITTING ONLY. Six speakers. "
                              "Never split into train/test -- speaker leakage is "
                              "unavoidable at this size."),
        "participants": entries,
    }, indent=2), encoding="utf-8")
    print("")
    print("  wrote %s" % MANIFEST.relative_to(ROOT))
    total = sum(e.get("duration_s", 0) for e in entries)
    if total:
        print("  total aphasic speech: %.1f minutes across %d speakers"
              % (total / 60, sum(1 for e in entries if e.get("audio"))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
