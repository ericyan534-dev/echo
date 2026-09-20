"""Episode-path resolution and container repair for the recovered SEP-28k audio.

WHY THIS EXISTS
---------------
`fetch_sep28k.py` accepted any download over 10 KB as a recovered episode. 27
of the 258 "recovered" episodes are not MP3 at all -- they are M4A/AAC served
under an .mp3 URL (`ftypM4A` in the first bytes), which libsndfile cannot open.
Every consumer was silently skipping them, and the loss was not random: it took
out ALL 4 HVSA episodes and 23 of 38 MyStutteringLife episodes, i.e. two whole
speaker pools, on top of the three shows already lost to link rot.

That is a recoverable loss, not a real one. ffmpeg reads them fine. This module
transcodes those episodes once to 16 kHz mono WAV and gives every consumer a
single `episode_path()` to call, so no script has to know which container a
given episode arrived in.

Transcoding to 16 kHz mono here also removes the resample-on-read step from
every downstream consumer, and with it the class of bug where a 32 kHz or
44.1 kHz episode is seeked with 16 kHz sample offsets.

    python scripts/sep28k_audio.py            # repair whatever needs it
    python scripts/sep28k_audio.py --check    # report only, transcode nothing
"""
from __future__ import annotations

import argparse
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EP_DIR = ROOT / "data" / "sep28k" / "episodes"
SR = 16_000


def episode_path(show: str, ep: str) -> Path | None:
    """Best available audio file for an episode, or None if there is none.

    Prefers the repaired 16 kHz WAV when it exists, so callers get a uniform
    sample rate; falls back to the original download otherwise.
    """
    wav = EP_DIR / show / ("%s.wav" % ep)
    if wav.exists() and wav.stat().st_size > 10_000:
        return wav
    mp3 = EP_DIR / show / ("%s.mp3" % ep)
    if mp3.exists() and mp3.stat().st_size > 10_000:
        return mp3
    return None


def is_readable(path: Path) -> bool:
    import soundfile as sf

    try:
        sf.info(str(path))
        return True
    except Exception:
        return False


def transcode(src: Path, dest: Path) -> bool:
    import imageio_ffmpeg

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
           "-i", str(src), "-vn", "-ac", "1", "-ar", str(SR),
           "-c:a", "pcm_s16le", str(dest)]
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        dest.unlink(missing_ok=True)
        return False
    return dest.exists() and dest.stat().st_size > 10_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    if not EP_DIR.is_dir():
        print("no episodes at %s -- run scripts/fetch_sep28k.py first" % EP_DIR)
        return 1

    need: list[Path] = []
    ok = Counter()
    bad = Counter()
    for show_dir in sorted(p for p in EP_DIR.iterdir() if p.is_dir()):
        for mp3 in sorted(show_dir.glob("*.mp3")):
            wav = mp3.with_suffix(".wav")
            if wav.exists() and wav.stat().st_size > 10_000:
                ok[show_dir.name] += 1
                continue
            if is_readable(mp3):
                ok[show_dir.name] += 1
            else:
                bad[show_dir.name] += 1
                need.append(mp3)

    print("SEP-28k episode container check")
    print("  %-22s %6s %6s" % ("show", "ok", "unreadable"))
    for show in sorted(set(ok) | set(bad)):
        print("  %-22s %6d %6d" % (show, ok[show], bad[show]))
    print("  %-22s %6d %6d" % ("TOTAL", sum(ok.values()), sum(bad.values())))

    if not need:
        print("\n  nothing to repair")
        return 0
    if args.check:
        first = need[0].read_bytes()[:12]
        print("\n  %d episodes need repair; first bytes of one: %r" % (len(need), first))
        return 0

    print("\n  transcoding %d episodes to 16 kHz mono WAV ..." % len(need))
    fixed = failed = 0
    for i, mp3 in enumerate(need, 1):
        if transcode(mp3, mp3.with_suffix(".wav")):
            fixed += 1
        else:
            failed += 1
            print("    FAILED %s" % mp3.relative_to(EP_DIR))
        if i % 5 == 0:
            print("    %d/%d" % (i, len(need)), flush=True)
    print("\n  repaired %d, failed %d" % (fixed, failed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
