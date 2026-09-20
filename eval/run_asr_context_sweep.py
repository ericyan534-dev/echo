"""How much context does a verbatim ASR need before it reports the filler?

A first measurement put CrisperWhisper's filler recall at 0.217 on
PodcastFillers clips. That number is misleading, and the misses said why: they
transcribed neighbouring WORDS ('I mean,', 'about', 'topic') rather than
producing nothing. Those clips are 1.00 s. Whisper-family models have a 30 s
receptive field; handed a one-second fragment they latch onto lexical content.
So 0.217 measures "Whisper on one-second fragments", not "Whisper in Echo",
where the recognizer sees continuous conversation.

This isolates the variable. Same annotated fillers, same model, window length
swept from 1 s to 12 s around the SAME event. If recall climbs with context,
the first number was a benchmark artifact. If it does not, the model genuinely
misses these fillers and the acoustic channel has to stay.

Positives: SEP-28k clips with Interjection >= 2/3 annotator agreement.
Negatives are NOT measured here on purpose: in a 10 s window of natural
conversation a neighbouring filler is likely and would be scored as a false
positive when it is actually a correct detection of a different event. Cleanly
attributing a detection to a specific event needs word-level timestamps; that is
a separate measurement, not a footnote to this one.

    python eval/run_asr_context_sweep.py --n 60
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "crisperwhisper"
EP_DIR = ROOT / "data" / "sep28k" / "episodes"
MANIFEST = ROOT / "data" / "sep28k" / "manifest.json"
OUT = ROOT / "eval" / "results" / "asr_context_sweep.json"

SR = 16000
WINDOWS = [1.0, 3.0, 6.0, 12.0]
# Apple's definition: an Interjection is "um"/"uh" OR a PERSON-SPECIFIC filler
# word the speaker uses to cope with their stutter (their example: "you know").
# So no fixed word list can be complete, and any lexical rule is a LOWER BOUND.
FILLER_RE = re.compile(
    r"\[(uh|um|uhm)\]|(?<![a-z])(uh+|um+|erm|er|mm+|hmm+)(?![a-z])"
    r"|you know|i mean|sort of|kind of", re.I)

# Repetition is the OTHER way dysfluency survives into a verbatim transcript:
# "that I like, that I like" / "she will actually, she will". Chrome smooths
# these away; a verbatim ASR renders them. For Echo the question is not which
# label applies, it is whether ANY evidence of the disfluency reaches the text.
_TOK = re.compile(r"[a-z']+")


def has_repetition(text: str) -> bool:
    toks = _TOK.findall((text or "").lower())
    for i in range(len(toks) - 1):
        if toks[i] == toks[i + 1]:
            return True                      # immediate word repetition
    for n in (2, 3):                          # short phrase repetition
        for i in range(len(toks) - 2 * n + 1):
            if toks[i:i + n] == toks[i + n:i + 2 * n]:
                return True
    return False


def preserves_dysfluency(text: str) -> bool:
    return bool(FILLER_RE.search(text or "")) or has_repetition(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60, help="annotated fillers to test")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not MANIFEST.exists() or not MODEL_DIR.exists():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED",
                                   "reason": "manifest or model missing"}, indent=2),
                       encoding="utf-8")
        print("SKIPPED"); return 0

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cands = [c for c in man["clips"]
             if c["usable"] and c["labels"]["Interjection"] == 1][:args.n * 3]

    dev = args.device if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR), dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()

    def transcribe(a):
        f = proc(a, sampling_rate=SR, return_tensors="pt").input_features.to(dev, dtype=dtype)
        with torch.no_grad():
            ids = model.generate(f, max_new_tokens=96)
        return proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    results = {w: {"hit": 0, "n": 0} for w in WINDOWS}
    samples = []
    used = 0
    for c in cands:
        if used >= args.n:
            break
        ep = EP_DIR / c["show"] / ("%s.mp3" % c["ep"])
        if not ep.exists():
            continue
        try:
            info = sf.info(str(ep))
        except Exception:
            continue
        # SEP-28k Start/Stop are in 16 kHz samples (Stop-Start == 48000 == 3 s),
        # but the source episodes are whatever the podcast was published at --
        # 32 kHz and 16 kHz both occur. Rescale into the file's own rate before
        # seeking, or every offset lands somewhere else in the episode.
        scale = info.samplerate / SR
        mid = int(((c["start"] + c["stop"]) // 2) * scale)
        row = {"show": c["show"], "ep": c["ep"], "clip": c["clip"], "texts": {}}
        ok_any = False
        for w in WINDOWS:
            half = int(w * info.samplerate / 2)
            a0, a1 = max(0, mid - half), min(info.frames, mid + half)
            try:
                a, sr = sf.read(str(ep), start=a0, stop=a1, dtype="float32")
            except Exception:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            if len(a) < sr // 2:
                continue
            if sr != SR:                    # Whisper expects 16 kHz
                import numpy as np
                idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
                a = np.interp(idx, np.arange(len(a)), a).astype("float32")
            txt = transcribe(a)
            det = preserves_dysfluency(txt)
            results[w]["n"] += 1
            results[w]["hit"] += int(det)
            row["texts"][str(w)] = {"text": txt[:120], "detected": det}
            ok_any = True
        if ok_any:
            used += 1
            if len(samples) < 6:
                samples.append(row)
        if used % 10 == 0 and used:
            print("  %d/%d events" % (used, args.n))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    summary = {str(w): {"n": v["n"], "recall": round(v["hit"] / v["n"], 4) if v["n"] else None}
               for w, v in results.items()}
    OUT.write_text(json.dumps({
        "status": "OK", "model": "nyralabs/CrisperWhisper2.0_turbo",
        "source": "SEP-28k, Interjection >= 2/3 agreement",
        "metric": ("dysfluency preserved in transcript = explicit filler token OR "
                   "discourse filler OR immediate word/phrase repetition. Lower "
                   "bound: interjections are person-specific per Apple's spec."),
        "n_events": used, "by_window_s": summary,
        "chrome_baseline_recall": 0.0,
        "note": ("Negatives deliberately not scored: a long window of natural "
                 "conversation may contain a neighbouring filler, which would "
                 "count as a false positive while being a correct detection of a "
                 "different event. That needs word-level timestamps."),
        "samples": samples,
    }, indent=2), encoding="utf-8")

    print("")
    print("FILLER RECALL vs CONTEXT LENGTH (%d annotated events)" % used)
    print("  %-12s %6s %8s" % ("window", "n", "recall"))
    for w in WINDOWS:
        s = summary[str(w)]
        print("  %-12s %6s %8s" % ("%.0f s" % w, s["n"],
                                   ("%.3f" % s["recall"]) if s["recall"] is not None else "-"))
    print("  %-12s %6s %8.3f   <- current stack" % ("Chrome", "5044", 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
