"""Does a verbatim ASR actually recover the fillers Chrome deletes?

This repo already measured the gap: on 5,044 PodcastFillers filler clips, the
live Chrome path fired the filler trigger 0 times -- recall 0.000 -- while a
"verbatim ASR" ablation fired on all 5,044 (recall 1.000). That ablation was
hypothetical: it assumed an ASR that emits the token. Echo's whole parallel
acoustic channel exists because no such ASR was in the stack.

This measures a real one. Same clips, same question: when the audio contains a
filler, does the transcript contain it?

Two error directions matter and are reported separately, because they cost
different things:
  * MISS on a filler clip  -> the stall signal is lost, same failure as Chrome.
  * FIRE on a Words clip   -> a false positive, which is what makes an aid nag.

    python eval/run_asr_filler_recall.py --n 150
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "crisperwhisper"
CLIPS = ROOT / "data" / "pfsd" / "clips" / "test"
OUT = ROOT / "eval" / "results" / "asr_filler_recall.json"

# CrisperWhisper marks fillers with explicit tokens; also accept the bare words
# in case a build spells them differently.
FILLER_RE = re.compile(r"\[(uh|um|uhm)\]|(?<![a-z])(uh+|um+|erm|er)(?![a-z])", re.I)


def has_filler(text: str) -> bool:
    return bool(FILLER_RE.search(text or ""))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150, help="clips per class")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not MODEL_DIR.exists() or not CLIPS.is_dir():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED",
                                   "reason": "model or PFSD clips missing"}, indent=2),
                       encoding="utf-8")
        print("SKIPPED -- model or clips missing")
        return 0

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    dev = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    proc = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR), dtype=dtype, low_cpu_mem_usage=True).to(dev).eval()

    groups = {"Uh": True, "Um": True, "Words": False}   # class -> is a filler clip
    rows, lat = [], []
    for cls, is_filler in groups.items():
        d = CLIPS / cls
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.wav"))[:args.n]:
            a, sr = sf.read(str(p), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            f = proc(a, sampling_rate=sr, return_tensors="pt").input_features.to(dev, dtype=dtype)
            t0 = time.perf_counter()
            with torch.no_grad():
                ids = model.generate(f, max_new_tokens=48)
            lat.append((time.perf_counter() - t0) * 1000)
            txt = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
            rows.append({"cls": cls, "is_filler": is_filler,
                         "text": txt, "detected": has_filler(txt)})
        print("  %-6s done (%d clips)" % (cls, sum(1 for r in rows if r["cls"] == cls)))

    pos = [r for r in rows if r["is_filler"]]
    neg = [r for r in rows if not r["is_filler"]]
    tp = sum(r["detected"] for r in pos)
    fp = sum(r["detected"] for r in neg)
    recall = tp / len(pos) if pos else 0.0
    fpr = fp / len(neg) if neg else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    lat.sort()

    res = {
        "status": "OK", "model": "nyralabs/CrisperWhisper2.0_turbo", "device": dev,
        "n_filler_clips": len(pos), "n_words_clips": len(neg),
        "filler_recall": round(recall, 4),
        "false_positive_rate_on_words": round(fpr, 4),
        "precision": round(prec, 4),
        "latency_ms_median": round(lat[len(lat) // 2], 1) if lat else None,
        "chrome_baseline_recall": 0.0,
        "chrome_baseline_source": "eval/results/stall_eval.json transcript_baseline (n=5044)",
        "rows": rows[:80],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print("")
    print("VERBATIM ASR vs THE CURRENT STACK -- filler recovery")
    print("  %-34s %8s %8s" % ("", "recall", "FP rate"))
    print("  %-34s %8.3f %8s" % ("Chrome (measured, n=5044)", 0.0, "n/a"))
    print("  %-34s %8.3f %8.3f" % ("CrisperWhisper (n=%d/%d)" % (len(pos), len(neg)),
                                   recall, fpr))
    print("")
    print("  precision %.3f | median %.0f ms/clip" % (prec, res["latency_ms_median"] or 0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
