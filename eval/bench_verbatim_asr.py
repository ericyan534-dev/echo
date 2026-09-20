"""Is a verbatim ASR fast enough to replace Chrome's recognizer, and on what?

WHY
---
Echo's v2 thesis is that consumer ASR erases the stall signals that matter:
Chrome deletes "um"/"uh" with no off switch, and every pipeline normalizes
"theeee" to "the". Echo works around that with a parallel acoustic channel.

CrisperWhisper is a Whisper variant retokenized and fine-tuned to transcribe
VERBATIM -- fillers, stutters, false starts, prolongations -- with word-level
timestamps. If it is fast enough, the workaround becomes unnecessary and the
transcript itself carries the evidence.

The blocking question is hardware, not quality: the 27B local LLM already holds
~15.2 GB of 16.4 GB VRAM, so ASR and prediction cannot both live on the GPU.
This measures the real cost on CPU and on GPU so the choice is made on numbers.

METRIC. Real-time factor (RTF) = processing seconds per audio second. RTF < 1 is
faster than real time. Echo transcribes short conversational bursts, not hours,
so the number that actually matters is wall-clock latency on a ~4 s utterance --
reported alongside RTF.

    python eval/bench_verbatim_asr.py --device cpu
    python eval/bench_verbatim_asr.py --device cuda     # stop llama-server first
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = ROOT / "models" / "crisperwhisper"
OUT = ROOT / "eval" / "results" / "verbatim_asr_bench.json"
# PodcastFillers clips: real speech containing real fillers. "Uh"/"Um" clips are
# exactly the signal Chrome throws away, so they are the interesting probe.
CLIP_DIRS = [ROOT / "data" / "pfsd" / "clips" / "test" / d for d in ("Uh", "Um", "Words")]


def pick_clips(n_per_class: int) -> list[Path]:
    clips: list[Path] = []
    for d in CLIP_DIRS:
        if d.is_dir():
            clips.extend(sorted(d.glob("*.wav"))[:n_per_class])
    return clips


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--n", type=int, default=6, help="clips per class")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    if not MODEL_DIR.exists():
        print("SKIPPED -- no model at %s" % MODEL_DIR)
        return 0
    clips = pick_clips(args.n)
    if not clips:
        print("SKIPPED -- no PFSD clips under data/pfsd/clips/test")
        return 0

    import soundfile as sf
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    if args.device == "cuda" and not torch.cuda.is_available():
        print("SKIPPED -- no CUDA")
        return 0

    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    print("loading CrisperWhisper on %s (%s) ..." % (args.device, args.dtype))
    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        str(MODEL_DIR), dtype=dtype, low_cpu_mem_usage=True)
    model.to(args.device).eval()
    load_s = time.perf_counter() - t0
    print("  loaded in %.1f s" % load_s)

    vram = None
    if args.device == "cuda":
        vram = torch.cuda.memory_allocated() / 1e6
        print("  VRAM allocated: %.0f MB" % vram)

    rows = []
    for path in clips:
        audio, sr = sf.read(str(path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / sr
        feats = processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = feats.input_features.to(args.device, dtype=dtype)
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(inputs, max_new_tokens=64)
        dt = time.perf_counter() - t0
        text = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        rows.append({"clip": path.parent.name + "/" + path.name,
                     "audio_s": round(dur, 2), "proc_s": round(dt, 3),
                     "rtf": round(dt / dur, 3) if dur else None, "text": text})
        print("  %-14s %4.1fs audio  %6.2fs proc  RTF %5.2f  %r"
              % (path.parent.name + "/" + path.stem, dur, dt, dt / dur, text[:52]))

    rtfs = sorted(r["rtf"] for r in rows if r["rtf"])
    procs = sorted(r["proc_s"] for r in rows)
    # A 4 s utterance is the realistic Echo unit: one stalled sentence.
    est_4s = statistics.median(rtfs) * 4.0
    summary = {
        "status": "OK", "device": args.device, "dtype": args.dtype,
        "model": "nyralabs/CrisperWhisper2.0_turbo",
        "load_s": round(load_s, 1), "vram_mb": (round(vram) if vram else None),
        "n_clips": len(rows),
        "rtf_median": round(statistics.median(rtfs), 3),
        "rtf_p90": round(rtfs[min(len(rtfs) - 1, int(0.9 * len(rtfs)))], 3),
        "proc_s_median": round(statistics.median(procs), 3),
        "est_latency_4s_utterance_s": round(est_4s, 2),
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if OUT.exists():
        try:
            prev = json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev[args.device] = summary          # keep both devices in one file
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")

    print("")
    print("  device=%s  RTF median %.2f  ->  a 4 s utterance costs ~%.1f s"
          % (args.device, summary["rtf_median"], est_4s))
    print("  (Echo's whole stall->word budget is ~1.5-2 s. ASR spends from the")
    print("   same budget as prediction, so RTF must leave room for the LLM.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
