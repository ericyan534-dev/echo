"""How accurate is the ASR on APHASIC speech? The number everything depends on.

WHY THIS EXISTS
---------------
Echo's ASR was chosen on dysfluency preservation -- does "[UM]" survive into
the transcript -- and `turbo` won that at 0.900 with the lowest latency. What
was never measured is whether the words AROUND the dysfluency are right, on the
speech this product is actually for.

They largely are not. Measured against APROCSA's clinician transcripts, at
the shipped streaming defaults:

    verbatim  WER 0.375     (0.402 before silence_commit_ms moved 280 -> 700)
    intended  WER 0.483

Two of every five words wrong. That number was read, for a while, as the
explanation for the two results that had been hard to interpret:

  * word prediction sits at the frequency-baseline floor (3/51 top-3). The
    predictor is given fragments like "And then [noise] [noise] and cut off
    [UM] Cut off And" when the speaker was reaching for "christmas".
  * detection metrics could not separate a detector from a metronome, partly
    because the evidence the detector reads is itself unreliable.

THE FIRST OF THOSE WAS TESTED AND DID NOT HOLD. Improving this number from
0.402 to 0.375 changed 46 of the 51 fragments the predictor sees and left
strict top-3 at 3/51, exactly where it was and exactly on the frequency
baseline (eval/run_aphasia_prediction.py, docs/VERSIONS.md). So "ASR accuracy
is the binding constraint" is a hypothesis this script can measure the input
to but has not confirmed: 2.7 points bought nothing downstream. A much larger
delta -- the offline arm is 0.288 -- is what would actually test it.

WHAT IS AND IS NOT SCORED
-------------------------
Fillers are stripped from BOTH sides before scoring. They are scored separately
by eval/bench_asr_models.py (dysfluency preservation), they are the one thing
verbatim mode deliberately adds, and leaving them in would let a model score
better on WER by transcribing hesitations the reference happens to spell the
same way. This measures the CONTENT words -- the ones a prediction has to be
built from.

Reference is `scripts/aprocsa_chat.clean_text`: what the clinician transcribed
the participant as actually producing, with CHAT annotation removed. Hypothesis
is the wearer-gated committed word stream, windowed to the utterance's own
media bullet.

    python eval/bench_asr_aphasia_wer.py --models turbo,large
"""
from __future__ import annotations

import argparse
import inspect
import json
import re
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from backend.schemas import Word  # noqa: E402
from backend.stt.verbatim import SR, VerbatimASR  # noqa: E402
from backend.timeline import norm  # noqa: E402
from eval.run_aphasia_eval import CACHE, TRANSCRIPTS, _decode, _encode, load_region  # noqa: E402
from eval.tune_aphasia_detector import apply_speaker, speaker_map  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "asr_aphasia_wer.json"
FILLER = re.compile(r"^(uh+|um+|uhm|erm|hmm+|mm+)$")
FRAME_MS = 100

# Read off the class rather than repeated as a literal. The shared cache key
# `asr_<pid>_<skip>_<region>_<mode>.json` is written by run_aphasia_eval.py at
# whatever VerbatimASR currently ships, so any literal here silently rots the
# moment a default moves -- which is exactly what happened when
# silence_commit_ms went 280 -> 700: this script would have relabelled a 700 ms
# stream as a 280 ms one and published the wrong WER against the wrong knob.
_SIG = inspect.signature(VerbatimASR.__init__).parameters
SHIPPED_SC = _SIG["silence_commit_ms"].default
SHIPPED_RW = _SIG["reset_window_s"].default


def levenshtein(a, b) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def transcribe_region(pid, model_name, mode, args):
    """Cached per (pid, model, mode) -- re-transcribing is the expensive part."""
    tag = "%s_%d_%d" % (pid, args.skip, args.region)
    sc = getattr(args, "silence_commit_ms", SHIPPED_SC)
    rw = getattr(args, "reset_window_s", SHIPPED_RW)
    suffix = ("" if (sc == SHIPPED_SC and rw == SHIPPED_RW)
              else "_sc%d_rw%.1f" % (sc, rw))
    key = "asr_%s_%s_%s%s" % (tag, mode, model_name, suffix)
    path = CACHE / (key + ".json")
    if path.exists():
        return [_decode(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    audio, _ = load_region(pid, args.region, args.skip)
    if audio is None:
        return None
    asr = VerbatimASR(model_name=model_name, mode=mode, sync=True,
                      silence_commit_ms=sc, reset_window_s=rw)
    step = int(SR * FRAME_MS / 1000)
    out = []
    t0 = time.perf_counter()
    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        for item in asr.feed(pcm):
            out.append((asr.now_ms, item))
    asr.close()
    print("      %s/%s/%s  %.0fs wall for %.0fs audio"
          % (pid, model_name, mode, time.perf_counter() - t0, len(audio) / SR),
          flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_encode(t, it) for t, it in out]), encoding="utf-8")
    return out


def transcribe_offline(pid, model_name, mode, args):
    """One pass over the whole region, no streaming, with word timestamps.

    The comparison that matters for the streaming wrapper: same audio, same
    scorer, only the incremental-commit policy differs. VerbatimASR
    force-commits whenever the VAD reports silence, and aphasic speech is full
    of long internal pauses -- so it may be committing hypotheses formed from a
    fraction of an utterance and locking in errors an offline pass would never
    make. Nobody had checked.
    """
    tag = "%s_%d_%d" % (pid, args.skip, args.region)
    path = CACHE / ("asr_%s_%s_%s_OFFLINE.json" % (tag, mode, model_name))
    if path.exists():
        return [_decode(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    audio, _ = load_region(pid, args.region, args.skip)
    if audio is None:
        return None
    from backend.stt.verbatim import get_model

    model = get_model(model_name)
    t0 = time.perf_counter()
    res = model.transcribe(audio, sr=SR, language=mode == "verbatim" and "en" or "en",
                           mode=mode, word_timestamps=True)
    out = []
    for w in (res.words or []):
        if w.start is None:
            continue
        out.append((int(w.end * 1000) if w.end else int(w.start * 1000),
                    Word(text=w.word.strip(), start_ms=int(w.start * 1000),
                         end_ms=int((w.end if w.end else w.start) * 1000),
                         is_final=True)))
    print("      %s/%s/%s OFFLINE  %.0fs wall, %d words"
          % (pid, model_name, mode, time.perf_counter() - t0, len(out)), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_encode(t, it) for t, it in out]), encoding="utf-8")
    return out


def wer_for(items, parsed, pid, args):
    items = apply_speaker(items, speaker_map(pid, args.skip, args.region),
                          args.skip * 1000, 0.35)
    words = [it for _, it in items if isinstance(it, Word)]
    off = args.skip * 1000
    err = nref = nhyp = 0
    per_utt = []
    for u in parsed[pid]["utterances"]:
        if not u["is_participant"] or u["start_ms"] is None:
            continue
        if u["start_ms"] < off or u["end_ms"] > off + args.region * 1000:
            continue
        ref = [w for w in u["text"].lower().split() if w and not FILLER.match(w)]
        if not ref:
            continue
        hyp = [norm(w.text) for w in words
               if u["start_ms"] - 250 <= w.end_ms + off <= u["end_ms"] + 250]
        hyp = [h for h in hyp if h and not FILLER.match(h)]
        e = levenshtein(ref, hyp)
        err += e
        nref += len(ref)
        nhyp += len(hyp)
        per_utt.append(min(1.0, e / len(ref)))
    return {"errors": err, "n_ref": nref, "n_hyp": nhyp,
            "wer": round(err / max(1, nref), 4),
            "median_utt_wer": round(float(np.median(per_utt)), 4) if per_utt else None,
            "n_utt": len(per_utt)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--models", default="turbo,large")
    ap.add_argument("--modes", default="verbatim")
    ap.add_argument("--silence-commit-ms", type=int, default=SHIPPED_SC,
                    help="silence before a forced commit; defaults to whatever "
                         "VerbatimASR ships (currently %d ms)" % SHIPPED_SC)
    ap.add_argument("--reset-window-s", type=float, default=SHIPPED_RW)
    ap.add_argument("--offline", action="store_true",
                    help="one-pass transcription instead of the streaming wrapper")
    ap.add_argument("--participants", default="1554,1713,1731,1738,1833,1944")
    args = ap.parse_args()

    parsed = load_all(TRANSCRIPTS)
    pids = [p.strip() for p in args.participants.split(",") if p.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    results = {}
    for model_name in models:
        for mode in modes:
            tot = {"errors": 0, "n_ref": 0, "n_hyp": 0, "n_utt": 0}
            per_pid = {}
            for pid in pids:
                # `turbo` is already cached under the original key from
                # run_aphasia_eval.py; reuse it rather than transcribing twice.
                if args.offline:
                    items = transcribe_offline(pid, model_name, mode, args)
                elif (model_name == "turbo"
                      and args.silence_commit_ms == SHIPPED_SC
                      and args.reset_window_s == SHIPPED_RW):
                    p = CACHE / ("asr_%s_%d_%d_%s.json"
                                 % (pid, args.skip, args.region, mode))
                    items = ([_decode(d) for d in
                              json.loads(p.read_text(encoding="utf-8"))]
                             if p.exists() else transcribe_region(pid, model_name, mode, args))
                elif args.offline:
                    items = transcribe_offline(pid, model_name, mode, args)
                else:
                    items = transcribe_region(pid, model_name, mode, args)
                if items is None:
                    continue
                r = wer_for(items, parsed, pid, args)
                per_pid[pid] = r
                for k in tot:
                    tot[k] += r[k]
            key = "%s/%s%s" % (model_name, mode,
                               "/offline" if args.offline
                               else ("" if args.silence_commit_ms == SHIPPED_SC
                                     else "/sc%d" % args.silence_commit_ms))
            results[key] = {
                "wer": round(tot["errors"] / max(1, tot["n_ref"]), 4),
                "n_ref_words": tot["n_ref"], "n_hyp_words": tot["n_hyp"],
                "n_utterances": tot["n_utt"], "per_participant": per_pid,
            }
            print("  %-18s WER %.3f  (%d ref words, %d utterances)"
                  % (key, results[key]["wer"], tot["n_ref"], tot["n_utt"]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged = {}
    if OUT.exists():
        try:
            merged = (json.loads(OUT.read_text(encoding="utf-8")) or {}).get("results", {})
        except Exception:
            merged = {}
    merged.update(results)
    results = merged
    OUT.write_text(json.dumps({
        "status": "OK",
        "corpus": "APROCSA -- 6 speakers with chronic post-stroke aphasia",
        "reference": "CHAT clean_text (clinician transcription), fillers stripped",
        "note": ("Fillers are stripped from both sides. They are measured "
                 "separately as dysfluency preservation; including them here "
                 "would reward a model for transcribing hesitations rather "
                 "than for getting the content words a prediction needs."),
        "region_s": args.region, "skip_s": args.skip,
        "results": results,
    }, indent=2), encoding="utf-8")

    print("")
    print("WORD ERROR RATE ON APHASIC SPEECH")
    print("  %-18s %8s" % ("model/mode", "WER"))
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["wer"]):
        print("  %-18s %8.3f" % (k, v["wer"]))
    print("")
    print("  Every downstream number is bounded by this. A fragment with two")
    print("  words in five wrong cannot support word prediction, however good")
    print("  the predictor or the trigger.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
