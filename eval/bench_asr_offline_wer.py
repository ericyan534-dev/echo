"""Is Echo's STREAMING WRAPPER costing accuracy, or is aphasic speech just hard?

WHY THIS EXISTS
---------------
`eval/bench_asr_aphasia_wer.py` measured WER 0.402 on APROCSA and that number
became the binding constraint on the whole product: the predictor is handed
fragments with two words in five wrong, so no prompt and no trigger can help.
But that 0.402 was measured through `VerbatimASR` -- LocalAgreement-2 over a
growing window, force-committed whenever the VAD reports silence -- and the
wrapper had never been separated from the model.

That separation matters here more than it would on fluent speech. Aphasic
speech is full of long INTERNAL pauses; a silence force-commit fires on every
one of them, locking in a hypothesis formed from a fraction of the utterance
and (when the window is then retired) throwing away the acoustic context
Whisper needs. If that is what the 0.402 is made of, it is a policy bug, not a
property of the model.

FOUR CONFIGURATIONS, EACH REMOVING ONE MORE THING
-------------------------------------------------
Every arm is scored by the SAME scorer (`bench_asr_aphasia_wer.wer_for`
semantics: per-utterance alignment against the clinician's CHAT text, fillers
stripped from both sides), so the differences are the configuration and
nothing else.

  stream          VerbatimASR at shipped defaults. 0.375 as it now ships
                  (silence_commit_ms=700); it was the 0.402 arm at 280 ms.
  stream_wts      Same, but word_timestamps=True. Isolates one specific
                  suspicion: with timestamps off the wrapper SPREADS words
                  evenly across the window's voiced span (see
                  `_assign_times`), and the scorer assigns hypothesis words to
                  reference utterances BY TIME. Interpolated times can push a
                  correctly recognised word into the neighbouring utterance,
                  where it counts twice -- a deletion here and an insertion
                  there. That is alignment error being reported as recognition
                  error, and it costs nothing to check.
  offline         One `model.transcribe()` over the whole region, the model's
                  own longform continuation strategy, word timestamps on. No
                  streaming, no force-commit, no window retirement.
  offline_oracle  Offline, plus the clinician's own utterance boundaries: each
                  utterance is decoded inside a padded window and words are
                  selected by timestamp. This is the accuracy FLOOR available
                  to any amount of streaming/segmentation work, because it is
                  handed the segmentation for free.

TWO WERS PER ARM, AND WHY
-------------------------
`wer` is per-utterance, the shipped metric. `wer_concat` scores the same words
after concatenating every in-region participant utterance into one reference
and one hypothesis. It is the same audio and the same words -- only the
boundaries are gone. If `wer_concat` is materially lower than `wer`, the gap
is the cost of assigning words to utterances by timestamp, and it is not
recognition error at all.

    python eval/bench_asr_offline_wer.py --configs stream,offline,offline_oracle
"""
from __future__ import annotations

import argparse
import json
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
from backend.stt.verbatim import SR, VerbatimASR, get_model  # noqa: E402
from backend.timeline import norm  # noqa: E402
from eval.bench_asr_aphasia_wer import FILLER, FRAME_MS, levenshtein  # noqa: E402
from eval.run_aphasia_eval import CACHE, TRANSCRIPTS, _decode, _encode, load_region  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "asr_offline_vs_stream.json"

# The scorer's window: a hypothesis word belongs to an utterance if its end
# lands within 250 ms of the clinician's media bullet. Kept identical to
# bench_asr_aphasia_wer so the arms are comparable to the shipped 0.402.
SLACK_MS = 250


# --------------------------------------------------------------------------
# hypothesis producers -- each returns [(text, end_ms_relative_to_region)]
# --------------------------------------------------------------------------
def hyp_stream(pid, audio, args, word_timestamps: bool = False, **kw):
    """Replay the region through VerbatimASR exactly as the live path does.

    Commit lag is collected here as well as words, because an accuracy gain
    bought with a longer window is not free: the window is what sets both the
    cost per transcribe and how long a word waits before the detector can see
    it. Reporting WER without it would hide the trade.

    The lag is measured on the AUDIO clock (`asr.now_ms`), not the wall clock,
    so a replay under GPU contention gives the same lag as an idle one --
    `sync=True` blocks feed() until the transcribe returns, so every commit
    lands at a deterministic audio offset. RTF is the opposite: it is wall
    time and it is only meaningful on an idle accelerator.
    """
    asr = VerbatimASR(model_name=args.model, mode=args.mode, sync=True,
                      word_timestamps=word_timestamps, **kw)
    step = int(SR * FRAME_MS / 1000)
    out: list[tuple[str, int]] = []
    lags: list[int] = []
    silent_since: int | None = None
    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        items = asr.feed(pcm)
        was_silent = silent_since is not None
        if asr.speech_prob >= 0.5:
            silent_since = None
        elif not was_silent:
            silent_since = asr.now_ms
        for item in items:
            if isinstance(item, Word):
                out.append((item.text, item.end_ms))
                # Only commits that land during silence matter for a stall:
                # that is the instant the predictor needs the last word.
                if silent_since is not None:
                    lags.append(asr.now_ms - silent_since)
    asr.close()
    return out, lags


def hyp_offline(pid, audio, args, **kw):
    """One pass over the whole region. No wrapper at all.

    Word timestamps are ON here even though the live path runs without them:
    the scorer needs a time per word to assign it to an utterance, and using
    the model's own alignment rather than an interpolation is the point of
    this arm -- it removes the streaming policy AND the interpolation, and
    `stream_wts` separates the two.
    """
    model = get_model(args.model)
    res = model.transcribe(audio, sr=SR, language="en", mode=args.mode,
                           word_timestamps=True, **kw)
    if res.words:
        return [(w.word, int((w.end if w.end is not None else 0) * 1000))
                for w in res.words]
    # No alignment came back: fall back to spreading, and say so, rather than
    # silently scoring a config whose times are fabricated.
    toks = (res.text or "").split()
    dur_ms = int(len(audio) * 1000 / SR)
    return [(t, int((i + 1) * dur_ms / max(1, len(toks)))) for i, t in enumerate(toks)]


def hyp_offline_oracle(pid, audio, args, utts, ctx_pad_s: float = 3.0, **kw):
    """Decode each clinician-bounded utterance inside a padded window.

    The padding is deliberate and is the difference between an oracle and a
    handicap. Whisper degrades badly on sub-second clips (this repo measured
    0.217 dysfluency recall on 1 s windows for exactly that reason), so
    decoding a 700 ms utterance in isolation would measure the short-context
    penalty, not the segmentation gain. The window carries `ctx_pad_s` of real
    neighbouring audio and words are then selected back to the utterance by
    their own timestamps.
    """
    model = get_model(args.model)
    off = args.skip * 1000
    out = []
    for u in utts:
        a0 = max(0, int((u["start_ms"] - off - ctx_pad_s * 1000) * SR / 1000))
        a1 = min(len(audio), int((u["end_ms"] - off + ctx_pad_s * 1000) * SR / 1000))
        if a1 - a0 < SR // 4:
            continue
        clip = audio[a0:a1]
        res = model.transcribe(clip, sr=SR, language="en", mode=args.mode,
                               word_timestamps=True)
        base_ms = int(a0 * 1000 / SR)
        lo, hi = u["start_ms"] - off - SLACK_MS, u["end_ms"] - off + SLACK_MS
        for w in (res.words or []):
            if w.end is None:
                continue
            e = base_ms + int(w.end * 1000)
            if lo <= e <= hi:
                out.append((w.word, e))
    return out


# --------------------------------------------------------------------------
def in_region(parsed, pid, args):
    off = args.skip * 1000
    return [u for u in parsed[pid]["utterances"]
            if u["is_participant"] and u["start_ms"] is not None
            and u["start_ms"] >= off and u["end_ms"] <= off + args.region * 1000]


def score(hyp_words, utts, args):
    """Per-utterance WER plus the boundary-free concatenated WER."""
    off = args.skip * 1000
    err = nref = nhyp = 0
    cat_ref: list[str] = []
    cat_hyp: list[str] = []
    per_utt = []
    for u in utts:
        ref = [w for w in u["text"].lower().split() if w and not FILLER.match(w)]
        if not ref:
            continue
        hyp = [norm(t) for t, e in hyp_words
               if u["start_ms"] - SLACK_MS <= e + off <= u["end_ms"] + SLACK_MS]
        hyp = [h for h in hyp if h and not FILLER.match(h)]
        e = levenshtein(ref, hyp)
        err += e
        nref += len(ref)
        nhyp += len(hyp)
        per_utt.append(min(1.0, e / len(ref)))
        cat_ref.extend(ref)
        cat_hyp.extend(hyp)
    cat_err = levenshtein(cat_ref, cat_hyp)
    return {"errors": err, "n_ref": nref, "n_hyp": nhyp,
            "wer": round(err / max(1, nref), 4),
            "cat_errors": cat_err,
            "wer_concat": round(cat_err / max(1, len(cat_ref)), 4),
            "median_utt_wer": round(float(np.median(per_utt)), 4) if per_utt else None,
            "n_utt": len(per_utt)}


CONFIGS = {
    "stream":         dict(fn="stream", kw={"word_timestamps": False}),
    "stream_wts":     dict(fn="stream", kw={"word_timestamps": True}),
    "offline":        dict(fn="offline", kw={}),
    "offline_oracle": dict(fn="oracle", kw={}),
}

# Streaming policy overrides ride on the config name so a sweep is one command
# and every arm is regenerable from the string in the results file:
#   stream@max_window_s=20@reset_window_s=10@silence_commit_ms=700
_FLOAT_KW = {"max_window_s", "reset_window_s", "max_turn_s", "chunk_duration",
             "stride", "ctx_pad_s"}


def parse_config(name: str) -> tuple[str, dict]:
    head, _, rest = name.partition("@")
    kw: dict = {}
    for part in rest.split("@") if rest else []:
        if not part:
            continue
        k, _, v = part.partition("=")
        if k in _FLOAT_KW:
            kw[k] = float(v)
        elif v.lstrip("-").isdigit():
            kw[k] = int(v)
        else:
            kw[k] = v
    return head, kw


def hyp_cached(name, pid, audio, args, utts):
    """Hypotheses are cached per (config, pid, model, mode, region) -- the
    transcription is the whole cost and none of it depends on the scorer."""
    head, extra_kw = parse_config(name)
    tag = "%s_%s_%s_%s_%d_%d" % (name.replace("@", "_").replace("=", ""), pid,
                                 args.model, args.mode, args.skip, args.region)
    path = CACHE / ("hyp_" + tag + ".json")
    if path.exists() and not args.force:
        d = json.loads(path.read_text(encoding="utf-8"))
        # Older caches (written before commit lag was collected) are a bare
        # word list rather than {"words": ..., "lags": ...}.
        if isinstance(d, list):
            return d, [], 0.0
        return d["words"], d.get("lags", []), 0.0
    spec = CONFIGS.get(head, {"fn": "stream", "kw": {}})
    kw = dict(spec["kw"])
    kw.update(extra_kw)
    t0 = time.perf_counter()
    lags: list[int] = []
    if spec["fn"] == "stream":
        words, lags = hyp_stream(pid, audio, args, **kw)
    elif spec["fn"] == "offline":
        words = hyp_offline(pid, audio, args, **kw)
    else:
        words = hyp_offline_oracle(pid, audio, args, utts, **kw)
    wall = time.perf_counter() - t0
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"words": words, "lags": lags}), encoding="utf-8")
    return words, lags, wall


def reuse_shipped(pid, args):
    """The shipped `stream` arm already has a cached item stream from
    run_aphasia_eval.py. Reuse it rather than re-transcribing: it is the exact
    byte-identical stream the detection and prediction evals were run on.

    That cache key carries no config, so it means "whatever VerbatimASR
    currently ships" and nothing else. Reusing it for a NON-default `stream`
    would silently score the wrong policy -- which is why the reuse is gated on
    `name == "stream"` and no overrides."""
    p = CACHE / ("asr_%s_%d_%d_%s.json" % (pid, args.skip, args.region, args.mode))
    if not p.exists():
        return None
    items = [_decode(d) for d in json.loads(p.read_text(encoding="utf-8"))]
    return [(it.text, it.end_ms) for _, it in items if isinstance(it, Word)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--configs", default="stream,stream_wts,offline,offline_oracle")
    ap.add_argument("--participants", default="1554,1713,1731,1738,1833,1944")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                    help="start a new results table instead of merging into the old one")
    args = ap.parse_args()

    parsed = load_all(TRANSCRIPTS)
    pids = [p.strip() for p in args.participants.split(",") if p.strip()]
    names = [c.strip() for c in args.configs.split(",") if c.strip()]

    results = {}
    if OUT.exists() and not args.fresh:
        # Arms are expensive and are run in separate invocations as the GPU
        # frees up; merging keeps one comparable table instead of a directory
        # of half-tables.
        try:
            results = json.loads(OUT.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    for name in names:
        tot = {"errors": 0, "n_ref": 0, "n_hyp": 0, "cat_errors": 0, "n_utt": 0}
        per_pid = {}
        wall = audio_s = 0.0
        all_lags: list[int] = []
        for pid in pids:
            audio, _ = load_region(pid, args.region, args.skip)
            if audio is None:
                continue
            utts = in_region(parsed, pid, args)
            words = None
            if name == "stream" and args.model == "turbo" and not args.force:
                words = reuse_shipped(pid, args)
            if words is None:
                words, lags, w = hyp_cached(name, pid, audio, args, utts)
                all_lags.extend(lags)
                wall += w
                audio_s += len(audio) / SR if w else 0.0
            r = score(words, utts, args)
            per_pid[pid] = r
            for k in tot:
                tot[k] += r[k]
            print("    %-40s %s  WER %.3f  concat %.3f  (%d ref)"
                  % (name, pid, r["wer"], r["wer_concat"], r["n_ref"]), flush=True)
        all_lags.sort()
        results[name] = {
            "wer": round(tot["errors"] / max(1, tot["n_ref"]), 4),
            "wer_concat": round(tot["cat_errors"] / max(1, tot["n_ref"]), 4),
            "n_ref_words": tot["n_ref"], "n_hyp_words": tot["n_hyp"],
            "n_utterances": tot["n_utt"],
            # Commit lag is on the audio clock and survives GPU contention.
            "commit_lag_ms_median": (all_lags[len(all_lags) // 2] if all_lags
                                     else results.get(name, {}).get("commit_lag_ms_median")),
            "commit_lag_ms_p90": (all_lags[int(0.9 * (len(all_lags) - 1))] if all_lags
                                  else results.get(name, {}).get("commit_lag_ms_p90")),
            "n_commit_lag_samples": len(all_lags) or None,
            # RTF is recorded ONLY when this run actually transcribed, and it
            # is wall time: it is a valid number only on an idle accelerator.
            # A cached arm reports null rather than a stale number.
            "rtf_contended": round(wall / audio_s, 3) if audio_s else None,
            "per_participant": per_pid,
        }
        print("  %-40s WER %.3f   concat %.3f   lag %s ms   (%d ref words, %d utts)"
              % (name, results[name]["wer"], results[name]["wer_concat"],
                 results[name]["commit_lag_ms_median"], tot["n_ref"], tot["n_utt"]),
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "corpus": "APROCSA -- 6 speakers with chronic post-stroke aphasia",
        "model": args.model, "mode": args.mode,
        "region_s": args.region, "skip_s": args.skip,
        "note": ("wer is per-utterance (the shipped metric). wer_concat scores "
                 "the same words with utterance boundaries removed; the gap "
                 "between them is the cost of assigning hypothesis words to "
                 "reference utterances by timestamp, not recognition error. "
                 "RTF is null for any arm served from cache."),
        "results": results,
    }, indent=2), encoding="utf-8")

    print("")
    print("STREAMING WRAPPER vs OFFLINE, SAME MODEL, SAME AUDIO")
    print("  %-44s %7s %7s %9s" % ("config", "WER", "concat", "lag_ms"))
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["wer"]):
        print("  %-44s %7.3f %7.3f %9s"
              % (k, v["wer"], v["wer_concat"], v.get("commit_lag_ms_median")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
