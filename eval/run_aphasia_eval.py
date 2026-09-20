"""Does Echo detect word-finding difficulty in REAL aphasic speech?

Every detection number this repo has published came from stuttered podcast
speech (SEP-28k) or fluent podcast speech (PodcastFillers). Neither is aphasia,
and aphasia is the entire point: stuttering is a MOTOR-SPEECH disorder where
the word is known and will not come out, while aphasia is a LANGUAGE disorder
where the word is not retrievable. They share surface evidence -- silent
blocks, filled pauses, repetitions -- which is why a stutter-trained detector
transfers at all, but "transfers" is a hypothesis until it is measured.

GROUND TRUTH
------------
APROCSA (Casilio et al. 2022): six people with chronic post-stroke aphasia,
transcribed in CHAT by clinicians, media-aligned. The CHAT codes mark word
finding directly -- retracings, abandoned utterances, phonological fragments,
filled pauses, timed pauses, paraphasias. See scripts/aprocsa_chat.py for
exactly which codes count and why.

    and &-um I have speech &-um (.) &-um (...) spring [//] Christmas
                                                       ^ wrong word, retraced

THE COMPARISON
--------------
A 2x2, so the two changes are separated instead of reported as one lump:

                    FillerNet acoustic      StutterNet acoustic
    intended ASR    old (shipped)           acoustic change only
    verbatim ASR    ASR change only         new

"intended" mode is a MEASURED stand-in for the browser recognizer, not a
guess: same model, same audio, it strips dysfluency to 0.060 preserved where
verbatim keeps 0.900, and Chrome itself measured 0.000 on 5,044 clips. Chrome
cannot be scripted offline, so this is the closest honest proxy and is labelled
as a proxy everywhere it appears.

The ASR is run ONCE per mode and its item stream replayed into all arms, so
the two acoustic arms see byte-identical transcripts and the 2x2 is exact
rather than approximately paired.

THIS UNDERSTATES THE CHANGE, DELIBERATELY
-----------------------------------------
Both arms here get the new VAD-gated silence ticks and the audio-derived turn
boundaries. The shipped browser path had neither: its pause trigger measured
gaps between browser transcript EVENTS, so a slow transcript looked exactly
like a speaker who had stopped. That clock fix is real and is not credited to
the "new" arm anywhere in this file. What is measured is only the difference
transcript CONTENT and the acoustic model make -- a lower bound on the total
improvement, chosen because it is the part that can be isolated cleanly.

    python eval/run_aphasia_eval.py --region-s 180
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
import soundfile as sf  # noqa: E402

from backend.acoustic.stream import AcousticStream  # noqa: E402
from backend.schemas import SilenceTick, TurnEnd, Word  # noqa: E402
from backend.stall_detector import StallDetector  # noqa: E402
from backend.stt.verbatim import SR, VerbatimASR  # noqa: E402
from backend.timeline import Timeline  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

DATA = ROOT / "data" / "aprocsa"
AUDIO = DATA / "audio"
TRANSCRIPTS = DATA / "transcripts"
FILLERNET = ROOT / "models" / "fillernet.pt"
STUTTERNET = ROOT / "models" / "stutternet.pt"
OUT = ROOT / "eval" / "results" / "aphasia_eval.json"

FRAME_MS = 100                 # the browser worklet's /ws/audio frame size
# A stall fires after the pause threshold, which lands after the last
# transcribed word of the utterance. The window is asymmetric for that reason.
PRE_MS = 250
POST_MS = 1500


def load_region(pid: str, region_s: float, skip_s: float):
    wav = AUDIO / ("%s.wav" % pid)
    if not wav.exists():
        return None, None
    info = sf.info(str(wav))
    a0 = int(skip_s * info.samplerate)
    a1 = min(info.frames, a0 + int(region_s * info.samplerate))
    audio, sr = sf.read(str(wav), start=a0, stop=a1, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        idx = np.linspace(0, len(audio) - 1, int(len(audio) * SR / sr))
        audio = np.interp(idx, np.arange(len(audio)), audio).astype("float32")
    return audio, int(skip_s * 1000)


CACHE = ROOT / "eval" / "results" / "cache" / "aprocsa"


def _encode(now_ms: int, item) -> dict:
    if isinstance(item, Word):
        return {"t": now_ms, "k": "w", "text": item.text,
                "s": item.start_ms, "e": item.end_ms}
    if isinstance(item, SilenceTick):
        return {"t": now_ms, "k": "s", "at": item.at_ms}
    if isinstance(item, TurnEnd):
        return {"t": now_ms, "k": "t"}
    return {"t": now_ms, "k": "a", "kind": item.kind,
            "at": item.at_ms, "conf": item.confidence}


def _decode(d: dict) -> tuple[int, object]:
    if d["k"] == "w":
        return d["t"], Word(text=d["text"], start_ms=d["s"], end_ms=d["e"], is_final=True)
    if d["k"] == "s":
        return d["t"], SilenceTick(at_ms=d["at"])
    if d["k"] == "t":
        return d["t"], TurnEnd()
    from backend.schemas import AcousticEvent

    return d["t"], AcousticEvent(kind=d["kind"], at_ms=d["at"], confidence=d["conf"])


def cached(key: str, build):
    """Persist a channel's item stream.

    The ASR is by far the most expensive part of this eval and its output does
    not depend on the detector's thresholds. Caching it is what makes it
    possible to iterate on the firing rate at all -- otherwise every threshold
    change costs a full re-transcription of five hours of audio.
    """
    path = CACHE / (key + ".json")
    if path.exists():
        return [_decode(d) for d in json.loads(path.read_text(encoding="utf-8"))]
    items = build()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([_encode(t, it) for t, it in items]), encoding="utf-8")
    return items


def run_asr(audio: np.ndarray, mode: str) -> list[tuple[int, object]]:
    """Replay the region through the ASR, keeping every item with the audio
    clock reading at the moment it was emitted."""
    asr = VerbatimASR(mode=mode, sync=True)
    step = int(SR * FRAME_MS / 1000)
    out = []
    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        for item in asr.feed(pcm):
            out.append((asr.now_ms, item))
    asr.close()
    return out


def run_acoustic(audio: np.ndarray, **kw) -> list[tuple[int, object]]:
    stream = AcousticStream(**kw)
    step = int(SR * FRAME_MS / 1000)
    out = []
    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        for ev in stream.feed(pcm):
            out.append((ev.at_ms, ev))
    return out


def run_detector(items: list[tuple[int, object]], pause_ms: int,
                 min_gap_ms: int = 0) -> list[dict]:
    """Merge the two channels by time and collect what the detector fires."""
    det = StallDetector(pause_ms=pause_ms, timeline=Timeline(), min_gap_ms=min_gap_ms)
    fires: list[dict] = []
    for _, item in sorted(items, key=lambda kv: kv[0]):
        if isinstance(item, Word):
            ev = det.observe_word(item)
        elif isinstance(item, SilenceTick):
            ev = det.observe_silence(item.at_ms)
        elif isinstance(item, TurnEnd):
            det.reset()
            continue
        else:
            ev = det.observe_acoustic(item)
        if ev is not None:
            fires.append({"at_ms": ev.at_ms, "trigger": ev.trigger,
                          "fragment": ev.fragment[-90:]})
    return fires


def score(fires: list[dict], utts: list[dict], offset_ms: int,
          region_ms: int) -> dict:
    """Utterance-level hit / false-alarm rates, plus partner-speech fires."""
    lo, hi = offset_ms, offset_ms + region_ms
    par = [u for u in utts
           if u["is_participant"] and u["start_ms"] is not None
           and u["start_ms"] >= lo and u["end_ms"] <= hi]
    inv = [u for u in utts
           if not u["is_participant"] and u["start_ms"] is not None
           and u["start_ms"] >= lo and u["end_ms"] <= hi]
    times = sorted(f["at_ms"] + offset_ms for f in fires)

    def fired_in(u, pre=PRE_MS, post=POST_MS) -> bool:
        a, b = u["start_ms"] - pre, u["end_ms"] + post
        i = np.searchsorted(times, a)
        return i < len(times) and times[i] <= b

    pos = [u for u in par if u["word_search"]]
    neg = [u for u in par if not u["word_search"]]
    hit = sum(fired_in(u) for u in pos)
    fa = sum(fired_in(u) for u in neg)
    # A fire while the CLINICIAN is speaking is unambiguously wrong -- it is
    # the "cannot isolate other speakers" defect, measured on real two-speaker
    # audio for the first time (the speaker gate was only ever validated on
    # synthetic mixes).
    partner = sum(fired_in(u, pre=0, post=0) for u in inv)
    return {
        "n_word_search": len(pos), "n_fluent": len(neg), "n_partner": len(inv),
        "recall": round(hit / len(pos), 4) if pos else None,
        "false_alarm_rate": round(fa / len(neg), 4) if neg else None,
        "partner_fire_rate": round(partner / len(inv), 4) if inv else None,
        "fires": len(fires),
        "fires_per_min": round(len(fires) / (region_ms / 60000.0), 2),
        "triggers": {t: sum(1 for f in fires if f["trigger"] == t)
                     for t in sorted({f["trigger"] for f in fires})},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-s", type=float, default=180.0)
    ap.add_argument("--skip-s", type=float, default=60.0,
                    help="skip the opening (consent/setup chatter)")
    ap.add_argument("--pause-ms", type=int, default=1300)
    ap.add_argument("--min-gap-ms", type=int, default=0,
                    help="detector-level refractory across all triggers")
    ap.add_argument("--sweep-gap", default="0,2000,3000,4000,6000",
                    help="min_gap_ms values to sweep for the shipped arm")
    ap.add_argument("--participants", default="")
    args = ap.parse_args()

    if not AUDIO.is_dir() or not any(AUDIO.glob("*.wav")):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED",
                                   "reason": "no APROCSA audio -- run scripts/fetch_aprocsa.py"},
                                  indent=2), encoding="utf-8")
        print("SKIPPED -- no APROCSA audio")
        return 0

    parsed = load_all(TRANSCRIPTS)
    pids = [p.strip() for p in args.participants.split(",") if p.strip()] or sorted(parsed)

    arms = {
        "old (intended ASR + FillerNet)": ("intended", {"model_path": str(FILLERNET)}),
        "verbatim ASR + FillerNet": ("verbatim", {"model_path": str(FILLERNET)}),
        "intended ASR + StutterNet": ("intended", {"stutter_model": str(STUTTERNET)}),
        "new (verbatim ASR + StutterNet)": ("verbatim", {"stutter_model": str(STUTTERNET)}),
    }
    if not STUTTERNET.exists():
        arms = {k: v for k, v in arms.items() if "StutterNet" not in k}
        print("NOTE: no StutterNet checkpoint; running ASR arms only")

    print("APROCSA aphasia evaluation -- %d participants, %.0f s each"
          % (len(pids), args.region_s))
    print("  ground truth: CHAT word-finding codes (scripts/aprocsa_chat.py)")

    per_arm: dict[str, list[dict]] = {k: [] for k in arms}
    sweep_rows: dict[tuple[str, int], list[dict]] = {}
    details = {}
    t_start = time.time()
    for pid in pids:
        audio, offset_ms = load_region(pid, args.region_s, args.skip_s)
        if audio is None:
            print("  %s: no audio, skipped" % pid)
            continue
        region_ms = int(len(audio) * 1000 / SR)
        utts = parsed[pid]["utterances"]
        print("  %s: %.0f s from %.0f s" % (pid, region_ms / 1000, args.skip_s), flush=True)

        # ASR once per mode; acoustic once per model. The arms are then exact
        # pairings of those streams rather than four independent runs.
        tag = "%s_%d_%d" % (pid, int(args.skip_s), int(args.region_s))
        modes = {m for m, _ in arms.values()}
        asr_items = {m: cached("asr_%s_%s" % (tag, m),
                               lambda m=m: run_asr(audio, m)) for m in modes}
        ac_keys = {json.dumps(kw, sort_keys=True) for _, kw in arms.values()}
        ac_items = {}
        for k in ac_keys:
            # The checkpoint fingerprint is part of the key: a retrained
            # StutterNet must not silently reuse the previous model's events.
            if "stutter_model" in k:
                st = STUTTERNET.stat()
                name = "stutter-%d" % int(st.st_mtime)
            else:
                name = "filler"
            ac_items[k] = cached("ac_%s_%s" % (tag, name),
                                 lambda k=k: run_acoustic(audio, **json.loads(k)))

        for name, (mode, kw) in arms.items():
            items = asr_items[mode] + ac_items[json.dumps(kw, sort_keys=True)]
            fires = run_detector(items, args.pause_ms, args.min_gap_ms)
            s = score(fires, utts, offset_ms, region_ms)
            s["participant"] = pid
            per_arm[name].append(s)
            details.setdefault(pid, {})[name] = {
                "score": s, "sample_fires": fires[:6],
                "n_words": sum(1 for _, it in asr_items[mode] if isinstance(it, Word)),
            }
            print("    %-34s recall %-6s FA %-6s partner %-6s (%d fires)"
                  % (name, s["recall"], s["false_alarm_rate"],
                     s["partner_fire_rate"], s["fires"]), flush=True)

        # Sweep the refractory on both end arms, so the interruption-rate
        # curve is reported as a curve rather than as one chosen point.
        for arm_name in (list(arms)[0], list(arms)[-1]):
            mode, kw = arms[arm_name]
            items = asr_items[mode] + ac_items[json.dumps(kw, sort_keys=True)]
            for gap in [int(g) for g in args.sweep_gap.split(",") if g.strip()]:
                f = run_detector(items, args.pause_ms, gap)
                sw = score(f, utts, offset_ms, region_ms)
                sweep_rows.setdefault((arm_name, gap), []).append(sw)

    def pooled(rows, key_num, key_den):
        num = sum(r[key_num] for r in rows)
        den = sum(r[key_den] for r in rows)
        return round(num / den, 4) if den else None

    summary = {}
    for name, rows in per_arm.items():
        if not rows:
            continue
        # Pooled over utterances, not averaged over participants: the six
        # differ by 3x in how much they say, and averaging rates would weight
        # a quiet participant the same as a talkative one.
        hits = sum(round((r["recall"] or 0) * r["n_word_search"]) for r in rows)
        fas = sum(round((r["false_alarm_rate"] or 0) * r["n_fluent"]) for r in rows)
        pfs = sum(round((r["partner_fire_rate"] or 0) * r["n_partner"]) for r in rows)
        n_pos = sum(r["n_word_search"] for r in rows)
        n_neg = sum(r["n_fluent"] for r in rows)
        n_par = sum(r["n_partner"] for r in rows)
        summary[name] = {
            "recall": round(hits / n_pos, 4) if n_pos else None,
            "false_alarm_rate": round(fas / n_neg, 4) if n_neg else None,
            "partner_fire_rate": round(pfs / n_par, 4) if n_par else None,
            "n_word_search": n_pos, "n_fluent": n_neg, "n_partner": n_par,
            "fires_per_min": round(float(np.mean([r["fires_per_min"] for r in rows])), 2),
            "per_participant": rows,
        }

    sweep_summary = []
    for (arm_name, gap), rows in sorted(sweep_rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        hits = sum(round((r["recall"] or 0) * r["n_word_search"]) for r in rows)
        fas = sum(round((r["false_alarm_rate"] or 0) * r["n_fluent"]) for r in rows)
        pfs = sum(round((r["partner_fire_rate"] or 0) * r["n_partner"]) for r in rows)
        n_pos = sum(r["n_word_search"] for r in rows)
        n_neg = sum(r["n_fluent"] for r in rows)
        n_par = sum(r["n_partner"] for r in rows)
        sweep_summary.append({
            "arm": arm_name, "min_gap_ms": gap,
            "recall": round(hits / n_pos, 4) if n_pos else None,
            "false_alarm_rate": round(fas / n_neg, 4) if n_neg else None,
            "partner_fire_rate": round(pfs / n_par, 4) if n_par else None,
            "fires_per_min": round(float(np.mean([r["fires_per_min"] for r in rows])), 2),
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "refractory_sweep": sweep_summary,
        "dataset": "APROCSA (Casilio et al. 2022) -- 6 speakers, chronic post-stroke aphasia",
        "region_s": args.region_s, "skip_s": args.skip_s, "pause_ms": args.pause_ms,
        "ground_truth": "CHAT word-finding codes; see scripts/aprocsa_chat.py",
        "proxy_note": ("'intended' mode stands in for the browser recognizer. Chrome "
                       "cannot be scripted offline; intended mode is the same model on "
                       "the same audio with dysfluency stripped (0.060 preserved vs "
                       "0.900 verbatim), and Chrome measured 0.000 on 5,044 clips."),
        "caveats": [
            "Six speakers. Not a population estimate.",
            ("A 'false alarm' is an utterance the clinician did not code as a word "
             "search. CHAT coding is utterance-level and conservative, so some of "
             "these are real word searches that were not coded."),
            "Threshold fitting on this set would invalidate it; pause_ms is the shipped default.",
        ],
        "wall_s": round(time.time() - t_start, 1),
        "summary": summary,
        "details": details,
    }, indent=2), encoding="utf-8")

    print("")
    print("POOLED OVER %d PARTICIPANTS" % len(pids))
    print("  %-34s %8s %8s %8s" % ("arm", "recall", "FA", "partner"))
    for name, s in summary.items():
        print("  %-34s %8s %8s %8s"
              % (name, s["recall"], s["false_alarm_rate"], s["partner_fire_rate"]))
    if sweep_summary:
        print("")
        print("REFRACTORY SWEEP (min_gap_ms -- minimum time between suggestions)")
        print("  %-34s %7s %8s %8s %8s %9s"
              % ("arm", "gap", "recall", "FA", "partner", "fires/min"))
        for r in sweep_summary:
            print("  %-34s %7d %8s %8s %8s %9s"
                  % (r["arm"], r["min_gap_ms"], r["recall"], r["false_alarm_rate"],
                     r["partner_fire_rate"], r["fires_per_min"]))

    print("")
    print("  recall  = word-search utterances Echo fired on (higher better)")
    print("  FA      = fluent utterances Echo fired on (lower better)")
    print("  partner = clinician utterances Echo fired on (lower better)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
