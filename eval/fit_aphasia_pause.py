"""How long is a word-search pause in aphasia? Measure it, then set the threshold.

Echo ships STALL_PAUSE_MS=1300. That number came from the stuttering and
disfluency literature, not from aphasic speech, and it is the single most
consequential constant in the detector: too low and the aid interrupts a
speaker who is merely thinking, too high and it arrives after they have given
up. Aphasic pause distributions are known to be population-specific, so
inheriting a threshold from another population is a guess wearing a citation.

WHAT THIS MEASURES
------------------
For every APROCSA participant utterance, the Silero VAD segments the audio
inside the utterance's own CHAT time bullet into speech and silence. Every
internal silence >= MIN_GAP_MS is one observation, labelled by whether the
clinician coded that utterance as a word search.

The output is a distribution, not a fitted model: "silences inside word-search
utterances vs silences inside fluent ones". A threshold follows from it, but
the distribution is the finding and survives any later change of decision rule.

WHY THIS IS NOT CIRCULAR
------------------------
The threshold is chosen against a property of the SPEECH (how long silences
are), not against Echo's own recall/false-alarm curve. Tuning directly on the
evaluation metric would make eval/run_aphasia_eval.py a training-set number.
The sweep at the bottom is printed as a sweep and labelled as one -- it is
context for the choice, not the choice.

    python eval/fit_aphasia_pause.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

from backend.acoustic.stream import VAD_CHUNK, _get_vad_instance  # noqa: E402
from backend.stt.verbatim import SR  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

AUDIO = ROOT / "data" / "aprocsa" / "audio"
TRANSCRIPTS = ROOT / "data" / "aprocsa" / "transcripts"
OUT = ROOT / "eval" / "results" / "aphasia_pause_fit.json"

MIN_GAP_MS = 200          # below this it is coarticulation, not a pause
CHUNK_MS = VAD_CHUNK * 1000 // SR       # 32 ms
SWEEP = [500, 700, 900, 1100, 1300, 1500, 1800, 2200, 2600, 3000]


def internal_gaps(audio: np.ndarray, vad) -> list[int]:
    """Silence runs strictly inside the utterance, in ms.

    Leading and trailing silence are dropped: those are turn boundaries, not
    a speaker stopping mid-sentence, and counting them would inflate every
    utterance's longest gap by however much room the transcriber left.
    """
    probs = []
    for i in range(0, len(audio) - VAD_CHUNK, VAD_CHUNK):
        chunk = torch.from_numpy(audio[i:i + VAD_CHUNK].copy())
        with torch.no_grad():
            probs.append(float(vad(chunk, SR).item()))
    voiced = [p >= 0.5 for p in probs]
    if not any(voiced):
        return []
    first, last = voiced.index(True), len(voiced) - 1 - voiced[::-1].index(True)
    gaps, run = [], 0
    for v in voiced[first:last + 1]:
        if v:
            if run * CHUNK_MS >= MIN_GAP_MS:
                gaps.append(run * CHUNK_MS)
            run = 0
        else:
            run += 1
    return gaps


def pct(xs: list[int], q: float) -> float:
    return float(np.percentile(xs, q)) if xs else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-utterances", type=int, default=250,
                    help="per participant, to bound runtime")
    args = ap.parse_args()

    if not AUDIO.is_dir() or not any(AUDIO.glob("*.wav")):
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED", "reason": "no APROCSA audio"},
                                  indent=2), encoding="utf-8")
        print("SKIPPED -- no APROCSA audio")
        return 0

    parsed = load_all(TRANSCRIPTS)
    vad = _get_vad_instance()

    search_gaps: list[int] = []
    fluent_gaps: list[int] = []
    per_utt: list[dict] = []
    per_participant: dict[str, dict] = {}

    for pid in sorted(parsed):
        wav = AUDIO / ("%s.wav" % pid)
        if not wav.exists():
            continue
        info = sf.info(str(wav))
        utts = [u for u in parsed[pid]["utterances"]
                if u["is_participant"] and u["start_ms"] is not None
                and u["end_ms"] > u["start_ms"]][:args.max_utterances]
        s_local: list[int] = []
        f_local: list[int] = []
        for u in utts:
            a0 = int(u["start_ms"] * info.samplerate / 1000)
            a1 = int(u["end_ms"] * info.samplerate / 1000)
            if a1 - a0 < info.samplerate // 2:
                continue
            try:
                a, sr = sf.read(str(wav), start=a0, stop=min(a1, info.frames),
                                dtype="float32")
            except Exception:
                continue
            if a.ndim > 1:
                a = a.mean(axis=1)
            if sr != SR:
                idx = np.linspace(0, len(a) - 1, int(len(a) * SR / sr))
                a = np.interp(idx, np.arange(len(a)), a).astype("float32")
            try:
                vad.reset_states()
            except AttributeError:
                pass
            gaps = internal_gaps(a, vad)
            (s_local if u["word_search"] else f_local).extend(gaps)
            per_utt.append({"pid": pid, "word_search": u["word_search"],
                            "max_gap_ms": max(gaps) if gaps else 0,
                            "n_gaps": len(gaps),
                            "dur_ms": u["end_ms"] - u["start_ms"]})
        search_gaps.extend(s_local)
        fluent_gaps.extend(f_local)
        per_participant[pid] = {
            "n_utterances": len(utts),
            "search_gap_median": round(pct(s_local, 50)) if s_local else None,
            "search_gap_p90": round(pct(s_local, 90)) if s_local else None,
            "fluent_gap_median": round(pct(f_local, 50)) if f_local else None,
            "fluent_gap_p90": round(pct(f_local, 90)) if f_local else None,
        }
        print("  %s: %d utterances, %d search gaps, %d fluent gaps"
              % (pid, len(utts), len(s_local), len(f_local)), flush=True)

    # Threshold sweep on the utterance-level rule "this utterance contains an
    # internal silence >= T". Youden's J = sensitivity + specificity - 1.
    pos = [u for u in per_utt if u["word_search"]]
    neg = [u for u in per_utt if not u["word_search"]]
    sweep = []
    for t in SWEEP:
        tpr = sum(u["max_gap_ms"] >= t for u in pos) / max(1, len(pos))
        fpr = sum(u["max_gap_ms"] >= t for u in neg) / max(1, len(neg))
        sweep.append({"threshold_ms": t, "sensitivity": round(tpr, 4),
                      "false_positive_rate": round(fpr, 4),
                      "youden_j": round(tpr - fpr, 4)})
    best = max(sweep, key=lambda r: r["youden_j"])

    result = {
        "status": "OK",
        "dataset": "APROCSA -- 6 speakers, chronic post-stroke aphasia",
        "method": ("Silero VAD inside each CHAT-aligned participant utterance; "
                   "internal silences >= %d ms; leading/trailing silence dropped"
                   % MIN_GAP_MS),
        "shipped_threshold_ms": 1300,
        "n_utterances": len(per_utt),
        "n_word_search": len(pos), "n_fluent": len(neg),
        "gap_distribution_ms": {
            "word_search": {"n": len(search_gaps),
                            "p50": round(pct(search_gaps, 50)),
                            "p75": round(pct(search_gaps, 75)),
                            "p90": round(pct(search_gaps, 90)),
                            "p95": round(pct(search_gaps, 95))},
            "fluent": {"n": len(fluent_gaps),
                       "p50": round(pct(fluent_gaps, 50)),
                       "p75": round(pct(fluent_gaps, 75)),
                       "p90": round(pct(fluent_gaps, 90)),
                       "p95": round(pct(fluent_gaps, 95))},
        },
        "sweep": sweep,
        "best_by_youden_j": best,
        "per_participant": per_participant,
        "caveats": [
            "Six speakers; between-participant spread is reported and is large.",
            ("The threshold is chosen against a property of the speech, not against "
             "Echo's recall/false-alarm curve -- tuning on that would make "
             "eval/run_aphasia_eval.py a training-set number."),
            ("Utterance boundaries come from clinician transcription, so a pause "
             "the transcriber treated as an utterance boundary is not counted as "
             "an internal gap even if the speaker was still searching."),
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    d = result["gap_distribution_ms"]
    print("")
    print("INTERNAL SILENCE INSIDE APHASIC UTTERANCES (ms)")
    print("  %-14s %6s %6s %6s %6s %6s" % ("", "n", "p50", "p75", "p90", "p95"))
    for k in ("word_search", "fluent"):
        r = d[k]
        print("  %-14s %6d %6d %6d %6d %6d"
              % (k, r["n"], r["p50"], r["p75"], r["p90"], r["p95"]))
    print("")
    print("UTTERANCE CONTAINS A GAP >= T")
    print("  %8s %10s %10s %8s" % ("T (ms)", "sens", "FPR", "J"))
    for r in sweep:
        mark = "  <- shipped" if r["threshold_ms"] == 1300 else (
            "  <- best J" if r is best else "")
        print("  %8d %10.3f %10.3f %8.3f%s"
              % (r["threshold_ms"], r["sensitivity"],
                 r["false_positive_rate"], r["youden_j"], mark))
    print("")
    print("  per-participant p90 of word-search gaps: %s"
          % {k: v["search_gap_p90"] for k, v in per_participant.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
