"""Can the speaker gate tell the wearer from a bystander -- and does it ever
mute the wearer?

The second question is the one that matters. Echo exists to give a person
their word back; a gate that wrongly suppresses the WEARER takes the product
away from the only user it has. So this harness reports two numbers per
condition and treats them asymmetrically:

  (a) suppression rate for the quieter speaker  -- the benefit. Nice to have.
  (b) false-suppression rate for the WEARER     -- the cost. MUST be ~0.
      A confidence of None (unknown) counts as NOT suppressed, because the
      contract is fail open: unknown never suppresses.

A harness that cannot fail proves nothing: (b) above 2% is reported as a
FAILED verdict and exits non-zero, whatever (a) says.

STIMULI. Synthetic two-speaker timelines, built from real speech when it is
available locally. Sources, in preference order (the one used is printed and
recorded in the JSON -- never quote a number from this harness without it):

  - "pfsd-by-episode": 1.0 s clips from the PFSD *test* split
    (data/pfsd/clips/test/Words), with the wearer's clips drawn from ONE
    podcast episode and the bystander's from a DIFFERENT one. Same episode
    means one recording channel, so each simulated talker is spectrally
    self-consistent -- which matters because the gate has a spectral-tilt
    term, and a "wearer" stitched from 20 unrelated podcasts would be
    penalised for channel variation no real wearer has.
  - "pfsd-random": same clips, episode grouping unavailable
    (PodcastFillers.csv missing). Honest but harsher on the tilt term.
  - "synthetic": generated pseudo-voice (harmonic stack, syllable-rate
    envelope) when no speech corpus exists on the machine.

The SAME clip assignment is reused at every separation, so only the mix LEVEL
differs between conditions -- separation is isolated from clip-instance
variance.

TIMELINE per trial: CALIB_S seconds of continuous WEARER speech (so the gate
has something to calibrate on), then alternating one-second turns
wearer / bystander / wearer / ... separated by noise-floor gaps. Every clip is
RMS-normalised to its speaker's nominal level, then jittered by
+/- `--jitter-db` to model natural word-to-word level variation.

WORD-LEVEL AGGREGATION. The gate scores 32 ms frames; the product decides per
WORD. Primary aggregation is the 90th percentile of the word's frame
confidences -- a word is judged by its loud part, which is what
`frontend/app.js`'s `wearerConfForWord` does, because the quiet frames of near
speech are indistinguishable in level from the loud frames of far speech. The
stricter median aggregation is reported alongside it in the same table so the
choice cannot hide anything.

WHAT THIS DOES NOT MODEL, and each of these flatters or distorts the result:
  - No overlapping speech. Turns strictly alternate.
  - No room: no reverb, no HF rolloff with distance. The bystander differs
    from the wearer ONLY in level (plus whatever channel difference the two
    episodes happen to have), so the gate's tilt term gets little evidence
    here. Real distance would give it some; real reverb would take some away.
    Net direction unknown.
  - VAD is oracle (ground-truth speech/silence labels are handed to the gate),
    which isolates the gate from VAD errors -- an upper bound.
  - A PFSD episode can contain more than one speaker, so "one episode" is one
    channel, not provably one voice. The gate uses no speaker identity, only
    level and tilt, so this does not confound the level measurement.

Therefore: this is an instrument-bench measurement of a LEVEL argument. The
speaker gate is UNVALIDATED IN REAL ROOMS. No real two-speaker recording of
this hardware exists, and nothing in this file may be described as room-tested.

    python eval/run_speaker_gate_eval.py
    python eval/run_speaker_gate_eval.py --source synthetic --trials 4

Writes eval/results/speaker_gate_eval.json ("status": "OK" | "SKIPPED").
Never hand-edit docs/EVAL.md; regenerate it with eval/make_report.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "eval" / "results" / "speaker_gate_eval.json"
CLIPS = ROOT / "data" / "pfsd" / "clips" / "test" / "Words"
META = ROOT / "data" / "pfsd" / "PodcastFillers.csv"

SR = 16000
FRAME = 512               # matches AcousticStream.VAD_CHUNK (32 ms @ 16k)
CLIP_S = 1.0              # PFSD clip length; the synthetic voice matches it
CALIB_S = 3.0             # wearer speech before any test word
GAP_S = 0.4               # noise-floor gap between turns
WEARER_DBFS = -26.0       # nominal wearer level at the mic
NOISE_DBFS = -70.0        # room noise floor (never digital silence)
SEPARATIONS_DB = (0.0, 3.0, 6.0, 12.0)
SPEECH_PROB = 0.95        # oracle VAD on a speech frame
SILENCE_PROB = 0.02       # oracle VAD in a gap
WEARER_FALSE_SUPPRESSION_MAX = 0.02   # the fail-open gate for this harness
AGGREGATIONS = (("p90", 0.9), ("median", 0.5))
PRIMARY_AGG = "p90"

DISCLAIMER = (
    "Synthetic two-speaker mixes only: alternating turns, level difference "
    "only, no reverb, no distance HF rolloff, oracle VAD. The speaker gate is "
    "unvalidated in real rooms."
)


# ---------------------------------------------------------------- primitives
def dbfs(x: np.ndarray) -> float:
    """RMS level in dBFS. Digital silence answers -120, never -inf/NaN."""
    if x.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))
    return 20.0 * float(np.log10(rms)) if rms > 1e-6 else -120.0


def scale_to_dbfs(x: np.ndarray, target_db: float) -> np.ndarray:
    """Rescale to an exact RMS level. A silent input is returned unchanged."""
    cur = dbfs(x)
    if cur <= -119.0:
        return x.astype(np.float32, copy=True)
    return (x * (10.0 ** ((target_db - cur) / 20.0))).astype(np.float32)


def percentile_of(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted list."""
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = min(max(q, 0.0), 1.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def word_conf(frame_confs: list[float | None], q: float = 0.9) -> float | None:
    """Collapse a word's per-frame confidences to one number.

    `q`-percentile over the frames the gate was willing to answer for; None
    when it answered for none of them. None is "unknown" and must never be
    counted as a suppression.
    """
    vals = sorted(float(c) for c in frame_confs if c is not None)
    return percentile_of(vals, q) if vals else None


def suppression_rate(confs: list[float | None], threshold: float) -> float:
    """Fraction of words that would be suppressed at `threshold`.

    None (unknown) counts as NOT suppressed: absent confidence must never
    suppress, so it must never be scored as a suppression either.
    """
    if not confs:
        return 0.0
    hit = sum(1 for c in confs if c is not None and float(c) < threshold)
    return hit / len(confs)


def unknown_rate(confs: list[float | None]) -> float:
    if not confs:
        return 0.0
    return sum(1 for c in confs if c is None) / len(confs)


# ------------------------------------------------------------------- stimuli
def find_pfsd_clips(clips_root: Path, n: int, seed: int) -> list[Path]:
    """n PFSD clip paths, seeded, ungrouped. Empty list when none exist."""
    if not clips_root.is_dir():
        return []
    paths = sorted(clips_root.glob("*.wav"))
    if not paths:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(paths))[:n]
    return [paths[int(i)] for i in idx]


def pfsd_episode_pools(clips_root: Path, meta_csv: Path,
                       min_clips: int) -> dict[str, list[Path]]:
    """episode -> its clip paths, for episodes with at least `min_clips`.

    Empty dict when the metadata CSV or the clips are absent -- the caller
    falls back to ungrouped draws rather than failing.
    """
    if not clips_root.is_dir() or not meta_csv.is_file():
        return {}
    have = {p.name: p for p in clips_root.glob("*.wav")}
    if not have:
        return {}
    pools: dict[str, list[Path]] = {}
    with meta_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            p = have.get(row.get("clip_name", ""))
            if p is not None:
                pools.setdefault(row.get("podcast_filename", "?"), []).append(p)
    return {k: sorted(v) for k, v in pools.items() if len(v) >= min_clips}


def plan_trials(clips_root: Path, meta_csv: Path, trials: int, need: int,
                per_trial: int, seed: int,
                allow_ungrouped: bool = True) -> tuple[str, list[tuple[list[Path], list[Path]]]]:
    """(source_label, [(wearer_paths, bystander_paths)]) or ("", []) if no speech.

    The wearer list holds `need` paths: the first `per_trial` are its test
    words, the rest are calibration (build_trial relies on that order).
    """
    rng = np.random.default_rng(seed)
    pools = pfsd_episode_pools(clips_root, meta_csv, need)
    names = sorted(pools)
    if len(names) >= 2:
        plan = []
        for _ in range(trials):
            a, b = rng.choice(len(names), size=2, replace=False)
            wp, bp = pools[names[int(a)]], pools[names[int(b)]]
            wi = rng.permutation(len(wp))[:need]
            bi = rng.permutation(len(bp))[:per_trial]
            plan.append(([wp[int(i)] for i in wi], [bp[int(i)] for i in bi]))
        return "pfsd-by-episode", plan
    if not allow_ungrouped:
        return "", []
    flat = find_pfsd_clips(clips_root, trials * (need + per_trial), seed)
    if not flat:
        return "", []
    plan, cursor = [], 0
    for _ in range(trials):
        wear = [flat[(cursor + i) % len(flat)] for i in range(need)]
        cursor += need
        byst = [flat[(cursor + i) % len(flat)] for i in range(per_trial)]
        cursor += per_trial
        plan.append((wear, byst))
    return "pfsd-random", plan


def load_clip(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:                      # PFSD is already 16k; guard anyway
        n = int(round(len(x) * SR / sr))
        x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    want = int(CLIP_S * SR)
    if len(x) < want:
        x = np.pad(x, (0, want - len(x)))
    return np.asarray(x[:want], dtype=np.float32)


def synth_voice(rng: np.random.Generator) -> np.ndarray:
    """One second of pseudo-voice: harmonic stack + syllable-rate envelope.

    Not speech. It exists so the harness still produces a number on a machine
    with no audio corpus, and the JSON says so.
    """
    n = int(CLIP_S * SR)
    t = np.arange(n, dtype=np.float64) / SR
    f0 = float(rng.uniform(90.0, 210.0))
    sig = np.zeros(n, dtype=np.float64)
    for k in range(1, 13):
        sig += (1.0 / k) * np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 2 * np.pi))
    syll = 0.5 + 0.5 * np.sin(2 * np.pi * float(rng.uniform(2.5, 4.5)) * t)
    sig *= syll
    sig += 0.05 * rng.standard_normal(n)          # breath
    return scale_to_dbfs(sig.astype(np.float32), WEARER_DBFS)


def build_trial(
    wearer_clips: list[np.ndarray],
    bystander_clips: list[np.ndarray],
    sep_db: float,
    jitter_db: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict], list[tuple[int, int]]]:
    """One timeline, its per-word labels, and every speech span.

    Returns (audio, words, spans):
      words  -- [{"speaker": "wearer"|"bystander", "start", "end", "level_db"}]
                in samples; calibration is deliberately NOT a word (it is the
                gate's warm-up, not a measurement)
      spans  -- (start, end) of EVERY speech segment including calibration,
                which is what the oracle VAD labels as speech

    `wearer_clips` must hold the test words first and the calibration clips
    after them, so no calibration audio is reused as a scored word.
    """
    calib_n = max(1, int(round(CALIB_S / CLIP_S)))
    n_words = min(len(wearer_clips) - calib_n, len(bystander_clips))
    segments: list[tuple[np.ndarray, str | None]] = []
    for i in range(calib_n):
        segments.append((wearer_clips[n_words + i], None))
    for i in range(n_words * 2):
        who = "wearer" if i % 2 == 0 else "bystander"
        src = (wearer_clips if who == "wearer" else bystander_clips)[i // 2]
        segments.append((src, who))

    gap = np.zeros(int(GAP_S * SR), dtype=np.float32)
    total = sum(len(s) for s, _ in segments) + len(gap) * len(segments)
    audio = scale_to_dbfs(rng.standard_normal(total).astype(np.float32), NOISE_DBFS)
    words: list[dict] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for src, who in segments:
        target = WEARER_DBFS - (sep_db if who == "bystander" else 0.0)
        target += float(rng.uniform(-jitter_db, jitter_db)) if jitter_db else 0.0
        seg = scale_to_dbfs(src, target)
        audio[pos:pos + len(seg)] += seg
        spans.append((pos, pos + len(seg)))
        if who is not None:
            words.append({"speaker": who, "start": pos, "end": pos + len(seg),
                          "level_db": round(target, 2)})
        pos += len(seg) + len(gap)
    return np.clip(audio, -1.0, 1.0), words, spans


# ---------------------------------------------------------------------- gate
def load_gate_class():
    """Import SpeakerGate lazily. Returns None when it does not exist yet.

    Imported inside the function on purpose: this harness must stay importable
    (and its scoring testable) without the backend gate.
    """
    try:
        from backend.acoustic.speaker_gate import SpeakerGate  # noqa: PLC0415
    except Exception:
        return None
    return SpeakerGate


def run_trial(gate, audio: np.ndarray, words: list[dict],
              spans: list[tuple[int, int]]) -> list[dict]:
    """Feed the whole timeline through one gate; score each labelled word."""
    import torch

    # oracle VAD: every segment we placed (calibration included) is speech
    speech = np.zeros(len(audio), dtype=bool)
    for a, b in spans:
        speech[a:b] = True

    per_frame: list[tuple[int, float | None]] = []
    for start in range(0, len(audio) - FRAME + 1, FRAME):
        frame = audio[start:start + FRAME]
        prob = SPEECH_PROB if speech[start:start + FRAME].mean() > 0.5 else SILENCE_PROB
        conf = gate.observe(torch.from_numpy(frame.copy()), prob)
        per_frame.append((start, None if conf is None else float(conf)))

    rows = []
    for w in words:
        confs = [c for (s, c) in per_frame if w["start"] <= s < w["end"]]
        row = {"speaker": w["speaker"], "level_db": w["level_db"],
               "frames": len(confs),
               "unknown_frames": sum(1 for c in confs if c is None)}
        for name, q in AGGREGATIONS:
            c = word_conf(confs, q)
            row["conf_" + name] = None if c is None else round(c, 4)
        rows.append(row)
    return rows


def compact(rows: list[dict]) -> dict:
    """Per-speaker confidence vectors for re-analysis, without dumping a row
    object per word (the full dump is ~10x bigger than every other eval JSON)."""
    out: dict[str, dict] = {}
    for who in ("wearer", "bystander"):
        sel = [r for r in rows if r["speaker"] == who]
        out[who] = {"level_db": [r["level_db"] for r in sel]}
        for agg, _ in AGGREGATIONS:
            out[who][agg] = [r["conf_" + agg] for r in sel]
    return out


def summarize(rows: list[dict], threshold: float, agg: str) -> dict:
    key = "conf_" + agg
    wearer = [r[key] for r in rows if r["speaker"] == "wearer"]
    other = [r[key] for r in rows if r["speaker"] == "bystander"]
    known_w = sorted(c for c in wearer if c is not None)
    known_o = sorted(c for c in other if c is not None)
    return {
        "n_wearer": len(wearer),
        "n_bystander": len(other),
        "bystander_suppression_rate": round(suppression_rate(other, threshold), 4),
        "wearer_false_suppression_rate": round(suppression_rate(wearer, threshold), 4),
        "wearer_unknown_rate": round(unknown_rate(wearer), 4),
        "bystander_unknown_rate": round(unknown_rate(other), 4),
        "wearer_conf_median": round(percentile_of(known_w, 0.5), 4) if known_w else None,
        "bystander_conf_median": round(percentile_of(known_o, 0.5), 4) if known_o else None,
    }


def skipped(out: Path, reason: str, source: str, extra: dict | None = None) -> int:
    payload = {"status": "SKIPPED", "reason": reason, "source": source,
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "disclaimer": DISCLAIMER}
    payload.update(extra or {})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SPEAKER GATE EVAL -- SKIPPED")
    print("  reason: %s" % reason)
    print("  wrote %s" % out)
    print("  reminder: the speaker gate is unvalidated in real rooms.")
    return 0


# ---------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Speaker-gate eval on synthetic "
                                             "two-speaker mixes.")
    ap.add_argument("--source", choices=("auto", "pfsd", "synthetic"), default="auto",
                    help="auto = real PFSD speech if present, else generated")
    ap.add_argument("--clips-root", default=str(CLIPS))
    ap.add_argument("--meta-csv", default=str(META),
                    help="PodcastFillers.csv, for episode grouping")
    ap.add_argument("--trials", type=int, default=24,
                    help="independent episode pairs; n words per cell = trials * "
                         "words-per-speaker")
    ap.add_argument("--words-per-speaker", type=int, default=4,
                    help="test words per speaker per trial")
    ap.add_argument("--threshold", type=float, default=0.35,
                    help="suppression threshold (matches wearer_conf_min default)")
    ap.add_argument("--jitter-db", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    per_trial = max(1, args.words_per_speaker)
    need = per_trial + max(1, int(round(CALIB_S / CLIP_S)))

    # --- audio source
    source, plan = "", []
    if args.source in ("auto", "pfsd"):
        try:
            import soundfile  # noqa: F401
            source, plan = plan_trials(Path(args.clips_root), Path(args.meta_csv),
                                       args.trials, need, per_trial, args.seed)
        except Exception as exc:
            source, plan = "", []
            if args.source == "pfsd":
                return skipped(out, "PFSD load failed: %s: %s"
                               % (type(exc).__name__, exc), "pfsd")
        if not plan:
            if args.source == "pfsd":
                return skipped(out, "no usable PFSD clips (or no soundfile) under %s"
                               % args.clips_root, "pfsd")
            source = "synthetic"
    else:
        source = "synthetic"

    gate_cls = load_gate_class()
    if gate_cls is None:
        return skipped(out, "backend.acoustic.speaker_gate.SpeakerGate not importable",
                       source)

    # Clip assignment is drawn ONCE and reused at every separation, so the only
    # thing that differs between conditions is level.
    rng = np.random.default_rng(args.seed)
    trials: list[tuple[list[np.ndarray], list[np.ndarray]]] = []
    try:
        for ti in range(args.trials):
            if plan:
                wear = [load_clip(p) for p in plan[ti][0]]
                byst = [load_clip(p) for p in plan[ti][1]]
            else:
                wear = [synth_voice(rng) for _ in range(need)]
                byst = [synth_voice(rng) for _ in range(per_trial)]
            trials.append((wear, byst))
    except Exception as exc:
        return skipped(out, "audio decode failed: %s: %s"
                       % (type(exc).__name__, exc), source)

    t0 = time.time()
    by_sep: dict[str, dict] = {}
    confs: dict[str, dict] = {}
    try:
        for sep in SEPARATIONS_DB:
            rows: list[dict] = []
            for ti, (wear, byst) in enumerate(trials):
                trng = np.random.default_rng(args.seed * 1000 + ti)
                audio, words, spans = build_trial(wear, byst, sep, args.jitter_db, trng)
                gate = gate_cls()
                for r in run_trial(gate, audio, words, spans):
                    r["sep_db"] = sep
                    r["trial"] = ti
                    rows.append(r)
            by_sep["%.0f" % sep] = {agg: summarize(rows, args.threshold, agg)
                                    for agg, _ in AGGREGATIONS}
            confs["%.0f" % sep] = compact(rows)
    except Exception as exc:   # gate API drift must not look like a gate result
        return skipped(out, "SpeakerGate API mismatch: %s: %s"
                       % (type(exc).__name__, exc), source)

    def worst(agg: str) -> float:
        return max(v[agg]["wearer_false_suppression_rate"] for v in by_sep.values())

    worst_false = worst(PRIMARY_AGG)
    ok = worst_false <= WEARER_FALSE_SUPPRESSION_MAX
    useful = [s for s, v in by_sep.items()
              if v[PRIMARY_AGG]["bystander_suppression_rate"] >= 0.5]
    payload = {
        "status": "OK",
        "source": source,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_s": round(time.time() - t0, 1),
        "params": {"trials": args.trials, "words_per_speaker": per_trial,
                   "threshold": args.threshold, "jitter_db": args.jitter_db,
                   "seed": args.seed, "wearer_dbfs": WEARER_DBFS,
                   "noise_dbfs": NOISE_DBFS, "calib_s": CALIB_S,
                   "frame_samples": FRAME, "separations_db": list(SEPARATIONS_DB),
                   "primary_aggregation": PRIMARY_AGG},
        "by_separation_db": by_sep,
        "wearer_false_suppression_max": WEARER_FALSE_SUPPRESSION_MAX,
        "worst_wearer_false_suppression": {agg: worst(agg) for agg, _ in AGGREGATIONS},
        "fail_open_respected": ok,
        "first_useful_separation_db": min(useful, key=float) if useful else None,
        "word_confidences": confs,
        "disclaimer": DISCLAIMER,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    src_label = {
        "pfsd-by-episode": "real PFSD test-split speech, one episode per talker",
        "pfsd-random": "real PFSD test-split speech, ungrouped (no episode metadata)",
        "synthetic": "GENERATED pseudo-voice (no speech corpus on this machine)",
    }.get(source, source)
    print("SPEAKER GATE EVAL -- synthetic two-speaker mixes")
    print("  audio source: %s" % src_label)
    print("  threshold=%.2f  trials=%d  words/speaker/trial=%d  jitter=+/-%.1f dB"
          % (args.threshold, args.trials, per_trial, args.jitter_db))
    print("")
    print("  %-7s %7s | %11s %12s | %11s %12s"
          % ("sep_dB", "n_each", "byst_supp", "WEARER_false",
             "byst_supp", "WEARER_false"))
    print("  %-7s %7s | %24s | %24s" % ("", "", "-- word = p90 of frames --",
                                        "-- word = median frame --"))
    for sep in ("%.0f" % s for s in SEPARATIONS_DB):
        p, m = by_sep[sep]["p90"], by_sep[sep]["median"]
        print("  %-7s %7d | %11.3f %12.3f | %11.3f %12.3f"
              % (sep, p["n_wearer"], p["bystander_suppression_rate"],
                 p["wearer_false_suppression_rate"],
                 m["bystander_suppression_rate"],
                 m["wearer_false_suppression_rate"]))
    print("")
    print("  byst_supp    = suppression rate for the quieter speaker -- the benefit")
    print("  WEARER_false = wearer wrongly suppressed -- the cost, must be ~0")
    print("  unknown confidence counts as NOT suppressed in both (fail open)")
    print("  p90 is the deployed rule (frontend/app.js wearerConfForWord);")
    print("  the median column is the stricter reading, shown so it cannot hide.")
    print("")
    if not ok:
        print("  VERDICT: FAILED -- wearer false-suppression %.3f > %.2f at some "
              "separation. On this bench the gate mutes the wearer; do not ship "
              "it enabled." % (worst_false, WEARER_FALSE_SUPPRESSION_MAX))
    elif not useful:
        print("  VERDICT: fail-open respected (wearer never wrongly muted), but the "
              "gate suppressed < 50% of bystander words at EVERY separation "
              "tested -- safe, and weak on this bench.")
    else:
        print("  VERDICT: fail-open respected (worst wearer false-suppression "
              "%.3f <= %.2f); bystander suppression reaches 50%% from %s dB "
              "separation." % (worst_false, WEARER_FALSE_SUPPRESSION_MAX,
                               min(useful, key=float)))
    print("")
    print("  LIMITS: alternating turns, level difference only, no reverb, no")
    print("          distance HF rolloff, oracle VAD, RMS-normalised clips.")
    print("          The speaker gate is unvalidated in real rooms.")
    print("  wrote %s" % out)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
