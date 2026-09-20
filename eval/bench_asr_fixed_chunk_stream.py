"""Can a FIXED-LENGTH rolling buffer close the streaming-vs-offline ASR gap?

WHY THIS EXISTS
---------------
`eval/bench_asr_offline_wer.py` established the gap on APROCSA (6 speakers with
chronic post-stroke aphasia, 300 s each, scored against the clinicians' CHAT
transcripts, fillers stripped from both sides):

    offline, one pass                      WER 0.288   concat 0.271   <- ceiling
    offline, chunk_duration=10 stride=8    WER 0.319   concat 0.300
    best streaming policy found            WER 0.375   concat 0.322

The interesting cell is the middle one. It is *chunked* decoding -- fixed 10 s
windows, 8 s stride, stitched -- and it lands far closer to the full offline
pass than any streaming configuration. `VerbatimASR` instead decodes a window
that GROWS from the last pause and is RETIRED at the next one, and aphasic
speech is mostly pause, so its decode span is frequently a second or two of a
fragment. The hypothesis under test is that the win in the middle row is simply
"always decode a long span, regardless of where the pauses fall" -- which a
streaming system can do with a rolling fixed-length buffer.

WHAT IS MEASURED
----------------
One decode pass per (buffer, base stride, speaker): every `stride` seconds,
transcribe the whole trailing `buffer` seconds. That pass is the entire GPU
cost and it is cached. THREE commit policies are then derived from it in pure
post-processing, which is what makes the latency question answerable rather
than assumed -- they differ only in WHEN a word is released, never in what was
decoded:

  scroll  Release a word only once its audio has scrolled out of the NEXT
          decode's buffer, i.e. once no future decode can revise it. This is
          the accuracy ceiling of the policy family, and its commit lag is
          about one buffer length by construction.
  agree   LocalAgreement-2 over the fixed buffer: release the longest prefix of
          the uncommitted tail that two consecutive decodes agree on (aligned
          by LCS on normalized text, so a shifted timestamp does not break the
          match), with `scroll` as a backstop so nothing is ever lost off the
          front of the buffer.
  now     Release every new word at the decode that first proposes it. No
          revision at all; lag is bounded by the stride.

THE CONSTRAINT THIS TRADES AGAINST
----------------------------------
Echo's stall fires ~1300 ms after silence onset and the predictor needs the
fragment complete by then. So WER is reported next to `frac_ready`: the
fraction of committed words that were committed within 1300 ms of the FIRST
silence onset after the word was spoken. A word that arrives after that instant
did not help the prediction it existed for. Lag is on the audio clock (the tick
at which the decode's buffer was complete) plus the measured decode wall time,
because both are real waiting.

    python eval/bench_asr_fixed_chunk_stream.py --buffers 8,10,12 --strides 2,4

WHAT THE FULL GRID CHANGED, AND WHAT IT DID NOT
-----------------------------------------------
The finding was first published from the buffer=8 row alone, with the
justification that larger buffers "can only be worse on lag". That reason is
wrong, and the completed grid says why: commit lag here has two components,
the policy's own wait and the decode wall, and only ONE of them is set by the
buffer.

  policy   lag on the policy's own clock, buffer 8 -> 10
  scroll   6720 -> 8720 ms   monotone in buffer, as assumed
  agree    3160 -> 3180 ms   flat -- set by the STRIDE
  now       920 ->  920 ms   identical -- set by the STRIDE

So buffer=12 could not have moved `agree` or `now` either, and the conclusion
stands -- but it stands because stride sets the lag, not because bigger
buffers cost more. Reported here because the first version of this claim was
right by accident.

Two things the buffer=8 row alone got wrong:

1. The best cell is buffer=10, not 8: 0.301 / 0.286 boundary-free against
   0.307 / 0.288. That closes 85% of the WER gap and 70% of the concat gap to
   the 0.288 / 0.271 offline ceiling. Decoding a long fixed span regardless of
   pause placement really is most of what offline was buying.

2. `now` is not uniformly hopeless. At stride 2 its policy lag is 920 ms --
   INSIDE the 1300 ms stall budget -- with 0.87 of words on time at zero
   decode cost and 0.74 at a 500 ms decode. Its measured ready of 0.27 is the
   decode wall, roughly 1.0-1.2 s of which is turbo's word-timestamp DTW that
   the live path does not even run (VerbatimASR sets word_timestamps=False for
   exactly this reason).

   It is still not an improvement. `b10 s2 now` scores 0.352 / 0.336 against
   the shipped 0.375 / 0.322 -- it buys 2.3 WER points and LOSES 1.4 points
   boundary-free, meaning its gain is better word timestamps rather than
   better recognition. The genuine accuracy lives in `agree`, whose 3160 ms
   policy-intrinsic lag is 2.4x the budget and is not reducible by any amount
   of model engineering.

RTF CAVEAT: every RTF in the grid was measured with a second process on the
GPU and includes word-timestamp DTW that the live path skips. Treat them as
upper bounds. They are not the published streaming figure.

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

from backend.stt.verbatim import SR, get_model  # noqa: E402
from backend.timeline import norm  # noqa: E402
from eval.bench_asr_offline_wer import in_region, score  # noqa: E402
from eval.run_aphasia_eval import CACHE, TRANSCRIPTS, load_region  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "asr_fixed_chunk_stream.json"

VAD_CHUNK = 512                  # silero requirement @16k (32 ms)
VAD_MS = VAD_CHUNK * 1000 // SR
STALL_MS = 1300                  # the deadline the predictor's fragment must meet


# --------------------------------------------------------------------------
# the decode pass -- the only thing that touches the GPU
# --------------------------------------------------------------------------
def decode_pass(pid, audio, buffer_s, stride_s, args):
    """Every `stride_s` seconds, transcribe the trailing `buffer_s` seconds.

    Word timestamps are ON. The stitcher needs to know which audio a word sits
    in, and the scorer assigns hypothesis words to reference utterances by
    time; the offline comparators in asr_offline_vs_stream.json were measured
    the same way, so this keeps the arms comparable. It is also the expensive
    choice (turbo's DTW alignment costs a roughly flat ~1 s per call) and that
    cost is included in the reported RTF rather than excused.
    """
    model = get_model(args.model)
    dur_ms = int(len(audio) * 1000 / SR)
    buf_ms = int(buffer_s * 1000)
    step_ms = int(stride_s * 1000)
    ticks = list(range(step_ms, dur_ms + 1, step_ms))
    if not ticks or ticks[-1] < dur_ms:
        ticks.append(dur_ms)
    out = []
    for t_ms in ticks:
        b0 = max(0, t_ms - buf_ms)
        clip = audio[int(b0 * SR / 1000):int(t_ms * SR / 1000)]
        if clip.size < SR // 4:
            continue
        t0 = time.perf_counter()
        res = model.transcribe(clip, sr=SR, language="en", mode=args.mode,
                               word_timestamps=True)
        wall = time.perf_counter() - t0
        words = []
        for w in (res.words or []):
            if w.end is None:
                continue
            s = b0 + int((w.start if w.start is not None else 0) * 1000)
            words.append([w.word, s, b0 + int(w.end * 1000)])
        out.append({"t": t_ms, "b0": b0, "words": words, "wall": round(wall, 4)})
    return out


def decodes_cached(pid, audio, buffer_s, stride_s, args):
    tag = "fcs_%s_%s_b%g_s%g_%d_%d" % (pid, args.mode, buffer_s, stride_s,
                                       args.skip, args.region)
    path = CACHE / (tag + ".json")
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8")), False
    d = decode_pass(pid, audio, buffer_s, stride_s, args)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d), encoding="utf-8")
    return d, True


def subsample(decodes, base_stride_s, stride_s):
    """Derive a coarser stride from a finer decode pass.

    A decode at tick t over buffer [t-B, t] does not depend on the stride at
    all, so the stride-4 pass is literally every other decode of the stride-2
    pass. Re-running the GPU for it would produce identical numbers.
    """
    m = int(round(stride_s / base_stride_s))
    if abs(m * base_stride_s - stride_s) > 1e-6 or m < 1:
        return None
    out = [d for i, d in enumerate(decodes[:-1]) if (i + 1) % m == 0]
    if decodes and (not out or out[-1] is not decodes[-1]):
        out.append(decodes[-1])          # the flush at end of audio
    return out


# --------------------------------------------------------------------------
# stitching
# --------------------------------------------------------------------------
def lcs_pairs(a, b):
    """Index pairs of a longest common subsequence of two token lists."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        row, nxt = dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = (nxt[j + 1] + 1 if a[i] == b[j] else max(nxt[j], row[j + 1]))
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _new_suffix(committed, d):
    """Words of decode `d` that are not already committed.

    Aligned by LCS against the committed words the buffer still covers rather
    than by timestamp: the same word gets a slightly different time in each
    decode, and a timestamp cut would either drop it or emit it twice. Emitting
    it twice is the worse failure here -- "you you recently" is a real word
    repetition and the single most useful thing in this transcript, so a
    stitcher that manufactures repetitions is not merely inaccurate, it is
    inaccurate in exactly the place the product reads.
    """
    tail = [w for w in committed if w[1] >= d["b0"]]
    cur = d["words"]
    pairs = lcs_pairs([norm(w[0]) for w in tail], [norm(w[0]) for w in cur])
    last = max((j for _, j in pairs), default=-1)
    return cur[last + 1:]


def commit(decodes, policy):
    """-> [(text, end_ms, commit_at_ms)] under one of the three release rules."""
    out = []
    prev = None
    for i, d in enumerate(decodes):
        nxt = decodes[i + 1] if i + 1 < len(decodes) else None
        # Audio before the NEXT decode's buffer start can never be re-decoded,
        # so anything in it must be released now whatever the policy says.
        scroll_b = nxt["b0"] if nxt is not None else d["t"]
        cand = _new_suffix(out, d)
        if policy == "scroll":
            # A word is only decodable while its audio is WHOLLY inside the
            # buffer, so the release test is on the word's START, not its end.
            # Testing the end instead defers every word that straddles the
            # next buffer's edge to a decode that can only see its second half,
            # which is the classic chunk-boundary truncation and it is
            # measurable: it costs ~1 WER point on the same decodes.
            take = [w for w in cand if w[1] < scroll_b]
        elif policy == "now":
            take = cand
        elif policy == "agree":
            conf = set()
            if prev is not None:
                conf = set(j for _, j in lcs_pairs(
                    [norm(w[0]) for w in prev["words"]],
                    [norm(w[0]) for w in d["words"]]))
            # `cand` is a suffix of d["words"], so shift indices to match.
            base = len(d["words"]) - len(cand)
            n_ok = 0
            for k, w in enumerate(cand):
                if (base + k) in conf or w[1] < scroll_b:
                    n_ok = k + 1
                else:
                    break
            take = cand[:n_ok]
        else:
            raise ValueError(policy)
        for w in take:
            out.append((w[0], w[2], d["t"]))
        prev = d
    return out


# --------------------------------------------------------------------------
# latency
# --------------------------------------------------------------------------
def vad_trace(pid, audio, args):
    """Absolute ms of every silence ONSET, on the VAD's own 32 ms grid.

    This is the clock the stall detector runs on, so it is the only clock on
    which "did the word arrive in time" has an answer.
    """
    tag = "fcs_vad_%s_%d_%d.json" % (pid, args.skip, args.region)
    path = CACHE / tag
    if path.exists() and not args.force:
        return json.loads(path.read_text(encoding="utf-8"))
    import torch

    from backend.acoustic.stream import _get_vad_instance
    vad = _get_vad_instance()
    onsets, prev, t = [], True, 0
    for i in range(0, len(audio) - VAD_CHUNK, VAD_CHUNK):
        with torch.no_grad():
            p = float(vad(torch.from_numpy(audio[i:i + VAD_CHUNK].copy()), SR).item())
        sp = p >= 0.5
        if prev and not sp:
            onsets.append(t)
        prev = sp
        t += VAD_MS
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(onsets), encoding="utf-8")
    return onsets


def latency(words, onsets, decode_wall_ms):
    """Commit lag, and the only latency question that decides anything.

    `frac_ready` is the fraction of committed words that were committed within
    STALL_MS of the first silence onset that followed them -- i.e. that were on
    the timeline at the instant the detector fired the stall the word was
    evidence for. A transcript that is more accurate but arrives after that
    instant is not a better transcript for this product.

    The wait charged to a word is the audio-clock wait plus the measured decode
    wall time: the buffer is only complete at the tick, and the words are only
    available once that decode returns.
    """
    lags, n = [], 0
    # `ready` is evaluated at three decode costs, not one. The measured cost is
    # what this configuration does today; 500 ms is what the same decode costs
    # with word timestamps off (turbo's DTW alignment is a roughly flat ~1 s
    # per call and dominates a buffer this short); 0 ms is the policy's own
    # floor. If the policy fails even at 0, no amount of engineering the model
    # rescues it, and that is the distinction the recommendation turns on.
    ready = {0.0: 0, 500.0: 0, decode_wall_ms: 0}
    k = 0
    # Sorted by the word's own end time, not release order: consecutive decodes
    # give the same word slightly different timestamps, so a released sequence
    # can step backwards by a few tens of ms. That is timestamp jitter, not a
    # duplicated word, but the onset cursor below must still advance monotonely
    # or a word would be checked against a silence that came after the one it
    # actually preceded.
    for _, end_ms, at_ms in sorted(words, key=lambda w: w[1]):
        lags.append(at_ms - end_ms + decode_wall_ms)
        while k < len(onsets) and onsets[k] < end_ms:
            k += 1
        if k < len(onsets):
            n += 1
            for w in ready:
                if at_ms + w <= onsets[k] + STALL_MS:
                    ready[w] += 1
    lags.sort()
    audio_lags = sorted(x - decode_wall_ms for x in lags)
    return {
        "lag_ms_median": int(lags[len(lags) // 2]) if lags else None,
        "lag_ms_p90": int(lags[int(0.9 * (len(lags) - 1))]) if lags else None,
        # The same wait with the decode charged at zero. This separates what
        # the POLICY costs from what the MODEL costs: the first is fixed by the
        # buffer and stride, the second falls ~4x if word timestamps are turned
        # off. Only the first is irreducible.
        "lag_audio_ms_median": (int(audio_lags[len(audio_lags) // 2])
                                if audio_lags else None),
        "frac_ready": round(ready[decode_wall_ms] / n, 4) if n else None,
        "frac_ready_wall0": round(ready[0.0] / n, 4) if n else None,
        "frac_ready_wall500": round(ready[500.0] / n, 4) if n else None,
    }


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--buffers", default="8,10,12")
    ap.add_argument("--strides", default="2,4")
    ap.add_argument("--base-stride", type=float, default=2.0)
    ap.add_argument("--policies", default="scroll,agree,now")
    ap.add_argument("--participants", default="1554,1713,1731,1738,1833,1944")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fresh", action="store_true")
    args = ap.parse_args()

    parsed = load_all(TRANSCRIPTS)
    pids = [p.strip() for p in args.participants.split(",") if p.strip()]
    buffers = [float(x) for x in args.buffers.split(",") if x.strip()]
    strides = [float(x) for x in args.strides.split(",") if x.strip()]
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]

    results = {}
    if OUT.exists() and not args.fresh:
        try:
            results = json.loads(OUT.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    # Load once: the audio, the reference utterances, the VAD silence onsets.
    data = {}
    for pid in pids:
        audio, _ = load_region(pid, args.region, args.skip)
        if audio is None:
            print("  MISSING audio for %s" % pid, flush=True)
            continue
        data[pid] = (audio, in_region(parsed, pid, args), vad_trace(pid, audio, args))
    pids = [p for p in pids if p in data]

    for buf in buffers:
        base = {}
        for pid in pids:
            audio = data[pid][0]
            t0 = time.perf_counter()
            d, fresh = decodes_cached(pid, audio, buf, args.base_stride, args)
            base[pid] = d
            if fresh:
                print("    decoded buf=%g pid=%s  %d ticks  %.0f s wall"
                      % (buf, pid, len(d), time.perf_counter() - t0), flush=True)
        for st in strides:
            sub = {}
            ok = True
            for pid in pids:
                s = subsample(base[pid], args.base_stride, st)
                if s is None:
                    ok = False
                    break
                sub[pid] = s
            if not ok:
                print("  skip stride %g (not a multiple of base %g)"
                      % (st, args.base_stride))
                continue
            # RTF for THIS stride: the mean cost of one decode over this buffer
            # times the number of decodes the stride actually performs. The
            # per-decode wall times come from the base pass, which decodes the
            # identical buffers.
            walls = [x["wall"] for pid in pids for x in base[pid]]
            mean_wall = float(np.mean(walls)) if walls else 0.0
            n_dec = sum(len(sub[pid]) for pid in pids)
            audio_s = sum(len(data[pid][0]) / SR for pid in pids)
            rtf = mean_wall * n_dec / audio_s if audio_s else None
            for pol in policies:
                key = "fixed@buffer=%g@stride=%g@commit=%s" % (buf, st, pol)
                tot = {"errors": 0, "n_ref": 0, "n_hyp": 0, "cat_errors": 0,
                       "n_utt": 0}
                per_pid = {}
                all_lat = {"lag_ms_median": [], "lag_ms_p90": [],
                           "lag_audio_ms_median": [], "frac_ready": [],
                           "frac_ready_wall0": [], "frac_ready_wall500": []}
                nw = 0
                for pid in pids:
                    words = commit(sub[pid], pol)
                    nw += len(words)
                    r = score([(t, e) for t, e, _ in words], data[pid][1], args)
                    r.update(latency(words, data[pid][2], mean_wall * 1000))
                    per_pid[pid] = r
                    for k in tot:
                        tot[k] += r[k]
                    for k in all_lat:
                        if r[k] is not None:
                            all_lat[k].append(r[k])
                results[key] = {
                    "buffer_s": buf, "stride_s": st, "commit": pol,
                    "wer": round(tot["errors"] / max(1, tot["n_ref"]), 4),
                    "wer_concat": round(tot["cat_errors"] / max(1, tot["n_ref"]), 4),
                    "n_ref_words": tot["n_ref"], "n_hyp_words": tot["n_hyp"],
                    "n_utterances": tot["n_utt"], "n_committed": nw,
                    "lag_ms_median": (int(np.median(all_lat["lag_ms_median"]))
                                      if all_lat["lag_ms_median"] else None),
                    "lag_ms_p90": (int(np.median(all_lat["lag_ms_p90"]))
                                   if all_lat["lag_ms_p90"] else None),
                    "lag_audio_ms_median": (int(np.median(all_lat["lag_audio_ms_median"]))
                                            if all_lat["lag_audio_ms_median"] else None),
                    "frac_ready": (round(float(np.mean(all_lat["frac_ready"])), 4)
                                   if all_lat["frac_ready"] else None),
                    "frac_ready_wall0": (round(float(np.mean(all_lat["frac_ready_wall0"])), 4)
                                         if all_lat["frac_ready_wall0"] else None),
                    "frac_ready_wall500": (round(float(np.mean(all_lat["frac_ready_wall500"])), 4)
                                           if all_lat["frac_ready_wall500"] else None),
                    "decode_wall_ms_mean": round(mean_wall * 1000, 1),
                    "rtf": round(rtf, 3) if rtf else None,
                    "per_participant": per_pid,
                }
                v = results[key]
                print("  %-40s WER %.3f  concat %.3f  lag %5s ms  ready %.2f  RTF %.2f"
                      % (key, v["wer"], v["wer_concat"], v["lag_ms_median"],
                         v["frac_ready"] or 0.0, v["rtf"] or 0.0), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "corpus": "APROCSA -- 6 speakers with chronic post-stroke aphasia",
        "model": args.model, "mode": args.mode,
        "region_s": args.region, "skip_s": args.skip,
        "stall_deadline_ms": STALL_MS,
        "note": ("Fixed rolling-buffer streaming. One decode pass per (buffer, "
                 "speaker) at the base stride; coarser strides are exact "
                 "subsets of it; the three commit policies are derived from "
                 "the same decodes and differ only in when a word is released. "
                 "lag_ms is audio-clock wait plus measured decode wall time. "
                 "frac_ready is the fraction of committed words on the timeline "
                 "within 1300 ms of the first silence onset after they were "
                 "spoken -- the deadline Echo's stall detector fires on."),
        "results": results,
    }, indent=2), encoding="utf-8")

    print("")
    print("FIXED-CHUNK ROLLING BUFFER -- FULL GRID")
    print("  %-44s %7s %7s %8s %7s %6s"
          % ("config", "WER", "concat", "lag_ms", "ready", "RTF"))
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["wer"]):
        print("  %-44s %7.3f %7.3f %8s %7.2f %6.2f"
              % (k, v["wer"], v["wer_concat"], v["lag_ms_median"],
                 v["frac_ready"] or 0.0, v["rtf"] or 0.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
