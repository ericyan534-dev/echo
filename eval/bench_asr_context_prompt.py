"""Does the continuation PROMPT buy accuracy, and what does it break?

THE LEAD
--------
CrisperWhisper2 was trained with a continuation objective: each longform chunk
is decoded with the previous chunk's last words in the decoder prompt as
``<ctx> ... <ectx>``. `model.transcribe()` only populates that slot from inside
the longform strategies, and those never run on the 3.5-8 s windows Echo
streams -- every one is under the 30 s encoder field, so the short path is
taken and the context slot is left empty. A prompt costs no latency, so if the
prompt is worth WER it is the only large-and-streamable ASR gain this project
has identified. (2.7 WER points of streaming-policy tuning bought exactly
nothing downstream; see docs/VERSIONS.md v5.)

THE CONFOUND THIS SCRIPT EXISTS TO SEPARATE
-------------------------------------------
The obvious ablation -- run the offline pass with the context suppressed --
does NOT isolate the prompt. The continuation strategy decodes 30 s chunks at a
26 s stride, so consecutive chunks share 4 s of audio, and the ONLY thing that
stops the second chunk re-transcribing that 4 s is the context prompt telling
it what has already been said. Remove the prompt and every seam duplicates
about ten words. That is a stitching failure, not a recognition failure, and it
is not something Echo's stream could ever recover: Echo retires a window only
inside silence and the next window starts on audio nobody has decoded, so its
windows do not overlap at all.

So the ablation is run in BOTH regimes:

    offline_ctx              stride 26 (4 s overlap), context ON   <- ships
    offline_noctx            stride 26 (4 s overlap), context OFF
    offline_nooverlap_ctx    stride 30 (no overlap),  context ON
    offline_nooverlap_noctx  stride 30 (no overlap),  context OFF

`offline_ctx` vs `offline_noctx` is the number the lead was written from.
`offline_nooverlap_ctx` vs `offline_nooverlap_noctx` is the one that predicts
anything about streaming, because it is the regime Echo is actually in: the
context describes audio the model cannot hear.

Then the streaming arms, same audio, same scorer, shipped defaults:

    stream_noctx             VerbatimASR as it ships
    stream_ctx               VerbatimASR(context_prompt=True)

DEGENERATE OUTPUTS ARE COUNTED SEPARATELY FROM WER
--------------------------------------------------
Prompt conditioning is famous for making Whisper transcribe its own prompt back
or fall into a repetition loop. A mean WER can improve while a handful of
outputs become garbage, and on a product that reads a suggestion out loud the
garbage is what the user hears -- so a WER win bought with more degenerate
outputs is not a win. Per decode call, against the context that call was given:

  echo   the hypothesis STARTS with >= 3 words that are the tail of its own
         context. In the overlapping offline regime this is not necessarily
         degenerate (the model really can hear those words again); in the
         no-overlap and streaming regimes it is, because that audio is gone.
  loop   some n-gram (n <= 4) repeats >= 3 times back to back. The threshold is
         3 and not 2 on purpose: "you you recently" is a WordRep and is the
         exact evidence verbatim mode exists to preserve. `loop_content`
         repeats the count with fillers removed first, because "[UM] [UM]
         [UM]" is a real hesitation in this corpus and not a decoder loop --
         it is the count to read when asking whether the prompt broke the
         decoder.
  empty  the decoder returned nothing at all for that window.

All three are counted for BOTH arms of every pair; the delta is the finding,
not the absolute, because aphasic speech produces real repetitions.

WHAT IT MEASURED
----------------
    arm                        WER    concat   n_hyp   echo  loop  cnt_loop
    offline_ctx              0.2876   0.2707    1605      0    11         9
    offline_noctx            0.3909   0.3698    1809     32    13         9
    offline_nooverlap_ctx    0.2918   0.2779    1588      0    13        10
    offline_nooverlap_noctx  0.2792   0.2677    1604      0    12        10
    stream_noctx             0.3752   0.3215    1532      0    57        63
    stream_ctx               0.4350   0.3758    1458      0    43        47

`offline_ctx` reproduces the shipped offline 0.2876/0.2707 exactly and
`stream_noctx` reproduces the shipped streaming 0.3752/0.3215 exactly, so the
harness and the scorer are the published ones.

THE PROMPT IS NOT WORTH 0.047, AND IN ECHO'S REGIME IT IS WORTH LESS THAN
NOTHING. The claim this script was written to test -- offline 0.288 with
context against 0.335 without, "0.047 of the gap is the prompt alone" -- does
not reproduce in either direction. Suppressing the prompt at the shipped
stride costs 0.103, not 0.047 (0.2876 -> 0.3909), and 32 of those 72 chunks
open by echoing their own context: that is the 4 s overlap being transcribed
twice, which is stitching, not recognition. Remove the overlap and the sign
flips -- 0.2792 without the prompt against 0.2918 with it, worse on 4 of 6
participants and better on none. (0.335 is almost certainly the `chunked_lcs`
arm from eval/results/asr_offline_vs_stream.json, 0.3347, which is a different
STRATEGY and not a context ablation.)

Streaming agrees, and much louder: 0.3752 -> 0.4350, worse on 5 of 6
participants. The mechanism is visible in the per-window log and is the
train/test mismatch named in backend/stt/context_prompt.py -- at training the
context words are re-heard at the head of the next chunk, so the right
behaviour is to skip them; Echo's windows do not overlap, so skipping them
DELETES speech nobody has transcribed:

    context   "later I couldn't walk for a [UM] I think about four months"
    no ctx    "Three or four months [UH] but [UM]"
    with ctx  "[UH] but [UM]"

Hence n_hyp 1532 -> 1458 words. Note what that means for this script's own
metrics: the feared failure never happened (echo stays 0, loops go DOWN, 57 ->
43), and the real failure is the opposite one -- the prompt makes the decoder
say less, not more. The degeneracy counters do not catch a deletion; the
hypothesis word count does, and it is in the table for that reason.

    python eval/bench_asr_context_prompt.py --arms offline
    python eval/bench_asr_context_prompt.py --arms stream
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.schemas import Word  # noqa: E402
from backend.stt.verbatim import SR, VerbatimASR, get_model  # noqa: E402
from backend.timeline import norm  # noqa: E402
from eval.bench_asr_aphasia_wer import FILLER, FRAME_MS  # noqa: E402
from eval.bench_asr_offline_wer import in_region, score  # noqa: E402
from eval.run_aphasia_eval import CACHE, TRANSCRIPTS, load_region  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

OUT = ROOT / "eval" / "results" / "asr_context_prompt.json"

# Read the shipped streaming policy off the class, never as a literal here:
# the same trap that would have relabelled a 700 ms stream as a 280 ms one.
_SIG = inspect.signature(VerbatimASR.__init__).parameters
SHIPPED = {k: _SIG[k].default for k in
           ("silence_commit_ms", "reset_window_s", "max_window_s",
            "word_time_policy", "context_words", "context_prompt")}


# --------------------------------------------------------------------------
# degeneracy
# --------------------------------------------------------------------------
def _toks(text, drop_fillers: bool = False):
    out = [t for t in (norm(w) for w in (text or "").split()) if t]
    if drop_fillers:
        out = [t for t in out if not FILLER.match(t)]
    return out


def echo_len(context, hyp) -> int:
    """Longest k with hyp[:k] == context[-k:], on normalized words."""
    c, h = _toks(context), _toks(hyp)
    best = 0
    for k in range(1, min(len(c), len(h)) + 1):
        if h[:k] == c[-k:]:
            best = k
    return best


def loop_run(hyp, max_n: int = 4, drop_fillers: bool = False) -> int:
    """Longest number of back-to-back repeats of any n-gram (n <= max_n)."""
    h = _toks(hyp, drop_fillers=drop_fillers)
    best = 1 if h else 0
    for n in range(1, max_n + 1):
        for i in range(len(h) - n + 1):
            gram = h[i:i + n]
            reps, j = 1, i + n
            while h[j:j + n] == gram:
                reps += 1
                j += n
            if reps > best:
                best = reps
    return best


def degeneracy(pairs, echo_min: int = 3, loop_min: int = 3) -> dict:
    n_echo = n_loop = n_loop_content = n_empty = 0
    worst_echo = worst_loop = 0
    examples = []
    for p in pairs:
        e = echo_len(p.get("context"), p.get("hyp"))
        r = loop_run(p.get("hyp"))
        rc = loop_run(p.get("hyp"), drop_fillers=True)
        worst_echo, worst_loop = max(worst_echo, e), max(worst_loop, r)
        bad = []
        if e >= echo_min:
            n_echo += 1
            bad.append("echo%d" % e)
        if r >= loop_min:
            n_loop += 1
            bad.append("loop%d" % r)
        if rc >= loop_min:
            n_loop_content += 1
            bad.append("cloop%d" % rc)
        if not _toks(p.get("hyp")):
            n_empty += 1
            bad.append("empty")
        if bad and len(examples) < 8:
            examples.append({"why": ",".join(bad),
                             "context": p.get("context"),
                             "hyp": (p.get("hyp") or "")[:180]})
    return {"n_calls": len(pairs), "n_echo": n_echo, "n_loop": n_loop,
            "n_loop_content": n_loop_content,
            "n_empty": n_empty, "max_echo_words": worst_echo,
            "max_loop_repeats": worst_loop, "examples": examples}


# --------------------------------------------------------------------------
# offline arms
# --------------------------------------------------------------------------
@contextlib.contextmanager
def no_context():
    """Suppress the ``<ctx> ... <ectx>`` prompt, and change nothing else.

    The longform strategy still computes and RECORDS the context it would have
    used (`ChunkResult.context`), so the degeneracy count can still ask "did
    this chunk echo the words it was told about" for the arm that was never
    told. `_transcribe_v2` imports PromptBuilder inside the function, so
    patching the module attribute reaches it.
    """
    import crisperwhisper.prompt as cwp

    base = cwp.PromptBuilder

    class _NoCtx(base):
        def _build(self, mode, hotwords=None, context=None):
            return base._build(self, mode, hotwords=hotwords, context=None)

    cwp.PromptBuilder = _NoCtx
    try:
        yield
    finally:
        cwp.PromptBuilder = base


OFFLINE_ARMS = {
    # stride 26 leaves the 4 s overlap the continuation objective was fitted
    # on; stride 30 == chunk_duration leaves none, which is Echo's regime.
    "offline_ctx":             dict(stride=26.0, ctx=True),
    "offline_noctx":           dict(stride=26.0, ctx=False),
    "offline_nooverlap_ctx":   dict(stride=30.0, ctx=True),
    "offline_nooverlap_noctx": dict(stride=30.0, ctx=False),
}


def run_offline(pid, audio, args, stride: float, ctx: bool):
    model = get_model(args.model)
    with contextlib.nullcontext() if ctx else no_context():
        res = model.transcribe(audio, sr=SR, language="en", mode=args.mode,
                               word_timestamps=True, chunk_duration=30.0,
                               stride=stride,
                               context_words=SHIPPED["context_words"])
    words = [(w.word, int((w.end if w.end is not None else 0) * 1000))
             for w in (res.words or [])]
    pairs = [{"context": c.context, "hyp": c.text, "chunk": c.chunk_idx}
             for c in (res.chunks or [])]
    return words, pairs


# --------------------------------------------------------------------------
# streaming arms
# --------------------------------------------------------------------------
STREAM_ARMS = {
    "stream_noctx": dict(context_prompt=False),
    "stream_ctx":   dict(context_prompt=True),
}


def run_stream(pid, audio, args, **kw):
    asr = VerbatimASR(model_name=args.model, mode=args.mode, sync=True,
                      context_log=True, **kw)
    step = int(SR * FRAME_MS / 1000)
    out = []
    for i in range(0, len(audio) - step, step):
        pcm = (audio[i:i + step] * 32767).astype("<i2").tobytes()
        for item in asr.feed(pcm):
            if isinstance(item, Word):
                out.append((item.text, item.end_ms))
    asr.close()
    return out, asr.context_log


# --------------------------------------------------------------------------
def cached(name, pid, args):
    path = CACHE / ("ctx_%s_%s_%s_%s_%d_%d.json"
                    % (name, pid, args.model, args.mode, args.skip, args.region))
    if path.exists() and not args.force:
        d = json.loads(path.read_text(encoding="utf-8"))
        return d["words"], d["pairs"]
    audio, _ = load_region(pid, args.region, args.skip)
    if audio is None:
        return None, None
    if name in OFFLINE_ARMS:
        spec = OFFLINE_ARMS[name]
        words, pairs = run_offline(pid, audio, args, spec["stride"], spec["ctx"])
    else:
        words, pairs = run_stream(pid, audio, args, **STREAM_ARMS[name])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"words": words, "pairs": pairs}), encoding="utf-8")
    return words, pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip", type=int, default=120)
    ap.add_argument("--region", type=int, default=300)
    ap.add_argument("--model", default="turbo")
    ap.add_argument("--mode", default="verbatim")
    ap.add_argument("--arms", default="offline",
                    help="'offline', 'stream', 'all', or explicit arm names")
    ap.add_argument("--participants", default="1554,1713,1731,1738,1833,1944")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    groups = {"offline": list(OFFLINE_ARMS), "stream": list(STREAM_ARMS),
              "all": list(OFFLINE_ARMS) + list(STREAM_ARMS)}
    names = []
    for a in args.arms.split(","):
        names.extend(groups.get(a.strip(), [a.strip()]))

    parsed = load_all(TRANSCRIPTS)
    pids = [p.strip() for p in args.participants.split(",") if p.strip()]

    results = {}
    if OUT.exists():
        try:
            results = json.loads(OUT.read_text(encoding="utf-8")).get("results", {})
        except Exception:
            results = {}

    for name in names:
        tot = {"errors": 0, "n_ref": 0, "n_hyp": 0, "cat_errors": 0, "n_utt": 0}
        per_pid = {}
        all_pairs = []
        for pid in pids:
            words, pairs = cached(name, pid, args)
            if words is None:
                continue
            all_pairs.extend(pairs)
            r = score(words, in_region(parsed, pid, args), args)
            per_pid[pid] = r
            for k in tot:
                tot[k] += r[k]
            print("    %-24s %s  WER %.3f  concat %.3f  (%d ref)"
                  % (name, pid, r["wer"], r["wer_concat"], r["n_ref"]), flush=True)
        deg = degeneracy(all_pairs)
        results[name] = {
            "wer": round(tot["errors"] / max(1, tot["n_ref"]), 4),
            "wer_concat": round(tot["cat_errors"] / max(1, tot["n_ref"]), 4),
            "n_ref_words": tot["n_ref"], "n_hyp_words": tot["n_hyp"],
            "n_utterances": tot["n_utt"], "degenerate": deg,
            "per_participant": per_pid,
        }
        print("  %-24s WER %.4f  concat %.4f  | %d calls, echo %d, loop %d/%d, "
              "empty %d"
              % (name, results[name]["wer"], results[name]["wer_concat"],
                 deg["n_calls"], deg["n_echo"], deg["n_loop"],
                 deg["n_loop_content"], deg["n_empty"]), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "corpus": "APROCSA -- 6 speakers with chronic post-stroke aphasia",
        "model": args.model, "mode": args.mode,
        "region_s": args.region, "skip_s": args.skip,
        "shipped_streaming_defaults": SHIPPED,
        "note": ("wer is per-utterance (the shipped metric); wer_concat removes "
                 "the utterance boundaries. Degeneracy is counted per decode "
                 "call against the context that call was given: echo = the "
                 "hypothesis starts with >=3 words that are the tail of its "
                 "own context, loop = an n-gram (n<=4) repeated >=3 times back "
                 "to back (n_loop_content repeats that count with fillers "
                 "removed, since '[UM] [UM] [UM]' is a real hesitation here "
                 "and not a decoder loop), empty = nothing decoded. Offline arms at stride 26 "
                 "overlap by 4 s, so an echo there is partly legitimate; the "
                 "nooverlap arms and the streaming arms have no such excuse."),
        "results": results,
    }, indent=2), encoding="utf-8")

    print("")
    print("CONTINUATION PROMPT, ON vs OFF")
    print("  %-26s %7s %7s %6s %5s %5s %7s %5s" %
          ("arm", "WER", "concat", "calls", "echo", "loop", "cnt_loop", "empty"))
    for k in names:
        v = results.get(k)
        if not v:
            continue
        d = v["degenerate"]
        print("  %-26s %7.4f %7.4f %6d %5d %5d %7d %5d"
              % (k, v["wer"], v["wer_concat"], d["n_calls"], d["n_echo"],
                 d["n_loop"], d.get("n_loop_content", -1), d["n_empty"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
