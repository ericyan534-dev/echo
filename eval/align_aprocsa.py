"""Marker-level ground truth for APROCSA: WHEN a word search happened.

THE PROBLEM THIS EXISTS TO FIX
------------------------------
eval/run_aphasia_eval.py scores a hit if the detector fires anywhere inside
[utterance_start - 250ms, utterance_end + 1500ms]. APROCSA utterances run long,
so that window credits a fire that has nothing to do with the moment of
difficulty: a detector that fires on a timer scores like one that detects. That
makes utterance-level recall unusable for deciding whether the system works.

What is needed is a timestamp on each CHAT marker, so recall can be measured
against an INSTANT with a tolerance we choose, rather than against a six-second
box the transcript happened to draw.

WHERE THE TIMES COME FROM (two independent sources, both reported)
------------------------------------------------------------------
1. The %wor tier. APROCSA ships a word-alignment tier under every participant
   utterance that is a character-identical copy of the *PAR line with media
   bullets inserted after words (verified here: 1486/1486 utterances match once
   whitespace is normalised). Where a bullet times exactly one token, that is
   the corpus's own alignment and nothing this script computes can beat it.

2. torchaudio MMS_FA forced alignment. The %wor tier leaves filled pauses,
   fragments and pauses untimed, and sometimes collapses a whole clause into a
   single bullet. Forced alignment fills those in. It is run on a +/-1000 ms
   PADDED window around the utterance bullet, deliberately: aligning strictly
   inside the bullet would make the containment sanity check vacuous -- every
   word would be inside by construction. With padding, "does the aligned word
   land inside the transcript's own bullet?" is a test the aligner can fail,
   and its failure rate is reported.

Precedence is source 1 over source 2, because source 1 is corpus data anchored
to the media and source 2 is a model guessing at aphasic speech. Both times are
written to the JSON so a consumer can disagree with that choice without
rerunning anything.

MARKER -> TIME ASSOCIATION RULE (stated, not buried)
----------------------------------------------------
The %wor line is tokenised left to right into audible tokens (words, &-filled
pauses, &+fragments) and markers. Then:

  &-um, &+xx        SELF-TIMED. The marker IS an audible token, so t_ms is its
                    own onset. It is the event, not a pointer to one.
  (.) (..) (...)    t_ms = END of the nearest preceding audible token, i.e. the
                    instant the silence begins. A pause has no acoustics of its
                    own, so its onset is the only defensible instant.
  [/] [//] [* ..]   POST-POSITIONED. CHAT writes these AFTER the material they
  +...              annotate, so t_ms = END of the nearest preceding audible
                    token. For a scoped group <a b c> [//] that resolves to c,
                    the last word before the revision, which is exactly the
                    moment a listener could know a revision was coming.

Every event also carries anchor_span_ms (the anchor token's [start, end]), so a
consumer who wants word-onset instead of word-end can recompute it directly.

WHAT IS NOT FABRICATED
----------------------
Markers that cannot be placed get t_ms: null plus a reason string, and are
counted in the report rather than dropped. The largest such class is
participant 1554, whose transcript contains a contiguous block of lines with no
media bullets at all -- no anchor exists there, so no honest timestamp does.

    python eval/align_aprocsa.py
    python eval/align_aprocsa.py --participants 1554 --force
"""
from __future__ import annotations

import argparse
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
import soundfile as sf  # noqa: E402

from scripts.aprocsa_chat import MARKERS, _COMPILED  # noqa: E402

DATA = ROOT / "data" / "aprocsa"
AUDIO = DATA / "audio"
TRANSCRIPTS = DATA / "transcripts"
CACHE = ROOT / "eval" / "results" / "cache" / "aprocsa"
REPORT = ROOT / "eval" / "results" / "aprocsa_alignment.json"

SR = 16000
PAD_MS = 1000          # padding around the utterance bullet; see docstring
CONTAIN_TOL_MS = 250   # slop allowed before a forced-aligned word is rejected

# Token classes. The alternation is ORDERED: the longest / most specific CHAT
# codes must win before the generic word pattern, or "[* s:ur]" becomes words.
TOK = re.compile(r"""
   \x15(?P<bul>\d+_\d+)\x15
 | (?P<error_code>\[\*\s[^\]]+\])
 | (?P<retracing>\[//\])
 | (?P<repetition>\[/\])
 | (?P<bracket>\[[^\]]*\])
 | (?P<trailing_off>\+\.\.\.)
 | (?P<long_pause>\(\.\.\.\))
 | (?P<medium_pause>\(\.\.\))
 | (?P<short_pause>\(\.\))
 | (?P<gesture>&=\S+)
 | (?P<fragment>&\+\S+)
 | (?P<filled_pause>&-\w+)
 | (?P<amp>&\S+)
 | (?P<scope>[<>])
 | (?P<word>[^\s<>\[\]]+)
""", re.X)
BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")

SELF_TIMED = ("filled_pause", "fragment")
PAUSES = ("short_pause", "medium_pause", "long_pause")
POST_POSITIONED = ("retracing", "repetition", "error_code", "trailing_off")
MARKER_NAMES = tuple(MARKERS)

# Confidence is a SOURCE PRIOR, not a probability. The numbers are stated here
# so that a consumer thresholding on them knows exactly what is being thresholded.
CONF_BASE = {
    "wor_exact": 0.95,       # the corpus's own single-token media bullet
    "fa_gap": 0.85,          # MMS_FA inside a window the corpus bullets pin down
    "fa": 0.75,              # MMS_FA, accepted only if it landed in the bullet
    "wor_coarse": 0.35,      # interpolated inside a multi-word corpus bullet
    "gap_interp": 0.30,      # interpolated between two timed anchors
    "utt_edge": 0.20,        # fell back to the utterance boundary
}
# Unintelligible material corrupts the alignment reference locally, so anything
# in such an utterance is worth less no matter which source timed it.
XXX_PENALTY = 0.6
# Mean CTC token score at which a forced-aligned anchor is treated as fully
# confident. Below it the confidence is scaled down, but only over the range
# [0.6, 1.0] of the source prior: a low CTC posterior on aphasic speech usually
# means the model is unsure WHICH label, not unsure WHERE the token was, and
# collapsing those events to ~0 would throw away the majority of the filled
# pauses. The raw score travels with the event as anchor_fa_score so a consumer
# can be stricter than this.
FA_SCORE_FLOOR = 0.30
FA_SCORE_WEIGHT = 0.4
# Slack added to each side of a corpus-pinned gap before re-aligning inside it.
# Bullet edges are themselves only good to a few tens of ms, so a token that
# genuinely starts on the boundary needs somewhere to go.
GAP_PAD_MS = 60
GAP_MIN_MS = 60          # shorter than this and there is nothing to align in


# --------------------------------------------------------------------------
# CHAT parsing
# --------------------------------------------------------------------------
def _records(path: Path) -> list[str]:
    """CHAT lines with tab-continuations folded into the tier above them."""
    text = path.read_text(encoding="utf-8", errors="replace")
    recs: list[str] = []
    for raw in text.splitlines():
        if raw[:1] in ("*", "@", "%"):
            recs.append(raw)
        elif raw.startswith("\t") and recs:
            recs[-1] += " " + raw.strip()
    return recs


def _norm(w: str) -> str:
    """A word reduced to the MMS_FA label set (a-z and apostrophe).

    "(a)n(d)" is CHAT for an unpronounced sound, so parenthesised letters are
    dropped rather than aligned -- the speaker did not say them. "@q" and its
    relatives are word-level CHAT flags, not phonemes.
    """
    w = w.lower()
    w = re.sub(r"@\S+$", "", w)
    w = re.sub(r"\([a-z']+\)", "", w)
    return re.sub(r"[^a-z']", "", w)


def parse_utterances(pid: str) -> list[dict]:
    """Participant utterances with an ordered token stream carrying %wor bullets.

    The %wor tier is the token source because it is the same string as the *PAR
    line plus bullets; parsing it once yields tokens and times together instead
    of having to reconcile two tokenisations afterwards.
    """
    recs = _records(TRANSCRIPTS / ("aprocsa%sa.cha" % pid))
    out: list[dict] = []
    for i, r in enumerate(recs):
        if not r.startswith("*PAR"):
            continue
        body = r.split(":", 1)[1]
        m = BULLET.search(body)
        wor = None
        for j in range(i + 1, len(recs)):
            if recs[j].startswith("*"):
                break
            if recs[j].startswith("%wor"):
                wor = recs[j].split(":", 1)[1]
                break
        src = wor if wor is not None else body

        toks: list[dict] = []
        for mm in TOK.finditer(src):
            kind = mm.lastgroup
            if kind == "bul":
                a, b = mm.group("bul").split("_")
                toks.append({"k": "bullet", "s": int(a), "e": int(b)})
            elif kind in ("bracket", "scope", "amp", "gesture"):
                continue          # annotation: neither speech nor a marker
            elif kind == "word":
                toks.append({"k": "word", "raw": mm.group(), "norm": _norm(mm.group())})
            else:
                toks.append({"k": "marker", "type": kind, "raw": mm.group()})

        # Cross-check against the marker regexes the rest of the repo uses. If
        # this tokeniser ever disagrees, the whole artifact is wrong, so it is
        # checked per utterance and surfaced rather than assumed.
        ref = {k: len(rx.findall(body)) for k, (rx, _) in _COMPILED.items()}
        mine = {k: sum(1 for t in toks if t["k"] == "marker" and t["type"] == k)
                for k in MARKER_NAMES}
        out.append({
            "index": len(out),
            "start_ms": int(m.group(1)) if m else None,
            "end_ms": int(m.group(2)) if m else None,
            "chat": BULLET.sub("", body).strip(),
            "has_wor": wor is not None,
            "has_xxx": "xxx" in body,
            "tokens": toks,
            "marker_counts": mine,
            "tokenizer_agrees": ref == mine,
        })
    return out


def audible(toks: list[dict]) -> list[int]:
    """Indices of tokens that made a sound: words, filled pauses, fragments.

    A word that normalises to empty made no alignable sound; including it would
    hand the aligner a zero-length reference token and raise.
    """
    idx = []
    for i, t in enumerate(toks):
        if t["k"] == "word" and (t["norm"] or t["raw"] == "xxx"):
            idx.append(i)
        elif t["k"] == "marker" and t["type"] in SELF_TIMED:
            idx.append(i)
    return idx


def wor_spans(toks: list[dict]) -> dict[int, tuple[int, int, int]]:
    """token index -> (bullet_start, bullet_end, tokens_sharing_that_bullet).

    A %wor bullet closes the run of audible tokens since the previous bullet,
    but it does NOT time everything in that run. CLAN puts bullets on WORDS;
    filled pauses and fragments sitting in front of a word are simply left
    untimed, and sweeping them into the bullet would smear the word's time
    backwards across seconds of hesitation. The corpus says so numerically: of
    839 multi-token runs, 597 have nothing but &-/&+ tokens ahead of the final
    word, and dividing their bullet across the whole run gives 160 ms per token
    against 330 ms for the single-word bullets -- i.e. the bullet is the length
    of ONE word, not of the run.

    So: the bullet is shared by the WORDS of the run (1 word => an exact time,
    several => coarse). Self-timed markers in the run get no bullet and fall
    through to forced alignment, which is the only thing that can place them.
    The exception is a run with no words at all -- "&-uh &+t <bullet>" -- where
    the bullet plainly belongs to the last token, so it is given to it.
    """
    res: dict[int, tuple[int, int, int]] = {}
    run: list[int] = []
    for i, t in enumerate(toks):
        if t["k"] == "bullet":
            words = [j for j in run if toks[j]["k"] == "word"]
            owners = words if words else run[-1:]
            for j in owners:
                res[j] = (t["s"], t["e"], len(owners))
            run = []
        elif (t["k"] == "word" and (t["norm"] or t["raw"] == "xxx")) or              (t["k"] == "marker" and t["type"] in SELF_TIMED):
            run.append(i)
    return res


def _ref_word(t: dict) -> str:
    """One alignment-reference token for one audible CHAT token.

    "xxx" is CHAT for unintelligible, so its content is unknown: it becomes the
    MMS_FA star token, which matches arbitrary audio. Aligning a literal "xxx"
    would force the model to find three /ks/ that were never said and would drag
    the neighbouring words out of position.
    """
    if t["k"] == "word":
        return "*" if t["raw"] == "xxx" else (t["norm"] or "*")
    return _norm(t["raw"].lstrip("&+-")) or "*"


# --------------------------------------------------------------------------
# Forced alignment
# --------------------------------------------------------------------------
class Aligner:
    def __init__(self, device: str = "cuda"):
        import torch
        import torchaudio

        self.torch = torch
        bundle = torchaudio.pipelines.MMS_FA
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model = bundle.get_model().to(self.device).eval()
        self.tokenizer = bundle.get_tokenizer()
        self.aligner = bundle.get_aligner()
        self.name = "torchaudio.pipelines.MMS_FA (torchaudio %s)" % torchaudio.__version__

    def __call__(self, wav: np.ndarray, words: list[str]) -> list[tuple[float, float, float]]:
        """(start_ms, end_ms, mean_token_score) per word, relative to wav[0]."""
        w = self.torch.from_numpy(wav)[None].to(self.device)
        with self.torch.inference_mode():
            emission, _ = self.model(w)
        # The model subsamples. Deriving frames->samples from the actual output
        # length is exact; assuming a 320x stride is not.
        ratio = w.shape[1] / emission.shape[1]
        # Star tokens at both ends absorb the padding audio -- including any
        # partner speech in it -- instead of the first and last real words being
        # stretched over it.
        spans = self.aligner(emission[0], self.tokenizer(["*"] + words + ["*"]))[1:-1]
        return [(sp[0].start * ratio / SR * 1000.0,
                 sp[-1].end * ratio / SR * 1000.0,
                 float(np.mean([x.score for x in sp]))) for sp in spans]


# --------------------------------------------------------------------------
# Per-participant build
# --------------------------------------------------------------------------
def build(pid: str, aligner: "Aligner | None") -> dict:
    wav_path = AUDIO / ("%s.wav" % pid)
    info = sf.info(str(wav_path))
    utts = parse_utterances(pid)

    out_utts: list[dict] = []
    events: list[dict] = []
    n_fa_words = n_contained = 0
    n_gap_ok = n_gap_fail = 0
    gap_widths: list[float] = []
    overshoots: list[float] = []
    wor_vs_fa: list[float] = []

    for u in utts:
        idx = audible(u["tokens"])
        rec = {"index": u["index"], "start_ms": u["start_ms"], "end_ms": u["end_ms"],
               "chat": u["chat"], "aligned": False, "fail_reason": None,
               "n_audible": len(idx), "words": []}
        wt = wor_spans(u["tokens"])
        fa: "list[tuple[float, float, float]] | None" = None
        audio: "np.ndarray | None" = None
        a0 = 0

        if u["start_ms"] is None:
            rec["fail_reason"] = "utterance has no media bullet in the transcript"
        elif not idx:
            rec["fail_reason"] = "no audible tokens to align"
        elif aligner is None:
            rec["fail_reason"] = "forced aligner disabled (--no-fa)"
        else:
            a0 = max(0, int((u["start_ms"] - PAD_MS) * SR / 1000))
            a1 = min(info.frames, int((u["end_ms"] + PAD_MS) * SR / 1000))
            if a1 - a0 < SR // 50:
                rec["fail_reason"] = "audio window shorter than 20 ms"
            else:
                audio, _ = sf.read(str(wav_path), start=a0, stop=a1, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                try:
                    fa = aligner(audio, [_ref_word(u["tokens"][i]) for i in idx])
                    rec["aligned"] = True
                except Exception as exc:                 # noqa: BLE001
                    # Nearly always "more reference tokens than emission frames"
                    # on a short bullet packed with words.
                    rec["fail_reason"] = "forced alignment raised %s" % type(exc).__name__
                    fa = None
                if fa is not None:
                    off = a0 * 1000.0 / SR
                    fa = [(s + off, e + off, sc) for s, e, sc in fa]

        # ---- resolve a time for every audible token ----------------------
        # value = (start_ms, end_ms, source, fa_score_or_nan)
        times: dict[int, tuple[float, float, str, float]] = {}
        constraint: dict[int, float] = {}   # width of the window a token was pinned into

        # Pass 1 -- the corpus's own exact word bullets. Nothing here beats
        # those, and they double as the anchors the next pass aligns between.
        for k, i in enumerate(idx):
            w = wt.get(i)
            f = fa[k] if fa is not None and k < len(fa) else None
            if f is not None and u["start_ms"] is not None:
                # THE FREE SANITY CHECK. The unconstrained pass above was given
                # +/-PAD_MS of room; how far did this word escape the
                # transcript's own bullet? Recorded for every word and reported
                # afterwards, whatever it says.
                n_fa_words += 1
                over = max(u["start_ms"] - f[0], f[1] - u["end_ms"], 0.0)
                overshoots.append(over)
                n_contained += int(over <= 0)
                if w is not None and w[2] == 1:
                    wor_vs_fa.append(abs(f[0] - w[0]))
            if w is not None and w[2] == 1:
                times[i] = (float(w[0]), float(w[1]), "wor_exact", float("nan"))

        # Pass 2 -- gap-constrained alignment. Everything still untimed is a
        # filled pause, a fragment, or a word under a coarse bullet, and it sits
        # in a window the corpus already pins down: from the previous exact
        # bullet's end to the next one's start, or the utterance edges.
        # Re-aligning ONLY those tokens against ONLY that audio cannot put them
        # outside the window -- which is exactly what the whole-utterance pass
        # does to 31% of them (measured on this corpus before this pass
        # existed). The window width is kept per token as the honest error bar.
        if aligner is not None and audio is not None and u["start_ms"] is not None:
            off = a0 * 1000.0 / SR
            anchors = [i for i in idx if i in times]
            for a, b in zip([None] + anchors, anchors + [None]):
                run = [i for i in idx if i not in times
                       and (a is None or i > a) and (b is None or i < b)]
                if not run:
                    continue
                lo = times[a][1] if a is not None else float(u["start_ms"])
                hi = times[b][0] if b is not None else float(u["end_ms"])
                if hi - lo < GAP_MIN_MS:
                    continue
                s0 = max(0, int((lo - GAP_PAD_MS - off) * SR / 1000))
                s1 = min(len(audio), int((hi + GAP_PAD_MS - off) * SR / 1000))
                if s1 - s0 < SR * GAP_MIN_MS // 1000:
                    continue
                try:
                    g = aligner(audio[s0:s1], [_ref_word(u["tokens"][i]) for i in run])
                except Exception:                        # noqa: BLE001
                    n_gap_fail += 1
                    continue
                n_gap_ok += 1
                base = off + s0 * 1000.0 / SR
                for n, i in enumerate(run):
                    times[i] = (g[n][0] + base, g[n][1] + base, "fa_gap", g[n][2])
                    constraint[i] = hi - lo
                    gap_widths.append(hi - lo)

        # Pass 3 -- whatever pass 2 could not do. The unconstrained alignment is
        # accepted only if it stayed inside the utterance bullet; then the
        # coarse corpus bullet; then interpolation.
        for k, i in enumerate(idx):
            if i in times:
                continue
            w = wt.get(i)
            f = fa[k] if fa is not None and k < len(fa) else None
            if f is not None and u["start_ms"] is not None and \
                    f[0] >= u["start_ms"] - CONTAIN_TOL_MS and \
                    f[1] <= u["end_ms"] + CONTAIN_TOL_MS:
                times[i] = (f[0], f[1], "fa", f[2])
            elif w is not None:
                # Coarse bullet: place the token at its proportional position
                # inside the phrase the bullet covers. Better than nothing, and
                # labelled so nobody mistakes it for a measurement.
                run = [j for j in idx if wt.get(j) == w]
                p = run.index(i)
                step = (w[1] - w[0]) / max(len(run), 1)
                times[i] = (w[0] + p * step, w[0] + (p + 1) * step,
                            "wor_coarse", float("nan"))

        # Interpolate the tokens still untimed between their nearest timed
        # neighbours. Only inside a media-aligned utterance: outside one there
        # is no anchor at all, and inventing a time there would be fabrication.
        if u["start_ms"] is not None:
            timed = [i for i in idx if i in times]
            for a, b in zip([None] + timed, timed + [None]):
                gap = [i for i in idx if i not in times
                       and (a is None or i > a) and (b is None or i < b)]
                if not gap:
                    continue
                lo = times[a][1] if a is not None else float(u["start_ms"])
                hi = times[b][0] if b is not None else float(u["end_ms"])
                hi = max(hi, lo)
                step = (hi - lo) / len(gap)
                for n, i in enumerate(gap):
                    times[i] = (lo + n * step, lo + (n + 1) * step,
                                "gap_interp", float("nan"))

        for k, i in enumerate(idx):
            t = u["tokens"][i]
            w = wt.get(i)
            tm = times.get(i)
            rec["words"].append({
                "token": t.get("raw", ""),
                "ref": _ref_word(t),
                "start_ms": round(tm[0]) if tm else None,
                "end_ms": round(tm[1]) if tm else None,
                "source": tm[2] if tm else None,
                "fa_start_ms": round(fa[k][0]) if fa else None,
                "fa_end_ms": round(fa[k][1]) if fa else None,
                "fa_score": round(fa[k][2], 4) if fa else None,
                "wor_start_ms": w[0] if w else None,
                "wor_end_ms": w[1] if w else None,
                "wor_exact": bool(w and w[2] == 1),
                "constraint_ms": round(constraint[i]) if i in constraint else None,
            })

        # ---- marker -> timed event ---------------------------------------
        for i, t in enumerate(u["tokens"]):
            if t["k"] != "marker":
                continue
            mtype = t["type"]
            ev = {"marker_type": mtype, "weight": MARKERS[mtype][1], "raw": t["raw"],
                  "utterance_index": u["index"], "t_ms": None, "source": None,
                  "confidence": 0.0, "anchor_token": None, "anchor_span_ms": None,
                  "anchor_fa_score": None, "anchor_constraint_ms": None,
                  "reason": None}
            anchor_score = float("nan")
            anchor_i = None
            if mtype in SELF_TIMED:
                tm = times.get(i)
                if tm is None:
                    ev["reason"] = rec["fail_reason"] or "no anchor in this utterance"
                else:
                    ev["t_ms"] = round(tm[0])            # onset of the audible event
                    ev["source"] = tm[2]
                    ev["anchor_token"] = t["raw"]
                    ev["anchor_span_ms"] = [round(tm[0]), round(tm[1])]
                    anchor_score = tm[3]
                    anchor_i = i
            else:
                prev = [j for j in idx if j < i and j in times]
                if prev:
                    tm = times[prev[-1]]
                    ev["t_ms"] = round(tm[1])            # end of the anchor token
                    ev["source"] = tm[2]
                    ev["anchor_token"] = u["tokens"][prev[-1]].get("raw", "")
                    ev["anchor_span_ms"] = [round(tm[0]), round(tm[1])]
                    anchor_score = tm[3]
                    anchor_i = prev[-1]
                elif u["start_ms"] is not None:
                    # Nothing audible precedes it in this utterance. The only
                    # defensible instant left is the utterance boundary.
                    edge = u["end_ms"] if mtype == "trailing_off" else u["start_ms"]
                    ev["t_ms"] = int(edge)
                    ev["source"] = "utt_edge"
                    ev["anchor_span_ms"] = [u["start_ms"], u["end_ms"]]
                else:
                    ev["reason"] = "utterance has no media bullet in the transcript"
            if anchor_score == anchor_score:                # not NaN
                ev["anchor_fa_score"] = round(anchor_score, 4)
            if anchor_i is not None and anchor_i in constraint:
                # The width of the window the corpus bullets pinned the anchor
                # into. This is the honest error bar on t_ms for everything the
                # gap pass placed: a scorer should widen its tolerance by it
                # rather than treat these timestamps as exact.
                ev["anchor_constraint_ms"] = round(constraint[anchor_i])
            if ev["source"]:
                c = CONF_BASE[ev["source"]]
                if ev["source"] in ("fa", "fa_gap") and anchor_score == anchor_score:
                    c *= (1.0 - FA_SCORE_WEIGHT) + FA_SCORE_WEIGHT * min(
                        1.0, max(anchor_score, 0.0) / FA_SCORE_FLOOR)
                if u["has_xxx"]:
                    c *= XXX_PENALTY
                ev["confidence"] = round(c, 3)
                if u["start_ms"] is not None and not (
                        u["start_ms"] <= ev["t_ms"] <= u["end_ms"]):
                    # Accepted forced alignments may sit up to CONTAIN_TOL_MS
                    # outside the bullet, and a gap interpolated off one of them
                    # inherits that. Flagged rather than clamped, so the artifact
                    # never claims more precision than it has.
                    ev["outside_utterance_bullet"] = True
            events.append(ev)

        out_utts.append(rec)

    # Control for the containment check: the corpus's OWN word bullets are
    # tested against the same utterance bullets. If they escaped too, the
    # containment metric would be measuring the transcript, not the aligner.
    wor_in = wor_tot = 0
    for u in utts:
        if u["start_ms"] is None:
            continue
        for _, (s_, e_, _n) in wor_spans(u["tokens"]).items():
            wor_tot += 1
            wor_in += int(s_ >= u["start_ms"] and e_ <= u["end_ms"])

    dur_min = info.frames / info.samplerate / 60.0
    placed = [e for e in events if e["t_ms"] is not None]
    ov = np.array(overshoots) if overshoots else np.array([0.0])
    ag = np.array(wor_vs_fa) if wor_vs_fa else np.array([0.0])
    return {
        "participant": pid,
        "audio": {"path": "data/aprocsa/audio/%s.wav" % pid,
                  "minutes": round(dur_min, 2), "sample_rate": info.samplerate},
        "aligner": aligner.name if aligner else None,
        "pad_ms": PAD_MS, "contain_tol_ms": CONTAIN_TOL_MS,
        "rules": {"self_timed": list(SELF_TIMED), "pauses": list(PAUSES),
                  "post_positioned": list(POST_POSITIONED),
                  "confidence_base": CONF_BASE, "xxx_penalty": XXX_PENALTY},
        "counts": {
            "utterances": len(utts),
            "utterances_media_aligned": sum(1 for u in utts if u["start_ms"] is not None),
            "utterances_fa_ok": sum(1 for r in out_utts if r["aligned"]),
            "utterances_fa_failed": sum(1 for r in out_utts if r["fail_reason"]),
            "tokenizer_disagreements": sum(1 for u in utts if not u["tokenizer_agrees"]),
            "markers": len(events),
            "markers_timed": len(placed),
            "markers_untimed": len(events) - len(placed),
            "by_source": {s: sum(1 for e in events if e["source"] == s)
                          for s in sorted({e["source"] for e in events if e["source"]})},
            "by_type": {m: sum(1 for e in events if e["marker_type"] == m)
                        for m in MARKER_NAMES},
            "timed_by_type": {m: sum(1 for e in placed if e["marker_type"] == m)
                              for m in MARKER_NAMES},
        },
        "quality": {
            "fa_words": n_fa_words,
            "fa_words_inside_bullet": n_contained,
            "fa_containment": round(n_contained / n_fa_words, 4) if n_fa_words else None,
            "fa_containment_tol250": round(float((ov <= 250).mean()), 4) if overshoots else None,
            "fa_escape_ms_p50": round(float(np.percentile(ov, 50)), 1),
            "fa_escape_ms_p90": round(float(np.percentile(ov, 90)), 1),
            "fa_escape_ms_p99": round(float(np.percentile(ov, 99)), 1),
            "gap_alignments_ok": n_gap_ok,
            "gap_alignments_failed": n_gap_fail,
            "gap_width_ms_p50": round(float(np.percentile(gap_widths, 50)), 1) if gap_widths else None,
            "gap_width_ms_p90": round(float(np.percentile(gap_widths, 90)), 1) if gap_widths else None,
            "wor_self_containment": round(wor_in / wor_tot, 4) if wor_tot else None,
            "wor_self_containment_n": wor_tot,
            "markers_outside_utterance_bullet":
                sum(1 for e in placed if e.get("outside_utterance_bullet")),
            "wor_vs_fa_n": len(wor_vs_fa),
            "wor_vs_fa_abs_ms_p50": round(float(np.percentile(ag, 50)), 1),
            "wor_vs_fa_abs_ms_p90": round(float(np.percentile(ag, 90)), 1),
        },
        "rate": {
            "audio_minutes": round(dur_min, 2),
            "markers_per_min": round(len(events) / dur_min, 2),
            "timed_markers_per_min": round(len(placed) / dur_min, 2),
            "strong_timed_per_min": round(
                sum(1 for e in placed if e["weight"] == "strong") / dur_min, 2),
        },
        "utterances": out_utts,
        "events": events,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    ap.add_argument("--no-fa", action="store_true",
                    help="use the %%wor tier only; skip the forced aligner")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not TRANSCRIPTS.is_dir():
        print("SKIPPED -- no APROCSA transcripts")
        return 0
    pids = [p.strip() for p in args.participants.split(",") if p.strip()] or \
        sorted(re.sub(r"\D", "", p.stem)[:4] for p in TRANSCRIPTS.glob("aprocsa*.cha"))
    pids = [p for p in pids if (AUDIO / ("%s.wav" % p)).exists()]
    if not pids:
        print("SKIPPED -- no APROCSA audio")
        return 0

    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [p for p in pids if args.force or not (CACHE / ("align_%s.json" % p)).exists()]
    aligner = None
    if todo and not args.no_fa:
        print("loading forced aligner ...", flush=True)
        aligner = Aligner(args.device)
        print("  %s on %s" % (aligner.name, aligner.device), flush=True)

    results: dict[str, dict] = {}
    t0 = time.time()
    for pid in pids:
        path = CACHE / ("align_%s.json" % pid)
        if pid in todo:
            print("  %s: aligning ..." % pid, flush=True)
            d = build(pid, aligner)
            path.write_text(json.dumps(d), encoding="utf-8")
        else:
            d = json.loads(path.read_text(encoding="utf-8"))
        results[pid] = d
        c, q = d["counts"], d["quality"]
        print("  %s  utt %4d (fa-ok %4d, fa-fail %3d)  markers %4d timed %4d  "
              "contain %.3f  wor-vs-fa p50 %5.0f ms"
              % (pid, c["utterances"], c["utterances_fa_ok"], c["utterances_fa_failed"],
                 c["markers"], c["markers_timed"], q["fa_containment"] or 0.0,
                 q["wor_vs_fa_abs_ms_p50"]), flush=True)

    # -------- pooled report ------------------------------------------------
    tot_m = sum(r["counts"]["markers"] for r in results.values())
    tot_t = sum(r["counts"]["markers_timed"] for r in results.values())
    tot_w = sum(r["quality"]["fa_words"] for r in results.values())
    tot_c = sum(r["quality"]["fa_words_inside_bullet"] for r in results.values())
    tot_min = sum(r["audio"]["minutes"] for r in results.values())
    wi = sum(r["quality"]["wor_self_containment_n"] * r["quality"]["wor_self_containment"]
             for r in results.values())
    wt_ = sum(r["quality"]["wor_self_containment_n"] for r in results.values())
    wi = int(round(wi))
    m_out = sum(r["quality"]["markers_outside_utterance_bullet"] for r in results.values())
    by_src: dict[str, int] = {}
    by_type = {m: [0, 0] for m in MARKER_NAMES}
    reasons: dict[str, int] = {}
    for r in results.values():
        for s, n in r["counts"]["by_source"].items():
            by_src[s] = by_src.get(s, 0) + n
        for m in MARKER_NAMES:
            by_type[m][0] += r["counts"]["by_type"][m]
            by_type[m][1] += r["counts"]["timed_by_type"][m]
        for e in r["events"]:
            if e["t_ms"] is None:
                k = e["reason"] or "unknown"
                reasons[k] = reasons.get(k, 0) + 1

    print("")
    print("POOLED -- %d participants, %.1f min audio" % (len(results), tot_min))
    print("  markers %d, timed %d (%.1f%%), untimed %d"
          % (tot_m, tot_t, 100.0 * tot_t / max(tot_m, 1), tot_m - tot_t))
    print("  forced-aligned words inside their utterance bullet: %.4f (%d/%d)"
          % (tot_c / max(tot_w, 1), tot_c, tot_w))
    print("    control -- corpus %%wor bullets inside the same utterance bullets: "
          "%.4f (%d/%d)" % (wi / max(wt_, 1), wi, wt_))
    print("  timed markers landing outside their utterance bullet: %d (%.2f%%)"
          % (m_out, 100.0 * m_out / max(tot_t, 1)))
    print("  markers/min %.2f, timed markers/min %.2f"
          % (tot_m / tot_min, tot_t / tot_min))
    print("")
    print("  %-14s %7s %7s" % ("marker", "total", "timed"))
    for m in MARKER_NAMES:
        print("  %-14s %7d %7d" % (m, by_type[m][0], by_type[m][1]))
    print("")
    print("  timing source: " + ", ".join("%s=%d" % kv for kv in sorted(by_src.items())))
    if reasons:
        print("  untimed because:")
        for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print("    %5d  %s" % (v, k[:86]))

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "status": "OK",
        "wall_s": round(time.time() - t0, 1),
        "pooled": {"participants": len(results), "audio_minutes": round(tot_min, 1),
                   "markers": tot_m, "markers_timed": tot_t,
                   "fa_words": tot_w, "fa_words_inside_bullet": tot_c,
                   "fa_containment": round(tot_c / max(tot_w, 1), 4),
                   "wor_self_containment": round(wi / max(wt_, 1), 4),
                   "markers_outside_utterance_bullet": m_out,
                   "markers_per_min": round(tot_m / tot_min, 2),
                   "by_source": by_src,
                   "by_type": {m: {"total": by_type[m][0], "timed": by_type[m][1]}
                               for m in MARKER_NAMES},
                   "untimed_reasons": reasons},
        "per_participant": {p: {"counts": r["counts"], "quality": r["quality"],
                                "rate": r["rate"]} for p, r in results.items()},
        "events_cache": "eval/results/cache/aprocsa/align_<pid>.json",
    }, indent=2), encoding="utf-8")
    print("")
    print("  wrote eval/results/aprocsa_alignment.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
