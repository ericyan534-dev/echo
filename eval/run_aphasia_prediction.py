"""Does Echo offer THE word the person was reaching for?

WHY THIS EXISTS (and why every detection metric here has failed to discriminate)
--------------------------------------------------------------------------------
Every detection number on this project has hit the same wall. In real aphasic
speech the clinician-coded dysfluency markers arrive every 1.2-2.3 s (8.19/min,
eval/results/aprocsa_alignment.json). "Is this person having word-finding
difficulty right now?" is therefore almost always yes, so a metronome scores
about as well as a detector and detection TIMING cannot separate systems on
this corpus.

What can separate them -- and what actually decides whether the product helps
anyone -- is whether the word Echo offers is the word the speaker wanted.

THE GROUND TRUTH IS ALREADY IN THE CORPUS
-----------------------------------------
APROCSA CHAT transcripts encode word searches AND their resolutions. A
retracing, `[//]`, means the speaker abandoned an attempt and restarted; what
follows is what they were trying to say:

    and &-um I have speech &-um (.) &-um (...) spring [//] Christmas
                                                       ^^^^^^ the intended word

So each well-timed `[//]` is a free (fragment, intended-word) pair: rewind the
ASR stream to the marker instant, ask the real predictor, and check the answer
against the clinician's transcript. That is a prediction benchmark nobody had
to label.

`+...` (trailing off) is the opposite case -- the speaker gave up and there is
no resolution. Those are exactly the moments an aid would matter most and there
is no ground truth for what was wanted, so they are EXCLUDED from accuracy and
counted separately as "cases with no recoverable target".

WHAT IS COMPARED
----------------
    verbatim+ctx    the shipped v3 system: verbatim ASR fragment + context
    intended+ctx    the pre-v3 browser-recognizer proxy (dysfluency stripped)
    verbatim        verbatim fragment, EMPTY context
    intended        intended fragment, EMPTY context
    context_only    no fragment at all -- if this scores near the full system,
                    the fragment is not contributing anything
    freq_english    always answer the commonest English content words (floor)
    freq_corpus     always answer the commonest content words in THIS corpus
                    (a deliberately unfair floor: it has seen the test set)

The 2x2 of (verbatim|intended) x (context|no context) separates the ASR-content
claim from the context machinery instead of reporting them as one lump. Every
arm sees byte-identical events, so the comparison is exactly paired.

HONEST LIMITS, STATED UP FRONT
------------------------------
* n is small. Only the cached 120-420 s region of each participant has ASR
  streams on disk, which is where the scorable retracings come from. The
  headline n is printed and repeated in the JSON; below ~40 the result is
  indicative only and the report says so.
* A retracing is not always a word search. "I went to and [//] I went to speech"
  is a grammatical restart. The `strict_lexical` subset (retracted span of <= 2
  words) is reported alongside the full set for that reason.
* If the target word is already sitting in the ASR fragment (the speaker said
  it in the abandoned attempt), predicting it is near-trivial and VERBATIM mode
  is the arm most likely to contain it -- that is a confound, so
  `target_not_in_fragment` is reported as its own subset.

    python eval/run_aphasia_prediction.py
    python eval/run_aphasia_prediction.py --limit 3          # smoke test
    python eval/run_aphasia_prediction.py --no-llm           # baselines only
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import get_settings  # noqa: E402  (also loads .env)
from backend.schemas import TurnEnd, Word  # noqa: E402
from backend.timeline import FILLERS, Timeline  # noqa: E402
from eval.align_aprocsa import parse_utterances  # noqa: E402
from eval.tune_aphasia_detector import speaker_map, streams  # noqa: E402
from scripts.aprocsa_chat import load_all  # noqa: E402

TRANSCRIPTS = ROOT / "data" / "aprocsa" / "transcripts"
ALIGN_CACHE = ROOT / "eval" / "results" / "cache" / "aprocsa"
PRED_CACHE = ROOT / "eval" / "results" / "cache" / "aphasia_prediction"
OUT = ROOT / "eval" / "results" / "aphasia_prediction.json"

# The only region with cached ASR streams on disk (see run_aphasia_eval.py).
# Re-transcribing outside it costs ~30 min/participant/mode, so the benchmark
# is bounded by what the cache already covers and says so rather than quietly
# shrinking n.
SKIP_S = 120
REGION_S = 300

# Timing quality gate, per the alignment artifact's own confidence semantics:
# `wor_exact` is the corpus's own media bullet and `fa_gap` is forced alignment
# inside a window those bullets pin down. Anything else (coarse bullets,
# interpolation, utterance edges) is not a measurement of WHEN, and a fragment
# cut at a fabricated instant would be a fabricated fragment.
GOOD_SOURCES = ("wor_exact", "fa_gap")
MIN_CONFIDENCE = 0.8

# Wearer-gate threshold. This is the value the detector tuning selected on the
# TUNE split (eval/results/aphasia_tuning.json), reused rather than re-searched:
# fitting it here on the same six speakers would be fitting on the test set.
WEARER_CONF_MIN = 0.35

CONTEXT_TURNS = 6          # the shipped default (backend.config CONTEXT_TURNS)
MAX_LOOKAHEAD = 6          # audible tokens after `[//]` searched for the target
STRICT_SPAN_WORDS = 2      # retracted span at or below this = lexical, not syntactic

# Closed-class words. A retracing that resolves to "the" or "was" tells us
# nothing about word RETRIEVAL, so the target is the first word outside this
# set. Kept explicit (not a POS tagger) so the rule is auditable and stable.
FUNCTION_WORDS = {
    "a", "an", "the", "this", "that", "these", "those",
    "i", "me", "my", "mine", "myself", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs", "there", "here",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "done", "have", "has", "had", "having",
    "will", "would", "shall", "should", "can", "could", "may", "might",
    "must", "let", "'s", "n't", "not", "no", "yes", "yeah", "okay", "ok",
    "and", "or", "but", "so", "if", "then", "than", "because", "as",
    "of", "to", "in", "on", "at", "for", "with", "from", "by", "about",
    "into", "over", "under", "up", "down", "out", "off", "back", "again",
    "very", "just", "too", "also", "only", "well", "now", "all", "some",
    "any", "one", "two", "more", "most", "much", "many", "other", "same",
    "what", "when", "where", "who", "how", "why", "which",
    "like", "know", "mean", "thing", "things", "stuff",
    # Interjections and backchannels. They are produced, but nobody searches
    # for them, and leaving them in lets the frequency baseline answer "mhm".
    "oh", "ah", "aw", "wow", "huh", "uhhuh", "mhm", "mm", "hm", "yep", "yup",
    "nope", "hey", "please", "thanks",
}
# Contraction stems, so "didn't" / "couldn't" are read as the closed-class
# items they are rather than as retrieved vocabulary.
_CONTRACTION_BASES = {
    "don", "doesn", "didn", "isn", "wasn", "aren", "weren", "won", "couldn",
    "wouldn", "shouldn", "hasn", "haven", "hadn", "ain", "mustn", "needn",
}
# `word [: gloss]` -- the clinician's own statement of what the speaker was
# trying to say when the production was a neologism or paraphasia. When CHAT
# supplies it, it IS the ground truth and the surface form is not.
_REPLACEMENT = re.compile(r"(\S+)\s+\[:\s*([^\]]+)\]")

# The floor. Three of the commonest content words in general English (SUBTLEX /
# COCA-style frequency lists agree on this neighbourhood). A system that does
# not beat this is not doing word prediction.
FREQ_ENGLISH = ["time", "people", "good"]


# --------------------------------------------------------------------------
# Matching rule
# --------------------------------------------------------------------------
def _norm_word(w: str) -> str:
    """Lowercase, letters and apostrophes only."""
    return re.sub(r"[^a-z']", "", str(w).lower()).strip("'")


def stem(w: str) -> str:
    """A crude, DECLARED English stemmer -- suffix stripping, nothing else.

    The scoring rule has to be stated exactly, because "did it offer the word"
    is otherwise a judgement call: "Christmas" vs "christmas" is the same
    answer, "stroke" vs "strokes" is the same answer, "stroke" vs "struck" is
    not (and this does not claim it is). No lemmatizer dependency, so anyone
    can recompute a judgement by hand. The number of judgements this changes
    versus plain exact match is reported in the output.
    """
    s = _norm_word(w)
    if len(s) <= 3:
        return s
    for suf, repl, keep in (("ies", "y", 3), ("ing", "", 3), ("ed", "", 3),
                            ("es", "", 3), ("ly", "", 4), ("s", "", 3)):
        if s.endswith(suf) and len(s) - len(suf) >= keep:
            return s[: len(s) - len(suf)] + repl
    return s


def matches(candidate: str, target: str, exact: bool = False) -> bool:
    """Does one predictor candidate hit the target word?

    A candidate may be a phrase ("blood pressure pills"), so it hits if ANY of
    its tokens matches -- an aid that shows "blood pressure pills" when the
    speaker wanted "pills" has offered the word. `exact` switches off stemming
    and is used only to measure how much stemming moves the result.
    """
    t = _norm_word(target) if exact else stem(target)
    if not t:
        return False
    toks = [w for w in re.split(r"[^A-Za-z']+", str(candidate)) if w]
    for tok in toks:
        if (_norm_word(tok) if exact else stem(tok)) == t:
            return True
    return False


def score_candidates(cands: list[str], target: str, exact: bool = False) -> tuple[int, int]:
    """(top1_hit, top3_hit) as 0/1."""
    top1 = int(bool(cands) and matches(cands[0], target, exact))
    top3 = int(any(matches(c, target, exact) for c in cands[:3]))
    return top1, top3


# --------------------------------------------------------------------------
# Ground truth: retracing -> intended word, from the CHAT transcript
# --------------------------------------------------------------------------
def _is_content(tok_raw: str) -> bool:
    w = _norm_word(tok_raw)
    if not w or len(w) < 2 or w == "xxx" or w in FILLERS or w in FUNCTION_WORDS:
        return False
    # "didn't" / "I'm" / "that's" are a closed-class item plus a clitic, not a
    # word anyone has to retrieve.
    base = w.split("'")[0]
    return not (base in FUNCTION_WORDS or base in _CONTRACTION_BASES)


def content_subwords(raw: str, replacement: str | None = None) -> list[str]:
    """The content words one CHAT token actually contributes.

    Three CHAT facts are handled here rather than being papered over:

    * `[: gloss]` -- the clinician wrote down what the speaker MEANT
      ("fe(...)@u [: aphasia]"). That gloss is the target; the neologism the
      speaker produced is not.
    * `@u` and friends without a gloss are non-words. There is no lexical item
      to predict, so the token contributes nothing and the scan moves on.
    * `_` joins a multiword unit ("you_know", "one_on_one", "p_t"). Splitting it
      keeps the content-word test meaningful: "you_know" is a filler, but
      "one_on_one therapy" still resolves to "therapy".
    """
    if replacement:
        src = replacement
    elif "@" in raw:
        return []
    else:
        src = raw
    return [_norm_word(p) for p in re.split(r"[_\s]+", src) if _is_content(p)]


def replacements(chat: str) -> dict[str, str]:
    """{produced_form: clinician gloss} for one utterance.

    Keyed by the surface string rather than by position because the two
    tokenisations (this file's and the %wor tier's) need not line up token for
    token, and a mis-keyed gloss would silently score the wrong word.
    """
    return {m.group(1): m.group(2).strip() for m in _REPLACEMENT.finditer(chat)}


def retracted_span_words(chat: str, k: int) -> int:
    """How many words the speaker retracted at the k-th `[//]` of this utterance.

    CHAT writes the abandoned material BEFORE the code, either scoped
    (`<a b c> [//]`) or as the single preceding word (`spring [//]`). A long
    span is a sentence restart; a one- or two-word span is a lexical
    substitution, which is the case Echo is actually built for. The two are
    reported separately instead of pooled.
    """
    pos = -1
    for _ in range(k + 1):
        pos = chat.find("[//]", pos + 1)
        if pos < 0:
            return -1
    before = chat[:pos].rstrip()
    if before.endswith(">"):
        open_i = before.rfind("<")
        if open_i < 0:
            return -1
        inner = before[open_i + 1: -1]
        return len([w for w in inner.split() if _norm_word(w)])
    return 1 if before.split() else 0


def target_after(tokens: list[dict], marker_i: int, repl: dict[str, str]) -> dict:
    """The word the speaker produced after a retracing -- the intended word.

    Scans forward over the utterance's audible tokens, skipping pauses, filled
    pauses, fragments and function words, and returns the first CONTENT word
    plus up to two more (some intentions are multi-word: "blood pressure").
    Stops at a `+...` -- past that point the utterance was abandoned and
    anything after it belongs to a different attempt. Gives up after
    MAX_LOOKAHEAD audible tokens of nothing but function words, because by then
    the retracing was a syntactic restart, not a word found.
    """
    seen = 0
    words: list[str] = []
    for t in tokens[marker_i + 1:]:
        if t["k"] == "marker":
            if t["type"] == "trailing_off":
                break
            continue
        if t["k"] != "word":
            continue
        raw = t.get("raw", "")
        if not _norm_word(raw):
            continue
        seen += 1
        got = content_subwords(raw, repl.get(raw))
        words.extend(got)
        if len(words) >= 3:
            break
        if not words and seen >= MAX_LOOKAHEAD:
            break
    words = words[:3]
    return {"target": words[0] if words else None, "target_span": words}


def marker_token_index(pid: str) -> list[tuple[int, int]]:
    """(utterance_index, token_index) for every marker, in align_*.json order.

    align_aprocsa.build() appends one event per marker while walking the same
    token stream this reproduces, so zipping the two recovers each event's
    POSITION in the utterance -- which the cached artifact does not store and
    which is exactly what is needed to read off what came next.
    """
    out = []
    for u in parse_utterances(pid):
        for i, t in enumerate(u["tokens"]):
            if t["k"] == "marker":
                out.append((u["index"], i))
    return out


# --------------------------------------------------------------------------
# What Echo would have had at that instant
# --------------------------------------------------------------------------
def conf_at(spans, t_ms):
    """Wearer confidence at an instant; None where no segment covers it.

    None must never suppress -- same fail-open contract as backend.timeline.
    """
    if not spans:
        return None
    for s in spans:
        if s["t0"] <= t_ms <= s["t1"]:
            return s["conf"]
    return None


def replay(items, spans, upto_rel_ms: int) -> tuple[str, list[str]]:
    """(fragment, recent_turns) as the live pipeline would have had them.

    The cached ASR stream is replayed into a real Timeline with the real wearer
    gate, stopped at the marker instant. The fragment is the open utterance's
    wearer text; the context is the last CONTEXT_TURNS completed turns. Partner
    speech is excluded by the gate, not by the transcript -- the system has no
    transcript at run time.
    """
    tl = Timeline(wearer_conf_min=WEARER_CONF_MIN)
    for t, it in sorted(items, key=lambda kv: kv[0]):
        if isinstance(it, Word):
            # Committed words only: a word still being spoken at the marker
            # instant is not something the pipeline could have used.
            if it.end_ms > upto_rel_ms:
                break
            tl.add_word(Word(text=it.text, start_ms=it.start_ms, end_ms=it.end_ms,
                             is_final=True,
                             wearer_conf=conf_at(spans, it.end_ms + SKIP_S * 1000)))
        elif isinstance(it, TurnEnd):
            if t > upto_rel_ms:
                break
            tl.mark_turn_boundary()
    # Deliberately NOT cleaned. The verbatim stream writes fillers as "[UM]" and
    # the live pipeline hands the predictor exactly that string; stripping them
    # here would delete the very dysfluency the v3 verbatim claim is about and
    # quietly turn the verbatim arm into the intended arm.
    return tl.utterance_text().strip(), tl.completed_turns()[-CONTEXT_TURNS:]


# --------------------------------------------------------------------------
# Event construction
# --------------------------------------------------------------------------
ARMS = {
    # name          -> (fragment mode or None, use context)
    "verbatim+ctx": ("verbatim", True),
    "intended+ctx": ("intended", True),
    "verbatim": ("verbatim", False),
    "intended": ("intended", False),
    "context_only": (None, True),
}


def build_events(pid: str, limit: int = 0) -> tuple[list[dict], dict]:
    align_path = ALIGN_CACHE / ("align_%s.json" % pid)
    if not align_path.exists():
        return [], {"reason": "no alignment cache"}
    align = json.loads(align_path.read_text(encoding="utf-8"))
    positions = marker_token_index(pid)
    if len(positions) != len(align["events"]):
        # A mismatch means the tokeniser and the cached artifact disagree, and
        # every target read off a position would be the wrong word. Refuse
        # rather than produce a plausible-looking wrong benchmark.
        return [], {"reason": "marker/event count mismatch (%d vs %d)"
                    % (len(positions), len(align["events"]))}
    utts = parse_utterances(pid)
    lo, hi = SKIP_S * 1000, (SKIP_S + REGION_S) * 1000

    st = {"retracings_total": 0, "retracings_in_region": 0, "retracings_well_timed": 0,
          "no_recoverable_target": 0, "trailing_off_in_region": 0,
          "empty_fragment_verbatim": 0, "empty_fragment_intended": 0}

    items = {m: streams(pid, SKIP_S, REGION_S, m, "none") for m in ("verbatim", "intended")}
    if any(v is None for v in items.values()):
        return [], {"reason": "no cached ASR stream for %d_%d" % (SKIP_S, REGION_S)}
    spans = speaker_map(pid, SKIP_S, REGION_S)

    # Which occurrence of "[//]" within its utterance each retracing is, so the
    # retracted span can be read back out of the raw CHAT string.
    seen_in_utt: Counter = Counter()
    events: list[dict] = []
    for (ui, ti), ev in zip(positions, align["events"]):
        if ev["marker_type"] == "trailing_off" and ev["t_ms"] is not None \
                and lo <= ev["t_ms"] <= hi:
            st["trailing_off_in_region"] += 1
        if ev["marker_type"] != "retracing":
            continue
        k = seen_in_utt[ui]
        seen_in_utt[ui] += 1
        st["retracings_total"] += 1
        if ev["t_ms"] is None or not (lo <= ev["t_ms"] <= hi):
            continue
        st["retracings_in_region"] += 1
        if ev["source"] not in GOOD_SOURCES or ev["confidence"] < MIN_CONFIDENCE:
            continue
        st["retracings_well_timed"] += 1

        u = utts[ui]
        tgt = target_after(u["tokens"], ti, replacements(u["chat"]))
        if not tgt["target"]:
            st["no_recoverable_target"] += 1
            continue

        rel = ev["t_ms"] - lo
        frags, ctxs = {}, {}
        for m in ("verbatim", "intended"):
            frags[m], ctxs[m] = replay(items[m], spans, rel)
            if not frags[m]:
                st["empty_fragment_%s" % m] += 1
        span_words = retracted_span_words(u["chat"], k)
        events.append({
            "pid": pid,
            "index": len(events),
            "utterance_index": ui,
            "t_ms": ev["t_ms"],
            "timing_source": ev["source"],
            "timing_confidence": ev["confidence"],
            "chat": u["chat"],
            "anchor_token": ev["anchor_token"],
            "target": tgt["target"],
            "target_span": tgt["target_span"],
            "retracted_span_words": span_words,
            "strict_lexical": 0 <= span_words <= STRICT_SPAN_WORDS,
            "fragment": frags,
            "context": ctxs,
            # The confound flag: verbatim keeps the abandoned attempt, so if the
            # speaker already SAID the target before retracing, verbatim can win
            # by copying rather than by predicting.
            "target_in_fragment": {m: matches(frags[m], tgt["target"])
                                   for m in ("verbatim", "intended")},
        })
        if limit and len(events) >= limit:
            break
    return events, st


# --------------------------------------------------------------------------
# Prediction (cached)
# --------------------------------------------------------------------------
def cache_key(ev: dict, arm: str) -> Path:
    mode, use_ctx = ARMS[arm]
    frag = ev["fragment"][mode] if mode else ""
    ctx = ev["context"][mode or "verbatim"] if use_ctx else []
    h = hashlib.sha1(json.dumps([arm, frag, ctx], sort_keys=True).encode("utf-8"))
    # The prompt hash is IN the key on purpose: change how the fragment is built
    # and the cache misses instead of serving an answer to a different question.
    return PRED_CACHE / ("%s_%03d_%s_%s.json" % (ev["pid"], ev["index"], arm,
                                                 h.hexdigest()[:10]))


async def predict_one(predictor, ev: dict, arm: str, sem, model: str) -> dict:
    path = cache_key(ev, arm)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    mode, use_ctx = ARMS[arm]
    frag = ev["fragment"][mode] if mode else ""
    ctx = ev["context"][mode or "verbatim"] if use_ctx else []
    rec = {"arm": arm, "model": model, "fragment": frag, "context": ctx,
           "candidates": [], "error": None}
    async with sem:
        for attempt in range(3):
            try:
                cands = await predictor.predict(list(ctx), frag)
                rec["candidates"] = [c.word for c in cands]
                rec["confidences"] = [round(float(c.confidence), 3) for c in cands]
                rec["error"] = None
                break
            except Exception as exc:                       # noqa: BLE001
                # Errors are recorded, never silently dropped: an arm that fails
                # half its calls would otherwise look like an arm that answers.
                rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:160])
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1), encoding="utf-8")
    return rec


async def run_predictions(events: list[dict], model: str, concurrency: int,
                          arms: list[str]) -> None:
    from backend.predictor.gemini import GeminiPredictor

    s = get_settings()
    predictor = GeminiPredictor(s.gemini_api_key, model=model, max_candidates=3)
    sem = asyncio.Semaphore(concurrency)
    tasks = [(ev, arm) for ev in events for arm in arms]
    done = 0
    results = await asyncio.gather(*[predict_one(predictor, ev, arm, sem, model)
                                     for ev, arm in tasks])
    for (ev, arm), rec in zip(tasks, results):
        ev.setdefault("predictions", {})[arm] = rec
        done += 1
    print("  %d predictions resolved (%d events x %d arms)"
          % (done, len(events), len(arms)))


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------
def corpus_frequency_words(n: int = 3) -> list[str]:
    """Commonest content words across all six participants' own speech.

    Deliberately unfair: it has seen the test corpus. If an arm cannot beat a
    baseline that cheats this way, the arm is not predicting anything.
    """
    c: Counter = Counter()
    for d in load_all(TRANSCRIPTS).values():
        for u in d["utterances"]:
            if not u["is_participant"]:
                continue
            for w in u["text"].split():
                for sub in content_subwords(w):
                    c[sub] += 1
    return [w for w, _ in c.most_common(n)]


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
def hits(cands: list[str], e: dict, exact: bool = False) -> dict:
    """Every scoring view of one (candidates, event) pair, as 0/1 flags.

    Four views, because "did it offer the word" has four defensible readings
    and quoting only the flattering one would be the whole problem:

      top1/top3      against THE target -- the first content word after the
                     retracing. Strict, and the headline.
      top1_span/     against ANY of the (up to 3) content words the speaker
      top3_span      produced as the resolution. Looser, and necessary: in
                     "[//] got ambulance and everything" the retrieved word is
                     plainly "ambulance", but the strict rule calls it "got".
    """
    t1, t3 = score_candidates(cands, e["target"], exact)
    span = e["target_span"]
    return {
        "top1": t1,
        "top3": t3,
        "top1_span": int(bool(cands) and any(matches(cands[0], w, exact) for w in span)),
        "top3_span": int(any(any(matches(c, w, exact) for c in cands[:3]) for w in span)),
    }


METRICS = ("top1", "top3", "top1_span", "top3_span")


def tally(events: list[dict], arm: str, getter, subset=None) -> dict:
    rows = [e for e in events if subset is None or subset(e)]
    n = empty = err = 0
    acc = {m: 0 for m in METRICS}
    acc_x = {m: 0 for m in METRICS}
    for e in rows:
        cands, error = getter(e, arm)
        n += 1
        if error:
            err += 1
        if not cands:
            empty += 1
            continue
        h = hits(cands, e)
        hx = hits(cands, e, exact=True)
        for m in METRICS:
            acc[m] += h[m]
            acc_x[m] += hx[m]
    out = {"n": n, "no_output": empty, "errors": err}
    for m in METRICS:
        out[m] = round(acc[m] / n, 4) if n else None
        out[m + "_hits"] = acc[m]
    out["exact_match_rule"] = {m: round(acc_x[m] / n, 4) if n else None for m in METRICS}
    out["judgements_changed_by_stemming"] = sum(acc[m] - acc_x[m] for m in METRICS)
    return out


def mcnemar_exact(only_a: int, only_b: int) -> float | None:
    """Two-sided exact McNemar p-value from the discordant pairs.

    Written out (math.comb, no scipy) because the whole point is that the
    reader can check it. With n around 50 the discordant counts are single
    digits and a difference of 4 vs 2 events is not evidence of anything; this
    is what says so numerically instead of leaving it to the eye.
    """
    import math

    m = only_a + only_b
    if m == 0:
        return None
    k = min(only_a, only_b)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) / (2 ** m)
    return round(min(1.0, 2 * tail), 4)


def paired(events: list[dict], arm_a: str, arm_b: str, getter_for, metric: str) -> dict:
    """Discordant-pair counts for two arms on the same events.

    With n around 50 a difference of one or two events is noise, and a table of
    rates hides that. This reports the only thing that carries information in a
    paired design: how many events arm A got and arm B did not, and vice versa.
    """
    only_a = only_b = both = neither = 0
    for e in events:
        ca, _ = getter_for(arm_a)(e, arm_a)
        cb, _ = getter_for(arm_b)(e, arm_b)
        a = hits(ca, e)[metric]
        b = hits(cb, e)[metric]
        both += int(a and b)
        only_a += int(a and not b)
        only_b += int(b and not a)
        neither += int(not a and not b)
    return {"metric": metric, "a": arm_a, "b": arm_b, "both": both,
            "only_a": only_a, "only_b": only_b, "neither": neither,
            "mcnemar_p_two_sided": mcnemar_exact(only_a, only_b)}


def llm_getter(e, arm):
    r = (e.get("predictions") or {}).get(arm)
    if not r:
        return [], None
    return r.get("candidates") or [], r.get("error")


def const_getter(words):
    return lambda e, arm: (list(words), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--participants", default="")
    ap.add_argument("--limit", type=int, default=0,
                    help="max scorable events per participant (smoke test)")
    ap.add_argument("--model", default="")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--no-llm", action="store_true", help="baselines only")
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args()

    if not TRANSCRIPTS.is_dir():
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"status": "SKIPPED",
                                   "reason": "no APROCSA transcripts"}, indent=2),
                       encoding="utf-8")
        print("SKIPPED -- no APROCSA transcripts")
        return 0

    pids = [p.strip() for p in args.participants.split(",") if p.strip()] or \
        sorted(re.sub(r"\D", "", p.stem)[:4] for p in TRANSCRIPTS.glob("aprocsa*.cha"))
    arms = [a.strip() for a in args.arms.split(",") if a.strip() in ARMS]
    model = args.model or get_settings().gemini_model

    print("APROCSA word-PREDICTION benchmark -- did Echo offer the right word?")
    print("  region %ds-%ds (the only span with cached ASR streams)"
          % (SKIP_S, SKIP_S + REGION_S))
    print("  ground truth: CHAT retracings [//]; target = first content word after")
    print("")

    t0 = time.time()
    events: list[dict] = []
    stats: dict[str, dict] = {}
    for pid in pids:
        evs, st = build_events(pid, args.limit)
        stats[pid] = st
        events.extend(evs)
        if "reason" in st:
            print("  %s: skipped -- %s" % (pid, st["reason"]))
        else:
            print("  %s: %3d retracings, %2d in region, %2d well-timed, "
                  "%2d scorable (%d no target)"
                  % (pid, st["retracings_total"], st["retracings_in_region"],
                     st["retracings_well_timed"], len(evs),
                     st["no_recoverable_target"]))

    n = len(events)
    print("")
    print("  SCORABLE EVENTS: %d" % n)
    if n == 0:
        OUT.write_text(json.dumps({"status": "SKIPPED", "reason": "no scorable events",
                                   "per_participant_counts": stats}, indent=2),
                       encoding="utf-8")
        print("  nothing to score")
        return 0
    if n < 40:
        print("  !! FEWER THAN 40 EVENTS -- INDICATIVE ONLY, NOT A RESULT !!")

    if not args.no_llm:
        print("")
        print("  predicting with %s (concurrency %d, cache %s) ..."
              % (model, args.concurrency, PRED_CACHE.relative_to(ROOT).as_posix()))
        asyncio.run(run_predictions(events, model, args.concurrency, arms))

    freq_corpus = corpus_frequency_words(3)
    baselines = {"freq_english": FREQ_ENGLISH, "freq_corpus": freq_corpus}
    all_arms = ([] if args.no_llm else list(arms)) + list(baselines)

    def getter_for(arm):
        return const_getter(baselines[arm]) if arm in baselines else llm_getter

    subsets = {
        "all": None,
        "strict_lexical": lambda e: e["strict_lexical"],
        "target_not_in_fragment": lambda e: not e["target_in_fragment"]["verbatim"]
        and not e["target_in_fragment"]["intended"],
    }

    results = {sub: {arm: tally(events, arm, getter_for(arm), fn)
                     for arm in all_arms}
               for sub, fn in subsets.items()}
    per_participant = {
        pid: {arm: tally([e for e in events if e["pid"] == pid], arm, getter_for(arm))
              for arm in all_arms}
        for pid in sorted({e["pid"] for e in events})
    }

    # ---- console report (ASCII only: GBK console) ------------------------
    print("")
    print("POOLED -- n=%d scorable retracings, %d participants"
          % (n, len({e["pid"] for e in events})))
    print("  %-16s %5s %7s %7s %10s %10s %7s %6s"
          % ("arm", "n", "top1", "top3", "top1_span", "top3_span", "no_out", "err"))
    for arm in all_arms:
        r = results["all"][arm]
        print("  %-16s %5d %7s %7s %10s %10s %7d %6d"
              % (arm, r["n"], r["top1"], r["top3"], r["top1_span"], r["top3_span"],
                 r["no_output"], r["errors"]))
    for sub in ("strict_lexical", "target_not_in_fragment"):
        m = results[sub][all_arms[0]]["n"]
        print("")
        print("SUBSET %s -- n=%d" % (sub, m))
        print("  %-16s %7s %7s %10s" % ("arm", "top1", "top3", "top3_span"))
        for arm in all_arms:
            r = results[sub][arm]
            print("  %-16s %7s %7s %10s" % (arm, r["top1"], r["top3"], r["top3_span"]))

    # The contrasts the whole exercise exists to settle, as discordant pairs.
    contrasts = [("verbatim+ctx", "intended+ctx"), ("verbatim+ctx", "verbatim"),
                 ("verbatim+ctx", "context_only"), ("verbatim+ctx", "freq_corpus"),
                 ("verbatim", "intended")]
    contrasts = [(a, b) for a, b in contrasts if a in all_arms and b in all_arms]
    pairs = [paired(events, a, b, getter_for, m)
             for a, b in contrasts for m in ("top3", "top3_span")]
    if pairs:
        print("")
        print("PAIRED DIFFERENCES (same %d events; only_a/only_b are the "
              "discordant pairs)" % n)
        print("  %-32s %-10s %6s %8s %8s %8s %10s"
              % ("contrast", "metric", "both", "only_a", "only_b", "neither",
                 "mcnemar_p"))
        for p in pairs:
            print("  %-32s %-10s %6d %8d %8d %8d %10s"
                  % ("%s vs %s" % (p["a"], p["b"]), p["metric"], p["both"],
                     p["only_a"], p["only_b"], p["neither"],
                     p["mcnemar_p_two_sided"]))

    # Six speakers differ enormously and one of them supplies 15 of the 51
    # events, so pooling alone could hide a single dominant speaker.
    print("")
    print("PER PARTICIPANT (top3 / top3_span)")
    print("  %-6s %4s " % ("pid", "n") + " ".join("%-15s" % a for a in all_arms))
    for pid, row in per_participant.items():
        cells = " ".join("%-15s" % ("%s / %s" % (row[a]["top3"], row[a]["top3_span"]))
                         for a in all_arms)
        print("  %-6s %4d %s" % (pid, row[all_arms[0]]["n"], cells))

    chg = sum(results["all"][a]["judgements_changed_by_stemming"] for a in all_arms)
    print("")
    print("  matching rule: case-insensitive suffix-stripped stem; a phrase "
          "candidate hits if any token matches")
    print("  judgements changed by stemming vs exact match: %d (across all arms "
          "and all four metrics)" % chg)
    print("  baselines: english=%s  corpus=%s" % (FREQ_ENGLISH, freq_corpus))
    print("  cases with NO recoverable target: %d retracings resolved to no "
          "content word; %d '+...' abandoned utterances in region"
          % (sum(s.get("no_recoverable_target", 0) for s in stats.values()),
             sum(s.get("trailing_off_in_region", 0) for s in stats.values())))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "status": "OK",
        "wall_s": round(time.time() - t0, 1),
        "question": ("When a person with aphasia is searching for a word, does "
                     "Echo offer THE word they were reaching for?"),
        "dataset": "APROCSA (Casilio et al. 2022) -- 6 speakers, chronic post-stroke aphasia",
        "ground_truth": ("CHAT retracing [//]; the target is the first content "
                         "word the participant produced after it, from the "
                         "transcript (not ASR). Function words are skipped; the "
                         "list is FUNCTION_WORDS in this file."),
        "region": {"skip_s": SKIP_S, "region_s": REGION_S,
                   "why": "the only span with cached ASR streams on disk"},
        "timing_gate": {"sources": list(GOOD_SOURCES), "min_confidence": MIN_CONFIDENCE},
        "wearer_conf_min": WEARER_CONF_MIN,
        "model": model,
        "matching_rule": ("case-insensitive; both sides reduced by a declared "
                          "suffix-stripping stemmer (ies/ing/ed/es/ly/s, min "
                          "stem 3); a multi-word candidate hits if ANY of its "
                          "tokens matches. top1_exact_match_rule / "
                          "top3_exact_match_rule give the same numbers without "
                          "stemming, and judgements_changed_by_stemming counts "
                          "the difference."),
        "arms": {a: {"fragment_mode": ARMS[a][0], "context": ARMS[a][1]}
                 for a in arms} if not args.no_llm else {},
        "baselines": baselines,
        "n_scorable": n,
        "indicative_only": n < 40,
        "prediction_cache": PRED_CACHE.relative_to(ROOT).as_posix()
        + "/<pid>_<event>_<arm>_<prompt-hash>.json",
        "counts": stats,
        "no_recoverable_target": {
            "retracing_without_content_word":
                sum(s.get("no_recoverable_target", 0) for s in stats.values()),
            "trailing_off_in_region":
                sum(s.get("trailing_off_in_region", 0) for s in stats.values()),
            "note": ("'+...' utterances are the cases an aid would matter most "
                     "for and there is no ground truth for what was wanted, so "
                     "they are excluded from accuracy and counted here."),
        },
        "pooled": results,
        "paired_differences": pairs,
        "per_participant": per_participant,
        "metrics": {
            "top1/top3": "the target = FIRST content word after the retracing",
            "top1_span/top3_span": ("ANY of the up to 3 content words the "
                                    "speaker produced as the resolution -- the "
                                    "looser and more forgiving reading"),
        },
        "caveats": [
            "Six speakers, one 5-minute region each. Not a population estimate.",
            ("A retracing is not always a word search -- some are grammatical "
             "restarts. See the strict_lexical subset (retracted span <= %d words)."
             % STRICT_SPAN_WORDS),
            ("Verbatim ASR keeps the abandoned attempt, so it can contain the "
             "target already. See the target_not_in_fragment subset."),
            ("'intended' mode is a MEASURED proxy for the browser recognizer, "
             "not the browser itself."),
            "Context is wearer-only: the partner's turns are removed by the "
            "speaker gate, exactly as in the live system.",
        ],
        "events": events,
    }, indent=2), encoding="utf-8")
    print("")
    print("  wrote %s" % OUT.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
