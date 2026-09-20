"""Generate docs/EVAL.md from the eval result JSONs.

Inputs (each optional -- missing data renders as PENDING so the report can be
regenerated at any point while the dataset downloads / the model trains):
  - eval/results/stall_eval.json      (eval/run_stall_eval.py)
  - eval/results/latency_bench.json   (eval/run_latency_bench.py)
  - eval/results/prolongation_eval.json (eval/run_prolongation_eval.py)
  - models/fillernet_metrics.json     (scripts/train_filler.py)
  - eval/results/prediction_eval.json (eval/run_prediction_eval.py; section
    rendered only when this file exists, with the context-ablation delta if
    eval/results/prediction_eval_ablated.json is also present)
  - eval/results/prediction_eval_longctx_v1.json /
    prediction_eval_longctx_v1_entity.json (eval/run_prediction_eval.py
    --dataset eval/data/prediction_eval_longctx_v1.jsonl [--entity-memory
    on]; renders a compact "Context-window extension (entity memory)"
    subsection under the prediction-accuracy section, only when the
    baseline (non-entity) longctx results exist)
  - eval/results/dual_channel_ablation.json (eval/run_dual_channel_ablation.py;
    system-level acoustic-vs-transcript-only ablation; section rendered only
    when this file exists)
  - eval/results/longconv_eval.json (eval/run_longconv_eval.py; long-
    conversation recall by turn depth, baseline vs ContextBuilder)
  - eval/results/noise_stress.json (eval/run_noise_stress.py; FillerNet
    noise-robustness degradation curve; section rendered only when this file
    exists)

docs/EVAL.md is the only file this writes outside eval/.

Usage: python eval/make_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.acoustic.stream import HOP_MS  # noqa: E402

STALL = ROOT / "eval" / "results" / "stall_eval.json"
LATENCY = ROOT / "eval" / "results" / "latency_bench.json"
PROLONG = ROOT / "eval" / "results" / "prolongation_eval.json"
TRAIN_METRICS = ROOT / "models" / "fillernet_metrics.json"
PRED = ROOT / "eval" / "results" / "prediction_eval.json"
PRED_ABL = ROOT / "eval" / "results" / "prediction_eval_ablated.json"
LONGCTX = ROOT / "eval" / "results" / "prediction_eval_longctx_v1.json"
LONGCTX_ENTITY = ROOT / "eval" / "results" / "prediction_eval_longctx_v1_entity.json"
DUAL = ROOT / "eval" / "results" / "dual_channel_ablation.json"
NOISE = ROOT / "eval" / "results" / "noise_stress.json"
LONGCONV = ROOT / "eval" / "results" / "longconv_eval.json"
SPKGATE = ROOT / "eval" / "results" / "speaker_gate_eval.json"
LONGCONV_LOCAL = ROOT / "eval" / "results" / "longconv_eval_local.json"
ASR_BENCH = ROOT / "eval" / "results" / "asr_model_bench.json"
APHASIA = ROOT / "eval" / "results" / "aphasia_eval.json"
APHASIA_PAUSE = ROOT / "eval" / "results" / "aphasia_pause_fit.json"
STUTTER_METRICS = ROOT / "models" / "stutternet_metrics.json"
SSL_METRICS = ROOT / "models" / "stutternet_ssl_v2_metrics.json"
SSL_MATRIX = ROOT / "eval" / "results" / "stutter_ssl_corpus_matrix.json"
SSL_HOSTLEAK = ROOT / "eval" / "results" / "stutter_ssl_hostleak.json"
PREDICT_APH_PRE = (ROOT / "eval" / "results"
                   / "_pre_sc700_aphasia_prediction.json")
MARKERS = ROOT / "eval" / "results" / "marker_scoring.json"
PREDICT_APH = ROOT / "eval" / "results" / "aphasia_prediction.json"
APH_WER = ROOT / "eval" / "results" / "asr_aphasia_wer.json"
OUT = ROOT / "docs" / "EVAL.md"

PENDING = "PENDING"


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _get(d: dict | None, *keys, default=PENDING):
    """Nested dict lookup that collapses to PENDING anywhere along the path."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _num(x, fmt: str = "{:.3f}") -> str:
    return fmt.format(x) if isinstance(x, (int, float)) else PENDING


def _hits(c) -> str:
    """Render a {'hits','n','rate'} counts dict as 'x/n (p%)' -- raw counts
    alongside the percentage so the small n stays plain."""
    if not isinstance(c, dict) or "hits" not in c or "n" not in c:
        return PENDING
    return f"{c['hits']}/{c['n']} ({_num(100 * c.get('rate', 0), '{:.1f}')}%)"


def _prediction_section(pred: dict | None, pred_abl: dict | None, section_no: int) -> str:
    """Markdown for the prediction-accuracy section; '' when no results exist."""
    if not isinstance(pred, dict):
        return ""

    prov = _get(pred, "provenance", default=None)
    has_abl = isinstance(pred_abl, dict)

    header = "| Split | Top-1 | Top-3 |"
    sep = "|---|---|---|"
    if has_abl:
        header += " Top-1 (context ablated) | Top-1 delta |"
        sep += "---|---|"

    def row(label: str, full_m, abl_m) -> str:
        cells = f"| {label} | {_hits(_get(full_m, 'top1', default=None))} | " \
                f"{_hits(_get(full_m, 'top3', default=None))} |"
        if has_abl:
            a1 = _get(abl_m, "top1", default=None)
            f1 = _get(full_m, "top1", default=None)
            delta = (f"{100 * (a1['rate'] - f1['rate']):+.1f} pts"
                     if isinstance(a1, dict) and isinstance(f1, dict) else PENDING)
            cells += f" {_hits(a1)} | {delta} |"
        return cells

    rows = [header, sep,
            row("Overall", _get(pred, "overall", default=None),
                _get(pred_abl, "overall", default=None))]
    for cat in ("concrete", "proper_context", "abstract_verb"):
        rows.append(row(cat,
                        _get(pred, "per_category", cat, default=None),
                        _get(pred_abl, "per_category", cat, default=None)))

    conf = _get(pred, "confidence", default=None)

    def _c(key: str) -> str:  # None (e.g. zero hits) renders as n/a, not "None"
        v = _get(conf, key, default=None)
        return _num(v) if isinstance(v, (int, float)) else "n/a"

    conf_line = (
        f"Mean predictor confidence (top-ranked candidate) when top-1 correct: "
        f"**{_c('mean_when_correct')}** (n={_get(conf, 'n_correct')}); "
        f"when wrong: **{_c('mean_when_wrong')}** (n={_get(conf, 'n_wrong')})."
        if isinstance(conf, dict) else ""
    )

    norm_rule = _get(prov, "normalization",
                     default='lowercase; strip one trailing "\'s"; strip punctuation; '
                             'collapse whitespace; strip one trailing plural "s"')
    freeze = _get(prov, "freeze_protocol",
                  default="frozen at first commit; see eval/run_prediction_eval.py")
    abl_note = (
        " The context-ablation column re-runs the identical items with the "
        "conversation context removed (`--ablate-context`); the collapse on "
        "`proper_context` items is the designed evidence that conversation "
        "context earns its place in the prompt."
        if has_abl else ""
    )

    return f"""## {section_no}. Prediction accuracy (frozen circumlocution set)

Run: provider **{_get(prov, 'provider')}** / model **{_get(prov, 'model')}**,
{_get(prov, 'timestamp_utc')} (`eval/run_prediction_eval.py`, dataset sha256
`{str(_get(prov, 'dataset_sha256'))[:12]}...`).

{chr(10).join(rows)}

{conf_line}{abl_note}

Read the headline with its scale in view: n=60, hand-authored (construction
below), and a SINGLE live run per condition at the shipped decoding settings
(temperature 0.2, `backend/predictor/gemini.py`) -- not a deterministic
decode, so a re-run could legitimately differ by a few items. The freeze
protocol forbids reacting to such variation by editing the set.

**Disclosure -- how this set was built and scored.** The eval set
(`eval/data/prediction_eval_set.jsonl`) is a hand-authored 60-item set written
by the team: 20 concrete everyday-object circumlocutions, 20 proper-noun items
where the answer appears only in earlier conversation turns (several use
invented names -- e.g. fictional businesses -- so the model cannot answer from
prior knowledge), and 20 abstract/verb items. Fragments mirror how the live
system's stalls look (fillers, repeats, trailing off); themes deliberately do
not overlap the few-shot examples shipped in `backend/prompts.py`. Each item's
`gold` list explicitly enumerates every acceptable surface form; scoring
beyond that list is only the normalization rule: {norm_rule}. Nothing fuzzier
(no embeddings, no LLM judging) is applied. Freeze protocol: {freeze}

**Prior art.** Purohit et al. 2023 (CSCW '23 Companion, "ChatGPT in
Healthcare: Exploring AI Chatbot for Spontaneous Word Retrieval in Aphasia")
is the offline precedent for this measurement: ChatGPT (GPT-3.5) retrieved the
intended word in 11/12 AphasiaBank circumlocution instances (91.67%). Their
n=12 items come from real aphasic speech transcripts with manual output
tagging; our set is larger (n=60) and hand-authored with a mechanical
acceptance rule, so the two numbers are methodologically not directly
comparable -- theirs establishes feasibility, ours measures Echo's shipped
prompt/provider pipeline.

"""


def _longctx_section(longctx: dict | None, longctx_entity: dict | None) -> str:
    """Markdown for the "Context-window extension (entity memory)" subsection
    -- '' when the baseline (non-entity) longctx results don't exist yet.
    Rendered as a subsection UNDER the prediction-accuracy section (##5), not
    a new top-level numbered section."""
    if not isinstance(longctx, dict):
        return ""

    prov = _get(longctx, "provenance", default=None)
    has_entity = isinstance(longctx_entity, dict)

    header = "| Condition | Top-1 | Top-3 |"
    sep = "|---|---|---|"

    def row(label: str, m) -> str:
        return f"| {label} | {_hits(_get(m, 'top1', default=None))} | " \
               f"{_hits(_get(m, 'top3', default=None))} |"

    rows = [header, sep, row("Baseline (windowed context, no injection)", _get(longctx, "overall", default=None))]
    if has_entity:
        rows.append(row("Entity memory ON (out-of-window entities injected)",
                        _get(longctx_entity, "overall", default=None)))

    entity_prov = _get(longctx_entity, "provenance", default=None) if has_entity else None
    entity_note = (
        f" A matched entity-memory run (provider **{_get(entity_prov, 'provider')}** / "
        f"model **{_get(entity_prov, 'model')}**, {_get(entity_prov, 'timestamp_utc')}) "
        f"is shown alongside it."
        if has_entity else
        " No matched --entity-memory on run exists yet; only the baseline is shown."
    )

    return f"""### Context-window extension (entity memory)

Every row above uses the frozen set's short (1-2 turn) contexts, well inside
`context_turns` -- it cannot exercise what happens once a name has scrolled
OUT of the window. This subsection uses a separate, versioned long-context
set (`eval/data/prediction_eval_longctx_v1.jsonl`, 20 items, category
`proper_context`) built for exactly that: each item names its target entity
ONCE in the first 2-4 of 10-16 context turns, then moves on to unrelated
small talk, so by the time the fragment stalls on it the mention is outside
the live system's `context_turns` window (**{_get(prov, "context_turns")}**,
matching `backend.config.Settings.context_turns`).{entity_note}

Run: provider **{_get(prov, 'provider')}** / model **{_get(prov, 'model')}**,
{_get(prov, 'timestamp_utc')} (`eval/run_prediction_eval.py --dataset
eval/data/prediction_eval_longctx_v1.jsonl`, dataset sha256
`{str(_get(prov, 'dataset_sha256'))[:12]}...`).

{chr(10).join(rows)}

**Mechanism disclosure.** Entity memory is a CAPITALIZATION HEURISTIC
(`backend/entities.py` `EntityTracker`), not a named-entity recognizer: it
tracks capitalized tokens not at sentence start, multi-word capitalized runs
(e.g. "Pete's Diner"), and sentence-start words that recur across turns --
disclosed in full in the module docstring. Only entities whose last mention
falls OUTSIDE the `context_turns` window are injected, as one line in the
predictor prompt (`backend/prompts.py` `build_user_text`); entities still
inside the window are not duplicated. The mechanism ships ON by default
(`ENTITY_MEMORY` env; the default was flipped from off to on on the strength
of this measurement, plus a disclosed self-reference denylist so the demo's
own product names are never tracked -- see `backend/entities.py`). All 20 longctx items
are constructed so their target entity is both extractable by EntityTracker
and out-of-window at `context_turns=6` by construction (verified in
`tests/test_prediction_eval.py::test_longctx_mechanism_fairness_gate`) --
this eval isolates and measures the INJECTION benefit specifically; it does
not measure the tracker's general extraction recall/precision on arbitrary
text (that is unit-tested separately in `tests/test_entities.py`, not scored
here). Freeze protocol: same as the frozen set above, applied to
`prediction_eval_longctx_v1.jsonl` -- frozen at first commit, never edited in
response to results.

"""


def _dual_channel_section(d: dict | None, section_no: int) -> str:
    """Markdown for the system-level dual-channel ablation; '' if no results."""
    if not isinstance(d, dict) or d.get("status") != "ok":
        return ""

    con = _get(d, "construction", default={})
    dbp = _get(d, "detected_before_pause", default={})
    lat = _get(d, "latency_from_onset_ms", default={})
    raw = _get(d, "raw_fire_triggers", default={})
    spur = _get(d, "spurious_acoustic_fires_during_fluent_speech", default={})

    dbp_line = (
        f"**{dbp.get('hits')}/{dbp.get('n')}** ({_num(100 * dbp.get('rate', 0), '{:.1f}')}%)"
        if isinstance(dbp.get("hits"), int) else PENDING
    )
    on_stats = lat.get("acoustic_on", {}) if isinstance(lat, dict) else {}
    off_stats = lat.get("pause_off", {}) if isinstance(lat, dict) else {}
    delta_stats = lat.get("paired_delta", {}) if isinstance(lat, dict) else {}

    t = [
        "| Condition | Median latency from filler onset | n |",
        "|---|---|---|",
        f"| Acoustic ON (fused, trigger=filler_acoustic) | "
        f"{_num(on_stats.get('median_ms'), '{:.0f}')} ms | {on_stats.get('n', PENDING)} |",
        f"| Acoustic OFF (transcript-only pause fallback) | "
        f"{_num(off_stats.get('median_ms'), '{:.0f}')} ms | {off_stats.get('n', PENDING)} |",
        f"| Paired delta (OFF - ON, positive = acoustic earlier) | "
        f"{_num(delta_stats.get('median_ms'), '{:.0f}')} ms | {delta_stats.get('n', PENDING)} |",
    ]

    return f"""## {section_no}. Dual-channel ablation (system-level, `eval/run_dual_channel_ablation.py`)

Every other latency number in this report is component-level (one isolated
clip through one detector). This section asks the SYSTEM-level question: in a
synthetic multi-minute conversation with embedded filler stalls, does the
fused acoustic+transcript StallDetector actually beat the transcript-only
pause fallback in practice, not just in isolated benchmarks?

**Construction:** {con.get('mix_ratio_note', PENDING)}, over
{con.get('n_fillers', PENDING)} embedded filler cycles /
{con.get('total_stream_seconds', PENDING)} s of synthetic audio (PFSD TEST
split only, seed {con.get('seed', PENDING)}). Each cycle is real Words clips
back-to-back, then a real Uh/Um clip with no gap (filler onset ==
the preceding word's end, matching how a stall actually starts), then
{con.get('silence_gap_ms', PENDING)} ms of true silence. A filler-stripping
(Chrome-condition) transcript and a {con.get('tick_ms', PENDING)} ms
SilenceTick timer drive the transcript-only detector; the identical raw audio
is also fed through the live `AcousticStream` (conf_thresh=
{con.get('conf_thresh', PENDING)}) to drive the fused detector -- both
detectors observe the SAME word/tick timeline, only the fused one also gets
`observe_acoustic`. **Attribution window:** {con.get('attribution_window', '')}
Construction caveat: {con.get('note', '')}

**Detected before the pause fallback would have fired:** {dbp_line} embedded
fillers -- {dbp.get('definition', '')}.

{chr(10).join(t)}

Raw fire triggers -- ON: {json.dumps(raw.get('on', {}))} (missed
{raw.get('on_missed', PENDING)}); OFF: {json.dumps(raw.get('off', {}))} (missed
{raw.get('off_missed', PENDING)}). Same many-speaker-concatenation caveat as
the stream-level false-alarm bench in Table 2 applies to this stream too.

**Spurious acoustic fires during fluent speech (disclosed, not credited):**
**{spur.get('count', PENDING)}** filler_acoustic fires
({_num(spur.get('rate_per_min'), '{:.2f}')}/min of
{spur.get('fluent_speech_seconds', PENDING)} s fluent speech) landed outside
every cycle's attribution window -- i.e. FillerNet fired during real fluent
speech, not on an embedded filler. {spur.get('definition', '')}

"""


def _noise_stress_section(d: dict | None, section_no: int) -> str:
    """Markdown for the noise-robustness stress test; '' if no results."""
    if not isinstance(d, dict) or d.get("status") != "ok":
        return ""

    conditions = _get(d, "conditions", default={})
    clipped = _get(d, "clipped_mixes", default={})
    order = ["clean", "15dB", "10dB", "5dB"]
    rows = ["| Condition | Standard F1 | Standard P/R | Operating-point F1 (conf>="
            f"{d.get('operating_point_conf_thresh', PENDING)}) | Operating-point P/R | Clipped mixes |",
            "|---|---|---|---|---|---|"]
    for label in order:
        c = conditions.get(label)
        if not isinstance(c, dict):
            continue
        s, o = c.get("standard", {}), c.get("operating_point", {})
        rows.append(
            f"| {label} | {_num(s.get('f1'))} | {_num(s.get('precision'))} / {_num(s.get('recall'))} | "
            f"{_num(o.get('f1'))} | {_num(o.get('precision'))} / {_num(o.get('recall'))} | "
            f"{clipped.get(label, PENDING)}/{c.get('n', PENDING)} |"
        )

    inter = _get(d, "interferer", default={})
    sample = _get(d, "sample", default={})

    return f"""## {section_no}. Noise-robustness stress test (`eval/run_noise_stress.py`)

EVAL-ONLY: no retraining, no threshold changes -- the shipped checkpoint and
the shipped conf={d.get('operating_point_conf_thresh', PENDING)} operating
point are evaluated exactly as they ship. Question: does FillerNet survive a
noisy demo hall?

**Sample.** n={sample.get('n', PENDING)} PFSD TEST-split clips
({sample.get('note', '')}). **Interferer.**
{inter.get('source', PENDING)} ({inter.get('pool_size', PENDING)} clips in the
pool); {inter.get('assignment', '')}. **SNR.** {d.get('snr_definition', '')}

{chr(10).join(rows)}

{d.get('clean_sanity_anchor', '')}

**Caveat.** {d.get('caveat', '')}

"""


def _speaker_gate_section(data: dict | None, section_no: int) -> str:
    """Proximity speaker gate (eval/run_speaker_gate_eval.py).

    Renders '' when results are absent or SKIPPED. This section is deliberately
    blunt: the gate's safety property holds and its usefulness does not, and a
    reader must not be able to take the first without the second.
    """
    if not isinstance(data, dict) or data.get("status") != "OK":
        return ""
    bs = data.get("by_separation_db") or {}
    params = data.get("params") or {}
    keys = sorted(bs, key=lambda x: float(x))
    rows = [
        "| Level separation | Bystander suppressed | Wearer WRONGLY muted |",
        "|---|---|---|",
    ]
    for k in keys:
        p = bs[k].get("p90") or {}
        rows.append("| %s dB | %s | %s |" % (
            k, _num(p.get("bystander_suppression_rate")),
            _num(p.get("wearer_false_suppression_rate"))))
    alt = [
        "| Level separation | Bystander suppressed | Wearer WRONGLY muted |",
        "|---|---|---|",
    ]
    for k in keys:
        m = bs[k].get("median") or {}
        alt.append("| %s dB | %s | %s |" % (
            k, _num(m.get("bystander_suppression_rate")),
            _num(m.get("wearer_false_suppression_rate"))))
    return "\n".join([
        "",
        "## %d. Speaker gate (proximity/energy only)" % section_no,
        "",
        "Two-speaker mixes, wearer and bystander drawn from DIFFERENT podcast",
        "episodes (`%s`), n=%s words per cell, threshold %s."
        % (data.get("source", "?"), (bs.get(keys[0], {}).get("p90") or {}).get("n_wearer", "?"),
           params.get("threshold", "?")),
        "",
        "**Deployed rule** (per-word p90 of frame confidences):",
        "",
        *rows,
        "",
        "The safety property holds: the wearer is **never** wrongly muted, which is",
        "the fail-open contract. The usefulness does not: below 12 dB of separation",
        "the gate suppresses **none** of the bystander's speech, and only ~45% at",
        "12 dB.",
        "",
        "**This is a measured negative result, and it is the honest headline: level",
        "alone does not separate two speakers in a room.** A more aggressive",
        "aggregation buys suppression only by muting the wearer, which is worse than",
        "not gating at all for an assistive device:",
        "",
        *alt,
        "",
        "So proximity-only gating does NOT solve the other-speaker problem. It is",
        "safe, it is nearly free, and it earns its place only in the lav mic's",
        "regime (DJI Mic 2S on the collar at ~5 cm, where a partner across a table sits far below",
        "12 dB down). For the laptop mic -- the primary demo path -- it is close to",
        "inert. Speaker-embedding verification with a short enrollment is the",
        "mechanism that would actually work, and it was deliberately deferred.",
        "",
        "**Unvalidated in real rooms.** These are synthetic mixes of real speech.",
        "No two-speaker recording of the actual hardware exists, and the VAD labels",
        "are oracle, so these numbers are an upper bound.",
        "",
        "Regenerate: `python eval/run_speaker_gate_eval.py`",
        "",
    ])


def _longconv_section(data: dict | None, section_no: int,
                      local: dict | None = None) -> str:
    """Long-conversation recall by turn depth (eval/run_longconv_eval.py).

    Renders '' when the results JSON is absent, like every other optional
    section, so the report never claims a number that was not measured.
    """
    if not isinstance(data, dict) or data.get("status") != "OK":
        return ""
    base = data.get("baseline_by_depth") or {}
    built = data.get("builder_by_depth") or {}
    order = ["early(<=10)", "mid(11-25)", "late(>25)"]
    rows = [
        "| Turn depth of the probe | Baseline (recent-turns tail) | With ContextBuilder | n |",
        "|---|---|---|---|",
    ]
    for b in order:
        if b not in base and b not in built:
            continue
        bs, bl = base.get(b, {}), built.get(b, {})
        rows.append("| %s | %s | %s | %d |" % (
            b, _num(bs.get("top1")), _num(bl.get("top1")),
            bl.get("n", bs.get("n", 0))))
    late_b = (base.get("late(>25)") or {}).get("top1", 0.0)
    late_c = (built.get("late(>25)") or {}).get("top1", 0.0)
    early_c = (built.get("early(<=10)") or {}).get("top1", 0.0)
    return "\n".join([
        "",
        "## %d. Long-conversation recall (top-1 by turn depth)" % section_no,
        "",
        "Each conversation introduces one target in turn 1 and never mentions it",
        "again, so a deep probe is answerable only if long-horizon memory works.",
        "Verbatim window: %s turns. Provider: **%s** / %s."
        % (data.get("verbatim_turns", "?"), data.get("provider", "?"),
           data.get("model", "?")),
        "",
        *rows,
        "",
        "Deep-recall delta from the ContextBuilder: **%+.3f** (%s -> %s in the"
        % (late_c - late_b, _num(late_b), _num(late_c)),
        "late bucket). Builder late-vs-early delta: **%+.3f**." % (late_c - early_c),
        "",
        "**Limits.** The conversations are hand-authored by the team, not",
        "transcripts of people with aphasia, and the filler turns are deliberately",
        "neutral (no competing entities). This is an upper bound on long-range",
        "recall under clean conditions, not field performance. n is small (see the",
        "table), so single-item swings move a bucket by ~0.08.",
        "",
        *(_longconv_local_rows(local) if local else []),
        "Regenerate: `python eval/run_longconv_eval.py`",
        "",
    ])


def _longconv_local_rows(local: dict) -> list[str]:
    """Same benchmark on the on-device model, so the cloud/local gap is visible
    rather than inferred."""
    if local.get("status") != "OK":
        return []
    b, c = local.get("baseline_by_depth") or {}, local.get("builder_by_depth") or {}
    out = ["**Same benchmark on the local model** (%s), for the cloud/local gap:"
           % local.get("model", "local"), ""]
    out += ["| Turn depth | Baseline | With ContextBuilder |", "|---|---|---|"]
    for k in ["early(<=10)", "mid(11-25)", "late(>25)"]:
        if k in b or k in c:
            out.append("| %s | %s | %s |" % (k, _num((b.get(k) or {}).get("top1")),
                                             _num((c.get(k) or {}).get("top1"))))
    out += ["",
            "The ContextBuilder delta is the SAME on both engines (+0.667 in the late",
            "bucket), which is the point worth taking away: the long-horizon memory fix",
            "is an architecture win, not a property of one model. The local model trails",
            "the cloud one on absolute deep recall (0.750 vs 0.917).",
            ""]
    return out



def _asr_section(d, n):
    """Which transcript source keeps the evidence, and what each size costs."""
    if not isinstance(d, dict) or d.get("status") != "OK":
        return ""
    types = ("Block", "Prolongation", "SoundRep", "WordRep", "Interjection")
    rows = ["| transcript source | preserved | " + " | ".join(types) + " | latency |",
            "|---|---|" + "---|" * (len(types) + 1)]
    for name, m in d.get("models", {}).items():
        bt = m.get("by_type", {})
        cells = " | ".join(_num(bt.get(t, {}).get("preserved")) for t in types)
        rows.append("| CrisperWhisper %s | **%s** | %s | %.0f ms |"
                    % (name, _num(m.get("preserved")), cells,
                       m.get("latency_ms_median") or 0))
    base = d.get("chrome_baseline", {})
    rows.append("| Chrome SpeechRecognition | **%s** | %s | -- |"
                % (_num(base.get("preserved")), " | ".join(["--"] * len(types))))
    body = "\n".join(rows)
    return """## %d. Verbatim ASR -- does the transcript keep the stall?

Echo's v2 thesis was that consumer ASR deletes the evidence, and it is
measured: on %s annotated filler clips the Chrome path recovered the filler
**%s** of the time. This section asks whether a recognizer that transcribes
verbatim on purpose changes that, and what size to pay for it.

%s

Source: `eval/bench_asr_models.py`, n=%s SEP-28k events, %.0f s windows.

Metric: %s

**`turbo` wins both axes** -- highest preservation and lowest latency -- and
`large` is *worse* at preserving dysfluency, which is what a bigger model's
stronger normalization prior buys you. Block is hardest for every model, as
expected: a block is silence, and no transcript represents silence.

The decisive control is the same model in `intended` mode on the same audio:
**0.060 preserved against 0.900 verbatim**. Every consumer recognizer makes
that choice silently and exposes no flag to change it.

""" % (n, "{:,}".format(base.get("n") or 0), _num(base.get("preserved")), body,
       d.get("n_events", PENDING), d.get("window_s") or 0, d.get("metric", ""))


def _stutter_section(d, n):
    if not isinstance(d, dict) or not isinstance(d.get("test"), dict):
        return ""
    rows = ["| type | n_pos | prevalence | AP | lift | F1 | precision | recall |",
            "|---|---|---|---|---|---|---|---|"]
    for t, m in d["test"].items():
        rows.append("| %s | %s | %s | **%s** | %s | %s | %s | %s |"
                    % (t, m.get("n_pos", PENDING), _num(m.get("prevalence")),
                       _num(m.get("ap")),
                       ("%sx" % m["ap_lift_over_chance"]
                        if m.get("ap_lift_over_chance") else "--"),
                       _num(m.get("f1")), _num(m.get("precision")), _num(m.get("recall"))))
    return """## %d. StutterNet -- five dysfluency types, trained on people who stutter

The shipped FillerNet is `["uh","um","speech","other"]` trained on
PodcastFillers: fluent podcast hosts saying "um". It has **no class for a
block** -- the silent struggle to initiate a word, and the strongest evidence a
speaker is stuck -- and it detects interjections, which fluent speakers produce
constantly. Both shipped complaints follow from that one fact.

%s

Split: **%s**. n_train=%s, n_val=%s, n_test=%s; %s parameters.

- %s
- %s

Thresholds are fitted on VAL and stored **inside the checkpoint**, so a retrain
cannot silently inherit the previous model's operating point.

""" % (n, "\n".join(rows), d.get("split", PENDING), d.get("n_train", PENDING),
       d.get("n_val", PENDING), d.get("n_test", PENDING),
       "{:,}".format(d.get("params") or 0), d.get("provenance", ""), d.get("caveat", ""))



TYPES = ["Block", "Prolongation", "SoundRep", "WordRep", "Interjection", "ANY"]


def _ssl_section(m, matrix, hostleak, n):
    """WavLM StutterNet: what replacing the representation bought, what more
    data did not buy, and the caveat that turned out to be false."""
    if not isinstance(m, dict):
        return ""
    meta, m = m, (m.get("test") or {})
    if not m.get("Block"):
        return ""
    out = ["""## %d. WavLM StutterNet -- the representation, not the data

The log-mel CNN above gives Block an AP of 0.256, and a calibrated downstream
fit had already given the Block head a weight of exactly **0.0** -- the channel
was carrying no signal worth using. Replacing the front end with WavLM Base+
(learned softmax over 13 hidden states, top 4 transformer layers unfrozen)
changes that, and the rest of this section is about how much, measured against
the ways such a number can be wrong.
""" % n]

    rows = ["| type | n_pos | prevalence | AP | lift | F1 | precision | recall |",
            "|---|---|---|---|---|---|---|---|"]
    for t in TYPES:
        d = m.get(t) or {}
        rows.append("| %s | %s | %s | **%s** | %s | %s | %s | %s |"
                    % (t, d.get("n_pos", PENDING), _num(d.get("prevalence")),
                       _num(d.get("ap")),
                       ("%sx" % d["ap_lift_over_chance"]
                        if d.get("ap_lift_over_chance") else "--"),
                       _num(d.get("f1")), _num(d.get("precision")),
                       _num(d.get("recall"))))
    out.append("\n".join(rows))
    out.append("""
Split: **%s**. n_train=%s, n_val=%s, n_test=%s; %s parameters, %s trainable.
Trained on: %s

Read against the CNN's table above with care: **these are different test
sets** -- the corpus grew from 20,124 to 30,962 clips and the split was
redrawn, so this headline sits below an earlier 0.384 without anything having
regressed. That earlier checkpoint no longer exists on disk; a clean rerun of
its configuration scores Block 0.371 / ANY 0.893. The same-test-set comparison
is the next table.

- %s
- %s
""" % (meta.get("split", PENDING), "{:,}".format(meta.get("n_train") or 0),
       "{:,}".format(meta.get("n_val") or 0), "{:,}".format(meta.get("n_test") or 0),
       "{:,}".format(meta.get("params_total") or 0),
       "{:,}".format(meta.get("params_trainable") or 0),
       meta.get("trained_on", PENDING), meta.get("provenance", ""),
       meta.get("caveat", "")))

    if isinstance(matrix, dict) and matrix.get("delta_clean"):
        drift = matrix.get("split_drift", {})
        out.append("""### Did the extra 10,838 clips help?

The obvious experiment -- score the old checkpoint on the new test set -- is
invalid here, and finding out why was most of the work. `make_splits` drew one
permutation per show from a **shared** RandomState, so growing the corpus
reshuffled every show: **%s of the %s new-test clips sit in the old model's
TRAIN set**. The old model scores Block %s on those and %s on the rest. The
only honest row is the intersection neither model trained on.
""" % ("{:,}".format(drift.get("new_test_in_old_train", 0)),
       "{:,}".format(drift.get("new_test_n", 0)),
       _num(_get(matrix, "old_on_new_dirty", "Block", "ap")),
       _num(_get(matrix, "old_on_new_clean", "Block", "ap"))))

        hdr = ["| arm | n | " + " | ".join(TYPES) + " |",
               "|---|---|" + "---|" * len(TYPES)]
        label = [("old_on_old", "v1 on its own test"),
                 ("old_on_new_all", "v1 on new test (31% leaked)"),
                 ("old_on_new_clean", "**v1 on clean intersection**"),
                 ("new_on_new_all", "v2 on new test"),
                 ("new_on_new_clean", "**v2 on clean intersection**")]
        for key, name in label:
            d = matrix.get(key) or {}
            hdr.append("| %s | %s | %s |"
                       % (name, "{:,}".format(d.get("n", 0)),
                          " | ".join(_num(_get(d, t, "ap")) for t in TYPES)))
        out.append("\n".join(hdr))

        ci = matrix.get("delta_clean_ci95") or {}
        dl = ["", "Paired bootstrap on the clean intersection, v2 minus v1:", "",
              "| type | delta AP | 95% CI | |", "|---|---|---|---|"]
        for t in TYPES:
            c = ci.get(t) or {}
            lo, hi = c.get("lo"), c.get("hi")
            real = (lo is not None and hi is not None and lo * hi > 0)
            dl.append("| %s | %+0.4f | [%s, %s] | %s |"
                      % (t, matrix["delta_clean"].get(t, 0.0), _num(lo), _num(hi),
                         "**real**" if real else "no effect"))
        out.append("\n".join(dl))
        out.append("""
**More data did not fix Block.** +0.001 AP, with a CI tight enough to rule out
anything past 0.03 in either direction, on a head whose positive count grew by
50%. Whatever limits block detection, it is not corpus size. What the extra
data bought is Interjection and a sliver of ANY -- the two heads that were
already working.
""")

    if isinstance(hostleak, dict) and hostleak.get("_mean_gap"):
        shows = [k for k in hostleak if not k.startswith("_")]
        hl = ["""### The caveat repeated for three versions, and false

Every episode-disjoint number since v3 carried "optimistic -- leaks the
podcast's recurring host". It sounded appropriately humble and nobody had
measured it. Measuring it means scoring **identical clips** with two models,
one trained having seen that show's host and one not, so host exposure is the
only thing that differs. Comparing a nine-show mixture against one show, as
was first tried, confounds host exposure with show difficulty instead.
""", "| show | n | " + " | ".join(TYPES) + " |",
              "|---|---|" + "---|" * len(TYPES)]
        for sh in shows:
            d = hostleak[sh]
            hl.append("| %s | %s | %s |"
                      % (sh, d.get("n", PENDING),
                         " | ".join("%+0.3f" % (d["gap"].get(t) or 0.0)
                                    for t in TYPES)))
        mg = hostleak["_mean_gap"]
        hl.append("| **mean gap** | | %s |"
                  % " | ".join("**%+0.3f**" % (mg.get(t) or 0.0) for t in TYPES))
        out.append("\n".join(hl))
        seen = {sh: (_get(hostleak, sh, "host_seen", "Block", "ap") or 0.0)
                for sh in shows}
        hi = max(seen, key=lambda k: seen[k])
        lo = min(seen, key=lambda k: seen[k])
        out.append("""
Mean gap on Block **%+0.3f**, on ANY **%+0.3f**, no per-show gap above 0.08,
and the sign is inconsistent. The caveat is retracted.

What actually moves the number is which show you test on: the same model
scores Block %s on %s and %s on %s. Show difficulty dominates host identity by
roughly an order of magnitude, and that is the caveat that should have been
written in its place.
""" % (mg.get("Block") or 0.0, mg.get("ANY") or 0.0,
       _num(seen[hi]), hi, _num(seen[lo]), lo))

    cal = meta.get("frame_calibration_detail") or {}
    fr = ["| type | frame recall | clean fire rate | logit threshold |",
          "|---|---|---|---|"]
    for t in TYPES:
        c = cal.get(t) or {}
        if c:
            fr.append("| %s | **%s** | %s | %s |"
                      % (t, _num(c.get("recall_at_threshold")),
                         _num(c.get("clean_fire_rate")),
                         _num(c.get("logit_threshold"))))
    out.append("""### What it costs, and the number the table above does not show

WavLM is 95.5M parameters against the CNN's 583k and costs 156 ms per 3 s
window on a full CPU against a 125 ms hop -- it is GPU-only (19 ms) until
distilled, which is why `STUTTER_BACKEND` defaults to `cnn`.

Every AP above is **clip-level**: one label for a 3 s clip. Echo never sees a
clip, it sees a stream, and it has to decide per frame. Calibrating the frame
head to a 2%% fire rate on clean speech -- roughly the most it can nag and stay
usable -- gives an operating point far harsher than the clip table implies.
The fit has to be done in logit space: peak logits reach 13-28, so fitting in
probability space returns exactly 1.0 under float32 sigmoid saturation.

%s

**The Block head recalls %s there.** Blocks are the dysfluency this product
most wants to catch, and at the interruption budget it actually runs at, it
catches about one in seven. That -- not the corpus, and not the encoder -- is
where the acoustic channel really stands.

""" % ("\n".join(fr), _num(_get(cal, "Block", "recall_at_threshold"))))
    return "\n".join(out)


def _aphasia_section(d, pause, n):
    if not isinstance(d, dict) or d.get("status") != "OK":
        return ""
    rows = ["| arm | recall | false alarm | partner fires |", "|---|---|---|---|"]
    n_pos = PENDING
    for name, m in d.get("summary", {}).items():
        n_pos = m.get("n_word_search", PENDING)
        rows.append("| %s | **%s** | %s | %s |"
                    % (name, _num(m.get("recall")), _num(m.get("false_alarm_rate")),
                       _num(m.get("partner_fire_rate"))))
    sweep = ""
    if d.get("refractory_sweep"):
        sw = ["| arm | min_gap_ms | recall | false alarm | fires/min |",
              "|---|---|---|---|---|"]
        for r in d["refractory_sweep"]:
            sw.append("| %s | %s | %s | %s | %s |"
                      % (r["arm"], r["min_gap_ms"], _num(r.get("recall")),
                         _num(r.get("false_alarm_rate")), r.get("fires_per_min", PENDING)))
        sweep = ("\n### Interruption rate\n\nAn aid that fires constantly is "
                 "unusable however good its recall, so the refractory is reported as "
                 "a curve rather than as one chosen point.\n\n" + "\n".join(sw) + "\n")
    pause_md = ""
    if isinstance(pause, dict) and pause.get("status") == "OK":
        g = pause["gap_distribution_ms"]
        at = lambda k, t=1300: next((r[k] for r in pause["sweep"]
                                     if r["threshold_ms"] == t), None)
        pause_md = """
### How long is a word-search pause in aphasia?

| internal silence | n | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|
| inside word-search utterances | %s | %s | %s | %s | %s |
| inside fluent utterances | %s | %s | %s | %s | %s |

The distributions **overlap heavily**. The shipped `STALL_PAUSE_MS=1300`
separates them at only %s sensitivity (FPR %s). Pause length alone is a weak
signal in aphasia -- which is an argument for the other channels, not for
tuning this constant. Source: `eval/fit_aphasia_pause.py`.
""" % (g["word_search"]["n"], g["word_search"]["p50"], g["word_search"]["p75"],
       g["word_search"]["p90"], g["word_search"]["p95"],
       g["fluent"]["n"], g["fluent"]["p50"], g["fluent"]["p75"],
       g["fluent"]["p90"], g["fluent"]["p95"],
       _num(at("sensitivity")), _num(at("false_positive_rate")))
    caveats = "\n".join("- " + c for c in d.get("caveats", []))
    return """## %d. Real aphasic speech (APROCSA)

Every other detection number in this report comes from stuttered or fluent
**podcast** speech. Neither is aphasia, and aphasia is the point: stuttering is
a motor-speech disorder where the word is known and will not come out; aphasia
is a language disorder where the word is not retrievable. They share surface
evidence, which is why a stutter-trained detector transfers at all -- but
"transfers" is a hypothesis until it is measured.

Ground truth is clinician CHAT coding, media-aligned: retracings, abandoned
utterances, phonological fragments, filled pauses, paraphasias. See
`scripts/aprocsa_chat.py` for exactly which codes count and why.

%s

n_word_search=%s, %.0f s region per participant, %s speakers.
%s%s
**This comparison understates the change.** Both arms get the new VAD-gated
silence ticks and audio-derived turn boundaries; the shipped browser path had
neither, because its pause trigger measured gaps between browser transcript
events. Only transcript content and the acoustic model are isolated here.

%s

%s

""" % (n, "\n".join(rows), n_pos, d.get("region_s") or 0,
       len(d.get("details", {})), sweep, pause_md, d.get("proxy_note", ""), caveats)


def _wer_section(d, n):
    """The number that bounds everything downstream."""
    if not isinstance(d, dict) or d.get("status") != "OK":
        return ""
    rows = ["| ASR configuration | WER on aphasic speech |", "|---|---|"]
    for k, v in sorted((d.get("results") or {}).items(), key=lambda kv: kv[1]["wer"]):
        rows.append("| %s | **%s** |" % (k, _num(v.get("wer"))))
    return """## %d. ASR accuracy on aphasic speech

Echo's recognizer was chosen on dysfluency preservation -- does "[UM]" survive
into the transcript -- and on latency. What was never measured is whether the
words AROUND the dysfluency are right, on the speech this product is for.

%s

Reference: %s. Corpus: %s.

Fillers are stripped from BOTH sides. They are measured separately in the
verbatim-ASR section, and leaving them in would let a model score better here
by transcribing hesitations rather than by getting the content words a
prediction has to be built from.

**This bounds every downstream number.** At two words in five wrong, the
predictor receives fragments like "And then [noise] [noise] and cut off [UM]
Cut off And" for a speaker reaching for *christmas*. No prompt, trigger or
acoustic model recovers from that. Source: `eval/bench_asr_aphasia_wer.py`.

""" % (n, "\n".join(rows), d.get("reference", ""), d.get("corpus", ""))


def _markers_section(d, n):
    if not isinstance(d, dict) or d.get("status") != "OK":
        return ""
    rows = ["| arm | recall | timer control | lift | precision | fires/min |",
            "|---|---|---|---|---|---|"]
    for a in d.get("arms", []):
        rows.append("| %s | %s | %s | **%s** | %s | %s |"
                    % (a.get("arm"), _num(a.get("recall")), _num(a.get("timer_recall")),
                       _num(a.get("lift_recall")), _num(a.get("precision")),
                       a.get("fires_per_min")))
    return """## %d. Detection against a metronome

Utterance-level scoring credits a fire anywhere inside a clinician-coded
utterance plus a margin -- a median window of 5.7 s. A detector firing on a
timer lands inside that routinely, so the metric cannot tell detection from
regular interruption. `eval/align_aprocsa.py` gives most CHAT markers real
timestamps (2,344 of 2,549), and this scores against those instants.

The control is the point: a TIMER arm fires at fixed intervals at the same rate
using no audio at all. Recall above it is the only evidence of detection.

%s

Window: -%s/+%s ms, asymmetric because a detector cannot fire before its
evidence exists -- the pause trigger fires 1300 ms after silence onset. The
control is scored through the identical window.

**Nothing clears the metronome by much**, and the reason is the finding:
clinician-coded markers occur every 1.2-2.3 s in aphasic speech. "Is this
person word-searching right now" is nearly always yes, so no timing metric on
this corpus separates a detector from a clock. Source: `eval/score_markers.py`.

""" % (n, "\n".join(rows), d.get("window_pre_ms", "?"), d.get("window_post_ms", "?"))


def _verbatim_vs_intended(d):
    """Does keeping the dysfluency in the fragment help the predictor?

    Earlier reports asserted "verbatim beats intended with no counterexamples".
    That was true of one run and is not a property of the method, so it is
    counted here from the paired table rather than restated.
    """
    pairs = [x for x in (d.get("paired_differences") or [])
             if {x.get("a"), x.get("b")} in ({"verbatim", "intended"},
                                             {"verbatim+ctx", "intended+ctx"})]
    if not pairs:
        return ""
    lines = ["The verbatim-vs-intended comparison, counted per event:", "",
             "| metric | arms | verbatim only | intended only | both | p |",
             "|---|---|---|---|---|---|"]
    wins = losses = 0
    for x in pairs:
        v, i = x.get("only_a", 0), x.get("only_b", 0)
        wins += v
        losses += i
        lines.append("| %s | %s vs %s | %s | %s | %s | %s |"
                     % (x.get("metric"), x.get("a"), x.get("b"), v, i,
                        x.get("both"), x.get("mcnemar_p_two_sided")))
    verdict = ("Verbatim leads on every comparison and there are no "
               "counterexamples." if losses == 0 else
               "Verbatim leads on aggregate (%d events to %d) but there are "
               "counterexamples, and no comparison approaches significance. "
               "This is consistent with the verbatim thesis and is not "
               "evidence for it." % (wins, losses))
    lines += ["", verdict]
    return "\n".join(lines)


def _aphasia_prediction_section(d, n, pre=None):
    ablation = _prediction_asr_ablation(pre, d)
    if not isinstance(d, dict):
        return ""
    # Metrics live under pooled["all"][arm]; d["arms"] only declares each
    # arm's configuration.
    arms = ((d.get("pooled") or {}).get("all")) or {}
    if not isinstance(arms, dict) or not arms:
        return ""
    rows = ["| arm | n | top-1 | top-3 | top-3 (span) |", "|---|---|---|---|---|"]
    for name, v in arms.items():
        if not isinstance(v, dict):
            continue
        rows.append("| %s | %s | %s | **%s** | %s |"
                    % (name, v.get("n", "--"), _num(v.get("top1")),
                       _num(v.get("top3")), _num(v.get("top3_span"))))
    return """## %d. Word prediction on real aphasic speech

The metric the product exists for. A CHAT retracing records both that a word
search happened and what the speaker was reaching for -- `spring [//]
Christmas` -- so each one is a free (fragment, intended word) pair. The
fragment is what Echo actually had at that instant, replayed from the cached
ASR stream with partner speech excluded.

%s

n = %s scorable events across six speakers.

**Echo is at the frequency-baseline floor on the strict metric**, and the
paired table shows the two hit disjoint events -- uncorrelated with a
constant-answer baseline rather than tied with it.

%s

%s
Source: `eval/run_aphasia_prediction.py`.

""" % (n, "\n".join(rows), d.get("n_scorable", d.get("n", "?")),
       _verbatim_vs_intended(d), ablation)


def _prediction_asr_ablation(pre, post):
    """Did improving the ASR move prediction? Held for a long time that it
    would; it is an ablation, so it can be run."""
    if not (isinstance(pre, dict) and isinstance(post, dict)):
        return ("The cause was assumed to be upstream ASR accuracy. That has "
                "not yet been tested here.")
    arms = ["verbatim+ctx", "context_only", "freq_corpus", "freq_english"]
    rows = ["| arm | top-3 hits before | after | top-3 span before | after |",
            "|---|---|---|---|---|"]
    for a in arms:
        b = _get(pre, "pooled", "all", a, default=None)
        c = _get(post, "pooled", "all", a, default=None)
        if not isinstance(b, dict) or not isinstance(c, dict):
            continue
        rows.append("| %s | %s | %s | %s | %s |"
                    % (a, b.get("top3_hits"), c.get("top3_hits"),
                       b.get("top3_span_hits"), c.get("top3_span_hits")))
    return """
### Was ASR accuracy the binding constraint? No.

This section previously closed by attributing the floor to ASR error
upstream. That is testable: the streaming policy changed between these two
runs and nothing else did, cutting WER on the same audio from **0.402 to
0.375**. Same 51 events, same model, same prompts.

%s

**Strict top-3 does not move at all** -- 3 hits before, 3 after, the same
count a corpus-frequency baseline gets. The looser span metric goes 11 to 9,
i.e. down, which at n=51 is noise in the other direction.

One number does improve: verbatim+ctx now beats context_only on top-3 span at
McNemar p=0.031, against 0.070 before. That is not the fragment arm getting
better -- it is context_only getting *worse* (5 hits to 3). Reading it as
progress would be reading a control's regression as a treatment effect.

So the diagnosis stated across the last two versions -- that everything
downstream is bounded by ASR accuracy -- is **not supported**. A 6.8%%
relative WER reduction bought exactly nothing. Either the remaining error
rate is still far above whatever threshold would matter, or word identity at
a word-search moment is not recoverable from the fragment at all. The
committed events file argues for the second: where the speaker was reaching
for *christmas*, the fragment Echo held reads "Alright, It was". That is not
a transcript that a better decoder rescues.
""" % ("\n".join(rows))

def main() -> int:
    stall = _load(STALL)
    latency = _load(LATENCY)
    prolong = _load(PROLONG)
    train = _load(TRAIN_METRICS)

    # Optional sections 5+, numbered sequentially only over the ones that
    # actually have results (each renders '' when its results JSON is absent).
    pred_raw, dual_raw, noise_raw = _load(PRED), _load(DUAL), _load(NOISE)
    section_no = 5
    pred_section = _prediction_section(pred_raw, _load(PRED_ABL), section_no)
    longctx_section = _longctx_section(_load(LONGCTX), _load(LONGCTX_ENTITY)) if pred_section else ""
    section_no += 1 if pred_section else 0
    dual_section = _dual_channel_section(dual_raw, section_no)
    section_no += 1 if dual_section else 0
    noise_section = _noise_stress_section(noise_raw, section_no)
    section_no += 1 if noise_section else 0
    longconv_section = _longconv_section(_load(LONGCONV), section_no,
                                        _load(LONGCONV_LOCAL))
    section_no += 1 if longconv_section else 0
    spkgate_section = _speaker_gate_section(_load(SPKGATE), section_no)
    section_no += 1 if spkgate_section else 0
    asr_section = _asr_section(_load(ASR_BENCH), section_no)
    section_no += 1 if asr_section else 0
    stutter_section = _stutter_section(_load(STUTTER_METRICS), section_no)
    section_no += 1 if stutter_section else 0
    ssl_section = _ssl_section(_load(SSL_METRICS), _load(SSL_MATRIX),
                               _load(SSL_HOSTLEAK), section_no)
    section_no += 1 if ssl_section else 0
    aphasia_section = _aphasia_section(_load(APHASIA), _load(APHASIA_PAUSE), section_no)
    section_no += 1 if aphasia_section else 0
    wer_section = _wer_section(_load(APH_WER), section_no)
    section_no += 1 if wer_section else 0
    markers_section = _markers_section(_load(MARKERS), section_no)
    section_no += 1 if markers_section else 0
    aphpred_section = _aphasia_prediction_section(
        _load(PREDICT_APH), section_no, _load(PREDICT_APH_PRE))
    section_no += 1 if aphpred_section else 0
    lim_no = section_no

    # Acoustic clip metrics: prefer the eval-run numbers, fall back to the
    # training script's test metrics (same evaluate(), same shape).
    acoustic = _get(stall, "acoustic_clip_metrics", default=None)
    acoustic_src = "eval/run_stall_eval.py"
    if not isinstance(acoustic, dict):
        acoustic = train
        acoustic_src = "models/fillernet_metrics.json (scripts/train_filler.py)"
    if not isinstance(acoustic, dict):
        acoustic, acoustic_src = None, "PENDING -- model not trained yet"

    counts = _get(stall, "test_split_counts", default=None)
    counts_line = (
        ", ".join(f"{k} n={v}" for k, v in counts.items())
        if isinstance(counts, dict) and counts
        else "PENDING -- test split not scanned yet (dataset may still be downloading)"
    )

    fb = _get(acoustic, "filler_binary", default=None) if acoustic else None
    base = _get(stall, "transcript_baseline", default=None)
    n_filler = _get(base, "n_filler_clips")
    chrome_recall = _get(base, "recall")
    verbatim_recall = _get(base, "verbatim_asr_ablation", "recall")

    # --- Table 1 rows ------------------------------------------------------
    t1 = [
        "| Detector | Condition | Precision | Recall | F1 |",
        "|---|---|---|---|---|",
        f"| Acoustic FillerNet (binary uh∪um) | PFSD test clips | "
        f"{_num(_get(fb, 'precision', default=None))} | "
        f"{_num(_get(fb, 'recall', default=None))} | "
        f"{_num(_get(fb, 'f1', default=None))} |",
        f"| Transcript-only filler trigger | live Chrome (fillers stripped by ASR) | "
        f"-- | {_num(chrome_recall)} | -- |",
        f"| Transcript-only filler trigger | verbatim-ASR ablation (token present) | "
        f"-- | {_num(verbatim_recall)} | -- |",
    ]
    per_class_rows = ["| Class | Precision | Recall | F1 | n |", "|---|---|---|---|---|"]
    if isinstance(acoustic, dict) and isinstance(acoustic.get("per_class"), dict):
        for cls, m in acoustic["per_class"].items():
            per_class_rows.append(
                f"| {cls} | {_num(m.get('precision'))} | {_num(m.get('recall'))} | "
                f"{_num(m.get('f1'))} | {m.get('n', PENDING)} |")
        acc_line = f"Overall 4-class accuracy: **{_num(acoustic.get('accuracy'))}**."
    else:
        per_class_rows.append(f"| {PENDING} | | | | |")
        acc_line = f"Overall 4-class accuracy: {PENDING}."

    # --- Table 2 rows ------------------------------------------------------
    pause_ms = _get(latency, "pause_baseline", "pause_ms")
    fil = _get(latency, "acoustic_filler", default=None)
    pro = _get(latency, "prolongation", default=None)

    def lat_cell(d) -> str:
        """Format a latency result dict as a Markdown table cell string.

        Task 3a fix: serving rows use 'n' (from _stats) not 'fired'/'runs'.
        Detect which keys are present and render accordingly.
        """
        if not isinstance(d, dict):
            return PENDING
        if "median_ms" not in d:
            return d.get("status", PENDING)
        fired = d.get("fired")
        runs = d.get("runs")
        n = d.get("n")
        if fired is not None and runs is not None:
            fire_str = f", {fired}/{runs} fired"
        elif n is not None:
            fire_str = f", n={n} runs"
        else:
            fire_str = ""
        return (f"{_num(d['median_ms'], '{:.0f}')} ms "
                f"(min {_num(d.get('min_ms'), '{:.0f}')}"
                f" / max {_num(d.get('max_ms'), '{:.0f}')}{fire_str})")

    # Task 3b: preamble_false_fires -- render in the acoustic-filler row.
    fil_pff = _get(latency, "acoustic_filler", "preamble_false_fires", default=None)
    fil_runs_n = _get(latency, "acoustic_filler", "runs", default=None)
    if isinstance(fil_pff, int) and isinstance(fil_runs_n, int):
        pff_note = (
            f" [{fil_pff} preamble false fires in {fil_runs_n} runs: "
            f"FillerNet fired on real-speech preamble before Um onset]"
        )
    else:
        pff_note = ""

    t2 = [
        "| Trigger | Detection latency from event onset | How measured |",
        "|---|---|---|",
        f"| Pause timeout (transcript baseline) | {_num(pause_ms, '{:.0f}')} ms | "
        f"by construction -- equals the configured threshold (STALL_PAUSE_MS) |",
        f"| Acoustic filler (FillerNet, {HOP_MS} ms hop) | {lat_cell(fil)} | "
        f"real Um clip at known onset after real-speech preamble, fed to "
        f"AcousticStream in 20 ms chunks; onset phase varied vs the hop grid"
        f"{pff_note} |",
        f"| Prolongation (rule-based) | {lat_cell(pro)} | "
        f"sustained vowel through ProlongationTracker; latency is "
        f"deterministic by construction (min_ms + one 50 ms frame), so "
        f"min = median = max -- reported as arithmetic, not statistics |",
    ]

    # Task 3c: 600 ms gate-miss disclosure (median read from the bench JSON).
    fil_median = _get(latency, "acoustic_filler", "median_ms", default=None)
    fil_median_str = _num(fil_median, "{:.0f}")
    ratio_str = (f"~{(pause_ms / fil_median):.1f}x"
                 if isinstance(fil_median, (int, float)) and fil_median
                 and isinstance(pause_ms, (int, float)) else "materially")
    gate_disclosure = (
        f"**Latency gate disclosure:** the plan set an acoustic filler "
        f"detection gate of <=600 ms median; the measured median is "
        f"{fil_median_str} ms, missing that gate. The dominant term is the "
        f"{HOP_MS} ms analysis hop plus the >=800 ms voiced gate at utterance "
        f"start; the channel is still {ratio_str} earlier than the "
        f"{_num(pause_ms, '{:.0f}')} ms pause baseline, which is the "
        f"comparison that matters for serving."
    )

    # --- Prolongation validation (rule-based, real PFSD audio) -------------
    # Task 3d: use new JSON keys; describe honest construction; add music +
    # stream-level numbers.
    pdet = _get(prolong, "detection", default=None)
    pff_speech = _get(prolong, "false_fires_speech", default=None)
    pff_music = _get(prolong, "false_fires_music", default=None)
    pff_stream = _get(prolong, "stream_falsefire", default=None)

    construction = (
        _get(pdet, "construction", default=None)
        if isinstance(pdet, dict) else None
    )
    construction_str = (
        f" (construction: {construction})"
        if isinstance(construction, str) else ""
    )

    det_line = (
        f"fires on **{pdet.get('fired')}/{pdet.get('scored')}** sustained real "
        f"vowels (rate {_num(pdet.get('detection_rate'))}){construction_str}"
        if isinstance(pdet, dict) and "fired" in pdet else PENDING
    )
    ff_speech_line = (
        f"**{pff_speech.get('false_fires')}** false fires in "
        f"{_num(pff_speech.get('running_speech_seconds'), '{:.0f}')} s of real running speech "
        f"({_num(pff_speech.get('false_fire_rate_per_min'), '{:.3f}')}/min)"
        if isinstance(pff_speech, dict) and "false_fires" in pff_speech else PENDING
    )
    ff_music_line = (
        f"**{pff_music.get('false_fires')}** false fires in "
        f"{_num(pff_music.get('music_seconds'), '{:.0f}')} s of music "
        f"({_num(pff_music.get('false_fire_rate_per_min'), '{:.3f}')}/min) "
        f"[near-static-envelope hazard; VAD gate limits live-path exposure]"
        if isinstance(pff_music, dict) and "false_fires" in pff_music else PENDING
    )
    stream_line = (
        f"**{pff_stream.get('filler_events')}** filler events in "
        f"{_num(pff_stream.get('speech_seconds'), '{:.0f}')} s of fluent speech "
        f"({_num(pff_stream.get('filler_events_per_min'), '{:.3f}')}/min)"
        if isinstance(pff_stream, dict) and "filler_events" in pff_stream
        and pff_stream.get("status") == "ok"
        else (
            pff_stream.get("status", PENDING)
            if isinstance(pff_stream, dict) else PENDING
        )
    )

    prolong_block = (
        f"The prolongation rule is rule-based (not part of the FillerNet "
        f"classification report) and is validated separately on real PFSD audio "
        f"(`eval/run_prolongation_eval.py`).\n\n"
        f"**Detection** (tracker-level): {det_line}. "
        f"PFSD has no labelled prolongations, so detection is synthetic and "
        f"we report the CONSERVATIVE construction: voiced 50 ms frames of "
        f"real Uh/Um clips palindrome-cycled to >=1.5 s, so every transition "
        f"is between frames adjacent in the real clip (natural jitter, "
        f"cos-sim ~0.95-0.98; no sim=1.0 tiling tautology, no artificial "
        f"wrap discontinuity). Read this as a LOWER BOUND: conversational "
        f"um/uh clips contain internal phone transitions (an 'um' closes "
        f"into the m), which a deliberately held vowel does not -- the "
        f"tracker is designed for the latter. Ground truth requires the "
        f"self-recorded held-vowel set (`eval/record_protocol.md`), still "
        f"unrecorded.\n\n"
        f"**False fires -- running speech (Words)**: {ff_speech_line}. "
        f"Concatenated PFSD 'Words' (lexical speech) clips; running speech "
        f"changes phones every ~100-150 ms, breaking the similarity streak.\n\n"
        f"**False fires -- music**: {ff_music_line}. "
        f"Concatenated PFSD 'Music' clips; music has near-static spectral "
        f"envelopes (the known false-fire hazard for the mel-envelope cosine "
        f"rule). In the live path the VAD gate (min_voiced_ms=800) prevents "
        f"music segments from reaching the prolongation tracker.\n\n"
        f"**Stream-level FillerNet false alarm** (full AcousticStream path, "
        f"VAD + FillerNet + confidence gate, 20 ms chunks): {stream_line}. "
        f"Source: PFSD 'Words' clips. Caveat: concatenating 1 s clips from "
        f"many speakers inserts a segment boundary every second, which "
        f"likely inflates the rate vs one continuous speaker -- treat as a "
        f"conservative upper bound. Downstream, the fused StallDetector's "
        f"confidence gate, debounce and re-arm windowing further limit how "
        f"many acoustic events become visible suggestions."
    )

    # --- Table 3 rows ------------------------------------------------------
    srv = _get(latency, "serving", default=None)
    pre = _get(srv, "prefetch", default=None)
    liv = _get(srv, "live_simulated", default=None)
    ext = _get(srv, "live_gemini_external_ms", default=None)
    ext_str = (f"{ext[0]}-{ext[1]} ms" if isinstance(ext, list) and len(ext) == 2
               else PENDING)
    cite = _get(srv, "live_gemini_citation", default="scripts/e2e_live.py")

    t3 = [
        "| Serving path | Stall -> candidates latency | Source |",
        "|---|---|---|",
        f"| Live Gemini round-trip | {ext_str} | measured in live e2e runs "
        f"(external constant; {cite}) |",
        f"| Live path, simulated 1500 ms predictor delay | "
        f"{lat_cell(liv)} | EchoPipeline mechanism check -- the live path "
        f"waits the full round-trip |",
        f"| Speculative prefetch (cache hit) | {lat_cell(pre)} | "
        f"EchoPipeline + MockPredictor, Prediction.latency_ms, served='prefetch' |",
    ]

    dual_n_fillers = _get(dual_raw, "construction", "n_fillers", default="?")
    dual_limitation = (
        f"- **Dual-channel ablation is one synthetic stream, not a distribution.** "
        f"A single seeded run (n={dual_n_fillers} embedded fillers) at one mix "
        f"ratio and one silence-gap length; each cycle's detection is scored "
        f"only inside its own tight attribution window (see the section above), "
        f"so a fire during a LATER cycle's fluent speech can never be credited "
        f"to an earlier filler -- but such fires are real and disclosed "
        f"separately as spurious acoustic fires, not hidden.\n" if dual_section else ""
    )
    noise_limitation = (
        "- **Noise-stress measures the classifier in isolation.** The live "
        "pipeline's VAD gate, voiced-time gate, and refractory may mitigate "
        "noise-induced misses in practice; that mitigation is not measured, "
        "only named as an open question.\n" if noise_section else ""
    )

    md = f"""# Echo -- Evaluation Report

*Generated {date.today().isoformat()} by `eval/make_report.py`. Cells marked
{PENDING} mean the corresponding artifact (dataset split, model checkpoint, or
bench result) did not exist at generation time; re-run the eval scripts and
regenerate.*

## 1. Methodology

**Dataset.** PodcastFillers (PFSD) 1.0 s clips, 16 kHz mono PCM16, official
splits (`train`/`validation`/`test`/`extra`). PFSD's consolidated vocabulary
is mapped to Echo's four classes (uh, um, speech, other) by
`backend.acoustic.model.LABEL_MAP`. All clip-level numbers below are on the
**official test split only**; counts at evaluation time:
{counts_line}.

**Acoustic filler metrics.** `eval/run_stall_eval.py` imports the *same*
`evaluate()` used by `scripts/train_filler.py`, so the metric definitions
(binary filler = uh∪um, per-class P/R/F1) are identical by construction to
`models/fillernet_metrics.json`. Echo's **primary filler-detection metric is
clip-level uh/um classification F1 on the official PFSD test split**, and the
>= 0.75 gate is defined on it. We deliberately do *not* report an
event-detection-with-tolerance F1: the PFSD test split is pre-segmented 1 s
clips, for which clip-level classification is the natural, standard benchmark
(and is what the published PFSD baselines report).

**The Chrome-condition baseline -- definition and honest framing.** Echo's
thesis is that consumer ASR (Chrome Web Speech and similar) strips filled
pauses from transcripts, so a transcript-only detector's *filler trigger*
cannot fire on live audio. We measure this at the **trigger level**: for each
filler clip in the test split we construct the transcript a filler-stripping
ASR would emit for it (no filler token), feed it through the real
`backend.stall_detector.StallDetector` after a content-word preamble (the
trigger's most favourable precondition), and count filler-trigger fires.
The result is 0 % recall **by construction** -- running it against the real
detector pins the claim to the shipped code and to a real n
(n = {n_filler} filler clips). Two honesty notes:
(1) a verbatim-ASR ablation (filler token present) is reported alongside to
show the detector logic itself fires given the token -- the bottleneck is the
ASR, not the detector; (2) the transcript-only system still catches the stall
*eventually* via the {_num(pause_ms, '{:.0f}')} ms pause timeout -- the acoustic channel's
contribution is firing earlier and on direct filler evidence, not detecting
otherwise-undetectable stalls.

**Latency benches.** `eval/run_latency_bench.py`, no network. Acoustic filler
latency uses real test clips composed into a synthetic mic stream with
sample-accurate onsets (onset phase varied against the {HOP_MS} ms classifier
hop using 24 evenly spaced phase offsets across 0..{HOP_MS - 1} ms to cover the
full hop period); prolongation latency tiles the loudest 50 ms frame of a real Uh clip
into a sustained vowel (tracker-level; acceptable for latency measurement since
we just need it to fire, distinct from the detection-rate eval which uses the
honest looped-frames construction). The live-LLM round-trip is never
re-measured by the bench; it is cited from live e2e runs.

## 2. Table 1 -- Filler detection (PFSD test split)

{chr(10).join(t1)}

The Chrome-condition recall of 0 is by construction (the documented
filler-stripping behaviour of consumer ASR), reported as
**transcript-only filler recall (Chrome condition)**.

Per-class acoustic metrics (source: {acoustic_src}):

{chr(10).join(per_class_rows)}

{acc_line}

## 3. Table 2 -- Detection latency

{chr(10).join(t2)}

The acoustic filler number is a conservative upper bound: the onset is the
start of the 1 s Um clip, while the voiced filler may begin some ms into it.
Prolongation is tracker-level; the live stream adds <= 50 ms frame buffering
and a >= 800 ms voiced gate at utterance start.

{gate_disclosure}

{prolong_block}

## 4. Table 3 -- End-to-end serving (stall -> candidates on screen)

{chr(10).join(t3)}

Prefetch shadow-predicts during fluent speech and serves a cached prediction
the instant a stall fires; the cache is only used when the spoken fragment
has drifted <= 2 content words past the cached one (see `backend/pipeline.py`).

{pred_section}{longctx_section}{dual_section}{noise_section}{longconv_section}{spkgate_section}{asr_section}{stutter_section}{ssl_section}{aphasia_section}{wer_section}{markers_section}{aphpred_section}## {lim_no}. Limitations

- **No contact with the target population yet.** No person with aphasia and
  no speech-language pathologist has used or reviewed this system, formally
  or informally. In particular, the central interaction assumption -- that a
  ranked word list plus a spoken cue mid-sentence RELIEVES word-finding
  effort rather than adding cognitive load during exactly the moment of
  least spare capacity -- is untested. Every number in this report measures
  the machine, not the interaction; an SLP-guided study is the necessary
  next step before any claim about helping people.
- **Domain shift.** FillerNet is trained and evaluated on podcast speech
  (PFSD). Aphasic word-finding speech differs in rate, prosody, and filler
  realization; podcast numbers are an optimistic proxy until the
  self-recorded set is collected.
- **Self-recorded eval set pending.** The 40-utterance two-speaker
  aphasia-style set (`eval/record_protocol.md`) has not been recorded yet;
  `run_stall_eval.py --wav-dir` is ready for it. Until then there is no
  utterance-level end-to-end accuracy number.
- **n counts.** All clip-level metrics are only as complete as the test split
  on disk at eval time (counts above); partial downloads shrink n, they do
  not bias the construction-level baseline result.
- **Trigger-level baseline.** The 0 % figure is the recall of one trigger
  under one (documented, common) ASR condition -- not "the baseline never
  detects stalls". The pause timeout remains as the baseline's catch-all at
  {_num(pause_ms, '{:.0f}')} ms.
- **Serving simulation.** The "live path, simulated" row injects a constant
  1500 ms delay; the real live range is the cited external measurement.
- **Acoustic latency gate miss.** The measured acoustic filler median
  exceeds the plan's original <=600 ms gate; see the disclosure under
  Table 2.
- **Prolongation detection is synthetic and read as a lower bound.** The
  palindrome-looped construction carries real frame-to-frame jitter but is
  built from conversational um/uh clips, which contain internal phone
  transitions a deliberately held vowel does not. The self-recorded
  held-vowel set (`eval/record_protocol.md`) is the ground-truth path.
- **Stream FillerNet false-alarm is a conservative upper bound.** The
  concatenated-clip stream inserts a speaker/segment boundary every second,
  inflating the rate vs one continuous speaker; downstream StallDetector
  gating further limits visible suggestions.
{dual_limitation}{noise_limitation}"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(md, encoding="utf-8")
    missing = [str(p.relative_to(ROOT)) for p in (STALL, LATENCY, TRAIN_METRICS)
               if not p.exists()]
    print(f"wrote {OUT}")
    if missing:
        print(f"PENDING inputs (report generated with placeholders): {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
