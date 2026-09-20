# Echo -- Evaluation Report

*Generated 2026-09-20 by `eval/make_report.py`. Cells marked
PENDING mean the corresponding artifact (dataset split, model checkpoint, or
bench result) did not exist at generation time; re-run the eval scripts and
regenerate.*

## 1. Methodology

**Dataset.** PodcastFillers (PFSD) 1.0 s clips, 16 kHz mono PCM16, official
splits (`train`/`validation`/`test`/`extra`). PFSD's consolidated vocabulary
is mapped to Echo's four classes (uh, um, speech, other) by
`backend.acoustic.model.LABEL_MAP`. All clip-level numbers below are on the
**official test split only**; counts at evaluation time:
Breath n=732, Laughter n=579, Music n=822, Uh n=2598, Um n=2446, Words n=2292.

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
(n = 5044 filler clips). Two honesty notes:
(1) a verbatim-ASR ablation (filler token present) is reported alongside to
show the detector logic itself fires given the token -- the bottleneck is the
ASR, not the detector; (2) the transcript-only system still catches the stall
*eventually* via the 1300 ms pause timeout -- the acoustic channel's
contribution is firing earlier and on direct filler evidence, not detecting
otherwise-undetectable stalls.

**Latency benches.** `eval/run_latency_bench.py`, no network. Acoustic filler
latency uses real test clips composed into a synthetic mic stream with
sample-accurate onsets (onset phase varied against the 125 ms classifier
hop using 24 evenly spaced phase offsets across 0..124 ms to cover the
full hop period); prolongation latency tiles the loudest 50 ms frame of a real Uh clip
into a sustained vowel (tracker-level; acceptable for latency measurement since
we just need it to fire, distinct from the detection-rate eval which uses the
honest looped-frames construction). The live-LLM round-trip is never
re-measured by the bench; it is cited from live e2e runs.

## 2. Table 1 -- Filler detection (PFSD test split)

| Detector | Condition | Precision | Recall | F1 |
|---|---|---|---|---|
| Acoustic FillerNet (binary uh∪um) | PFSD test clips | 0.949 | 0.918 | 0.933 |
| Transcript-only filler trigger | live Chrome (fillers stripped by ASR) | -- | 0.000 | -- |
| Transcript-only filler trigger | verbatim-ASR ablation (token present) | -- | 1.000 | -- |

The Chrome-condition recall of 0 is by construction (the documented
filler-stripping behaviour of consumer ASR), reported as
**transcript-only filler recall (Chrome condition)**.

Per-class acoustic metrics (source: eval/run_stall_eval.py):

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| uh | 0.876 | 0.783 | 0.827 | 2598 |
| um | 0.883 | 0.923 | 0.902 | 2446 |
| speech | 0.802 | 0.774 | 0.788 | 2292 |
| other | 0.844 | 0.940 | 0.889 | 2133 |

Overall 4-class accuracy: **0.852**.

## 3. Table 2 -- Detection latency

| Trigger | Detection latency from event onset | How measured |
|---|---|---|
| Pause timeout (transcript baseline) | 1300 ms | by construction -- equals the configured threshold (STALL_PAUSE_MS) |
| Acoustic filler (FillerNet, 125 ms hop) | 755 ms (min 41 / max 1203, 24/24 fired) | real Um clip at known onset after real-speech preamble, fed to AcousticStream in 20 ms chunks; onset phase varied vs the hop grid [3 preamble false fires in 24 runs: FillerNet fired on real-speech preamble before Um onset] |
| Prolongation (rule-based) | 650 ms (min 650 / max 650, 24/24 fired) | sustained vowel through ProlongationTracker; latency is deterministic by construction (min_ms + one 50 ms frame), so min = median = max -- reported as arithmetic, not statistics |

The acoustic filler number is a conservative upper bound: the onset is the
start of the 1 s Um clip, while the voiced filler may begin some ms into it.
Prolongation is tracker-level; the live stream adds <= 50 ms frame buffering
and a >= 800 ms voiced gate at utterance start.

**Latency gate disclosure:** the plan set an acoustic filler detection gate of <=600 ms median; the measured median is 755 ms, missing that gate. The dominant term is the 125 ms analysis hop plus the >=800 ms voiced gate at utterance start; the channel is still ~1.7x earlier than the 1300 ms pause baseline, which is the comparison that matters for serving.

The prolongation rule is rule-based (not part of the FillerNet classification report) and is validated separately on real PFSD audio (`eval/run_prolongation_eval.py`).

**Detection** (tracker-level): fires on **5/40** sustained real vowels (rate 0.125) (construction: palindrome-looped real vowel frames -- natural jitter, no wrap discontinuity). PFSD has no labelled prolongations, so detection is synthetic and we report the CONSERVATIVE construction: voiced 50 ms frames of real Uh/Um clips palindrome-cycled to >=1.5 s, so every transition is between frames adjacent in the real clip (natural jitter, cos-sim ~0.95-0.98; no sim=1.0 tiling tautology, no artificial wrap discontinuity). Read this as a LOWER BOUND: conversational um/uh clips contain internal phone transitions (an 'um' closes into the m), which a deliberately held vowel does not -- the tracker is designed for the latter. Ground truth requires the self-recorded held-vowel set (`eval/record_protocol.md`), still unrecorded.

**False fires -- running speech (Words)**: **0** false fires in 120 s of real running speech (0.000/min). Concatenated PFSD 'Words' (lexical speech) clips; running speech changes phones every ~100-150 ms, breaking the similarity streak.

**False fires -- music**: **11** false fires in 120 s of music (5.500/min) [near-static-envelope hazard; VAD gate limits live-path exposure]. Concatenated PFSD 'Music' clips; music has near-static spectral envelopes (the known false-fire hazard for the mel-envelope cosine rule). In the live path the VAD gate (min_voiced_ms=800) prevents music segments from reaching the prolongation tracker.

**Stream-level FillerNet false alarm** (full AcousticStream path, VAD + FillerNet + confidence gate, 20 ms chunks): **29** filler events in 300 s of fluent speech (5.800/min). Source: PFSD 'Words' clips. Caveat: concatenating 1 s clips from many speakers inserts a segment boundary every second, which likely inflates the rate vs one continuous speaker -- treat as a conservative upper bound. Downstream, the fused StallDetector's confidence gate, debounce and re-arm windowing further limit how many acoustic events become visible suggestions.

## 4. Table 3 -- End-to-end serving (stall -> candidates on screen)

| Serving path | Stall -> candidates latency | Source |
|---|---|---|
| Live Gemini round-trip | 1250-1800 ms | measured in live e2e runs (external constant; scripts/e2e_live.py (live e2e measurement of Gemini stall-to-word round-trip)) |
| Live path, simulated 1500 ms predictor delay | 1500 ms (min 1500 / max 1501, n=5 runs) | EchoPipeline mechanism check -- the live path waits the full round-trip |
| Speculative prefetch (cache hit) | 0 ms (min 0 / max 0, n=20 runs) | EchoPipeline + MockPredictor, Prediction.latency_ms, served='prefetch' |

Prefetch shadow-predicts during fluent speech and serves a cached prediction
the instant a stall fires; the cache is only used when the spoken fragment
has drifted <= 2 content words past the cached one (see `backend/pipeline.py`).

## 5. Prediction accuracy (frozen circumlocution set)

Run: provider **gemini** / model **gemini-3.5-flash**,
2026-07-07T16:36:51+00:00 (`eval/run_prediction_eval.py`, dataset sha256
`9fd89ea1d59b...`).

| Split | Top-1 | Top-3 | Top-1 (context ablated) | Top-1 delta |
|---|---|---|---|---|
| Overall | 58/60 (96.7%) | 59/60 (98.3%) | 40/60 (66.7%) | -30.0 pts |
| concrete | 20/20 (100.0%) | 20/20 (100.0%) | 20/20 (100.0%) | +0.0 pts |
| proper_context | 19/20 (95.0%) | 19/20 (95.0%) | 2/20 (10.0%) | -85.0 pts |
| abstract_verb | 19/20 (95.0%) | 20/20 (100.0%) | 18/20 (90.0%) | -5.0 pts |

Mean predictor confidence (top-ranked candidate) when top-1 correct: **0.992** (n=58); when wrong: **0.950** (n=1). The context-ablation column re-runs the identical items with the conversation context removed (`--ablate-context`); the collapse on `proper_context` items is the designed evidence that conversation context earns its place in the prompt.

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
beyond that list is only the normalization rule: lowercase; strip one trailing "'s"; strip punctuation; collapse whitespace; strip one trailing plural "s". Nothing fuzzier
(no embeddings, no LLM judging) is applied. Freeze protocol: The dataset eval/data/prediction_eval_set.jsonl is frozen at its first commit. After the first live run, items may never be edited, added, or removed in response to results. One headline live run per condition (full and context-ablated). Any future change to the set requires a new versioned filename and both old and new results files must be kept.

**Prior art.** Purohit et al. 2023 (CSCW '23 Companion, "ChatGPT in
Healthcare: Exploring AI Chatbot for Spontaneous Word Retrieval in Aphasia")
is the offline precedent for this measurement: ChatGPT (GPT-3.5) retrieved the
intended word in 11/12 AphasiaBank circumlocution instances (91.67%). Their
n=12 items come from real aphasic speech transcripts with manual output
tagging; our set is larger (n=60) and hand-authored with a mechanical
acceptance rule, so the two numbers are methodologically not directly
comparable -- theirs establishes feasibility, ours measures Echo's shipped
prompt/provider pipeline.

### Context-window extension (entity memory)

Every row above uses the frozen set's short (1-2 turn) contexts, well inside
`context_turns` -- it cannot exercise what happens once a name has scrolled
OUT of the window. This subsection uses a separate, versioned long-context
set (`eval/data/prediction_eval_longctx_v1.jsonl`, 20 items, category
`proper_context`) built for exactly that: each item names its target entity
ONCE in the first 2-4 of 10-16 context turns, then moves on to unrelated
small talk, so by the time the fragment stalls on it the mention is outside
the live system's `context_turns` window (**6**,
matching `backend.config.Settings.context_turns`). A matched entity-memory run (provider **gemini** / model **gemini-3.5-flash**, 2026-07-08T06:15:50+00:00) is shown alongside it.

Run: provider **gemini** / model **gemini-3.5-flash**,
2026-07-08T06:15:03+00:00 (`eval/run_prediction_eval.py --dataset
eval/data/prediction_eval_longctx_v1.jsonl`, dataset sha256
`2c8bd5d5193e...`).

| Condition | Top-1 | Top-3 |
|---|---|---|
| Baseline (windowed context, no injection) | 0/20 (0.0%) | 0/20 (0.0%) |
| Entity memory ON (out-of-window entities injected) | 19/20 (95.0%) | 19/20 (95.0%) |

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

## 6. Dual-channel ablation (system-level, `eval/run_dual_channel_ablation.py`)

Every other latency number in this report is component-level (one isolated
clip through one detector). This section asks the SYSTEM-level question: in a
synthetic multi-minute conversation with embedded filler stalls, does the
fused acoustic+transcript StallDetector actually beat the transcript-only
pause fallback in practice, not just in isolated benchmarks?

**Construction:** 1 embedded filler event per cycle of 6 real Words clips (measured 240.0s of real speech / 40 fillers = 6.00s speech per filler on average), over
40 embedded filler cycles /
344.0 s of synthetic audio (PFSD TEST
split only, seed 13). Each cycle is real Words clips
back-to-back, then a real Uh/Um clip with no gap (filler onset ==
the preceding word's end, matching how a stall actually starts), then
1600 ms of true silence. A filler-stripping
(Chrome-condition) transcript and a 100 ms
SilenceTick timer drive the transcript-only detector; the identical raw audio
is also fed through the live `AcousticStream` (conf_thresh=
0.75) to drive the fused detector -- both
detectors observe the SAME word/tick timeline, only the fused one also gets
`observe_acoustic`. **Attribution window:** per cycle: (onset, onset + filler_clip_duration + silence_gap_ms] -- bounded by THAT cycle's own silence gap ending, never by the next cycle's onset, so a fire during the following cycle's fluent speech cannot be credited as a (falsely late) detection of this cycle's filler
Construction caveat: clips concatenated from many speakers/episodes across PFSD's TEST split only (never train/validation); every cycle boundary inserts a segment/speaker discontinuity absent from one continuous speaker's prosody -- same caveat as eval_stream_falsefire's stream-level false-alarm bench. Transcript channel text is placeholder (StallDetector only checks FILLERS/HEDGES membership and content-word count, never lexical identity).

**Detected before the pause fallback would have fired:** **36/39** (92.3%) embedded
fillers -- ON fired trigger=='filler_acoustic' strictly before the ms at which OFF's in-window fire occurred, both attributed within the same cycle's tight [onset, onset+filler+gap] window (see attribution_window above).

| Condition | Median latency from filler onset | n |
|---|---|---|
| Acoustic ON (fused, trigger=filler_acoustic) | 760 ms | 36 |
| Acoustic OFF (transcript-only pause fallback) | 1300 ms | 40 |
| Paired delta (OFF - ON, positive = acoustic earlier) | 540 ms | 36 |

Raw fire triggers -- ON: {"filler_acoustic": 36, "pause": 3} (missed
1); OFF: {"pause": 40} (missed
0). Same many-speaker-concatenation caveat as
the stream-level false-alarm bench in Table 2 applies to this stream too.

**Spurious acoustic fires during fluent speech (disclosed, not credited):**
**19** filler_acoustic fires
(4.75/min of
240.0 s fluent speech) landed outside
every cycle's attribution window -- i.e. FillerNet fired during real fluent
speech, not on an embedded filler. ON fires with trigger=='filler_acoustic' whose at_ms falls outside EVERY cycle's [onset, onset+filler+gap] window -- i.e. FillerNet fired during real fluent speech, not on an embedded filler. Never credited toward detected_before_pause or latency_from_onset_ms; a fused system that 'wins' by firing everywhere isn't winning, so this is disclosed on its own.

## 7. Noise-robustness stress test (`eval/run_noise_stress.py`)

EVAL-ONLY: no retraining, no threshold changes -- the shipped checkpoint and
the shipped conf=0.75 operating
point are evaluated exactly as they ship. Question: does FillerNet survive a
noisy demo hall?

**Sample.** n=300 PFSD TEST-split clips
(items drawn via scripts/train_filler.py:scan_split(['test'], limit=n) -- identical code path and shuffle (random.Random(13)) as the training-time test scan, sliced to n for runtime. class_balance (uh/um/speech/other counts) is reported so the clean-condition F1's comparability to the full-test-split 0.933 anchor is auditable, not asserted.). **Interferer.**
PFSD TEST-split 'Music' clips (real recorded music, an ambient-noise proxy for a demo hall; zero new downloads, fully reproducible) (822 clips in the
pool); one Music clip pre-assigned per eval item via numpy default_rng(seed), reused unchanged across all SNR levels so only the mix level differs between conditions -- isolates the SNR variable from interferer-instance variance. **SNR.** target_noise_rms = rms(signal) / 10**(snr_db/20); interferer scaled to that RMS then added; mix clipped to [-1, 1] (clipping incidence reported per condition below; 'clean' is the unmixed signal, never clipped)

| Condition | Standard F1 | Standard P/R | Operating-point F1 (conf>=0.75) | Operating-point P/R | Clipped mixes |
|---|---|---|---|---|---|
| clean | 0.929 | 0.941 / 0.919 | 0.903 | 0.973 / 0.843 | 0/300 |
| 15dB | 0.891 | 0.960 / 0.831 | 0.843 | 0.992 / 0.733 | 1/300 |
| 10dB | 0.836 | 0.984 / 0.727 | 0.735 | 1.000 / 0.581 | 3/300 |
| 5dB | 0.684 | 0.989 / 0.523 | 0.498 | 1.000 / 0.331 | 14/300 |

docs/EVAL.md Table 1 reports binary filler F1=0.933 on the FULL ~9.4k-clip test split (scripts/train_filler.py evaluate()); this script's 'clean'/'standard' row uses the identical decision rule on a random n=300 subset of that same split, so it should land close to 0.933 within sampling noise, not reproduce it exactly.

**Caveat.** measures the FillerNet CLASSIFIER IN ISOLATION on fixed 1.0 s clips. The live AcousticStream pipeline additionally applies a Silero VAD gate, a >=800 ms accumulated-voiced-time gate, and a 1200 ms per-kind refractory (backend/acoustic/stream.py) that may mitigate noise-induced misses in practice -- that mitigation is NOT measured here, only named as an open question.


## 8. Long-conversation recall (top-1 by turn depth)

Each conversation introduces one target in turn 1 and never mentions it
again, so a deep probe is answerable only if long-horizon memory works.
Verbatim window: 6 turns. Provider: **gemini** / gemini-3.5-flash.

| Turn depth of the probe | Baseline (recent-turns tail) | With ContextBuilder | n |
|---|---|---|---|
| early(<=10) | 1.000 | 1.000 | 12 |
| mid(11-25) | 0.083 | 0.750 | 12 |
| late(>25) | 0.250 | 0.917 | 12 |

Deep-recall delta from the ContextBuilder: **+0.667** (0.250 -> 0.917 in the
late bucket). Builder late-vs-early delta: **-0.083**.

**Limits.** The conversations are hand-authored by the team, not
transcripts of people with aphasia, and the filler turns are deliberately
neutral (no competing entities). This is an upper bound on long-range
recall under clean conditions, not field performance. n is small (see the
table), so single-item swings move a bucket by ~0.08.

**Same benchmark on the local model** (Qwen3.8-27B-Q4_K_M), for the cloud/local gap:

| Turn depth | Baseline | With ContextBuilder |
|---|---|---|
| early(<=10) | 1.000 | 1.000 |
| mid(11-25) | 0.083 | 0.750 |
| late(>25) | 0.083 | 0.750 |

The ContextBuilder delta is the SAME on both engines (+0.667 in the late
bucket), which is the point worth taking away: the long-horizon memory fix
is an architecture win, not a property of one model. The local model trails
the cloud one on absolute deep recall (0.750 vs 0.917).

Regenerate: `python eval/run_longconv_eval.py`

## 9. Speaker gate (proximity/energy only)

Two-speaker mixes, wearer and bystander drawn from DIFFERENT podcast
episodes (`pfsd-by-episode`), n=96 words per cell, threshold 0.35.

**Deployed rule** (per-word p90 of frame confidences):

| Level separation | Bystander suppressed | Wearer WRONGLY muted |
|---|---|---|
| 0 dB | 0.000 | 0.000 |
| 3 dB | 0.000 | 0.000 |
| 6 dB | 0.000 | 0.000 |
| 12 dB | 0.448 | 0.000 |

The safety property holds: the wearer is **never** wrongly muted, which is
the fail-open contract. The usefulness does not: below 12 dB of separation
the gate suppresses **none** of the bystander's speech, and only ~45% at
12 dB.

**This is a measured negative result, and it is the honest headline: level
alone does not separate two speakers in a room.** A more aggressive
aggregation buys suppression only by muting the wearer, which is worse than
not gating at all for an assistive device:

| Level separation | Bystander suppressed | Wearer WRONGLY muted |
|---|---|---|
| 0 dB | 0.292 | 0.260 |
| 3 dB | 0.490 | 0.219 |
| 6 dB | 0.750 | 0.198 |
| 12 dB | 1.000 | 0.177 |

So proximity-only gating does NOT solve the other-speaker problem. It is
safe, it is nearly free, and it earns its place only in the lav mic's
regime (DJI Mic 2S on the collar at ~5 cm, where a partner across a table sits far below
12 dB down). For the laptop mic -- the primary demo path -- it is close to
inert. Speaker-embedding verification with a short enrollment is the
mechanism that would actually work, and it was deliberately deferred.

**Unvalidated in real rooms.** These are synthetic mixes of real speech.
No two-speaker recording of the actual hardware exists, and the VAD labels
are oracle, so these numbers are an upper bound.

Regenerate: `python eval/run_speaker_gate_eval.py`
## 10. Verbatim ASR -- does the transcript keep the stall?

Echo's v2 thesis was that consumer ASR deletes the evidence, and it is
measured: on 5,044 annotated filler clips the Chrome path recovered the filler
**0.000** of the time. This section asks whether a recognizer that transcribes
verbatim on purpose changes that, and what size to pay for it.

| transcript source | preserved | Block | Prolongation | SoundRep | WordRep | Interjection | latency |
|---|---|---|---|---|---|---|---|
| CrisperWhisper small | **0.883** | 0.750 | 0.750 | 0.923 | 0.923 | 1.000 | 389 ms |
| CrisperWhisper medium | **0.883** | 0.833 | 0.750 | 0.769 | 1.000 | 1.000 | 814 ms |
| CrisperWhisper turbo | **0.900** | 0.750 | 0.917 | 0.923 | 0.846 | 1.000 | 327 ms |
| CrisperWhisper large | **0.850** | 0.667 | 0.667 | 0.923 | 0.923 | 1.000 | 1072 ms |
| Chrome SpeechRecognition | **0.000** | -- | -- | -- | -- | -- | -- |

Source: `eval/bench_asr_models.py`, n=60 SEP-28k events, 6 s windows.

Metric: dysfluency preserved = explicit filler token OR discourse filler OR cut-off word fragment ('f-') OR immediate word/phrase repetition. LOWER BOUND: Apple's Interjection definition includes person-specific fillers no fixed list can enumerate.

**`turbo` wins both axes** -- highest preservation and lowest latency -- and
`large` is *worse* at preserving dysfluency, which is what a bigger model's
stronger normalization prior buys you. Block is hardest for every model, as
expected: a block is silence, and no transcript represents silence.

The decisive control is the same model in `intended` mode on the same audio:
**0.060 preserved against 0.900 verbatim**. Every consumer recognizer makes
that choice silently and exposes no flag to change it.

## 11. StutterNet -- five dysfluency types, trained on people who stutter

The shipped FillerNet is `["uh","um","speech","other"]` trained on
PodcastFillers: fluent podcast hosts saying "um". It has **no class for a
block** -- the silent struggle to initiate a word, and the strongest evidence a
speaker is stuck -- and it detects interjections, which fluent speakers produce
constantly. Both shipped complaints follow from that one fact.

| type | n_pos | prevalence | AP | lift | F1 | precision | recall |
|---|---|---|---|---|---|---|---|
| Block | 440 | 0.137 | **0.256** | 1.86x | 0.339 | 0.254 | 0.511 |
| Prolongation | 315 | 0.098 | **0.484** | 4.94x | 0.498 | 0.519 | 0.479 |
| SoundRep | 277 | 0.086 | **0.306** | 3.55x | 0.358 | 0.265 | 0.549 |
| WordRep | 386 | 0.120 | **0.254** | 2.11x | 0.309 | 0.230 | 0.471 |
| Interjection | 744 | 0.232 | **0.734** | 3.17x | 0.686 | 0.665 | 0.708 |
| ANY | 1769 | 0.551 | **0.786** | -- | 0.737 | 0.633 | 0.882 |

Split: **episode-disjoint (host leakage measured on the SSL model: mean gap Block +0.003, ANY -0.004 -- see eval/eval_stutter_ssl_hostleak.py)**. n_train=14619, n_val=2296, n_test=3209; 582,949 parameters.

- SEP-28k reconstructed from Apple's official labels; 258/385 episodes recovered, 3 shows lost to link rot. NOT comparable to published SEP-28k numbers (different corpus).
- Stuttered speech, not aphasic speech. Transfers because the surface evidence overlaps; it is not an aphasia measurement.

Thresholds are fitted on VAL and stored **inside the checkpoint**, so a retrain
cannot silently inherit the previous model's operating point.

## 12. WavLM StutterNet -- the representation, not the data

The log-mel CNN above gives Block an AP of 0.256, and a calibrated downstream
fit had already given the Block head a weight of exactly **0.0** -- the channel
was carrying no signal worth using. Replacing the front end with WavLM Base+
(learned softmax over 13 hidden states, top 4 transformer layers unfrozen)
changes that, and the rest of this section is about how much, measured against
the ways such a number can be wrong.

| type | n_pos | prevalence | AP | lift | F1 | precision | recall |
|---|---|---|---|---|---|---|---|
| Block | 563 | 0.128 | **0.325** | 2.55x | 0.414 | 0.330 | 0.556 |
| Prolongation | 435 | 0.099 | **0.519** | 5.27x | 0.553 | 0.543 | 0.563 |
| SoundRep | 425 | 0.096 | **0.628** | 6.52x | 0.624 | 0.693 | 0.567 |
| WordRep | 412 | 0.093 | **0.737** | 7.89x | 0.736 | 0.696 | 0.782 |
| Interjection | 966 | 0.219 | **0.860** | 3.93x | 0.819 | 0.774 | 0.871 |
| ANY | 2293 | 0.520 | **0.882** | -- | 0.823 | 0.766 | 0.889 |

Split: **episode-disjoint (host leakage measured: mean gap Block +0.003, ANY -0.004)**. n_train=23,727, n_val=2,824, n_test=4,411; 95,564,930 parameters, 29,536,610 trainable.
Trained on: SEP-28k reconstruction, 30962 clips, 9 speaker pools (FluencyBank, HVSA, HeStutters, IStutterSoWhat, MyStutteringLife, StrongVoices, StutterTalk, StutteringIsCool, WomenWhoStutter); includes HF mirror data/sep28k_hf, which declares NO LICENCE -- partial reconstruction, NOT comparable to published SEP-28k figures; see docs/DATA_PROVENANCE.md

Read against the CNN's table above with care: **these are different test
sets** -- the corpus grew from 20,124 to 30,962 clips and the split was
redrawn, so this headline sits below an earlier 0.384 without anything having
regressed. That earlier checkpoint no longer exists on disk; a clean rerun of
its configuration scores Block 0.371 / ANY 0.893. The same-test-set comparison
is the next table.

- SEP-28k reconstructed from Apple's official labels; 258/385 episodes recovered, 3 shows lost to link rot. NOT comparable to published SEP-28k numbers (different corpus).
- Stuttered speech, not aphasic speech. Transfers because the surface evidence overlaps; it is not an aphasia measurement. Thresholds and epoch selection are fitted on VAL only.

### Did the extra 10,838 clips help?

The obvious experiment -- score the old checkpoint on the new test set -- is
invalid here, and finding out why was most of the work. `make_splits` drew one
permutation per show from a **shared** RandomState, so growing the corpus
reshuffled every show: **1,380 of the 4,411 new-test clips sit in the old model's
TRAIN set**. The old model scores Block 0.410 on those and 0.304 on the rest. The
only honest row is the intersection neither model trained on.

| arm | n | Block | Prolongation | SoundRep | WordRep | Interjection | ANY |
|---|---|---|---|---|---|---|---|
| v1 on its own test | 3,209 | 0.371 | 0.521 | 0.570 | 0.791 | 0.875 | 0.893 |
| v1 on new test (31% leaked) | 4,411 | 0.339 | 0.513 | 0.633 | 0.731 | 0.847 | 0.880 |
| **v1 on clean intersection** | 2,833 | 0.304 | 0.508 | 0.595 | 0.644 | 0.810 | 0.853 |
| v2 on new test | 4,411 | 0.325 | 0.519 | 0.628 | 0.737 | 0.860 | 0.882 |
| **v2 on clean intersection** | 2,833 | 0.305 | 0.518 | 0.607 | 0.683 | 0.840 | 0.864 |

Paired bootstrap on the clean intersection, v2 minus v1:

| type | delta AP | 95% CI | |
|---|---|---|---|
| Block | +0.0011 | [-0.027, 0.028] | no effect |
| Prolongation | +0.0100 | [-0.022, 0.042] | no effect |
| SoundRep | +0.0119 | [-0.019, 0.044] | no effect |
| WordRep | +0.0382 | [-0.018, 0.095] | no effect |
| Interjection | +0.0304 | [0.015, 0.045] | **real** |
| ANY | +0.0115 | [0.001, 0.022] | **real** |

**More data did not fix Block.** +0.001 AP, with a CI tight enough to rule out
anything past 0.03 in either direction, on a head whose positive count grew by
50%. Whatever limits block detection, it is not corpus size. What the extra
data bought is Interjection and a sliver of ANY -- the two heads that were
already working.

### The caveat repeated for three versions, and false

Every episode-disjoint number since v3 carried "optimistic -- leaks the
podcast's recurring host". It sounded appropriately humble and nobody had
measured it. Measuring it means scoring **identical clips** with two models,
one trained having seen that show's host and one not, so host exposure is the
only thing that differs. Comparing a nine-show mixture against one show, as
was first tried, confounds host exposure with show difficulty instead.

| show | n | Block | Prolongation | SoundRep | WordRep | Interjection | ANY |
|---|---|---|---|---|---|---|---|
| HeStutters | 647 | +0.019 | +0.001 | -0.008 | +0.026 | +0.020 | +0.006 |
| StutterTalk | 633 | -0.009 | -0.020 | +0.009 | +0.025 | -0.004 | -0.006 |
| StutteringIsCool | 458 | -0.001 | +0.017 | +0.079 | +0.035 | -0.033 | -0.012 |
| **mean gap** | | **+0.003** | **-0.001** | **+0.027** | **+0.029** | **-0.006** | **-0.004** |

Mean gap on Block **+0.003**, on ANY **-0.004**, no per-show gap above 0.08,
and the sign is inconsistent. The caveat is retracted.

What actually moves the number is which show you test on: the same model
scores Block 0.430 on StutterTalk and 0.222 on StutteringIsCool. Show difficulty dominates host identity by
roughly an order of magnitude, and that is the caveat that should have been
written in its place.

### What it costs, and the number the table above does not show

WavLM is 95.5M parameters against the CNN's 583k and costs 156 ms per 3 s
window on a full CPU against a 125 ms hop -- it is GPU-only (19 ms) until
distilled, which is why `STUTTER_BACKEND` defaults to `cnn`.

Every AP above is **clip-level**: one label for a 3 s clip. Echo never sees a
clip, it sees a stream, and it has to decide per frame. Calibrating the frame
head to a 2% fire rate on clean speech -- roughly the most it can nag and stay
usable -- gives an operating point far harsher than the clip table implies.
The fit has to be done in logit space: peak logits reach 13-28, so fitting in
probability space returns exactly 1.0 under float32 sigmoid saturation.

| type | frame recall | clean fire rate | logit threshold |
|---|---|---|---|
| Block | **0.151** | 0.020 | 29.295 |
| Prolongation | **0.425** | 0.021 | 45.500 |
| SoundRep | **0.545** | 0.020 | 43.725 |
| WordRep | **0.707** | 0.020 | 26.622 |
| Interjection | **0.654** | 0.020 | 14.714 |

**The Block head recalls 0.151 there.** Blocks are the dysfluency this product
most wants to catch, and at the interruption budget it actually runs at, it
catches about one in seven. That -- not the corpus, and not the encoder -- is
where the acoustic channel really stands.

## 13. Real aphasic speech (APROCSA)

Every other detection number in this report comes from stuttered or fluent
**podcast** speech. Neither is aphasia, and aphasia is the point: stuttering is
a motor-speech disorder where the word is known and will not come out; aphasia
is a language disorder where the word is not retrievable. They share surface
evidence, which is why a stutter-trained detector transfers at all -- but
"transfers" is a hypothesis until it is measured.

Ground truth is clinician CHAT coding, media-aligned: retracings, abandoned
utterances, phonological fragments, filled pauses, paraphasias. See
`scripts/aprocsa_chat.py` for exactly which codes count and why.

| arm | recall | false alarm | partner fires |
|---|---|---|---|
| old (intended ASR + FillerNet) | **0.818** | 0.732 | 0.159 |
| verbatim ASR + FillerNet | **0.837** | 0.704 | 0.153 |
| intended ASR + StutterNet | **0.824** | 0.694 | 0.141 |
| new (verbatim ASR + StutterNet) | **0.818** | 0.648 | 0.159 |

n_word_search=159, 300 s region per participant, 6 speakers.

### Interruption rate

An aid that fires constantly is unusable however good its recall, so the refractory is reported as a curve rather than as one chosen point.

| arm | min_gap_ms | recall | false alarm | fires/min |
|---|---|---|---|---|
| new (verbatim ASR + StutterNet) | 0 | 0.818 | 0.648 | 18.53 |
| new (verbatim ASR + StutterNet) | 2000 | 0.805 | 0.639 | 14.2 |
| new (verbatim ASR + StutterNet) | 3000 | 0.805 | 0.611 | 11.87 |
| new (verbatim ASR + StutterNet) | 4000 | 0.748 | 0.491 | 10.1 |
| new (verbatim ASR + StutterNet) | 6000 | 0.616 | 0.491 | 8.0 |
| old (intended ASR + FillerNet) | 0 | 0.818 | 0.732 | 17.67 |
| old (intended ASR + FillerNet) | 2000 | 0.830 | 0.704 | 14.7 |
| old (intended ASR + FillerNet) | 3000 | 0.792 | 0.685 | 12.17 |
| old (intended ASR + FillerNet) | 4000 | 0.755 | 0.593 | 10.33 |
| old (intended ASR + FillerNet) | 6000 | 0.679 | 0.463 | 8.07 |

### How long is a word-search pause in aphasia?

| internal silence | n | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|
| inside word-search utterances | 1227 | 672 | 1168 | 1856 | 2464 |
| inside fluent utterances | 167 | 416 | 848 | 1248 | 1926 |

The distributions **overlap heavily**. The shipped `STALL_PAUSE_MS=1300`
separates them at only 0.370 sensitivity (FPR 0.051). Pause length alone is a weak
signal in aphasia -- which is an argument for the other channels, not for
tuning this constant. Source: `eval/fit_aphasia_pause.py`.

**This comparison understates the change.** Both arms get the new VAD-gated
silence ticks and audio-derived turn boundaries; the shipped browser path had
neither, because its pause trigger measured gaps between browser transcript
events. Only transcript content and the acoustic model are isolated here.

'intended' mode stands in for the browser recognizer. Chrome cannot be scripted offline; intended mode is the same model on the same audio with dysfluency stripped (0.060 preserved vs 0.900 verbatim), and Chrome measured 0.000 on 5,044 clips.

- Six speakers. Not a population estimate.
- A 'false alarm' is an utterance the clinician did not code as a word search. CHAT coding is utterance-level and conservative, so some of these are real word searches that were not coded.
- Threshold fitting on this set would invalidate it; pause_ms is the shipped default.

## 14. ASR accuracy on aphasic speech

Echo's recognizer was chosen on dysfluency preservation -- does "[UM]" survive
into the transcript -- and on latency. What was never measured is whether the
words AROUND the dysfluency are right, on the speech this product is for.

| ASR configuration | WER on aphasic speech |
|---|---|
| turbo/verbatim/offline | **0.288** |
| turbo/verbatim | **0.375** |
| large/verbatim | **0.408** |
| medium/verbatim | **0.408** |
| turbo/intended | **0.483** |

Reference: CHAT clean_text (clinician transcription), fillers stripped. Corpus: APROCSA -- 6 speakers with chronic post-stroke aphasia.

Fillers are stripped from BOTH sides. They are measured separately in the
verbatim-ASR section, and leaving them in would let a model score better here
by transcribing hesitations rather than by getting the content words a
prediction has to be built from.

**This bounds every downstream number.** At two words in five wrong, the
predictor receives fragments like "And then [noise] [noise] and cut off [UM]
Cut off And" for a speaker reaching for *christmas*. No prompt, trigger or
acoustic model recovers from that. Source: `eval/bench_asr_aphasia_wer.py`.

## 15. Detection against a metronome

Utterance-level scoring credits a fire anywhere inside a clinician-coded
utterance plus a margin -- a median window of 5.7 s. A detector firing on a
timer lands inside that routinely, so the metric cannot tell detection from
regular interruption. `eval/align_aprocsa.py` gives most CHAT markers real
timestamps (2,344 of 2,549), and this scores against those instants.

The control is the point: a TIMER arm fires at fixed intervals at the same rate
using no audio at all. Recall above it is the only evidence of detection.

| arm | recall | timer control | lift | precision | fires/min |
|---|---|---|---|---|---|
| OLD  gap=0 | 0.742 | 0.697 | **0.045** | 0.536 | 18.53 |
| OLD  gap=2500 | 0.657 | 0.534 | **0.124** | 0.527 | 14.03 |
| OLD  gap=5000 | 0.405 | 0.388 | **0.017** | 0.507 | 9.27 |
| v3   thr=0.20 | 0.837 | 0.848 | **-0.011** | 0.512 | 21.33 |
| v3   thr=0.35 | 0.758 | 0.668 | **0.090** | 0.535 | 19.27 |
| v3   thr=0.50 | 0.668 | 0.590 | **0.079** | 0.543 | 17.37 |
| v4   thr=0.20 | 0.865 | 0.803 | **0.062** | 0.522 | 21.6 |
| v4   thr=0.35 | 0.781 | 0.758 | **0.022** | 0.540 | 19.77 |
| v4   thr=0.50 | 0.685 | 0.691 | **-0.006** | 0.547 | 17.87 |

Window: -250/+2000 ms, asymmetric because a detector cannot fire before its
evidence exists -- the pause trigger fires 1300 ms after silence onset. The
control is scored through the identical window.

**Nothing clears the metronome by much**, and the reason is the finding:
clinician-coded markers occur every 1.2-2.3 s in aphasic speech. "Is this
person word-searching right now" is nearly always yes, so no timing metric on
this corpus separates a detector from a clock. Source: `eval/score_markers.py`.

## 16. Word prediction on real aphasic speech

The metric the product exists for. A CHAT retracing records both that a word
search happened and what the speaker was reaching for -- `spring [//]
Christmas` -- so each one is a free (fragment, intended word) pair. The
fragment is what Echo actually had at that instant, replayed from the cached
ASR stream with partner speech excluded.

| arm | n | top-1 | top-3 | top-3 (span) |
|---|---|---|---|---|
| verbatim+ctx | 51 | 0.020 | **0.059** | 0.176 |
| intended+ctx | 51 | 0.020 | **0.020** | 0.118 |
| verbatim | 51 | 0.020 | **0.059** | 0.157 |
| intended | 51 | 0.020 | **0.059** | 0.098 |
| context_only | 51 | 0.000 | **0.039** | 0.059 |
| freq_english | 51 | 0.000 | **0.020** | 0.039 |
| freq_corpus | 51 | 0.020 | **0.059** | 0.098 |

n = 51 scorable events across six speakers.

**Echo is at the frequency-baseline floor on the strict metric**, and the
paired table shows the two hit disjoint events -- uncorrelated with a
constant-answer baseline rather than tied with it.

The verbatim-vs-intended comparison, counted per event:

| metric | arms | verbatim only | intended only | both | p |
|---|---|---|---|---|---|
| top3 | verbatim+ctx vs intended+ctx | 3 | 1 | 0 | 0.625 |
| top3_span | verbatim+ctx vs intended+ctx | 5 | 2 | 4 | 0.4531 |
| top3 | verbatim vs intended | 2 | 2 | 1 | 1.0 |
| top3_span | verbatim vs intended | 6 | 3 | 2 | 0.5078 |

Verbatim leads on aggregate (16 events to 8) but there are counterexamples, and no comparison approaches significance. This is consistent with the verbatim thesis and is not evidence for it.


### Was ASR accuracy the binding constraint? No.

This section previously closed by attributing the floor to ASR error
upstream. That is testable: the streaming policy changed between these two
runs and nothing else did, cutting WER on the same audio from **0.402 to
0.375**. Same 51 events, same model, same prompts.

| arm | top-3 hits before | after | top-3 span before | after |
|---|---|---|---|---|
| verbatim+ctx | 3 | 3 | 11 | 9 |
| context_only | 3 | 2 | 5 | 3 |
| freq_corpus | 3 | 3 | 5 | 5 |
| freq_english | 1 | 1 | 2 | 2 |

**Strict top-3 does not move at all** -- 3 hits before, 3 after, the same
count a corpus-frequency baseline gets. The looser span metric goes 11 to 9,
i.e. down, which at n=51 is noise in the other direction.

One number does improve: verbatim+ctx now beats context_only on top-3 span at
McNemar p=0.031, against 0.070 before. That is not the fragment arm getting
better -- it is context_only getting *worse* (5 hits to 3). Reading it as
progress would be reading a control's regression as a treatment effect.

So the diagnosis stated across the last two versions -- that everything
downstream is bounded by ASR accuracy -- is **not supported**. A 6.8%
relative WER reduction bought exactly nothing. Either the remaining error
rate is still far above whatever threshold would matter, or word identity at
a word-search moment is not recoverable from the fragment at all. The
committed events file argues for the second: where the speaker was reaching
for *christmas*, the fragment Echo held reads "Alright, It was". That is not
a transcript that a better decoder rescues.

Source: `eval/run_aphasia_prediction.py`.

## 17. Limitations

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
  1300 ms.
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
- **Dual-channel ablation is one synthetic stream, not a distribution.** A single seeded run (n=40 embedded fillers) at one mix ratio and one silence-gap length; each cycle's detection is scored only inside its own tight attribution window (see the section above), so a fire during a LATER cycle's fluent speech can never be credited to an earlier filler -- but such fires are real and disclosed separately as spurious acoustic fires, not hidden.
- **Noise-stress measures the classifier in isolation.** The live pipeline's VAD gate, voiced-time gate, and refractory may mitigate noise-induced misses in practice; that mitigation is not measured, only named as an open question.
