# Echo v6 — research, model comparison, and overhaul proposals

*Written 2026-09-07. Everything measured here was measured on this laptop (RTX
4090 Laptop GPU; 32 logical cores; google-genai 1.39.1) against the tree at
`2f9b5c6` plus the prototype in §4. Numbers in `docs/EVAL.md` are not touched.
Exploratory numbers not backed by a committed results file are marked (scratch).*

## 1. Landscape (cited)

### 1.1 Real-time disfluency / word-finding detection from audio

**SOTA on the corpora Echo trains on is low, and Block is the worst class
everywhere.** A 315M wav2vec2-XLSR fine-tune (SEP-28k-E + KSoF + AS-70) reports
SEP-28k-E per-class F1 Block 0.33 / Interjection 0.77 / Prolongation 0.51 /
SoundRep 0.53 / WordRep 0.71 (arXiv 2603.26939,
https://arxiv.org/html/2603.26939). Attention-MIL over Whisper/WavLM gets Block
0.35 clip-level and frame-level F1 Whisper 0.70 / WavLM 0.56 (arXiv 2606.20338,
https://arxiv.org/html/2606.20338). A 3B-LLM stack on clean Mandarin clinical
data still has Block F1 < 0.5 (arXiv 2505.22005,
https://arxiv.org/html/2505.22005v1). Echo's CNN (Block AP 0.256) and WavLM Base+
(0.325) sit where a 583k- and a base-size model should against this table; the
WavLM base-vs-large gap is 0.747 vs 0.803 binary clip F1 (arXiv 2409.10704,
https://arxiv.org/html/2409.10704v1).

**Real-time is a solved non-issue, so it is not a differentiator.** A 616k
log-mel CNN on 3 s windows — essentially Echo's CNN — runs 1.34 ms end-to-end on
an iPhone, at simple-CNN macro-F1 ~0.28 on SEP-28k and AUC 0.58 for predicting
an upcoming event (arXiv 2604.27279, https://arxiv.org/html/2604.27279). No
2024–2026 paper reports a causal/streaming encoder with a latency-vs-F1 curve for
audio stutter detection; the only streaming-disfluency latency work is text-side
(arXiv 2205.00620, https://arxiv.org/pdf/2205.00620).

**The transcript deletes the evidence** (Echo's v2 thesis, now quantified):
Whisper transcribes 13.1% of filled pauses and 10.2% of repetitions in
FluencyBank (Romana et al., JSLHR 2024,
https://pubs.asha.org/doi/10.1044/2024_JSLHR-24-00070) and hallucinates on >20%
of sound-repetition clips (Interspeech 2025,
https://www.isca-archive.org/interspeech_2025/sridhar25_interspeech.html).
CrisperWhisper restores verbatim fillers with timestamps
(https://arxiv.org/abs/2408.16589), which Echo v3 shipped.

**Aphasia-specific event-level acoustic detection does not exist in 2024–2026.**
Speaker-level aphasia detection from pause statistics reaches 86.6%
(https://arxiv.org/abs/2408.14082); pause distributions differ by PPA variant and
person (https://www.tandfonline.com/doi/full/10.1080/02687038.2024.2366285); the
closest "what word were they reaching for" audio model is a closed-set 15-word
Boston Naming classifier (Frontiers in AI 2026,
https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1786757/full).
Nobody publishes per-event precision/recall for "this pause is a word-finding
stall." Echo's APROCSA marker-level result — no detector beats a metronome by
more than +0.12 recall because clinician-coded markers occur every 1.2–2.3 s — is
therefore not an Echo failure; it is what the field has not solved either. No
stuttering challenge at Interspeech 2025
(https://www.interspeech2025.org/challenges) or ICASSP 2026
(https://2026.ieeeicassp.org/sp-grand-challenges/); the last was SLT 2024 on
AS-70 (https://arxiv.org/abs/2409.05430).

### 1.2 LLM-based intended-word recovery

- **Purohit et al., CSCW '23 Companion:** ChatGPT recovered the intended word in
  11/12 AphasiaBank circumlocutions (n=12,
  https://dl.acm.org/doi/10.1145/3584931.3606993). Their own future-work
  describes an agent that would intervene during discussion — i.e. Echo — which
  they did not build.
- **Kim, Storai, Hwang, Findings of EMNLP 2024, GradSelect** (the only
  real-patient benchmark, ~2,500 unintended words from 353 AphasiaBank Cinderella
  sessions): challenge set GradSelect EM 0.3271 / Acc@5 0.5420 vs GPT-4 0.3081 /
  0.4395; original set 0.4301 / 0.6496 vs 0.3196 / 0.4939
  (https://aclanthology.org/2024.findings-emnlp.616/,
  https://arxiv.org/html/2506.14203). **A frontier LLM on real aphasic
  circumlocutions is a 30–45% top-1 system.**
- Adikari et al., Sci Rep 2025: GPT-4o with memory reconstructs whole utterances
  at ~0.80–0.82 rated accuracy (~1,980 utterances / 180 participants,
  https://www.nature.com/articles/s41598-025-24725-x) — utterance reconstruction,
  not exact-word EM, not comparable.
- Reverse dictionary with clean definitions: 57.5% top-1, 73.8% with five
  candidates (https://www.sciencedirect.com/science/article/abs/pii/S0952197624007541);
  TREC tip-of-the-tongue R@1 ~0.17
  (https://trec.nist.gov/pubs/trec33/papers/Overview_tot.pdf). Even the easy
  version is a coin flip at top-1; multiple candidates is what clears 50%.
- SpeakFaster (Nature Comms 2024): conditioning on the partner's turn gives 77%
  abbreviation-expansion for typed AAC
  (https://www.nature.com/articles/s41467-024-53873-3) — strongest evidence
  context helps, for cognitively intact typists.
- Zero-shot LLM judgments on aphasic language are unreliable; few-shot matters
  (arXiv 2606.15696, https://arxiv.org/abs/2606.15696) — Echo's three few-shots
  are the right call.
- **Topic-only conditioning for anomia is unpublished** (AphasiaBank tasks are
  monologic, so a "last 6 turns" eval can't run on the standard benchmark) — the
  cheapest open experiment Echo can own (proposal 3).

### 1.3 Speculative / anticipatory prediction

Production voice agents already do what Echo's prefetch does and publish the cost.
LiveKit preemptive generation starts on a stable partial and discards on
mismatch, "increases LLM token usage"
(https://docs.livekit.io/agents/logic/sessions/); third-party measurements: 150–350
ms saving, discards "below 5 percent of turns"
(https://futureagi.com/blog/how-to-optimize-livekit-latency-2026/). Deepgram
Flux's eager end-of-turn trims 100–200 ms for "50–70% more LLM calls"
(https://developers.deepgram.com/docs/flux/voice-agent-eager-eot). The 2026
endpoint-anticipation paper reports 505 ms average reduction for 28.4% more
speculative compute (https://arxiv.org/abs/2606.13450); RelayS2S keeps 99% of
cascaded quality with a verifier deciding when the speculative prefix may stand
(https://arxiv.org/abs/2603.23346).

The incremental-processing literature is the cautionary half: 90.5% edit overhead
and only 58.6% of words immediately correct in a baseline incremental recognizer
(https://aclanthology.org/N09-1043.pdf); partials are "generally unstable" and
hypothesis age is the best stability predictor
(https://aclanthology.org/W11-2014.pdf); ~95% of hypotheses stable after ~0.5 s
(https://napier-repository.worktribe.com/OutputFile/3127047); the IU framework in
which any hypothesis can be revoked (https://aclanthology.org/E09-1081/). **Echo's
prefetch has no revoke path** — the drift guard serves a prediction made on a
prefix and never re-checks it. §2 measures the cost.

### 1.4 Gemini Flash latency (Sept 2026)

Official: the Flash line is 3.5 / 3.6 / 3.7 / 3.8-flash plus 3.5- and
3.1-flash-lite (https://ai.google.dev/gemini-api/docs/models). On Gemini 3.x
`thinking_budget` is "accepted for backwards compatibility" and Flash/Flash-Lite
"do not support full thinking-off"; the documented control is `thinking_level`
(default `medium` on 3.5-flash) (https://ai.google.dev/gemini-api/docs/thinking).
Google recommends temperature 1.0 on Gemini 3
(https://ai.google.dev/gemini-api/docs/gemini-3). Implicit context caching needs
4,096 tokens on 3.x Flash
(https://ai.google.dev/gemini-api/docs/generate-content/caching); Echo's prompt is
~520 tokens, so caching is worth nothing here. Third-party TTFT benches (overstate
Echo's, 10k-token prompts): 3.5-flash `minimal` 0.99 s
(https://artificialanalysis.ai/models/gemini-3-5-flash-minimal/providers);
3.7-flash `low` 0.70 s; 2.5-flash-lite 0.29 s; a 600-call voice bench put
2.5-flash-lite at 381 ms TTFT / 674 ms total
(https://dev.to/karimgeh/i-tested-6-gemini-models-for-voice-ai-latency-the-results-will-change-how-you-build-1kbm).

**Measured here** (shipped predictor: `thinking_budget=0`, response_schema, three
few-shots, ~517 prompt tokens):

| model / setting (SDK 1.39.1) | n | median ms | notes |
|---|---|---|---|
| gemini-3.5-flash, budget 0, non-stream (shipped) | 16 | **1047** | thoughts=0 every call — thinking IS off |
| gemini-3.5-flash, budget 0, stream | 16 | **876** | identical top words on all 16 |
| gemini-2.5-flash-lite, budget 0, non-stream | 16 | 778 | |
| gemini-2.5-flash-lite, budget 0, stream | 16 | 691 | |
| gemini-3.5-flash-lite, budget 0 | — | 400 INVALID_ARGUMENT | rejected; ~860–910 ms with no thinking_config (scratch) |
| gemini-3.7-flash, budget 0/none | 4 | 1770–2070 | 241–248 thought tok/call; can't disable on this SDK (scratch) |
| gemini-3.8-flash, budget 0 | 4 | 7815 | 241–247 thought tok; p95 8.8 s (scratch) |

First four rows: `eval/results/predictor_latency_bench.json`. Removing the schema
did NOT help (1212 vs 1133 ms, n=6); dropping few-shots saved ~130 ms at an
unmeasured accuracy cost; `max_output_tokens` 64 vs 256 changed nothing. The
docs' claim that `thinking_budget=0` may not be honoured is false for 3.5-flash
on this SDK and true for 3.7/3.8; `thinking_level` needs SDK 2.x.

### 1.5 Competing / adjacent products

Nothing shipping listens beneath the transcript, and nothing is unprompted — but
two products own "AI listens and suggests" in buyers' minds:

- **Lingraphica Conversations** (July 2026, insurance-funded devices, SLP
  distribution): transcribes the partner and suggests tap-responses; on-screen +
  spoken (https://finance.yahoo.com/technology/ai/articles/lingraphica-supports-spontaneous-communication-conversations-154700678.html).
- **Broca AI Speech** (2024–25): listens to the conversation, offers phrase cards
  to tap; no evidence published (https://www.brocaaispeech.com/).
- **AphasiaGPT** (Nov 2025, free, by a stroke survivor): camera word finder +
  practice (https://apps.apple.com/us/app/aphasiagpt/id6753925046).
- **WordFinder** (QARC+AWS): photo → Rekognition → Claude related words; manual
  (https://aws.amazon.com/blogs/machine-learning/wordfinder-app-harnessing-generative-ai-on-aws-for-aphasia-communication/).
  **Spoken** (LLM next-word for typed AAC, https://spokenaac.com/).
- Symbol/phrase AAC (Proloquo, TouchChat, Grid, Predictable, Lingraphica
  SmallTalk): tap-driven, no listening
  (https://lingraphica.com/smalltalk-aphasia-apps/,
  https://touchchatapp.com/touchchat-hd-aac-with-wordpower,
  https://thinksmartbox.com/voco-chat/).
- Therapy apps with evidence: Constant Therapy virtual RCT, WAB-AQ +6.75 vs +0.38
  (https://www.frontiersin.org/journals/neurology/articles/10.3389/fneur.2021.626780/full);
  Tactus 6-level cue hierarchy whose guidance says cues should NOT be automatic
  (https://tactustherapy.com/cueing-hierarchy-word-finding-aphasia/).
- **Google Relate/Euphonia, Apple Live Speech / Listen for Atypical Speech,
  Azure Speech, Voiceitt** — personalised ASR for impaired speech; none detects a
  stall or supplies a word (https://sites.research.google/relate/,
  https://www.apple.com/newsroom/2025/05/apple-unveils-powerful-accessibility-features-coming-later-this-year/,
  https://www.voiceitt.com/). On dysarthric speech they beat Echo's transcript.
- Private-delivery precedent: head-worn glanceable vocabulary for PWA (CHI 2015,
  https://dl.acm.org/doi/10.1145/2702123.2702484); earpiece feedback on the
  wearer's own fillers (WSCoach 2025, https://arxiv.org/abs/2507.04238);
  smartglasses co-design found the form factor "socially conspicuous"
  (https://www.nature.com/articles/s41598-025-22253-2). **No aphasia system
  cues the speaker from their own live audio.**

Clinical timing/load literature bearing on Echo's defaults: phonological
(first-sound) cues help every anomia profile and whole-word modelling is the
bottom of the hierarchy
(https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2021.747391/full);
PWA mean silent pause ~1.0–1.3 s vs ~0.9 s in controls
(https://pmc.ncbi.nlm.nih.gov/articles/PMC11047180/); a 1 s vs 5 s response delay
helped 3 of 39 PWA and hurt 3
(https://www.frontiersin.org/journals/human-neuroscience/articles/10.3389/fnhum.2019.00406/full);
an auditory stimulus during naming raises omission and phonological errors in PWA
(https://www.tandfonline.com/doi/full/10.1080/02687038.2023.2253567).

## 2. Where Echo stands, honestly

**Real and unmatched:** the acoustic channel runs beneath a transcript that
provably deletes the evidence; no shipping product listens to the wearer's own
speech for a stall; every competitor's latency is "however long it takes you to
tap". The engineering is measured and regenerable beyond what the §1.5 startups attempt.

**What the numbers actually say:**

1. **Word prediction on real aphasic speech is at the frequency floor** (3/51
   strict top-3, `EVAL.md` §16) and stayed there through five improvements. The
   literature's best real-patient number is GPT-4 EM 0.31 / Acc@5 0.44, so
   96.7% on the hand-authored set is a prompt sanity check, not a capability claim.
2. **The acoustic latency number describes a model the live path does not run.**
   `run_latency_bench.py` measures FillerNet (755 ms) while the shipped default is
   the StutterNet CNN, which fires at **531 ms median (min 412 / max 729) but
   misses 3/24** (scratch); the WavLM SSL model on CUDA fires at 567 ms with 9/24
   misses. The 755 ms is window physics, not compute (FillerNet is 2.0 ms/hop).
3. **The 0 ms prefetch row hides an accuracy cost the drift guard permits** (cache
   served when the fragment drifted up to 2 content words past the cached one).
   Re-running the frozen 60-item set with trailing content words removed (scratch,
   gemini-3.5-flash):

   | fragment | top-1 | top-3 | concrete | proper | abstract_verb |
   |---|---|---|---|---|---|
   | full (published) | 58/60 | 59/60 | 20/20 | 19/20 | 19/20 |
   | last 1 content word removed | **45/60** | 50/60 | 20/20 | 18/20 | **7/20** |
   | last 2 removed | **50/60** | 54/60 | 20/20 | 17/20 | 13/20 |

   The trailing word is where the description lands ("…so you don't get" → "wet").
   A stale prefetch hit is a 13–22 point top-1 loss on exactly the informative
   items — and a fresh (drift-0) cache is rarely available (0/14 at 1.5–2.0 w/s,
   7/14 at a 1300 ms pause stall, scratch). Prefetch hides the LLM by answering an
   earlier question; a correction path is proposal 1.
4. **Block recall is 0.151 at the product's fire budget**, the scorer gives the
   Block head weight 0.0, and the pause feature is negative — the acoustic
   channel's contribution on aphasic speech is the Interjection and Prolongation
   heads.
5. **No AphasiaBank eval, no clinician, no user, and the cognitive-load
   assumption has published contrary evidence** (dual-task naming in PWA, §1.5).
   Clinicians' norm is user-initiated cueing and the cue hierarchy places the
   whole word last. Echo's default (automatic, ~1–2 s, whole word) is the
   configuration most likely to add load; an armed/opt-in mode and a first-sound
   cue are the defensible versions, and neither exists.
6. Against Google/Apple/Microsoft/Voiceitt on a dysarthric wearer, Echo's
   transcript (WER 0.375 on APROCSA) loses, and the fragment sent to the predictor
   is least reliable exactly at the stall.

**Claimable today:** a dual-channel stall trigger that fires earlier than a pause
timeout on synthetic streams; a working private delivery loop; a live round-trip
of ~0.9–1.05 s; an eval harness designed for Kim et al.'s AphasiaBank split.
**Not claimable:** any detection F1 on aphasic speech above a clock, any
intended-word accuracy above the GPT-4 30–45% band, any benefit to a person with
aphasia.

## 3. Model comparison — Gemini 3.5 vs 3.7 vs 3.8 Flash

Generated by `eval/compare_gemini_models.py` from
`eval/results/gemini_model_compare.json` (run 2026-09-07). Frozen 60-item set
(sha256 `9fd89ea1d59b…`), 3× per condition; latency pools every successful call
(180/row); google-genai 1.39.1. Incumbent default `gemini-3.5-flash` (unchanged
in `backend/config.py`). Settings: `thinking_off` = shipped
(thinking_budget=0, max_output 256); `thinking_default` = model-default thinking
under 256 cap; `thinking_default_4k` = model default at 4096.

| model | setting | top-1 | top-3 | p50 | p95 | errors |
|---|---|---|---|---|---|---|
| `3.5-flash` | thinking_off | 97.8% (58.67/60) | 100% (60/60) | 1016 | **1247** | 0 |
| `3.5-flash` | thinking_default | 8.3% (5/60) | 8.3% | 1740 | 2007 | 165 empty |
| `3.5-flash` | thinking_default_4k | 97.8% | 100% | 2179 | 3293 | 0 |
| `3.7-flash` | thinking_off | 96.7% (58/60) | 98.9% | 1247 | 1750 | 2 empty |
| `3.7-flash` | thinking_default | 61.7% (37/60) | 62.2% | 1827 | 3377 | 68 empty |
| `3.7-flash` | thinking_default_4k | 97.8% | 100% | 1747 | 3370 | 0 |
| `3.8-flash` | thinking_off | 97.8% | 99.4% | 1914 | 8847 | 1 empty |
| `3.8-flash` | thinking_default | 37.2% (22.33/60) | 37.2% | 2458 | 8103 | 113 empty |
| `3.8-flash` | thinking_default_4k | 97.8% | 100% | 2752 | 8392 | 0 |

`errors` = exceptions + zero-parseable-candidate responses (empty text,
MAX_TOKENS, bad JSON); all score as misses. Under `thinking_default` an empty
response means the model spent the 256-token cap on thoughts — the failure
`backend/predictor/gemini.py` avoids with thinking_budget=0. thinking_budget=0 is
NOT fully honoured by 3.7-flash (mean 67.3 thought tok/call) or 3.8-flash (68.5):
those still bill some reasoning, and adopting them would require raising
`max_output_tokens`. n=60×3, so a one-item top-1 difference is 1.7 pts —
differences under ~two items are noise.

**Rule:** switch away from 3.5-flash only if a candidate under `thinking_off` has
top-1 ≥2 items/rep higher OR equal top-1 with p95 ≥15% lower, AND no new
failures, AND ≤10% p95 increase. Result — **keep `gemini-3.5-flash` with
`thinking_off`**: 3.7-flash DO NOT SWITCH (top-1 −0.67, p95 +40%, +2 failures);
3.8-flash DO NOT SWITCH (top-1 +0, p95 +609%, +1 failure); keep thinking off on
all three. Change only via the `GEMINI_MODEL` env var. Regenerate:
`python eval/compare_gemini_models.py gemini-3.5-flash gemini-3.7-flash
gemini-3.8-flash --reps 3 --sleep 0.25` (`--report-only` re-renders).

> Live default note: the shipped predictor is now **DeepSeek `deepseek-flash`**
> (see `README.md` / `VERSIONS.md`); this Gemini comparison remains the basis for
> the Gemini fallback choice.

## 4. Ranked proposals (all within current hardware)

Ranked by expected value against risk. New scripts write NEW files under
`eval/results/`.

**1. Prefetch serve-then-correct (revoke path).** On a cache hit with drift > 0,
serve cached candidates immediately as `served="prefetch-provisional"` AND launch
the live call on the full fragment; if top-1 differs, emit a second `Prediction`
`served="live-correction"` (`backend/pipeline.py`, `schemas.py`; the card and
TTS must handle the second `prediction` — coordinate `session.py`/`frontend`).
*Gain:* recovers the 13–22
top-1 points from §2.3 while first paint stays ~0 ms; correction lands ~0.9 s
later, inside the 1.3 s budget (LiveKit/RelayS2S verifier pattern; 5–30%
overhead). *Measure:* `run_prediction_eval.py --dataset` on versioned truncated
sets vs full; extend the serving bench. *Risk:* medium (two words reaching the
speaker is worse than one late word if the card/TTS policy is wrong). *Effort:*
6–8 h.

**2. Gemini streaming transport (`GEMINI_STREAM`) — prototyped, §4.** *Gain:*
−171 ms median on every live-path call (1047→876 ms), same candidates; helps live
stalls, rejects, and shadow prefetch (876 ms row hits a drift-0 cache 10–11/14 vs
0–7/14). *Measure:* `python eval/bench_predictor_latency.py`. *Risk:* low (same
request, 7 offline tests). *Effort:* 2 h (spent).

**3. Topic-conditioned prediction on APROCSA (the v6 item-1 experiment).** Add a
`topic+ctx` arm to `eval/run_aphasia_prediction.py` (new prompt variant, unused by
the live path) that first asks the topic from the rolling summary, then words
likely sought in it; scored against the 51 retracing events + frequency
baselines. Optional `kim-style` arm drops the fragment. *Gain:* unknown, and
that's the point — the only untested hypothesis for the metric the product exists
for; no topic-only anomia result in the literature (§1.2); cached ASR streams +
CHAT alignments already exist. *Risk:* none to production. *Effort:* 4 h + ~300
API calls.

**4. Make the latency bench measure the shipped acoustic model.** Add
`--stutter-model`/`--stutter-backend` arms so `run_latency_bench.py` runs the CNN
(live default) and, on CUDA, SSL, next to FillerNet. *Gain:* the CNN measures
531 ms median (n=24, 21/24 fired, 1 preamble false fire) which MEETS the 600 ms
gate the report discloses as missed — at a recall cost that must print next to it;
also closes "where does the time go" (a 62 ms hop makes CNN misses WORSE, 17/24 —
hop is not the lever; compute is 2.0 ms FillerNet, 6.6 ms CNN CPU, 97 ms SSL CPU,
9.4 ms SSL CUDA). *Risk:* none. *Effort:* 2 h.

**5. Flash-lite on the shadow path; conditional thinking config.** Attach
`thinking_config` only when the model accepts it (3.5-flash-lite returns 400 with
budget 0); optional separate `shadow_predictor` (`GEMINI_SHADOW_MODEL`) so the
prefetch shadow runs faster than the live/reject path. *Gain:* 2.5-flash-lite
streamed 691 ms vs 876 (−185 ms) at **56/60 vs 58/60** top-1
(`prediction_eval_gemini25flashlite.json`; loses three abstract verbs +
"clothespin"→"pegs"). 3.7/3.8-flash ruled out on SDK 1.39.1 (thinking can't be
disabled). *Risk:* low-medium (−2 items). *Effort:* 2 h.

**6. Run SSL StutterNet on the 4090 and refit the scorer.** Env only for the
model (`STUTTER_BACKEND=ssl ACOUSTIC_DEVICE=cuda`); re-run
`eval/fit_stall_scorer.py` with SSL features so `DEFAULT_WEIGHTS` fit the running
model (today `acou_block` is 0.0 for the CNN). *Gain:* Block AP 0.325 vs 0.256,
ANY 0.882 vs 0.786 become reachable live (SSL 9.4 ms/3 s window on GPU, max 17 ms,
under the 125 ms hop); it fired 15/24 at 567 ms with 0 preamble false fires (more
conservative, not faster). Whether the refit lifts APROCSA recall above the
metronome is the open question. *Risk:* medium (GPU shared with CrisperWhisper;
NumPy-2 ABI warning to check). *Effort:* 3 h.

**7. Calibrate to the DJI Mic 2S, not the podcast prior.** Record 2 min wearer +
2 min partner across a table with the DJI lav and the laptop mic
(`scripts/mic_probe.py`), measure real level separation / SNR, then pick
`STUTTER_SCALE` on the DJI recording (`eval/calibrate_stutter_frames.py`) at the
2% clean-fire budget, and only then decide `WEARER_GATE`. *Gain:* lowering
`ACOUSTIC_CONF` 0.75→0.70 buys clip recall 0.844→0.863 for 5.8→6.4 false
alarms/min on FillerNet, but the live default is StutterNet (`stutter_scale`), and
noise-stress shows recall 0.843→0.733 at 15 dB SNR — the DJI's SNR is worth more
than any threshold move, and it's unmeasured. The wearer gate suppresses ~45% of
bystander speech at 12 dB separation and 0.000 below it. *Risk:* low. *Effort:*
3 h + recording.

**8. Personalisation: rerank with the speaker's accepted words.** Keep a session
lexicon of accepted top words (needs an `accept` message on `/ws` — `app.py`/
`frontend`) and pass it as one prompt line + a deterministic post-rank boost.
*Gain:* unmeasurable now (frozen set has no repeated targets, no users; a
four-minute demo shows nothing). *Risk:* low. *Effort:* 4 h. Last on evidence,
not cost.

**Not proposed:** prompt caching (below 4,096-token minimum), removing the
response schema (no gain), Gemini 3.7/3.8 (thinking can't be disabled on the SDK),
shorter FillerNet hop for the CNN (worse), any `STALL_PAUSE_MS` retune (aphasic
pause distributions overlap, `EVAL.md` §13).

## 5. Prototype: `GEMINI_STREAM` (proposal 2)

`backend/predictor/gemini.py` gained a `stream` flag: the identical request
(system prompt, few-shots, response schema, `thinking_budget=0`, temp 0.2, 256
tokens) goes through `generate_content_stream`, chunks concatenated through the
same tolerant `_parse`. `backend/config.py` reads `GEMINI_STREAM` (default off);
`__init__.py` wires it through `get_predictor`; `app.py`/`session.py`/
`audio_sources.py`/`frontend` untouched. **Tests:** `tests/test_gemini_stream.py`,
7 tests, no network (off→`generate_content` only; on→stream only, reassembles
chunks split mid-key/value, skips thought-signature chunks; identical candidates;
empty stream → `[]`; env round-trips). Full suite **475 passed** (468 before).

Before/after (`eval/results/predictor_latency_bench.json`; 4 fragments × 4 rounds,
interleaved, 0 errors):

| arm | n | median | p90 | min | max | top-1 agreement |
|---|---|---|---|---|---|---|
| gemini-3.5-flash non-stream (shipped) | 16 | **1047** | 1162 | 914 | 1697 | — |
| gemini-3.5-flash stream | 16 | **876** | 1065 | 819 | 1339 | 16/16 same top word |
| gemini-2.5-flash-lite non-stream | 16 | 778 | 857 | 677 | 1124 | 16/16 |
| gemini-2.5-flash-lite stream | 16 | 691 | 787 | 666 | 1131 | 16/16 |

The saving is the non-streaming end-of-response wait, not generation (the JSON is
25–45 tokens, one or two chunks, so a partial-JSON parser adds nothing).
`gemini-3.5-flash-lite` arms are 16 errors each (rejects thinking_budget=0,
proposal 5). **Regenerate:** `python eval/bench_predictor_latency.py --rounds 4
--models gemini-3.5-flash-lite,gemini-2.5-flash-lite` then `python -m pytest
tests`. **Enable:** `GEMINI_STREAM=on`; default stays off until a live
`scripts/e2e_live.py` run with it on. It does not touch accuracy, does not help a
cache hit (already ~0 ms), and does not address §2.3 staleness (that's proposal 1).
