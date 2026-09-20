# Echo — pitch notes (judge-facing)

## The problem (30s)
**Over 2,000,000 Americans live with aphasia** — most after a stroke — and
**~180,000 acquire it every year** (National Aphasia Association; NIDCD). The most
common symptom is **anomia**: you know exactly what you want to say, but the word
won't come. People describe around it — *"the thing you put bread in… it gets
hot…"* — while the conversation moves on without them.

Existing tools don't help **in the moment**: AAC apps make you type or tap
symbols; therapy apps are drills for later; smart-reply tools listen to the
*other* person and hand you canned responses. Nothing helps you finish **your own
sentence**.

Scope, stated plainly: Echo targets the **anomia / word-finding** profile — someone
who knows the word and can read a suggested one. It is not designed for
comprehension-impaired, severe Broca's, or global aphasia, where a
ranked-candidate display may not be legible; the 2M figure is the population
living with aphasia broadly, not a claim that all of them are served.

## What Echo does (15s)
Echo listens while you speak and stays silent while you're fluent. The moment you
stall — a pause, a filler, a stretched *"theeee…"*, a circumlocution — it predicts
the word you're reaching for and offers it within ~1–2 seconds (or instantly, via
prefetch). You say it, and keep going.

## The technical core (v2): the dual-channel thesis
**Consumer ASR erases exactly the speech phenomena that matter for aphasia.**
- Chrome's recognizer **suppresses "um/uh"** — no off switch
  (github.com/GoogleCloudPlatform/golang-samples/issues/1373; Google STT has no
  disfluency parameter, unlike AssemblyAI `disfluencies=true` / Deepgram
  `filler_words=true`).
- **Every** transcript pipeline normalizes prolongations: `'uhhhh'→'uh'`,
  `'theeee'→'the'` (developers.deepgram.com/docs/filler-words; Whisper strips
  fillers in training normalization).

So the two most direct acoustic signatures of a word-finding stall are invisible
in text. Echo v2 runs a **parallel acoustic channel** — raw 16 kHz → Silero VAD →
FillerNet (compact ~136k-param CNN on PodcastFillers, 85,803 labeled clips) + a
rule-based prolongation detector (sustained near-static mel envelope ≥ 600 ms;
Shriberg 1993 / Eklund 2001) — fused with the transcript channel in one stall
detector. Live: the transcript lane prints a clean "the" while the acoustic lane
lights **PROLONGATION**.

**Pre-empt** — *"Why not Deepgram `filler_words=true`?"* That restores filler
**text** only: no prolongations (normalized by design), no prosody, a cloud
dependency on the privacy-critical path, and detection still gated on ASR
finalization latency — our acoustic channel decides on a 125 ms hop.

## Headline numbers (from docs/EVAL.md)
| Metric | Value |
|---|---|
| Acoustic filler F1 (PFSD official test split; gate ≥ 0.75) | **0.933** clip-level (binary uh∪um; precision 0.949, recall 0.918; 4-class accuracy 0.852). Added a binary uh∪um BCE auxiliary loss on top of the 4-class CE (the product trigger fires on that binary decision); chosen from a committed validation-only grid; extra-split leakage found, removed, retrained on episode-disjoint data |
| Stall detection latency | 1300 ms (pause threshold) → **755 ms median** via the acoustic channel (~1.7× earlier; original ≤600 ms gate missed — dominant term 125 ms hop + 800 ms voiced gate; EVAL Table 2) |
| Word serving | **~0 ms** (mechanism bench, n=20) / <300 ms (live e2e cache hit) on prefetch vs ~1.25–1.8 s live LLM |

## The novelty claim (judge-safe — do not overclaim)
> To our knowledge, Echo is the **first real-time, speaker-side co-pilot for
> anomia**: it passively detects a word-finding stall *as it happens* and surfaces
> the intended word fast enough to ride inside the speaker's own sentence. The
> concept was proposed (Purohit et al., CSCW 2023; Kim et al., Findings of EMNLP
> 2024) — **but only as offline studies on pre-transcribed text**. The real-time
> system — live stall detection, low latency, in-conversation delivery — is the
> part no one had built.

Positioning (know these cold):
- **Broca AI Speech** — listens to the *partner*, suggests whole replies. Echo
  recovers *your* word.
- **Voiceitt / Whispp** — convert impaired *speech audio* to clear speech. No word
  prediction.
- **Lingraphica / Tactus / AAC boards** — therapy drills and type-to-talk. Not
  live conversation.
- **Purohit 2023 (CSCW)** — showed ChatGPT can identify the intended word from
  circumlocution (11/12 = 91.67% in their AphasiaBank sample; verified against the
  PDF, cited in `docs/EVAL.md` §5) — *offline, on transcripts*. We built the live
  system they proposed.

## How it works (45s)
```
 mic (laptop, or DJI Mic 2S lav on the speaker)
      transcript channel: mic -> Chrome STT -> words ──┐
                                                        ├─► fused StallDetector ─► prefetch cache / LLM ─► word card + TTS (browser)
      acoustic channel:   mic -> VAD+FillerNet+       ──┘
                          prolongation rule (raw 16kHz PCM)
```
- **StallDetector** (our core engineering): pure-Python, deterministic, per-word
  state machine. Five triggers — **pause** (>1.3 s), **filler** text, **hedge**
  ("the thing"), **filler_acoustic** (FillerNet heard the "um" Chrome deleted),
  **prolongation** — through one shared debounce. Clause-windowed and re-armed;
  prolongation rule 0 false fires on 120 s of running speech; stream-level
  FillerNet false-alarm upper bound 5.8/min before detector gating (conservative;
  disclosed in EVAL).
- **Speculative prefetch**: while fluent, Echo shadow-predicts every 3 content
  words; on a stall the cached word serves at ~0 ms (bench, n=20) / <300 ms (live
  e2e) (`⚡ prefetch`) instead of the ~1.25–1.8 s round-trip. Drift guard +
  stale-completion sequence numbers keep cached words honest.
- **Predictor**: LLM, thinking disabled for latency, JSON-schema constrained,
  few-shot primed; conversation context grounds names ("my sister… *Maria*").
  Provider-abstracted — Claude/DeepSeek/local are one-env-var swaps; EchoLM
  (`docs/ROADMAP.md`) drops in behind the same interface.
- **Delivery — the agency thesis**: the aid must **never talk over the
  speaker**. The word arrives as a large on-screen card, spoken aloud only when
  the speaker confirms it (autospeak off by default); the speaker chooses
  whether to use it.

## What's real (we can show all of this)
- Live dual-channel mode in the browser — speak, stall, get the word; watch the
  transcript lane miss what the acoustic lane catches.
- Live LLM predictions verified e2e — `toaster` from "toast in the…", **`Maria`
  from conversation context**, `Tokyo`.
- Prediction accuracy on a frozen 60-item circumlocution set (freeze protocol
  upheld): **top-1 58/60 (96.7%), top-3 59/60**. Context ablation drops
  proper-noun recovery **19/20 → 2/20** — the context pipe is load-bearing,
  proven. (Team-written set, mechanically scored — construction and rule disclosed
  in `docs/EVAL.md` §5.)
- Entity memory beyond the context window (`backend/entities.py`, on by default):
  on a frozen 20-item long-context set where the target name appeared once, 10–16
  turns back, the predictor recovers **0/20 without** vs **19/20 (95%) with** it
  (EVAL §5). Without it, invented businesses confabulate into famous brands; with
  it, the speaker's own "Whitmore Hardware" comes back.
- System-level dual-channel ablation: the acoustic channel catches **36/39
  (92.3%)** embedded filler stalls strictly before the 1300 ms pause fallback —
  median **760 ms vs 1300 ms**. A speed win, honestly framed: the pause fallback
  still catches the rest, later, and spurious acoustic fires during fluent speech
  are counted separately (4.75/min on this stream, consistent with the 5.8/min
  bound).
- Prefetch serving ~0 ms (bench, n=20) / <300 ms (live e2e) vs ~1.25–1.8 s live.
- Prolongation rule on real PFSD audio: **0 false fires on 120 s** (operating
  point from the committed sweep `eval/tune_stall_thresholds.py`). Detection rate
  on palindrome-looped clips 5/40 (conservative lower bound; deliberate held-vowel
  ground truth awaits the self-recorded set).
- 503 tests + replay harness + 5-case live e2e, all passing.
- A DJI Mic 2S wireless lav on the speaker, measured against the laptop array
  and auto-recognised by the console (`docs/MICROPHONE.md`).

## If a judge presses (answer confidently)
- Word-level timing from browser STT is approximate; the detector is tuned
  conservatively (miss > nag).
- FillerNet trains on podcast speech, not aphasic speech; AphasiaBank eval awaits
  consortium access (faculty request pending).
- Measured noise fragility (EVAL §7): under heavy noise the classifier misses
  rather than false-alarms — precision holds 0.99–1.00 while recall falls to 0.33
  at 5 dB SNR. Failure mode is silence, consistent with miss-over-nag; mic
  proximity (the lav on the speaker) is the mitigation.
- Browser STT is cloud-backed (Chrome); the fully-local path (faster-whisper +
  EchoLM) is architected behind the same interfaces but not yet the demo path.
- Echo is an assistive tool for the anomia profile; an SLP-guided validation study
  is the planned next step.

## Designed against documented communication-support principles

Every choice below traces to a citable principle from the Supported Conversation
for Adults with Aphasia (SCA) framework and aphasia-friendly formatting research.
Clinical validation alongside SLPs is the roadmap.

| Echo decision | Documented principle | Source |
|---|---|---|
| **Pause tolerance before offering** — silent through ordinary fluent pauses, reacts only once sustained silence follows content words (the 1300 ms constant is ours, tuned by the false-positive sweep; SCA's guidance is qualitative) | SCA time/pacing: increase wait time, give the person sufficient time to respond independently | Kagan, A. (1998). Aphasiology 12(9), 816–830; Aphasia Institute "Communicative Access & SCA" (aphasia.ca) |
| **Multimodal delivery** — large-type on-screen card + optional spoken cue, never audio-only | SCA multi-modal technique set; aphasia-friendly formatting (larger font, simpler layout) measurably improves comprehension | Kagan 1998; Rose et al. (2003) Aphasiology 17(10), 947–963; Rose et al. (2011) IJSLP 13(4), 335–347 |
| **Verify, don't correct** — ranked candidates, an explicit "not it" reject, the speaker finishes their own sentence; autospeak off + confirm-to-speak (`docs/DEMO_SCRIPT.md`) | SCA's distinction between acknowledging and revealing competence — the partner verifies rather than supplying the message | Kagan, A. (1995). Topics in Stroke Rehabilitation 2(1), 15–28; Aphasia Institute SCA verification technique (aphasia.ca) |

### Judge Q&A — clinical framing
- *"Is this clinically validated?"* — Built on the established SCA framework
  (Kagan 1998) and aphasia-friendly formatting (Rose 2003, 2011); the table maps
  each decision to its source. Clinical validation with SLPs is the next step.
- *"Why 1300 ms, not shorter?"* — SCA's guidance is to extend wait time. Echo's
  threshold operationalizes "give sufficient time" as a number, tuned so restraint
  wins over nagging (`docs/EVAL.md`).
- *"Why a screen *and* speech, not just speak it?"* — SCA is itself multi-modal,
  and formatting research shows larger type + simpler layout help comprehension.
  Multi-channel delivery mirrors the literature, not decoration.
- *"When Echo guesses wrong?"* — The speaker sees ranked candidates and can reject
  the top with one tap ("not it" promotes #2 and requests a replacement excluding
  rejected words). Echo never overrides — it verifies and waits.
- *"Does Echo ever speak for the person?"* — No, by design: autospeak defaults off
  with confirm-to-speak, so the speaker decides when a word is said aloud.

## Impact stats for slides (cited)
- ">2,000,000 people in the U.S. live with aphasia" — National Aphasia
  Association, aphasia.org/statistics
- "~180,000 new cases each year" — NAA FAQ
- "About 1/3 of stroke survivors have aphasia" — NIDCD (nidcd.nih.gov/health/aphasia)
- Concept precedent: Purohit, Upadhyaya, Holzer, *CSCW '23 Companion*
  (talkbank.org/aphasia/publications/2023/Purohit23.pdf); Kim, Storaï, Hwang,
  *Findings of EMNLP 2024* (aclanthology.org/2024.findings-emnlp.616)
- Filler suppression: github.com/GoogleCloudPlatform/golang-samples/issues/1373;
  developers.deepgram.com/docs/filler-words
- Dataset: PodcastFillers, zenodo.org/records/7121457 (annotations:
  non-commercial research use)
