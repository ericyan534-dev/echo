# Echo — version history

Five versions, each defined by a change in what the system *believes about the
problem*, not by a batch of commits. Every number here is regenerable from a
script in `eval/`, and every version records what it got wrong as well as what
it got right — because in this project the mistakes have been more informative
than the wins.

---

## v1 — transcript-only

**Belief:** a word-finding stall is visible in the transcript. Watch for pauses,
fillers and circumlocutions; when one appears, ask an LLM for the intended word.

Browser `SpeechRecognition` → words over `/ws` → `StallDetector` (pause / filler
/ hedge) → Gemini → word card.

**What it got right:** the shape. Transcript in, stall out, word back, fast
enough to ride inside a sentence. Every later version still has this skeleton.

**What killed it:** the transcript does not contain the evidence.

---

## v2 — dual-channel

**The measurement that forced it:** on 5,044 annotated filler clips, the live
Chrome path fired the filler trigger **0 times**. Recall 0.000. Chrome deletes
"um" and "uh" with no off switch, and every pipeline normalises "theeee" to
"the". A transcript-only detector is structurally blind to the two most direct
signatures of a stall.

**Belief:** if the transcript will not show the stall, listen to the audio in
parallel.

Added a second sensing channel on raw PCM — Silero VAD, FillerNet (a 136k-param
CNN trained on PodcastFillers), and a rule-based prolongation detector — fused
into the same detector. Plus speculative prefetch (shadow-predict during fluent
speech so a stall is served from cache at ~0 ms) and a long-horizon context
overhaul: an append-only `Timeline`, extractive summarisation, and
token-budgeted context assembly.

**What it got right:** the workaround worked, and the context work was real —
out-of-window proper-noun recovery went 0/20 → 19/20 with entity memory.

**What it got wrong, discovered in v3:**
- FillerNet's classes are `uh / um / speech / other`, trained on **fluent
  podcast hosts saying "um"**. There is no class for a *block* — the silent
  struggle to initiate a word, and the strongest evidence a speaker is stuck.
  It could not detect the most important event, and it fired on the speech of
  people who were not struggling at all.
- Every number came from stuttered or fluent podcast speech. **No aphasic
  speech had ever been used to evaluate any component.**

---

## v3 — verbatim transcripts, and the first real aphasic evaluation

**The measurement that forced it:** CrisperWhisper transcribes verbatim *on
purpose*, and the mode is a flag. Same model, same audio, 60 SEP-28k events:

| transcript | dysfluency preserved |
|---|---|
| Chrome (v1/v2 path) | **0.000** (n=5,044) |
| CrisperWhisper `intended` | 0.060 |
| CrisperWhisper `verbatim` | **0.900** |

`intended` is the control that matters: the same weights strip the evidence
exactly the way Chrome does. That is what every consumer recognizer silently
gives you.

**Belief:** the transcript can carry the stall after all. Keep the acoustic
channel — a block is silence and reaches no transcript — but stop working
around a recognizer we can replace.

- **`backend/stt/verbatim.py`** — server-side CrisperWhisper reading the *same
  PCM the acoustic channel already receives*. LocalAgreement-2 streaming (Macháček et al.
  2023) plus a force-commit on VAD silence, because a stall *is* a silence and
  waiting for agreement hands the predictor a sentence missing its last word.
  Both channels now share one sample clock, so `Word.end_ms` and
  `AcousticEvent.at_ms` are finally comparable.
- **`backend/acoustic/stutter.py`** — StutterNet: five independent dysfluency
  types (Block / Prolongation / SoundRep / WordRep / Interjection) trained on
  SEP-28k, weakly supervised with frame-level output.
- **APROCSA** — six people with chronic post-stroke aphasia, clinician-coded in
  CHAT, openly licensed. The first aphasic speech this project has ever been
  measured on. 1,486 participant utterances, 686 coded as word searches.

**Measured, and stated plainly:** on real aphasic speech v3 matched v2's recall
(0.843) with better precision (false alarm 0.704 → 0.611). Better, not
decisively better.

**What it got wrong:**
- Model choice was made on dysfluency preservation and latency. **Accuracy was
  never measured** — see v4.
- Clip-level thresholds were compared against per-frame probabilities, two
  different scales. The detector fired 25 times a minute until it was caught.
- A first RTF diagnosis blamed the streaming window policy for an 8× slowdown
  that was actually GPU contention with a training run.

---

## v4 — better acoustics, real speaker attribution, and the number that reframed the project

Three things landed, and the third matters most.

**1. WavLM replaces the hand-rolled CNN.** Same task, same split, same metrics:

| type | v3 CNN | **v4 WavLM** |
|---|---|---|
| Block | 0.256 | **0.384** |
| SoundRep | 0.306 | **0.574** |
| WordRep | 0.254 | **0.824** |
| ANY | 0.786 | **0.894** |

> **These four numbers cannot be reproduced from disk, and v5 does not ship
> the model that produced them.** The checkpoint was destroyed; a clean rerun
> of the same configuration scores Block 0.371 / ANY 0.893, so the column is
> right to within noise, but it is now a claim about a rerun rather than about
> an artifact. See *A checkpoint this project lost* under v5.

Block +0.129, bootstrap 95% CI [+0.089, +0.169]. Block was the head a
calibrated fit had given *literally zero weight*; the representation was the
bottleneck. Cost: 156 ms per window on a full CPU against a 125 ms hop, so it
runs on the GPU or not at all until distilled.

**2. Speaker attribution by voice.** ECAPA-TDNN embeddings, clustered per
session against a short enrolment: **0.963 accuracy** over 328 segments. The
proximity gate it replaces measured **0.000** bystander suppression. This
reverses an earlier decision to avoid voice enrolment; the evidence for
revisiting it was Echo firing during the clinician's speech on up to 0.465 of
their utterances, and putting their words into the fragment sent to the
predictor.

**3. Corpus recovery.** A verified mirror restored all three shows lost to link
rot plus FluencyBank's audio: labels matched Apple's on **31,908/31,908 rows
across all 12 annotator columns**, audio sample-exact where independently
checkable. 20,124 → 30,962 clips, Block positives +50%, 5 → 9 speaker pools.

### The measurement that reframed everything

Two results refused to make sense together: v2/v3 both scored ~0.84 detection
recall, while recovering the coded evidence into the transcript 0.025 and 0.560
of the time respectively. Both numbers cannot describe detection.

Marker-level scoring settled it. `eval/align_aprocsa.py` found APROCSA already
ships per-word media bullets, giving 2,344 of 2,549 markers real timestamps.
`eval/score_markers.py` scores fires against those instants — against a **timer
control** that fires on a clock using no audio at all.

**Nothing clears the metronome by much.** Best lift is +0.12 recall. And the
reason is the finding: clinician-coded markers occur every **1.2–2.3 seconds**
in aphasic speech. "Is this person word-searching right now" is nearly always
*yes*. No timing metric on this corpus can separate a detector from a clock —
which means detection timing was the wrong target all along, for every version.

So the question became the one the product actually exists to answer: **is the
offered word the word they were reaching for?** CHAT retracings record exactly
that (`spring [//] Christmas`). Measured on 51 scorable events:

| | strict top-3 |
|---|---|
| Echo (verbatim + context) | **3/51** |
| corpus frequency baseline | **3/51** |

At the floor — and hitting *disjoint* events, so uncorrelated with the baseline
rather than tied with it. The one consistent signal is that verbatim beats
intended with no counterexamples (5–0 without context), which is the cleanest
support the v3 thesis has ever had on outcome rather than on detection. n is 51.
*(That 5–0 was re-measured at the improved v5 ASR and did not survive — see the
end of v5.)*

**And the cause, apparently quantified:**

| transcript | WER on aphasic speech |
|---|---|
| `intended` | 0.477 |
| `verbatim` | **0.402** |

Two words in five are wrong. The predictor is handed `"And then [noise] [noise]
and cut off [UM] Cut off And"` when the speaker was reaching for *christmas*.
No prompt, no trigger and no acoustic model recovers from that.

**The binding constraint on this product is ASR accuracy on disordered speech.**
Everything downstream is bounded by it, and that is where v5 goes.

---

## v5 — the ASR layer, and a diagnosis that did not survive its own test

**The measurement that forced it.** v4 ended with two results that would not
reconcile: detection could not beat a metronome, and word prediction sat exactly
at a frequency baseline (3/51 top-3). The obvious suspects — the trigger logic,
the acoustic model, the prompt — had all been improved without moving either.

So the thing nobody had measured got measured: **is the transcript even right?**

| transcript | WER on aphasic speech |
|---|---|
| `intended` (browser-equivalent) | 0.477 |
| `verbatim` (shipped) | 0.402 |

Two words in five wrong. The predictor was being handed `"And then [noise]
[noise] and cut off [UM] Cut off And"` for a speaker reaching for *christmas*,
and answering — sensibly and uselessly — "Berlin, Germany, home" for target
*burlington*. No prompt, trigger or acoustic model recovers from that. **Every
downstream number in v1–v4 was bounded by this, and the ASR had been selected
on dysfluency preservation and latency, never on accuracy.** That was the
diagnosis. Improving the number and re-running the downstream benchmark is how
it was tested; it did not hold, and the test is at the end of this section.

### What did not work, recorded because the search cost real time

- **Model size is irrelevant.** turbo 0.402 / large 0.408 / medium 0.408, all
  within noise, against a per-speaker spread of 0.33–0.56. Speaker variation
  dominates model choice entirely.
- **The commit threshold was not the cause.** The prediction was that the
  280 ms force-commit chopped aphasic hesitations in half, since word-search
  pauses have a median of 672 ms. Every increase made it worse: 900 → 0.426,
  1000 → 0.419, 1500 → 0.456, 2500 → 0.444.
- **Window size is flat**: reset_window_s 3.5 → 0.402, 12.0 → 0.413.

### A correction to a number this project published

Streaming was first reported as costing 11.4 WER points against offline
decoding. That was partly a measurement artifact of the scorer, which assigns
hypothesis words to reference utterances **by timestamp** — and streaming words
carry VAD-derived times while offline words carry DTW times. Scoring the same
words with utterance boundaries removed puts the real gap at **7.6 points**
(0.347 vs 0.271), not 11.4. Large and real, but overstated when first claimed.

### What shipped

| | WER | boundary-free | commit lag |
|---|---|---|---|
| v4 defaults | 0.4024 | 0.3474 | 200 ms |
| **v5 defaults** | **0.3752** | **0.3215** | 400 ms |

`silence_commit_ms=700` **and** `word_time_policy="incremental"`. Neither half
works alone — 0.398 and 0.383 respectively — and a sweep of either axis in
isolation concludes both are dead ends. The improvement is only visible in
combination, which is a lesson about the search, not just the parameter.

What carried it was **word-time assignment**, the thing initially dismissed as
a scoring artifact. It was both: part of the gap really was mis-assignment in
the scorer, *and* fixing how the stream assigns word times is a genuine product
improvement, because Echo windows the predictor's fragment by time.

Re-timed live on 180 s of APROCSA (`eval/bench_asr_stream.py`): commit lag
median 200 → 350 ms and p90 300 → 700 ms, still less than half the 1300 ms
stall budget, and streaming RTF 0.57 → 0.59 — measured while another job held
the GPU, so both figures are upper bounds, and the comparison is paired under
identical contention. One real cost showed up there: a window is only retired
inside silence, so requiring a longer silence means the buffer twice reached
its hard cap and took the last-resort turn break that can clip a word at the
seam. Never at 280 ms, twice in 180 s at 700 ms.

### And it did not propagate

The v5 diagnosis is only worth what the downstream metric says, so the product
benchmark was re-run on ASR streams regenerated at the new defaults. 46 of the
51 fragments the predictor sees changed, so the improvement did reach the
prompt:

| strict top-3 | v4 ASR (0.402) | v5 ASR (0.375) |
|---|---|---|
| Echo (verbatim + context) | 3/51 | **3/51** |
| corpus frequency baseline | 3/51 | 3/51 |

**Unchanged, still exactly at the frequency floor.** The looser span metrics
moved the wrong way (top-3-span 11/51 → 9/51, top-1-span 7/51 → 5/51). And v4's
one positive signal did not survive: verbatim beat intended 5–0 on discordant
events without context, and at the better ASR it is 6–3, p = 0.51. That was
noise all along. (The predictor is sampled, not deterministic, so the span
movements are within re-run noise; the strict top-3 is the number to read, and
it did not move at all. The two frequency baselines are deterministic and are
identical across both runs, which is the control that says the harness itself
did not shift.)

Detection moved about as little (`eval/run_aphasia_eval.py`, shipped arm):
recall 0.843 → 0.818, false-alarm 0.611 → 0.648, and the one clear gain,
partner-speech fires 0.247 → 0.159.

So **"ASR accuracy is the binding constraint" is not supported by the only
experiment that could support it.** 2.7 WER points bought nothing downstream.
n is 51 and one step of one knob is not a curve — the honest reading is that
the v4/v5 diagnosis is unproven, and the next test is a much larger ASR delta
(offline is 0.288) rather than another parameter, or another prompt.

### A caveat this project repeated for three versions, now retracted

Every episode-disjoint number since v3 was labelled "optimistic — leaks the
podcast's recurring host". It was a reasonable prior and it is measurably
wrong. Trained with a show's host seen versus unseen, scoring identical test
clips, the mean gap across three held-out shows is **Block +0.003, ANY
−0.004**, no per-show gap above 0.08, and the sign is inconsistent.

Getting there needed two corrections of its own. Comparing a nine-show mixture
against one show confounds host exposure with show difficulty, and raw AP is
prevalence-sensitive — Block prevalence ranges 0.09–0.15 by show and ANY
0.35–0.65, so HeStutters' ANY 0.904 "beating" the reference 0.882 was an
artifact; normalised to lift over chance it is 1.52 against 1.70, i.e.
*below*. What actually moves the number is which show you test on: the same
model scores Block 0.430 on StutterTalk and 0.222 on StutteringIsCool. Show
difficulty dominates host identity by roughly an order of magnitude.

It survived three versions because it sounded appropriately humble, not because
anyone had measured it. `eval/eval_stutter_ssl_hostleak.py` measures it.

And the corpus expansion that v4 celebrated — +54% clips, +50% Block positives,
4 new speaker pools — bought **+0.001 AP on Block** on an identical clean test
set. Whatever limits Block detection is not data volume. Measuring that needed
care of its own: 1,380 of the 4,411 new-test clips sit in the old checkpoint's
training set, and scoring there reads Block 0.410 against 0.304 clean.

### A checkpoint this project lost

`models/stutternet_ssl.pt` — the model behind v4's published Block 0.384 /
ANY 0.894 — was overwritten by a `--limit 200 --epochs 1` smoke run that was
checking whether multi-source loading worked. The trainer's default output
path is the shipped path, the smoke run took it, and `*.pt` is gitignored, so
there was no copy anywhere. The file now embeds its own confession:
`n_train=200, epochs=1, n_unfreeze=0, Block 0.2127`.

What survives is a clean rerun of the same configuration,
`stutternet_ssl_v1repro.pt`, at Block 0.371 / ANY 0.893 — within noise of the
published column, which is the only reason this is a lost artifact rather than
a lost result. It was found because the retrain agent checked the checkpoint's
embedded metadata instead of trusting the filename.

Two fixes, both in `scripts/train_stutter_ssl.py`: a truncated run
(`--limit`, or fewer than two epochs) now refuses to write to a default
checkpoint name and demands an explicit `--out`; and `trained_on` was
hardcoded to *"SEP-28k (74% episode subset, 5 of 8 shows)"* while ignoring
`--extra`, so every v2 checkpoint had been misdescribing its own training
data. It is now derived from what was actually loaded. **A model file that
cannot say what it was trained on is how the first problem stayed invisible.**

### The split moved under the corpus

The leak above — 1,380 clips — was not bad luck. `make_splits` drew one
permutation per show from a **single shared** `RandomState(13)`, so a show's
assignment depended on every show processed before it. Growing one show
changed its permutation length, shifted the stream, and reshuffled every show
downstream. On the real corpus, 18 episodes crossed from train into test,
including episodes of shows that had not changed at all.

`split_version="stable"` assigns each episode by hashing `(seed, show,
episode)` and thresholding, so membership is a function of the episode's own
identity: growth can add to a split but never move across one. Measured on the
real corpus, legacy moves 18 episodes and stable moves 0.
`legacy` stays the default, because every published number was produced under
it and silently changing how they were made would be worse than the bug.
`tests/test_split_stability.py` pins both behaviours, including the one case
"stable" cannot fix — a show with four episodes cannot be both stratified and
growth-invariant, and per-show test coverage was chosen over invariance for
the two shows that small.

### What v5 ships for acoustics

`models/stutternet_ssl_v2.pt`, at **Block 0.325 / ANY 0.882** on its own
episode-disjoint test set (n=4411).

That headline is *lower* than v4's 0.384/0.894 and it is not a regression:
the two are different test sets, and the v2 set is larger and harder. On the
2,833 clips that are clean for both models the honest comparison is a wash —
Block +0.001 [−0.027, +0.028], with real gains only on Interjection (+0.030
[+0.015, +0.046]) and ANY (+0.012 [+0.001, +0.022]).

It ships anyway, for three reasons that are not accuracy: it is never worse,
it is trained on 9 speaker pools rather than 5 (the relevant prior for a user
nobody has heard before), and the alternative no longer exists on disk.

One operational number worth stating plainly: at the 2% clean-fire budget the
calibration targets, **the Block head recalls 0.151**. Blocks are the
dysfluency this product most wants to catch, and at the operating point it
actually runs at, it catches one in seven.

### Fixed-chunk streaming: right conclusion, wrong reason

The strongest structural lead in the ASR work was that chunked *offline*
decoding (0.319/0.300) sits close to full offline (0.288/0.271), suggesting
that always decoding a long fixed span — regardless of where the speaker's
pauses fall — is most of what offline is buying. Made streaming, the full
6-speaker grid says it is:

| config | WER | concat | policy lag | ready | |
|---|---|---|---|---|---|
| `b10 s2 agree` | **0.301** | **0.286** | 3,180 ms | 0.03 | closes 85% of the gap |
| `b8 s2 agree` | 0.307 | 0.288 | 3,160 ms | 0.02 | |
| `b10 s2 now` | 0.352 | 0.336 | **920 ms** | 0.87† | fits the budget |
| shipped | 0.375 | **0.322** | 400 ms | — | |

† at zero decode cost; 0.74 at a 500 ms decode.

**The accuracy is real and unreachable.** `agree` gets there by refusing to
commit a word until a later decode of the same audio agrees, which takes 3.2 s
of policy-intrinsic wait against a 1,300 ms stall budget. No amount of model
engineering reduces that — it is the definition of the policy.

**And the one member that fits the budget is not an improvement.** `now` at
stride 2 waits only 920 ms, and its apparent hopelessness in the first report
was a decode-wall artifact (roughly 1.0–1.2 s of it is turbo's word-timestamp
DTW, which the live path does not even run). But it scores 0.336 boundary-free
against the shipped 0.322: it buys 2.3 WER points and *loses* 1.4 points once
utterance boundaries are removed, so its gain is better word timestamps, not
better recognition.

**A correction to how this was first closed.** The finding was published from
the buffer=8 row alone, justified by "larger buffers can only be worse on lag".
That reason is wrong. Commit lag has two components and only one is set by the
buffer: `scroll` goes 6,720 → 8,720 ms across buffer 8→10, while `agree`
(3,160 → 3,180) and `now` (920 → 920) are flat, because for those the stride
sets the lag. Buffer=12 could not have changed anything — but not for the
reason given, and the first claim was right by accident.

The conclusion stands: **the offline advantage IS the latency.** Offline is
accurate precisely because it can wait for context a live system does not have
yet. The remaining 0.322 → 0.271 gap is not reachable by deferring commits.

### Decoder context: a number that was wrong, and a lead that is closed

`eval/asr_context_model.py` had been built on the claim that CrisperWhisper's
continuation prompt — the `<ctx>...<ectx>` slot the checkpoint was trained
with, which `model.transcribe()` never fills on short windows — was worth
**0.047 WER** of the offline advantage. A prompt costs no latency, so if that
transferred it was the large-and-streamable gain the project had been looking
for after 2.7 points bought nothing.

**The claim does not reproduce, in either direction.**

| arm | WER | concat | hyp words | context echoes |
|---|---|---|---|---|
| offline, stride 26, context ON (ships) | **0.2876** | 0.2707 | 1,605 | 0 |
| offline, stride 26, context OFF | 0.3909 | 0.3698 | 1,809 | **32 of 72** |
| offline, stride 30, context ON | 0.2918 | 0.2779 | 1,588 | 0 |
| offline, stride 30, context OFF | **0.2792** | 0.2677 | 1,604 | 0 |

The naive ablation is **0.103, not 0.047** — and it is a *stitching* number,
not a recognition one. The strategy decodes 30 s chunks at a 26 s stride, and
the prompt is the only thing stopping the next chunk from transcribing the
shared 4 s again: remove it and 32 of 72 chunks open by echoing their own
context, inflating the hypothesis from 1,605 words to 1,809. Remove the
*overlap* instead — which is Echo's actual regime — and the sign flips: 0.2792
without the prompt against 0.2918 with it, worse on four of six participants
and better on none.

**Streaming confirms it.** Wired in behind `context_prompt=False`, with context
drawn only from words committed out of retired audio so no hypothesis can lose
its committed prefix:

| arm | WER | concat | hyp words |
|---|---|---|---|
| shipped (no context) | **0.3752** | 0.3215 | 1,532 |
| with context | 0.4350 | 0.3758 | 1,458 |

Six WER points worse, on five of six speakers.

**And the failure was not the one the harness was built to catch.** Prompt
conditioning is famous for making Whisper echo its prompt or loop, so the
bench counts both separately from WER. Echoes: zero in both arms. Loops went
*down*, 57 to 43. The actual mechanism is **deletion**: at training the
context words are re-heard at the head of the next chunk, so skipping them is
correct behaviour — but Echo's windows do not overlap, so the skip deletes
speech nobody transcribed.

    context   "later I couldn't walk for a [UM] I think about four months"
    no ctx    "Three or four months [UH] but [UM]"
    with ctx  "[UH] but [UM]"

Hypothesis word count 1,532 → 1,458. The degeneracy counters would never have
found this; the hypothesis word count did, which is why it is in the table.

So: the offline pass is not accurate because of the prompt. It is accurate
because it **overlaps its chunks and can wait**, and the prompt is the
bookkeeping that makes the overlap work. A stream that cannot overlap gets
nothing from the prompt and loses six points to it. The only surviving version
of the idea is to give the stream a real overlap so the context describes
audio the model can actually hear — which breaks the "every window boundary
sits in silence" invariant and re-opens the double-emission bug. That is a
redesign, not a parameter.

### One published table now predates its own code

`min_gap_ms` is documented as "minimum time between two fires, across all
triggers", and exists because a served word takes 1.5–2 s to arrive and be
read. It was not surviving a turn boundary: `reset()` cleared the episode list
that `_emit` reads to enforce the refractory, and `TurnEnd` calls `reset()`.
Measured consequence — a pause stall fires at 2,500 ms, a turn ends, and a
filler stall fires at 3,650 ms: **two suggestions 1,150 ms apart under a
4,000 ms refractory.**

Fixing it changes what the detector does whenever the refractory is non-zero.
Replaying the cached APROCSA streams through the corrected detector:

| `min_gap_ms` | fires before | fires after | recall | false alarm |
|---|---|---|---|---|
| 0 | 585 | 585 | unchanged | unchanged |
| 1000 | 536 | 536 | unchanged | unchanged |
| 2000 | 449 | 448 | 0.8491 → 0.8491 | 0.7130 → 0.7130 |
| 3000 | 371 | 368 | 0.8365 → 0.8365 | 0.6574 → 0.6574 |
| **4000 (shipped)** | **314** | **304** | **0.7736 → 0.7484** | **0.5556 → 0.5370** |
| 6000 | 247 | 235 | 0.6604 → 0.6604 | 0.4907 → 0.4630 |

**Every headline number in this repo is measured at `min_gap_ms=0` and is
bit-identical.** What is stale is the `refractory_sweep` block in
`eval/results/aphasia_eval.json` and the corresponding sweeps in
`aphasia_tuning.json`, `stack_comparison.json` and `marker_scoring.json`, for
rows with a non-zero refractory.

Those rows have been left exactly as they were measured rather than quietly
regenerated. They are correct measurements of the code that produced them, and
they are labelled here as predating the fix. Re-running the sweep is a
deliberate act for v6, not a silent tidy-up during a seal.

### Still open

Offline one-pass decoding remains the ceiling at **0.2876 / 0.2707**, and
chunked offline (10 s chunks, 8 s stride) sits at 0.3190 / 0.2997 — close
enough to suggest that always decoding a long fixed span, regardless of where
the pauses fall, is what offline is really buying. Whether that survives being
made streaming inside the 1300 ms latency budget is the open question.

---

## v5 sealed — what this version actually is

Sealed after three adversarial reviews (runtime correctness, repository
hygiene, test-suite mutation). 435 tests pass. Every number below regenerates
from a committed script; the one table that does not is named above.

### What got better, measured

| | v4 | v5 | |
|---|---|---|---|
| ASR WER, aphasic speech | 0.402 | **0.375** | better |
| boundary-free WER | 0.347 | **0.322** | better |
| dysfluency reaches the transcript | 0.000 | 0.900 | better |
| evidence recovery per utterance | 0.025 | 0.560 | better |
| acoustic ANY AP | 0.786 | 0.882 | better |
| Block AP (clip-level) | 0.256 | 0.325 | better |
| speaker attribution | 0.000 | 0.963 | better |
| corpus | 20k clips / 5 pools | 31k / 9 pools | better |
| SSL acoustic model reachable from the product | no | yes | fixed |
| acoustic feed on the event loop | 438 ms/hop | 10.0 ms/hop | fixed |

### What did not get better, and is not hidden

| | status |
|---|---|
| word prediction, strict top-3 | **3/51 — exactly the frequency baseline** |
| detection vs a metronome | **no lift beyond +0.12; the metric is saturated** |
| Block recall at the 2% fire budget | **0.151 — one block in seven** |

Word prediction is the metric the product exists for, and in v5 it resisted a
better acoustic model, a better ASR, a better trigger, a larger corpus, and
decoder context. It sat at the frequency floor through all five.

### Claims this version retracted

- *"ASR accuracy is the binding constraint on everything downstream."* Tested
  by cutting WER 0.402 → 0.375 and re-running the product benchmark on
  regenerated streams. 46 of 51 fragments changed; strict top-3 did not move at
  all. Not refuted — n is 51 and one step of one knob is not a curve — but
  unproven, and the incremental version of the plan is ruled out.
- *"Episode-disjoint numbers are optimistic, they leak the podcast's recurring
  host."* Carried since v3 on plausibility. Measured across three shows on
  identical clips: mean gap Block +0.003, ANY −0.004, sign inconsistent.
- *"The decoder continuation prompt is worth 0.047 WER."* It is 0.103, it is a
  stitching artifact rather than a recognition gain, and in Echo's
  non-overlapping regime the sign flips.
- *"Verbatim beats intended 5–0 without context."* 6–3, p = 0.51. Noise.

### Defects this version shipped and then fixed

Eight, from adversarial review rather than from use. The two that mattered:
the duplicate-word bug had returned by a second route (a silence-armed forced
commit was never cancelled when speech resumed, retiring the window
mid-utterance), and the pause trigger was measuring our own commit lag instead
of the speaker — firing on a 200 ms pause it computed as 1,374 ms, which is
the "it nags" failure the whole trigger design exists to avoid.

### What the test count is worth

435 pass. Mutation testing found the LocalAgreement core, split stability and
the speaker gate genuinely resistant — every mutation caught. It also found
that the two tests written to guard the newest incidents were the weakest in
the suite: one was a source grep defeated by moving the literal into a comment,
and one had had its fixture tuned away from the seam it was named for. Both are
fixed, and the acoustic decision path — which production always takes and no
test had ever executed — now has 21 tests against real checkpoints.

The number is worth more than most such numbers and less than it was being sold
for. That is the honest version.

---

## v6 — what has to change

Ordered by what a user would feel.

1. **Decide whether word prediction is the right target.** Five independent
   improvements have failed to move it. The two live hypotheses: the fragment
   does not contain the information (where the speaker reached for
   *christmas*, Echo held `"Alright, It was"` — no decoder recovers that), or
   the target has to come from the conversation's topic rather than its syntax.
   The second has never been tested and is the cheapest real experiment left.
2. **Frame-level Block recall, 0.151 at a 2% fire budget.** Clip-level AP hid
   this for three versions. If the acoustic channel is to carry any weight,
   this is the number to move — not AP.
3. **Re-run the refractory sweep** against the corrected `min_gap_ms`, or
   revert the fix. Documented above; deliberately left for a decision rather
   than folded into the seal.
4. **A much larger ASR delta, or none at all.** 2.7 points bought nothing.
   Offline reaches 0.288 and is measurably unreachable live (5,339 ms commit
   lag). The only surviving idea is giving the stream real overlap so decoder
   context describes audio the model can hear — which breaks the
   "every window boundary sits in silence" invariant and re-opens the
   double-emission bug. A redesign, not a parameter.
5. **Two coverage holes remain**: `EchoSession.handle_audio_events` and the
   `/ws/audio` lifecycle have no test, and they are the join point of the whole
   live path.
6. **Distil the SSL acoustic model** or accept it is GPU-only. 95.5M params,
   156 ms per window on CPU against a 125 ms hop, which is why the default is
   still the CNN.

## v6 in progress — the console became a terminal, the mic became a measured input, and the model question closed

Sealed v5 said what had to change. This increment does not close that list;
it changes the surface the judges see, adds the first commercial input, and
answers three questions with numbers before any of the v6 items are touched.

### What shipped

- **The console is an operator terminal, not a demo harness.** A navigation
  rail (Console / Sessions / Settings, `alt 1..3`), an input tree listing
  the host's audio inputs with the recognised one marked, a session bar, an
  inspector with an audio-source sheet and an event log, a Sessions page
  (every served prediction, JSON export), and a Settings page that reads the
  server runtime from `/healthz` and `/api/config`. Every element ID
  `app.js` binds to is preserved; `frontend/console.js` owns the shell.
- **DJI Mic 2S as the on-speaker audio tier.** `backend/audio_sources.py`
  carries microphone profiles; `GET /api/audio/sources` and
  `POST /api/audio/source` let the console report what is feeding the
  server, and `/healthz` shows it. The console auto-prefers the recognised
  lav and passes its capture constraints (DJI: AEC, NS and AGC off, one
  channel). `scripts/mic_probe.py` reproduces every DJI number in
  `MICROPHONE.md`. Measured, both mics open at once, music as the program:
  DJI level at the capsule **+4.8 / +5.5 dB** over the laptop array, but the
  array's idle floor is 12 dB lower, so the SNR proxy favours the array;
  the DJI's advantage is unprocessed level on the speaker, not a quieter
  floor. No accuracy claim is made for it: nothing in `eval/` scores
  unlabeled program audio.
- **Model question closed.** `eval/compare_gemini_models.py`, 60 frozen
  items x 3 reps x 3 models x 3 thinking settings, 1,620 calls, 0 errors:
  every model lands at 58-59/60 top-1 with thinking off; p95 latency is
  1247 ms (3.5-flash) vs 1750 ms (3.7) vs 8847 ms (3.8). 3.7 and 3.8 do not
  honour `thinking_budget=0` and leak ~68 thought tokens per call, which
  twice exhausted the 256-token cap. **Stay on 3.5-flash.** Table and
  interpretation: `docs/V6_RESEARCH.md` §3.
- **Streaming transport, behind `GEMINI_STREAM` (default off).** n=16 per
  arm, interleaved, identical top words: 1047 -> 876 ms median. Regenerate
  with `eval/bench_predictor_latency.py`; results in
  `eval/results/predictor_latency_bench.json`.
- **Research record.** `docs/V6_RESEARCH.md`: landscape with citations,
  where Echo stands, eight ranked proposals. Two measured facts it adds:
  the published 755 ms acoustic latency describes FillerNet, while the
  shipped StutterNet CNN fires at 531 ms median on the same construction at
  a recall cost; and prefetch served one or two content words early costs
  58/60 -> 45/60 or 50/60 top-1 on the frozen set, so the "0 ms prefetch"
  row carries an accuracy cost the table does not show.

### What did not get better

None of the six v6 items above is closed. Word prediction on aphasic
speech is still at its floor; frame-level Block recall is unchanged; the
refractory sweep is still owed; the SSL model is still GPU-only. The DJI
adds level, not a measured detection gain. The console shows more, and
claims nothing it did not before.

### Tests

475 passing (435 at the v5 seal; +33 audio-source, +7 streaming). The two
coverage holes named at the seal are still open.

---

### v6.1 -- DeepSeek, SSL sensitivity, offline, and the GX10 appliance

A follow-on increment, still measured, nothing in `docs/EVAL.md` changed:

- **DeepSeek `deepseek-flash`** added as a predictor and made the live default:
  **59/60 top-1 (98.3%), p95 1023 ms, 0 failures** on the frozen set
  (`eval/results/prediction_eval_deepseek.json`), ~18% faster than Gemini. It is
  a reasoning model, so the provider pins `reasoning_effort:"none"`.
- **Detection sensitivity:** the SSL StutterNet on GPU catches **0.554 recall vs
  the CNN's 0.396** at a fraction of the false-fire cost that threshold-lowering
  would inflict (`eval/results/sensitivity_sweep.json`,
  [`DETECTION_TUNING.md`](DETECTION_TUNING.md)); shipped via
  `STUTTER_BACKEND=ssl ACOUSTIC_DEVICE=cuda`. Stronger real-mic training
  augmentation is staged in `scripts/train_stutter.py` for a GX10 retrain.
- **Offline mode** ([`DEPLOYMENT.md`](DEPLOYMENT.md)): the full loop runs with no
  internet on a local LLM -- 95% top-1, prefetch hiding the local latency.
- **Private appliance** ([`DEPLOYMENT.md`](DEPLOYMENT.md)): Echo runs entirely on one ASUS
  Ascent GX10, so no audio or text leaves the box.

### v6.2 -- "detected but no word": two root causes, found in rehearsal

Rehearsal symptom: the stall lit up on the console and nothing was served,
seven rounds in a row. The server log and a 7-stall probe over the real `/ws`
protocol found two independent faults, each fixed with a failing test first
(commit `26a3b4b`; `tests/test_clock_domains.py`, `tests/test_failover_predictor.py`):

- **Two clocks compared as one.** In browser-ASR mode the words and silence
  ticks carry the browser clock (ms since *Start*, restarted at 0 on every
  Start and reload) while acoustic events carry the session's audio clock, and
  `StallDetector` keeps a single `_last_fire_ms` across all triggers for the 4 s
  min-gap. After any fire, a reload put the next stall "400 ms after" the last
  by timestamp -- minutes later in the room -- and it was suppressed; an
  acoustic fire on the audio clock muted every transcript trigger for as long as
  the clocks differed. `EchoSession` now owns **one clock**; browser timestamps
  are lifted onto it at the `/ws` boundary, re-pinned when the browser clock
  steps backwards. The Simulate tab's synthetic clock is untouched.
- **A provider outage was an empty card.** DeepSeek answered HTTP 503 / timed
  out for 30+ consecutive stalls; the provider "never raises", so each became
  `[]`. `FailoverPredictor` now races DeepSeek against Gemini (hedge at 1.2 s,
  hand-off on empty/error, bounded at 3.5 s) via `PREDICTOR_FALLBACKS`;
  `/healthz` shows the chain.
- **`scripts/run_live.ps1`** pins the whole live stack so a restart cannot drop
  a knob. Verified: **7/7** cold stalls at human cadence produced a word
  (bus / watering can / grill ...), across simulated reloads. Two stalls
  inside 4 s of each other are still one suggestion -- that is the min-gap,
  by design.

## Where each claim can be re-run

| claim | script |
|---|---|
| dysfluency preserved by transcript | `eval/bench_asr_models.py` |
| ASR accuracy on aphasic speech | `eval/bench_asr_aphasia_wer.py` |
| streaming RTF and commit lag | `eval/bench_asr_stream.py` |
| acoustic model, per type | `scripts/train_stutter.py`, `scripts/train_stutter_ssl.py` |
| frame operating point | `eval/calibrate_stutter_frames.py` |
| detection on aphasic speech | `eval/run_aphasia_eval.py` |
| detection vs a metronome | `eval/score_markers.py` |
| word prediction on aphasic speech | `eval/run_aphasia_prediction.py` |
| ASR streaming vs offline, policy sweep | `eval/bench_asr_offline_wer.py` |
| aphasic pause distribution | `eval/fit_aphasia_pause.py` |
| speaker attribution | `eval/make_speaker_map.py` |
| fixed-chunk streaming grid | `eval/bench_asr_fixed_chunk_stream.py` |
| decoder context prompt | `eval/bench_asr_context_prompt.py` |
| corpus growth and host leakage | `eval/eval_stutter_ssl_matrix.py`, `eval/eval_stutter_ssl_hostleak.py` |
| split stability under corpus growth | `tests/test_split_stability.py` |
| Gemini model and thinking-setting comparison | `eval/compare_gemini_models.py` |
| predictor transport latency (`GEMINI_STREAM`) | `eval/bench_predictor_latency.py` |
| DJI Mic 2S capture, floor, pipeline timing | `scripts/mic_probe.py` |

Corpus provenance, and what each corpus forbids claiming, is in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).
