# Echo v2 — Architecture

Echo detects the moment a person with aphasia stalls hunting for a word and
offers the predicted word fast enough to ride inside their own sentence. v1 did
this from the transcript alone; v2 adds a parallel **acoustic channel**.

## The dual-channel thesis

**Consumer ASR erases exactly the speech phenomena that matter for aphasia.** A
word-finding stall announces itself as fillers ("um", "uh"), prolongations
("theeee…"), and pauses — transcript pipelines destroy the first two:

- **Chrome's SpeechRecognition suppresses "um/uh" entirely** — Google STT strips
  "Um" with no off switch
  (github.com/GoogleCloudPlatform/golang-samples/issues/1373); the Web Speech /
  Google STT APIs expose **no disfluency parameter**, unlike AssemblyAI
  (`disfluencies=true`) and Deepgram (`filler_words=true`).
- **All transcript pipelines normalize prolongations** — Deepgram documents
  `'uhhhh'→'uh'` even with filler words on
  (developers.deepgram.com/docs/filler-words); Whisper's training-text
  normalization strips fillers.

So a transcript-only detector is blind to the two most direct signatures. Echo
runs **two sensing channels in parallel** and fuses them:

1. **Transcript channel** — browser `SpeechRecognition` words → `/ws` (JSON).
   Catches pauses, surviving filler text, hedges ("the thing", "what's it called").
2. **Acoustic channel** — raw 16 kHz PCM16 from the browser `AudioWorklet`
   (laptop mic or the DJI Mic 2S lav on the speaker) → `/ws/audio` (binary).
   Catches fillers Chrome deleted (FillerNet) and
   prolongations no transcript can represent (rule-based).

> *"Why not Deepgram `filler_words=true`?"* — Restores filler **text** only: no
> prolongations (normalized by design), no prosody, a cloud dependency on the
> privacy-critical path, and detection gated on ASR finalization latency — our
> acoustic channel decides on a 125 ms hop.

## v3 — the transcript channel stops lying

**CrisperWhisper transcribes verbatim on purpose**, and the mode is controllable.
Measured on 60 annotated SEP-28k events, same model, same audio
(`eval/bench_asr_models.py`):

| transcript source | dysfluency preserved |
|---|---|
| CrisperWhisper `verbatim` (turbo) | **0.900** |
| CrisperWhisper `intended` | 0.060 |
| Chrome SpeechRecognition | **0.000** (n=5,044, measured) |

`intended` is the important row: same weights, same audio, stripping the evidence
exactly the way Chrome does — the difference is that here it's a flag. So v3 sets
`ASR_PROVIDER=crisper` and transcribes **the same PCM the worklet already
sends**. Consequences: (1) the acoustic channel stays — a block is silence and
never reaches text; (2) both channels finally share one clock (`Word.end_ms` and
`AcousticEvent.at_ms` are the same counter, so fusing is sound); (3) the pause
trigger measures the speaker, not the pipeline (silence ticks are VAD-gated).

Model choice measured, not assumed: `turbo` wins on both axes — 0.900 preserved
at 327 ms — and `large` is *worse* at preserving dysfluency (0.850), a bigger
model's stronger normalization prior. Word timestamps cost a flat ~1.4 s on turbo
and the live path reads exactly one (the last committed word's end, which the VAD
already knows), so they are off.

**Streaming a non-streaming model.** Echo uses LocalAgreement-2 (Macháček et al.
2023): transcribe a growing buffer every `step_ms`, commit only the prefix two
hypotheses agree on, plus a force-commit whenever the VAD reports silence —
because a stall *is* a silence and waiting for agreement would hand the predictor
a sentence missing its last word. See `backend/stt/verbatim.py`.

**StutterNet replaces FillerNet's task.** FillerNet's classes are
`["uh","um","speech","other"]` (fluent podcast hosts) — **no class for a block**
(the silent struggle to initiate a word, the strongest sign a speaker is stuck),
and it fires on interjections fluent speakers produce constantly. **StutterNet**
(`backend/acoustic/stutter.py`) uses five independent labels (Block /
Prolongation / SoundRep / WordRep / Interjection) trained on SEP-28k — people who
actually stutter. Multi-label is forced by the data (3,009 clips carry ≥2 types at
≥2/3 agreement), weakly-supervised with frame-level output (a 3 s clip verdict
localizes to ±1.5 s while the whole stall-to-word budget is ~1.5 s).

## Data flow

```
                         ┌──────────────────────── browser (Chrome) ───────────────────────┐
 🎤 mic ──┬─► SpeechRecognition ──► words {text, ms, is_final} ──► /ws (JSON) ─────────┐    │
 (laptop, └─► getUserMedia ─► AudioWorklet @16kHz ─► PCM16 frames ─► /ws/audio (bin) ──┤    │
  or DJI Mic 2S lav on the speaker)                                                    │    │
        ┌── backend ────────────────────────────────────────────────────────────────── ▼ ──┐
        │  /ws/audio ─► AcousticStream: Silero VAD (512-samp) · ProlongationTracker (50ms)   │
        │              · FillerNet (1s window / 125ms hop) ─► AcousticEvent{filler|prolong}  │
        │  /ws ─► Word / SilenceTick / TurnEnd / context                                     │
        │        ┌────────── StallDetector (fused) ──────────┐                               │
        │        │ pause · filler · hedge · filler_acoustic · │                               │
        │        │ prolongation — shared debounce, re-arm     │                               │
        │        └────────────────┬───────────────────────────┘  StallEvent{fragment,trigger}│
        │        EchoPipeline ── prefetch cache hit? ──yes──► serve cached (~0 ms, ⚡)        │
        │             │           └──no──► WordPredictor (~1.25–1.8s measured)                │
        │        Prediction{candidates, served, latency_ms}                                   │
        │             └─► /ws ─► UI word cards + dual-channel timeline + TTS                  │
        └───────────────────────────────────────────────────────────────────────────────────┘
```

Everything joins in one `EchoSession` (`backend/session.py`) so transcript
triggers, acoustic triggers, prefetch, and delivery stay coherent.

## WebSocket protocols

Full field-by-field reference (every message, default, example, edge case) in
[`PROTOCOL.md`](PROTOCOL.md). Two sockets: `/ws` (UI JSON: `word`, `silence`,
`turn_end`, `context`, `reject`, `ping` in; `prediction`, `acoustic_event`,
`transcript_word`, `pong` out) and `/ws/audio` (client→server binary PCM16
16 kHz mono, each connection its own `AcousticStream`; only one connection at a
time transcribes under `crisper`).

**Card interaction** (`frontend/app.js`): a served prediction renders as one
dominant word + up to two chips + a "not it" button, **never auto-speaking by
default** (`autospeak` off — the speaker taps a word or checks the box; see
`DEMO_SCRIPT.md` confirm-to-speak). "not it" sends `reject` and promotes the next
candidate locally while the replacement is in flight; four new final words with no
accept implicitly dismisses a stale card. Card lifecycle fires `CustomEvent`s
(`echo:card-rendered/accepted/rejected`) so the latency counter and tally hook in
without coupling.

**Delivery thesis: the aid must never talk over the speaker.** The word is a
glanceable card; speech is confirm-to-speak and off by default, so the cue is
non-interruptive and the speaker keeps agency.

## StallDetector (`backend/stall_detector.py`)

Pure-Python, deterministic, per-word state machine. Five triggers, one debounce:

| Trigger | Source | Condition |
|---|---|---|
| `pause` | transcript timing | mid-utterance silence ≥ 1300 ms (`STALL_PAUSE_MS`) with ≥ 2 content words |
| `filler` | transcript text | "um/uh/er/…" after ≥ 1 content word |
| `hedge` | transcript text | ends in a circumlocution phrase ("the thing", "what's it called", …) |
| `filler_acoustic` | acoustic | FillerNet hears uh/um the transcript dropped |
| `prolongation` | acoustic | sustained near-static vowel |

Shared mechanics (all through one `_emit`): **debounce/re-arm** — after firing,
quiet until ≥ 1 new content word (a recovered word + fresh trigger = a new
search). **Episode annotation (v3, replaced clause windowing)** — the fragment is
the FULL utterance, never truncated; earlier searches in the turn are recorded as
`SearchEpisode`s and their offered words ride along on `StallEvent.already_served`.
(v2 advanced a window past the abandoned attempt, which deleted the sentence:
`'on some'` returned "toast, bread, eggs" — words the speaker had already said —
while the full utterance returns "butter, jam, jelly".) **Interim safety** —
`is_final=false` hypotheses ignored. **Acoustic gating** — events require ≥ 1
transcribed content word. **Negative-tested** — trailing "you know", short pauses,
leading fillers never fire ("you know" is deliberately excluded from the hedge
list as a fluent discourse marker).

## AcousticStream (`backend/acoustic/stream.py`)

Per-feed pipeline over the rolling PCM buffer:

1. **Silero VAD** on 512-sample (32 ms) chunks — gates utterance accumulation;
   ≥ 1000 ms silence ends the utterance and zeroes the voiced-time counter.
2. **ProlongationTracker** on 50 ms frames (rule-based, below).
3. **FillerNet** every 125 ms hop on the last 1.0 s window — compact ~136k-param
   CNN, 4 classes, PodcastFillers (85,803 labeled 1 s clips, official splits —
   zenodo.org/records/7121457; non-commercial research). The `speech` class forces
   the filler-vs-word boundary. Recipe: since the deployed trigger fires on the
   binary uh∪um decision, an auxiliary binary BCE loss (weight 1.0) sits on top of
   the 4-class CE so the optimized objective matches the deployed metric; the
   checkpoint was chosen from a 5-variant validation-only grid, test touched once
   (numbers in `eval/results/threshold_sweep.json` under `training_grid`).

Event gating: **voiced-time gate** (no event until ≥ 800 ms accumulated voiced
speech, `min_voiced_ms`); **confidence gate** (filler events require
`P(uh)+P(um) ≥ 0.75`, `ACOUSTIC_CONF`, *and* uh/um as argmax — the 0.75 point
chosen to minimize stream false alarms subject to clip recall ≥ 0.837: at 0.75,
clip recall 0.844, stream false alarms 5.8/min); **per-kind refractory** (1200 ms).

**Prolongation rule** (`backend/acoustic/prolongation.py`): keys on the
spectral-envelope half of the signature only (F0 not computed) — consecutive
50 ms voiced frames whose time-averaged mel envelopes stay nearly identical
(cosine sim ≥ 0.94) for ≥ 600 ms. Deliberately not learned (published
prolongation classifiers reach only ~0.5–0.7 F1). The operating point is from a
committed sweep (`eval/tune_stall_thresholds.py`): real held vowels jitter at
~0.95–0.98, so 0.94 puts that band above threshold with margin (0.985 missed real
vowels), and 600 ms was the shortest hold keeping the hard constraint — **0 false
fires on 120 s of running speech**. Validation on real PFSD audio
(`eval/run_prolongation_eval.py`, 4 blocks): 0/120 s speech false fires (lead
result); palindrome-looped detection **5/40** (conservative lower bound — held
vowels lack the internal phone transitions conversational clips have; ground truth
needs the self-recorded set, `eval/record_protocol.md`); music 5.5 fires/min
(VAD-gated live).

## Speculative prefetch (`backend/pipeline.py`)

The LLM round-trip is ~1.25–1.8 s (measured). Prefetch hides it:

- **Shadow cadence:** whenever the cache is cold (turn start, or right after a
  stall consumed it) the pipeline shadow-predicts on the very next content word —
  so even the *first* stall of a turn is a cache hit — then refreshes every 3 new
  content words (`PREFETCH_EVERY`), one shadow in flight at a time.
- **Drift guard:** a cached prediction serves only if the live fragment still
  starts with the cached fragment *and* has drifted ≤ 2 content words past it.
- **Staleness:** shadow completions carry a sequence number; a consumed stall or a
  `turn_end` bumps it, orphaning an in-flight shadow (it can't write into the
  cleared cache and doesn't chase-relaunch); out-of-order/failed completions drop
  silently.
- **Result:** on a hit the word serves `served="prefetch"` at **~0 ms** (bench,
  n=20) / **<300 ms** (live e2e) instead of ~1.25–1.8 s.

## Predictor (`backend/predictor/`)

Interface `WordPredictor.predict(context_turns, fragment) -> [Candidate]`.
Default LLM with `thinking_budget=0` (a thinking model otherwise spends the whole
output budget on thoughts and returns empty — and it's faster without thinking),
JSON-schema-constrained, few-shot primed, grounded in the last 6 turns
(`CONTEXT_TURNS`). Verified live: context "My sister Maria visited" → stall "I
need to call, um" → predicts **Maria** at 1.0. Provider-abstracted
(`PREDICTOR_PROVIDER=gemini|claude|deepseek|local|mock`, one env var); anything
implementing `WordPredictor` drops in with zero refactor — the EchoLM integration
point (`docs/ROADMAP.md`).

## Failure modes & graceful degradation

| Missing / broken | Behavior | Where |
|---|---|---|
| No FillerNet checkpoint | Acoustic runs **prolongation-only**; `/healthz` reports it | `AcousticStream.__init__` |
| No StutterNet checkpoint, none named | Logs at ERROR naming the missing path + the FillerNet that runs instead; `/healthz` `fillernet+prolongation`. Never silent | `AcousticStream.__init__` |
| Named checkpoint (`STUTTER_MODEL`/`_BACKEND`) missing | `FileNotFoundError` at connect — an explicit request is not degraded | `AcousticStream.__init__` |
| `STUTTER_BACKEND=ssl` on CPU | Refused at startup (438 ms/hop is a stalled session); `STUTTER_ALLOW_CPU=on` for offline checks only | `config.get_settings` |
| No/invalid API key, missing SDK | Falls back to `MockPredictor`; UI shows "mock fallback" | `session.make_predictor` |
| DJI Mic 2S present | Browser `AudioWorklet` feeds `/ws/audio` from the receiver (auto-preferred via `GET /api/audio/sources`) | `frontend/app.js`, `backend/audio_sources.py` |
| No DJI | Same worklet from the laptop array; the level-based speaker gate loses its physical basis | `frontend/app.js` |
| No mic / permission lost | **⌨ Simulate** tab drives the identical pipeline | `frontend/` |
| Venue Wi-Fi dies | Hotspot for the predictor; worst case `PREDICTOR_PROVIDER=mock` + Simulate is fully offline | — |
| Shadow prefetch fails | Silent; the stall is served by the live path | `EchoPipeline._shadow_predict` |
| Reject re-predict fails/times out | Returns `None`; UI keeps its locally-promoted candidate; rejected words stay recorded | `EchoPipeline.reject` |

Audio-source ladder in one line: **DJI Mic 2S (on the speaker) → laptop mic
(on the table) → Simulate (no mic)**. Which one is capturing is reported by
`POST /api/audio/source` and `/healthz` `audio_source`; measured DJI numbers in
`docs/MICROPHONE.md`.

## Empirical verification to capture (5 min — the thesis made visible)

Standalone probe **`eval/chrome_filler_probe.html`** (no server): open in Chrome,
click *Start listening*, read *"Every morning I, um, make some, uh, toast."* then,
stretching the vowel, *"I'll take theeee… train."* Screenshot the transcript pane
— **um/uh never appear** and **theeee lands as a clean "the"**. Repeat in the
**Live tab** and screenshot the dual-channel timeline (transcript clean, acoustic
lighting FILLER / PROLONGATION) for the paired view. Save both to the Devpost
gallery, captioned with the evidence links. Restraint check:
`python -m eval.run_prolongation_eval` → 0/120 s running-speech false fires
(headline), 11/120 s music (5.5/min, VAD-gated live), 5/40 palindrome detection
(explicit lower bound), stream-level FillerNet false alarms.
