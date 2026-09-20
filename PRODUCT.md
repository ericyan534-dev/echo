# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

**Primary (the screen's audience): technical judges and observers.** During both
the HackMIT demo and every rehearsal, the laptop screen is read by someone
*other* than the person Echo serves — a judge standing beside or behind the
operator, 2–4 ft away, in a noisy venue with ambient glare. They are skeptical,
time-boxed (roughly four minutes), and evaluating whether the mechanism is real.

**Served, but not the screen's audience: the person with aphasia/anomia.** They
receive the predicted word as a large word card and, when enabled, a short
spoken cue (TTS) — a glance, never a read. The card is deliberately one word in
one large face: an aid that forces the speaker to study a screen mid-sentence
competes with the conversation it is supposed to protect.

**Secondary: the operator.** The person running the demo (and, later, a
partner or clinician) drives mic start/stop, the Simulate fallback, and
accept/reject taps.

## Product Purpose

Echo detects the moment a person with aphasia stalls searching for a word and
offers the intended word in ~1–2 s — or instantly when speculative prefetch has
already guessed right — so they finish *their own* sentence rather than being
finished by someone else.

The screen's job is narrower and should not be confused with the product's:
**make the invisible mechanism legible.** A judge must be able to see, without
narration, that two sensing channels are running, that one of them caught
something the other erased, and how long the person actually waited.

## Positioning

Consumer speech recognition erases exactly the signals that matter for aphasia:
Chrome's recognizer silently deletes "um"/"uh" with no off switch, and every
transcript pipeline normalizes prolongations ("theeee…" → "the"). Echo runs a
parallel acoustic channel (Silero VAD → a ~136k-param CNN → a rule-based
prolongation detector) fused with the transcript in one stall detector.

The claim a neighboring product cannot truthfully copy: **the stall is detected
beneath the transcript, in real time, and the word is delivered privately
without interrupting the speaker.** Prior work (CSCW '23, Findings of EMNLP '24)
showed LLMs can recover intended words from circumlocutions — on pre-transcribed
text, offline. The real-time loop is the contribution.

## Operating Context

- **The judging scene is the design constraint.** Laptop on a table, judge
  standing, 2–4 ft viewing distance, venue noise, uncontrolled lighting,
  no second chance. Anything that requires leaning in has failed.
- **Four minutes, four beats** (`docs/DEMO_SCRIPT.md`): silence while fluent →
  a stall the transcript misses → the word arriving in context → prefetch
  landing it instantly.
- **Three degradation tiers, all of which must look deliberate rather than
  broken:** DJI Mic 2S wireless lav on the speaker → laptop mic → Simulate tab
  (typed input driving the identical pipeline). Venue wifi is assumed hostile.
- **The input is visible on screen.** The rail lists the host's audio inputs
  and marks the recognised one (the DJI Mic 2S), because judges ask which
  microphone is actually feeding the pipeline.
- Hardware is a single commercial part: the DJI Mic 2S wireless lavalier, whose
  measured behaviour is in `docs/MICROPHONE.md`.

## Capabilities and Constraints

- Two live modes share one output region: **Live** (browser SpeechRecognition +
  raw PCM over a second WebSocket) and **Simulate** (typed text through the
  identical pipeline). Output — suggested words, meta, history — is shared.
- Interaction model is **confirm-to-speak**: autospeak defaults OFF; the
  operator taps the word to accept, or "not it" to reject. Rejection re-predicts
  with the rejected word excluded. Implicit reject after 4 further words.
- **Silence is a feature.** Most of the detector exists to *not* fire during
  fluent speech. The UI's resting state must read as calm and idle, not as a
  dashboard demanding attention.
- Two WebSockets: `/ws` (transcript JSON) and `/ws/audio` (binary PCM16).
  Full reference: `docs/PROTOCOL.md`.
- Frontend is dependency-free static assets (`frontend/index.html`, `app.js`,
  `console.js`, `styles.css`, `pcm-worklet.js`) served by FastAPI. `console.js`
  loads first and owns the shell (pages, presence tree, event log, source
  profiles); `app.js` owns the session and calls its hooks. **No build step, no
  framework, no external fonts, no CDN** — the venue may have no usable network.
- Element IDs and class names in `index.html` are load-bearing: `app.js` binds
  to them and dispatches `echo:card-rendered` / `-accepted` / `-rejected`
  CustomEvents that tests and the e2e suite observe. Renaming is a behavioral
  change, not a cosmetic one.
- Chrome is the only supported browser (SpeechRecognition + AudioWorklet).

## Brand Commitments

- Name: **Echo**.
- `#38d39f` (mint) is the exact brand anchor and must not be shifted.
- Two typographic voices, already established and binding: a humanist sans
  carries the *speaker's* words (transcript, the dominant suggested word) and a
  monospace carries everything the *machine* measures (chips, latency, tallies,
  acoustic tokens). The pairing mirrors the dual-channel thesis.
- Voice is measured and confident. The project's own documents lead with what
  is measured; the interface states results plainly and does not overclaim
  beyond them.
  The session tally is explicitly labeled session-local and "not an efficacy
  claim" — that kind of hedge is a commitment, not clutter to be cleaned up.
- A healthcare tool for the anomia word-finding profile; assistive
  positioning, with clinical validation as the roadmap. No fabricated clinical
  or diagnostic claims.

## Evidence on Hand

Real, measured, and regenerable — every number below comes from
`eval/results/*.json` via `eval/make_report.py` into `docs/EVAL.md`:

- Fused detector catches **36/39 (92.3%)** of embedded filler stalls in a full
  synthetic conversation stream, a median **540 ms before** the transcript-only
  fallback fires.
- Prediction: **58/60 top-1 (96.7%)**, 59/60 top-3, on a frozen hand-authored
  60-item circumlocution set. Context ablation collapses proper-noun recovery
  19/20 → 2/20.
- Acoustic filler detection: **755 ms median** — a disclosed **miss** against the
  team's own ≤600 ms design gate, still ~1.7× earlier than the 1300 ms pause
  baseline. FillerNet binary F1 **0.933** on the PodcastFillers official test split.
- Prolongation rule: **0 false fires per 120 s** of real running speech;
  detection quoted only as a conservative **5/40** synthetic lower bound.
- 169 tests; 5/5 live end-to-end against the real API.
- One real screenshot exists: `docs/img/ui-idle.png` (headless Chrome capture of
  the live UI at rest). **It is regenerated from the running app — if the
  interface changes, that file is stale and must be recaptured.**

**Absences future work must not fabricate:** no contact with the target
population yet; no AphasiaBank evaluation; no clinician validation; no users, no
testimonials, no deployment. The central interaction assumption — that a ranked
word list plus a spoken cue *relieves* word-finding effort rather than adding
cognitive load — is untested.

## Product Principles

1. **Silence is the default state.** A word-finding aid that nags is worse than
   no aid. The interface earns attention only at event moments.
2. **Show the mechanism, don't assert it.** The dual-channel timeline exists so
   a judge can watch the transcript lane stay clean while the acoustic lane
   fires. Evidence over adjectives.
3. **Never overclaim on screen what the docs disclose.** Hedges and scoping
   labels are load-bearing content.
4. **Degrade visibly and deliberately.** Every fallback tier must look like a
   designed state, never like a failure.
5. **The speaker keeps the floor.** Delivery is a glanceable card and an
   optional short cue; the aid never talks over the person it serves.

## Accessibility & Inclusion

**WCAG 2.2 AA is a binding constraint** (confirmed by the project owner), which
for this surface means: every text and non-text contrast pair meets AA at its
rendered size; focus is visible on every interactive element; nothing is
signaled by color alone (state carries a glyph or text label too); motion
respects `prefers-reduced-motion`; and interactive targets are large enough to
hit reliably.

Product-specific need beyond the standard: the served population has a
*language* impairment, not a vision impairment — so word-level legibility,
short labels, and low reading load matter more than raw information density
wherever the speaker could plausibly see the screen.
