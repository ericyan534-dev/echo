# Echo — demo runbook, video script, and pre-event prep

Covers the live demo (4 beats, ~2.5 min), the judge Q&A quick answers, the
fallback drill, the 60-second Devpost video, and the human critical path before
the event. Numbers trace to `docs/EVAL.md`; quote that, not memory.

## Setup checklist (before judges arrive)

1. **Network for the predictor** — the laptop needs internet for the live LLM
   (or run the offline stack, `docs/DEPLOYMENT.md`). The DJI Mic 2S is a USB
   receiver; it needs no network.
2. `uvicorn backend.app:app --port 8000` · open **http://localhost:8000** in **Chrome**.
3. Check the chips: `connected` · model provider · `acoustic: full`.
   - **Model chip "mock fallback"** → API key isn't loading; check `.env`.
   - **Acoustic chip "prolongation-only"** → no `models/fillernet.pt`; the
     prolongation beat still works, skip the filler claim.
   - **With `ASR_PROVIDER=crisper`, open only one console page** — only the
     first `/ws/audio` connection transcribes; whichever connects first owns
     the transcript, released when that socket drops.
   - Console is three pages (Console / Sessions / Settings, Alt+1..3). The
     rail's **Inputs** tree lists the host's audio inputs and marks the DJI Mic
     2S when its receiver is present; judges asking "which mic is that?" get
     the tree and the Audio-source panel. **Settings** shows the live server
     runtime from `/healthz` and `/api/config`.
4. Grant mic permission; confirm the VU bar moves. With the **DJI Mic 2S**
   receiver plugged in, the Input picker auto-selects "Wireless Mic Rx", the deck
   shows a DJI badge, and the Audio-source panel reports profile/format/floor.
   Clip the transmitter on the speaker (`docs/MICROPHONE.md`).
5. **Quiet-ish corner.** TTS self-hearing: the software gate is primary (while
   Echo speaks, mic PCM is suppressed via `flushPCM` + 300 ms tail + 10 s safety
   cap); it covers the acoustic lane only, so moderate TTS volume (or an earbud)
   is the backup for the *transcript* lane.
6. Pre-run one stall so the first judge-visible prediction isn't a cold start.

## The 4-beat core demo (Live tab)

**Beat 0 — calm by default (5 s).** Press *Start listening*, say nothing for
2–3 s. The UI sits idle, just the lanes' faint breathing pulse. "This is what
Echo does most of the time: nothing. It only exists in the moment you need it."

**Beat 1 — fluent = silent (15 s).** Say smoothly:
> "Hi, I'm demoing Echo — it's a co-pilot for people with aphasia."

Point at the **dual-channel panel**: words in the transcript lane, acoustic lane
quiet, no suggestion. "Echo runs two channels — what the recognizer admits to,
and what the mic actually heard. While you're fluent, both stay quiet. That
restraint is the hard part."

**Beat 2 — the stall the transcript can't see (25 s).** Say, stretching the vowel:
> "Every morning I make some toast in theeee…" *(hold ~1 s, then stop.)*

Point first at the transcript lane: a bare **"the"** with an **'ASR heard'**
strike-through marking what the recognizer flattened. Hold one beat — the
sequencing *is* the argument. THEN the acoustic lane lights **PROLONGATION** and
the **toaster** card arrives. The amber **ms counter** in the Suggested-words
header climbs from detection (let it tick — the visible wait is the cost of a
stall) and freezes green on the card ("NNN ms perceived", real wall-clock, next
to the backend's latency badge). Line: **"The transcript never saw that. Chrome
deletes um's and flattens 'theeee' to 'the' — every consumer recognizer does.
Echo hears the audio itself. That's the v2 thesis."** (Tune the reveal delay via
`--reveal-ms` in `styles.css`.)

**Beat 3 — context (25 s).** Say fluently:
> "My sister Maria visited me yesterday."

Then **click Stop listening** (do NOT wait silently with the mic open — the pause
trigger fires at 1300 ms into any mid-turn silence with ≥2 content words). Stop
sends a `turn_end` that moves the sentence into context. Press Start again:
> "I really need to call, um…" *(stop.)*

→ the card shows **Maria**. "It pulled her *name* from the conversation — the
speaker keeps agency." **Confirm-to-speak:** autospeak defaults **OFF**; when the card appears
Echo stays silent, the presenter taps the dominant word and Echo speaks it.
"Echo never takes the turn. The speaker decides if — and when — the word gets
said aloud." Finishing past a served card ticks the **"stalls recovered: N this
session"** tally (session-local, never quoted as efficacy).

**Beat 4 — prefetch: the word is just there (20 s).** Say:
> "After dinner I washed all of the dirty…" *(stop.)*

→ the card appears with the **prefetch** badge; the ms counter freezes at tens of
ms right next to Beat 2's three-digit stamp. Point at the two numbers: "While I
was fluent, Echo was shadow-predicting every few words. A live round-trip is
about 1.5 s — measured — but on a stall it serves from cache, essentially
instantly. That contrast is the whole speculative-prefetch story, and every digit
is real." **Drift recovery:** if the badge shows **live**, the cache missed
(stall too many words past the last shadow) — "the live path is the fallback,
still ~1.5 s, still faster than giving up," then show a past prefetch run in
Session history. Drill this beat solo so the sentence lands inside the drift window.

**Wrong-word recovery micro-beat (15 s).** Use a known-ambiguous circumlocution
(drilled to reliably produce an off top guess). Top card wrong → tap **"not it"**
→ candidate #2 is right → say it yourself and finish. "A miss costs one tap, and
the sentence is still mine." **Honesty rule:** the miss must be genuine, or shown
from Session history as a past miss — never a canned fake. If the top guess
happens to be right on demo day, say so and show a real historical miss.

**Closing line (10 s).**
> "2 million Americans know exactly what they want to say and can't retrieve the
> word. Echo lets them finish **their own sentence**."

## Judge Q&A quick answers

- *"Just a chatbot?"* — No. Speaker-side: never replies, never takes a turn.
  Recovers **your** word, triggered by **your** stall.
- *"What's novel?"* — The live system, and the acoustic channel. Prior work
  (CSCW'23, EMNLP'24) proved prediction works **offline on transcripts**;
  transcripts erase the stall signals themselves (see `PITCH.md`).
- *"Why not Deepgram `filler_words=true`?"* — Restores filler *text* only: no
  prolongations (normalized by design), no prosody, cloud on the privacy path,
  and gated on ASR finalization latency vs our 125 ms acoustic hop.
- *"What if it's wrong?"* — Ranked candidates, speaker keeps agency; nothing is
  spoken until confirmed. A miss costs one tap: **"not it"** promotes #2 and
  requests a replacement excluding rejected words; a wrong word left alone fades
  after a few spoken words (implicit reject).
- *"Latency?"* — Prefetch hit ~0 ms (mechanism bench, n=20) / <300 ms live e2e;
  live LLM ~1.25–1.8 s, measured. Acoustic detection 755 ms median (≤600 ms gate
  missed; ~1.7× earlier than the 1300 ms pause baseline).
- *"Filler accuracy?"* — Gate clip-level F1 ≥ 0.75 on the PFSD official test
  split; clean number **0.933** (precision 0.949, recall 0.918) after removing
  extra-split leakage and adding a binary uh∪um auxiliary loss.
- *"False-alarm rate?"* — Prolongation rule: 0 false fires on 120 s of running
  speech. Stream-level FillerNet upper bound 5.8/min before detector gating
  (conservative). Both in `docs/EVAL.md`.
- *"Works in a loud room?"* — Under noise the model misses rather than
  false-alarms: precision holds 0.99–1.00 while recall drops (0.33 at 5 dB SNR at
  the shipped gate). A loud hall makes Echo quieter, not noisier; the lav on
  the speaker keeps the mic centimeters from the mouth. A missed stall still
  hits the 1300 ms pause fallback.

Full 27-question drill: `docs/QA_DRILL.md`.

## 3-tier fallback drill (rehearse the handoffs)

| Tier | Symptom | Move |
|---|---|---|
| 1→2 | DJI receiver missing / transmitter dead | **Laptop mic** — the Input picker falls back to the array; the AudioWorklet feeds the same `/ws/audio`, identical pipeline. |
| 2→3 | Mic/STT breaks (permission, too loud) | **⌨ Simulate** tab — same pipeline, scripted words. |
| 3 | Network/API dies | `PREDICTOR_PROVIDER=demo_fallback` keeps the live path in front and serves a local backup on timeout, disclosed in the console. Fully-offline: `PREDICTOR_PROVIDER=mock` + Simulate, or `python -m scripts.replay`. |

**Tier 3 detail (`demo_fallback`).** In `.env` set
`PREDICTOR_PROVIDER=demo_fallback` (inner provider defaults `gemini`; override
`DEMO_FALLBACK_INNER`). Wraps the live predictor with its own
`DEMO_FALLBACK_TIMEOUT_S` (default 1.5 s, under the pipeline's 4 s). On a
timeout/exception it serves a committed local lookup covering the four rehearsed
beats (toast/toaster, call-Maria, dirty-dishes/dishwasher, Tokyo) — and **only**
those: a live call that legitimately returns nothing still returns nothing.
Rehearse once with Wi-Fi off, confirming each beat serves its backup and the
console prints `[demo-fallback] network path failed; served local backup for
fragment: …`. **Disclosure rule (never break):** if asked "is that live?" answer
truthfully — **"Gemini live, with a local backup if the venue network drops — the
backup only ever fires on a network failure and logs when it does."**

Other moves: STT mishears the setup → repeat once (Beat 2 only needs "…toast in
theeee"); prediction slow (>3 s) → keep talking, show Session history of earlier
sub-2 s / ⚡ runs; mic permission lost → `chrome://settings/content/microphone`
→ allow localhost, or Simulate.

## Demo-day env checklist

- `.env`: valid predictor key (`PREDICTOR_PROVIDER` + key), `PREFETCH=on`.
- `models/fillernet.pt` present (`/healthz` → `"acoustic":"fillernet+prolongation"`).
- `python -m pytest -q` → all green (one test needs the local PFSD dataset).
- `python -m scripts.e2e_live` → 5/5 PASS (proves key + model + pipeline).
- DJI Mic 2S receiver plugged in and selected in the Input picker (or the
  laptop array, knowingly).

---

## 60-second Devpost video

Shot-by-shot, mapped to the 4 beats and **real UI elements** (dual-channel lanes,
"ASR heard" strike-through, amber ms counter, session tally, word card, prefetch
badge). Nothing is staged — every number and UI state is produced by the live
pipeline.

| Sec | Shot | Spoken | Highlight |
|---|---|---|---|
| 0:00–0:05 | Cold open, **no UI** — presenter mid-conversation, reaches for a word, trails off | "Every morning I make some toast in theeee—" (genuine stall) | none — the human problem |
| 0:05–0:10 | Cut to Live tab, listening; presenter resumes fluently, lanes calm | "Echo listens while you talk — and stays silent while you're fluent." | both lanes scrolling; "Silent while fluent" |
| 0:10–0:14 | Push in on transcript lane; repeat the toast line, hold the vowel ~1 s | "…toast in theeee…" | transcript prints "…toast in the." |
| 0:14–0:20 | Hold both lanes: "the" strikes through (`ASR heard`), one beat later acoustic lights PROLONGATION; ms counter ticks | "The transcript shows a clean 'the.' Chrome deleted what I actually said." | `asr-missed` strike-through; PROLONGATION; `#latency-counter` |
| 0:20–0:26 | Card renders "toaster" + 2 chips + "not it"; counter freezes green | "My microphone caught it — the word showed up before I even had to ask." | `.word-card`; "770 ms perceived" |
| 0:26–0:33 | Insert context line ("My sister Maria visited yesterday"); stall on "I need to call, um—"; close on the card: "Maria" | "It even remembers who I was talking about, turns back — and hands me her name." | word card |
| 0:33–0:40 | "After dinner I washed all of the dirty—"; card renders instantly with prefetch badge; counter at tens of ms beside the earlier ~770 ms | "While I was talking, Echo already guessed. This time the word was just… there." | `[prefetch NNN ms]` `.served-fast`; two-counter contrast |
| 0:40–0:48 | Genuinely ambiguous stall, top guess wrong (real miss); tap "not it", #2 promotes, presenter says it and keeps talking | "It's not always right. A miss costs one tap — and I still finish my own sentence, not Echo's." | `.not-it`; candidate promotion; session tally |
| 0:48–0:55 | Wide: laptop + presenter wearing the DJI transmitter, finishing; glance at Session history | "Two million Americans know exactly what they want to say." | `#history` |
| 0:55–1:00 | Clean logo card | "Echo helps them finish their own sentence." | "Echo — finish your own sentence." |

**Constraint check:** human problem opens with no tech (0:00–0:05); ASR-erasure
reveal (strike-through → PROLONGATION) at 0:14–0:20; two-counter prefetch
contrast at 0:33–0:40; one honest limitation spoken on camera at 0:40–0:48 (a
real "not it" tap, not a disclaimer slide); closing five seconds is the tagline.

**Capture:** 1920×1080, 30 fps min (60 preferred for vowel-hold / strike-through
timing); record in Chrome; browser zoom 100%; kill notification banners; capture
screen and presenter shots as separate takes. Mic: the DJI Mic 2S on the
presenter (laptop fan noise drags FillerNet down); keep the SAME take's audio for any
clip where on-screen timing must sync; TTS monitor low / earbud. **Warm up
before recording:** run one full stall-to-card cycle off-camera (the first stall
pays VAD load + TLS cold-cache, not representative); say a throwaway 3-word
sentence right before the prefetch line so the cache is warm; confirm the model
chip reads the real provider and acoustic reads "fillernet+prolongation". Expect
3–5 takes for the vowel hold, 5–10 for context+prefetch (timing-sensitive; stall
must land within the drift window, `_PREFETCH_DRIFT`), 3–5 for "not it"; budget
45–60 min.

**Never fake a number.** Every ms stamp is real instrumentation
(`frontend/app.js` `startLatencyCounter`/`freezeLatencyCounter`). If a live call
is slow on camera: prefer the real prefetch path for timed beats (a prefetch hit
still shows an honest badge + sub-second stamp); cut dead air in editing, never
change the displayed number; if a live call fails outright, that's real footage
of the disclosed fallback (`demo_fallback` serves within 1.5 s and logs) — re-take
or keep it, but don't narrate a fallback word as an unqualified live call. Never
splice a pre-recorded "fast" stall over a slow live one.

---

## Pre-event prep (the human critical path)

Everything automatable is committed (tests green, live e2e, docs audited); what
remains is physical. Do in order; each names its source doc and done-gate.

**1. Microphone bring-up (~1 h).**
- [ ] Software-only smoke test — GATE: Simulate serves "toaster".
- [ ] DJI Mic 2S charged (transmitter + receiver); receiver plugged in; `python
      scripts/mic_probe.py --list` shows it as the `dji-mic-2s` profile.
- [ ] Live-tab bring-up on the DJI — GATE: Input picker auto-selects "Wireless
      Mic Rx", VU moves from the transmitter, acoustic lane fires on a real "um".
- [ ] Dress rehearsal — GATE: `/healthz` reports the DJI under `audio_source`;
      full 4-beat run on the lav.

**2. The last weak number (1–2 h).**
- [ ] Record the 40-utterance held-vowel set (`eval/record_protocol.md`, 1–2
      speakers), run `python eval/run_stall_eval.py --wav-dir <dir>`. Converts
      the 5/40 synthetic lower bound into a real detection number — the biggest
      remaining evidence upgrade. Hand results to the orchestrator to fold
      through `eval/make_report.py`.

**3. Evidence captures (30 min).**
- [ ] Chrome filler-stripping screenshots via `eval/chrome_filler_probe.html`
      (say "um", watch Chrome delete it).
- [ ] One clean Live-tab screenshot mid-demo (card + lanes + counter) for the
      README/Devpost header.

**4. Rehearsal (3 clean runs min).**
- [ ] Run the beats end-to-end 3× consecutive clean; drill the wrong-word beat
      until the known-ambiguous circumlocution reliably misses.
- [ ] Drill `docs/QA_DRILL.md` — at least the 5 `[LIKELY]` questions, out loud.
- [ ] Exercise every fallback tier once: wifi physically off, receiver unplugged
      (laptop mic), Simulate.

**5. Video (1–2 h after rehearsals).**
- [ ] Record per the shot table above; link it at the top of `docs/DEVPOST.md`.

**6. Venue day.**
- [ ] Fresh clone → `scripts/setup_venue.ps1 -WithDev` (~7 min to tests-green).
- [ ] Fresh API key into `.env` (never commit). `-WithE2E` for the 5/5.
- [ ] Optional Anthropic side-by-side: `ANTHROPIC_API_KEY`, then
      `python eval/run_prediction_eval.py --provider claude`.
- [ ] Network up, DJI transmitter + receiver charged, spare USB-C cable packed.

**Done = every claim measured and audited, every failure mode rehearsed, the
mic on the speaker, the five hardest questions answered out loud before any
judge asks.**
