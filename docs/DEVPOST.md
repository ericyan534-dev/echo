# Echo - Devpost

**Echo - a real-time word-finding co-pilot for aphasia.**
Health & Accessibility.

![Echo's console at rest - two sensing channels running, calm until a stall happens](img/ui-idle.png)

---

## Inspiration

People forget words all the time. You are talking with friends and suddenly
stop in the middle of a sentence: *"I put the bread in the... the thing that
heats it..."* Most people push through an awkward pause and move on.

For someone with **aphasia** or **anomia**, that moment happens far more often,
and it makes everyday conversation exhausting. The person knows exactly what
they want to say - they just cannot retrieve the word fast enough. We built
Echo to help in that exact moment: it offers a prompt without taking over the
conversation, so the speaker finishes the sentence in their own voice.

## What it does

Echo listens to the speaker in real time and stays silent while speech is
fluent. When it detects a stall - a long pause, a repeated sound, a stretched
vowel - it predicts the intended word from the unfinished sentence and the
conversation around it.

If someone says *"Every morning I put two slices of bread into the..."*, Echo
suggests **"toaster."** The word appears as a large card in the browser and,
if enabled, is spoken once by text-to-speech, so the speaker can glance at it
and keep talking. Echo also predicts *while speech is still fluent* and caches
the guess - so when the speaker does get stuck, a cached word can appear almost
instantly.

## How we built it

Echo runs two sensing channels over the same raw audio, because consumer speech
recognition erases exactly the signals a word-finding stall is made of - it
deletes "um"/"uh" and normalizes "theeee..." to "the."

- **Transcript channel.** CrisperWhisper transcribes verbatim, preserving
  fillers, false starts, and repeated words - and gives the predictor the
  linguistic context it needs.
- **Acoustic channel.** The same PCM16 audio goes to Silero VAD (finds speech),
  then StutterNet - our own dysfluency model scoring blocks, repetitions, and
  more - plus a rule-based detector for prolonged vowels. A deterministic state
  machine fuses these with a 1.3-second pause detector; debounce and refractory
  rules keep Echo from interrupting.

When Echo detects a stall it checks the prefetch cache or asks an LLM. The
predictor is pluggable: the live default is **DeepSeek (`deepseek-flash`)** -
98.3% top-1 on our set at ~1 second - with Gemini, Claude, and a **fully local**
model behind the same interface. The backend is FastAPI with two WebSockets:
`/ws/audio` (microphone) and `/ws` (browser console).

For hardware, the speaker wears a **DJI Mic 2S** wireless lavalier. Its USB
receiver is a plain audio input; the console recognises it, prefers it over the
laptop array, and turns off Chrome's gain control for it so the level at the
capsule - which is what the speaker gate reads - reaches the detector
untouched. We measured it against the laptop mic before trusting it
(`docs/MICROPHONE.md`).

## Challenges we ran into

- **Getting onto the ASUS GX10.** We first tried to reach the box over its own
  hotspot; a compatibility issue blocked it, so we borrowed a mouse and monitor
  and set it up directly.
- **Speed vs. accuracy.** The suggestion has to land in ~1-2 seconds while
  staying right, so we benchmarked several LLMs to find the fastest accurate one
  (DeepSeek won) and pushed the acoustic detector's latency down.
- **Stability under live load.** Live audio, transcription, prediction, and
  the browser all at once - any dropped WebSocket or slow request could break
  the moment. Making the whole loop degrade gracefully took real
  work.

## Accomplishments that we're proud of

- We **trained our own acoustic dysfluency model** and wired it into a complete
  real-time loop. Echo does not analyze a recording after the fact - it detects
  a stall, predicts the word, and delivers it *while the person is still
  speaking.*
- The system carries **503 automated tests** plus five live end-to-end cases,
  all passing. In a full synthetic conversation stream the fused detector caught
  **36 of 39 embedded stalls (92.3%)**, a median **540 ms before** a
  transcript-only system could.
- It **runs fully offline** on a local model, and can run **entirely on one ASUS
  Ascent GX10** - so a clinic's conversation never leaves the room.

## What we learned

Putting a real microphone on a real speaker taught us that a prediction is only
useful if it reaches the person in a form they can use in a second - the size
of the word on screen, the timing of the spoken cue, the network, and where the
microphone sits all shape the experience.

And we learned that our definition of "fun" now includes debugging live audio,
training speech models, and measuring two microphones against each other at
midnight.

## What's next

Our most important next step is **evaluating Echo with people who have aphasia.**
Our prediction set was written by our team, and StutterNet was trained on
stuttered podcast speech, not aphasic speech - real-user testing comes before
any claim about clinical usefulness. We are also making the on-device path the
default (it already runs on the GX10) so the whole system is private and local,
and running the recorded-speech protocol through the DJI Mic 2S and the laptop
mic side by side so the input's effect on detection is measured, not assumed.
