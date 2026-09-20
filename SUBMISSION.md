# Echo — HackMIT 2026 (Health & Accessibility)

**Echo is a real-time word-finding co-pilot for aphasia.** It listens while a
person speaks, detects the moment they stall reaching for a word — *beneath the
transcript*, where consumer speech recognition erases the signal — and offers
the intended word in ~1-2 seconds as an on-screen word card with optional
speech, so they finish their own sentence.

![Echo console at rest](docs/img/ui-idle.png)

**What it does that nothing else does:** the stall is caught in the raw audio,
in real time, and the word is delivered without ever taking the speaker's turn.
Consumer ASR deletes "um/uh" and normalizes "theeee…" to "the"; Echo runs a
parallel acoustic channel and catches **36/39 (92.3%)** of word-finding stalls a
median **540 ms before** a transcript-only system could (`docs/EVAL.md`). Its
word prediction is the intended word **58/60 (96.7%)** on the frozen eval set.

---

## Run it in two minutes

```bash
git clone <this-repo> echo && cd echo
python -m venv .venv && .venv\Scripts\activate      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add a predictor API key (Gemini or DeepSeek)
uvicorn backend.app:app --port 8000
```

Open **http://localhost:8000** in **Chrome**. Press *Start listening* and stall
mid-sentence, or use the **Simulate** tab (identical pipeline, no microphone).

**Runs fully offline** (no cloud, no key) with a local LLM — see
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Or with no key at all for a quick look:
`PREDICTOR_PROVIDER=mock uvicorn backend.app:app --port 8000`.

## Private, on-device deployment (ASUS track)

Echo runs entirely on one **ASUS Ascent GX10** — detection, word prediction, and
speech-to-text all local, so a clinic's aphasia conversations never leave the
box. The full plan, topology, and bring-up: [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md).

## What runs on a fresh clone

- **The full application** — dual-channel stall detector, speculative prefetch,
  and the three-page operator console (Console / Sessions / Settings).
- **The acoustic channel** — `models/fillernet.pt` (544 KB) ships, so detection
  works out of the box; heavier models (SSL StutterNet, the local LLM,
  CrisperWhisper) are fetched/trained locally and the app uses them when present.
- **The word predictor** — Gemini or DeepSeek with your key in `.env`, a local
  LLM for the offline path, or `mock` for a no-key look.

## Verify the claims

```bash
python -m pytest tests          # full suite, offline, no key needed
```

Every published number is generated from `eval/` into
[`docs/EVAL.md`](docs/EVAL.md) — the source of truth, never hand-edited.

## Where to read next

- **[`docs/README.md`](docs/README.md)** — the full, role-routed documentation
  index (pitch / microphone / backend / demo-day).
- [`docs/PITCH.md`](docs/PITCH.md) — the one-page pitch, with the design mapped
  to the aphasia communication-support literature.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — how it works.
- [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) — private on-device deployment (the
  ASUS GX10 appliance) and the offline stack.

## Hardware

The only hardware is an optional **DJI Mic 2S** wireless lavalier worn by the
speaker; its USB receiver is a plain audio input that the console recognises
and auto-prefers. Measured compatibility and the constraints the console
applies: [`docs/MICROPHONE.md`](docs/MICROPHONE.md). The mic is optional for
the software demo — it runs on a laptop mic alone.

## Scope and licensing

Echo targets the **anomia / word-finding** profile: someone who knows the word
and can read a suggested one. Clinical validation alongside SLPs is the roadmap.
CrisperWhisper (optional local STT) is under a non-commercial research license;
the default demo path does not require it. A repository license is not yet set —
ask the authors before reuse.
