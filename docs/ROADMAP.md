# Echo — Roadmap (post-v2)

v2 ships dual-channel stall detection (transcript + acoustic), speculative
prefetch, and a recognised wireless-lav input (DJI Mic 2S). This is what comes
next.

## Phase 3 — EchoLM: on-device word prediction (deferred, recipe validated)

Replace the cloud LLM with a fine-tuned small model running locally, so the
privacy-critical path never leaves the device and serving latency drops below the
prefetch threshold for *every* stall, not just cache hits.

There is direct precedent that a fine-tuned small model **beats GPT-4 on real
aphasic circumlocutions**: Kim, Storaï & Hwang (Findings of EMNLP 2024,
aclanthology.org/2024.findings-emnlp.616) report GradSelect EM 0.327 / Acc@5
0.542 vs GPT-4's 0.308 / 0.440 on AphasiaBank. Word-from-description is narrow;
task-specific fine-tuning dominates general capability.

The recipe (validated end-to-end on paper, not yet trained):

| Step | Choice | Why |
|---|---|---|
| Base | Qwen2.5-1.5B-Instruct | strong small instruct model; QLoRA on one consumer GPU |
| Method | QLoRA via Unsloth | cheapest path to a competent adapter |
| Data 1 | **3D-EX** — 2.27M definition→word rows (HF `LM-Lexicon/3D-EX`) | the reverse-dictionary shape of the task |
| Data 2 | WordNet glosses | dense, clean definition→word pairs |
| Data 3 | Synthetic circumlocutions | bridge dictionary→"the thing you put bread in"; method per arXiv 2510.24817 |
| Serving | llama.cpp + **GBNF grammar** on the candidate-JSON schema | <300 ms local, structurally valid |
| Integration | implements `WordPredictor` | **zero refactor** — `PREDICTOR_PROVIDER` was built for this (`backend/predictor/base.py`) |

Deferred because training + eval is a multi-day GPU job that competes with the
system work that is the actual demo; the interface seam means deferral costs
nothing — EchoLM drops in the day the checkpoint exists.

## Phase 4 — exploratory

- **AphasiaBank evaluation (pending).** The honest target is real aphasic speech.
  Access requires consortium membership — **faculty request pending**. Until then
  detection metrics come from PodcastFillers (fillers) and recorded validation
  clips (prolongations); see `docs/EVAL.md`.
- **SLP-guided pilot.** Echo is a prototype, not a medical device. The next
  credibility step is a small pilot designed with an SLP: cueing-timing
  preferences, visual-vs-TTS delivery, and per-user threshold tuning
  (`STALL_PAUSE_MS`, prolongation `min_ms`) — anomia presents very differently
  across people.

## Smaller engineering items

- Server-side streaming STT (Deepgram skeleton in `backend/stt/`) for word-level
  timestamps and external-mic transcript routing.
- Fully-local STT (faster-whisper) behind the same `stt` interface to complete the
  no-cloud path alongside EchoLM.
- Multi-session backend (one `EchoSession` per user instead of per process).
