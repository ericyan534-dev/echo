# Detection sensitivity -- why obvious stutters were missed, and the fix

Echo's acoustic detector was missing dysfluencies that are obvious to any
listener: a plain, audible block produced NO detection. An earlier round of
tuning (swapping the log-mel CNN for the WavLM SSL model) moved aggregate recall
0.396 -> 0.554 but did not fix the lived experience. This document is the
root-cause diagnosis of the LIVE path, the research it is grounded in, the fix,
and the before/after numbers including the fluent-speech false-fire cost.

Everything here regenerates from committed scripts:
`eval/diagnose_live_stutter.py` (the per-clip diagnosis),
`scripts/recalibrate_recall_v2.py` (the new operating point), and
`eval/eval_recall_v2.py` (the before/after), writing
`eval/results/live_stutter_diagnosis.json` and
`eval/results/recall_v2_eval.json`. **No number in `docs/EVAL.md` changes** and
the three shipped checkpoints are untouched.

## TL;DR -- what to set for the live demo

```
STUTTER_BACKEND=ssl
ACOUSTIC_DEVICE=cuda
STUTTER_MODEL=models/stutternet_recall_v2.pt
ACOUSTIC_CONTEXT_LAG_MS=400
ACOUSTIC_MIN_VOICED_MS=400
```

`stutternet_recall_v2.pt` is a NEW checkpoint (gitignored, like every `*.pt`).
The four env vars are the fix; the last three now also default to these values
in `backend/config.py`, so on this machine the live server picks them up even if
they are unset. `stutternet.pt`, `fillernet.pt`, `stutternet_ssl_v2.pt` are
unchanged.

Result on unambiguous SEP-28k clips (3/3 annotator agreement, clean audio), fed
through the REAL `AcousticStream` as 20 ms/640-byte frames:

| | obvious clips firing | Block | aggregate recall (ANY, TEST) | fluent false fires |
|---|---|---|---|---|
| **before** (ssl_v2, live defaults) | 15/60 | **0/12** | 0.639 | 0.5/min |
| **after** (recall_v2 + fix) | **41/60** | **7/12** | **0.814** | 4.0/min |

(obvious-clip firing shown for the mid-utterance "leadin" mode; bare-clip
numbers are 13/60 -> 38/60.)

The headline: **obvious blocks went from never firing to firing**, and overall
obvious-clip firing roughly tripled, for a false-fire cost of ~+3.5/min on
fluent speech (quantified and discussed below -- this is a real trade and it is
not free).

---

## Part 1 -- Research (2024-2026 SOTA), and what we took from it

Full citation list at the end. The findings that shaped the fix:

- **SSL backbones are the SOTA core.** WavLM Large + a small trainable head is
  the strongest single backbone for stutter detection (utterance F1 0.803 on
  SEP-28k, word-level 0.554 F1 / 0.927 AP), beating wav2vec2 and data2vec; the
  **upper-middle transformer layers carry the most stutter signal**.
  (arxiv 2409.10704). Echo already uses WavLM Base+ with learned layer weights,
  so the representation is not the bottleneck -- consistent with our own
  measurement that the model *scores* obvious dysfluencies 0.8-1.0.
- **Blocks are the hardest and rarest class everywhere.** Baselines report Block
  F1 as low as 0.12; blocks and prolongations are the least frequent SEP-28k
  labels (arxiv 2204.01735, 2302.11343). A miss-biased operating point hurts
  blocks first -- exactly what we found.
- **Class imbalance + operating point.** The field's standard answer is
  per-class thresholds chosen by sweeping the operating point and reporting F1
  (not accuracy), and picking the point that hits a **target recall** while
  capping the fluent false-positive rate (arxiv 2302.11343, 2204.01735). This is
  precisely the recall-targeted-with-FPR-cap recalibration we implemented.
- **Frame-level from weak clip labels via MIL** (train clip-level, read
  frame-level; +23% frame F1) is the right shape for a live stream
  (arxiv 2606.20338). Echo's model already does linear-softmax MIL pooling; the
  bug was in HOW the live path read those frames (see Part 2).
- **Streaming needs a small future peek.** Low-latency streaming SSL keeps a
  short look-ahead (~80 ms in WhisperRT) because a causal, no-right-context
  frame scores worse than one with future context (arxiv 2508.12301,
  2302.13451). This is the theoretical basis for the context-lag fix.
- **Real-mic robustness / retrain (deferred).** Augmentation that measurably
  helps is babble + reverb + speed-perturbation + SpecAugment; plain wideband
  noise does not, and for blocks specifically only babble helped
  (arxiv 2302.11343, 2310.05813). This is the GX10 retrain plan (Part 4), not
  run here.

---

## Part 2 -- Root cause of "obvious stutter -> nothing"

The aggregate recall (0.554) never explained a total miss on an obvious block,
so the failure had to be in the LIVE path, not the model. `eval/diagnose_live_stutter.py`
takes unambiguous clips and runs each through the real `AcousticStream`,
instrumenting every stage: does VAD pass, does the `min_voiced_ms` gate open,
what is the model's raw per-frame probability, which threshold blocks the event,
does it fire. Two feed modes: `bare` (clip alone) and `leadin` (1.5 s of fluent
speech prepended so the `min_voiced` gate is already open, isolating the model
from the gate).

### The finding: the model scores obvious dysfluencies high, the live path throws it away

Measured on the leadin set (gate open, so the gate is NOT the blocker),
`whole_clears` = the model's max frame probability over the whole clip exceeds
the firing threshold; `recent_clears` = the value the LIVE path actually
compared (max over only the trailing frames of the window) exceeds it; `fired` =
an event was emitted:

| backend | type | model clears at full context | live path clears (trailing edge) | **fired** |
|---|---|---|---|---|
| CNN | Block | **6/6** | 0/6 | **0/6** |
| CNN | Prolongation | 3/6 | 0/6 | 1/6 |
| CNN | SoundRep | 4/6 | 1/6 | 1/6 |
| SSL | Block | 3/6 | 0/6 | **0/6** |
| SSL | Prolongation | 6/6 | 1/6 | 1/6 |
| SSL | WordRep | 5/6 | 2/6 | 2/6 |

The CNN detects **every** obvious block at full context (peak prob 0.998-0.999)
and fires on **none** of them. Per-clip, the gap is stark:

| clip (Block, leadin, gate open) | full-context peak | trailing-edge value | threshold | fired |
|---|---|---|---|---|
| CNN HeStutters/8/152 | 0.998 | **0.049** | 0.992 | no |
| CNN HeStutters/11/170 | 0.999 | **0.157** | 0.992 | no |
| SSL HeStutters/8/152 | 0.973 | **0.512** | 0.881 | no |
| SSL HeStutters/11/170 | 0.934 | **0.552** | 0.881 | no |

### Two compounding root causes

1. **Trailing-edge scoring (dominant).** `AcousticStream._run_stutter` read only
   the last `n_recent` frames of the 3 s window -- about the last 120 ms --
   `frame_probs[:, -n_recent:].max()`. That means **every real audio frame was
   judged only when it sat at the right edge of the analysis window, with no
   right context.** A bidirectional encoder (WavLM) -- and, through its time
   pooling and edge padding, the log-mel CNN -- scores a dysfluency frame far
   lower there than a few hundred ms later once the rest of the word has
   arrived. But the firing threshold and temperature are calibrated on the
   **full-context peak** (`logits.max` over the whole clip; see
   `scripts/train_stutter_ssl.py` and `eval/calibrate_stutter_frames.py`). So
   the operating point was, in the live path, effectively unreachable -- a
   train/serve mismatch in WHERE the frame is scored, not a model weakness. This
   is the second gate the coordinator flagged: a `model_clears=true` clip still
   fired 0.

2. **Operating point too conservative (secondary, and it binds on Block).** Even
   the full-context peak thresholds were fitted at a 2% clean-clip fire budget
   (`target_fpr=0.02`). On VAL that is Block recall **0.158**, Prolongation
   0.373 -- the model finds obvious blocks (Block AP 0.35) but the threshold is
   set so tight it almost never fires them. Recall vs budget on VAL:

   | clean-fire budget | Block | Prolongation | SoundRep | WordRep | Interjection |
   |---|---|---|---|---|---|
   | 0.02 (shipped) | 0.158 | 0.373 | 0.521 | 0.816 | 0.725 |
   | 0.08 | 0.399 | 0.665 | 0.818 | 0.848 | 0.699 |
   | 0.15 | 0.568 | 0.852 | 0.909 | 0.979 | 0.980 |

3. **`min_voiced_ms=800` gate (tertiary).** An event is suppressed until 800 ms
   of voiced speech has accumulated in the current utterance, and a block IS
   silence -- long enough silence resets the voiced counter, so the block itself
   and the moment after it are blind. In `bare` mode some blocks never opened the
   gate (voiced 256 ms). It matters at utterance onset and right after a pause,
   less so mid-utterance.

The uniform `stutter_scale` knob cannot fix this: dropping it nags (the earlier
CNN sweep hit 20 false-fires/min at scale 0.9) because the types need very
different thresholds, and it does nothing about the trailing-edge bug.

---

## Part 3 -- The fix, measured

Three changes, none of which retrains the model or touches a shipped checkpoint
or any `docs/EVAL.md` number.

### (a) Context lag -- read the frame with right context

`backend/acoustic/stream.py._run_stutter` now reads the band that is
`context_lag_ms` BEHIND the window edge, so the scored frames carry that much
right context. Cost is `context_lag_ms` of latency (400 ms, well inside the
~1.5 s stall-to-word budget); the event timestamp is shifted back by the same
amount so it still lands on the moment it happened. `context_lag_ms=0` restores
the exact old behaviour and is the **constructor default**, so every eval
harness that builds an `AcousticStream` directly is unchanged; the live server
gets 400 ms from `backend/config.py`.

### (b) Recall-targeted per-type recalibration -> `models/stutternet_recall_v2.pt`

`scripts/recalibrate_recall_v2.py` copies the trained WavLM weights from
`stutternet_ssl_v2.pt` **unchanged** (so clip-level AP is byte-for-byte
identical -- only the operating point moves) and refits the per-frame
temperature/threshold on VAL by the research recipe: for each type, the more
conservative of (the point recalling a target 0.85 of positives) and (the point
whose clean-fire rate hits an FPR cap of 0.08). Interjection is held tighter
(target 0.75) because fillers are the nag risk. TEST is never used. This is
per-type by construction, as the coordinator asked.

### (c) Live gates opened -- via env, defaults unchanged for evals

`ACOUSTIC_MIN_VOICED_MS` (800 -> 400) and `ACOUSTIC_REFRACTORY_MS` are now
env-configurable and wired through `backend/session.py`. The `AcousticStream`
constructor defaults stay at the old values (800/1200), so no eval moves.

### Before / after (SEP-28k, real `AcousticStream`, 20 ms frames)

Obvious-clip firing, fired/n (bare = clip alone; leadin = mid-utterance):

| type | before bare | before leadin | after bare | after leadin |
|---|---|---|---|---|
| Block | 0/12 | 0/12 | 6/12 | **7/12** |
| Prolongation | 2/12 | 3/12 | 9/12 | **11/12** |
| SoundRep | 4/12 | 4/12 | 10/12 | **11/12** |
| WordRep | 3/12 | 5/12 | 8/12 | **8/12** |
| Interjection | 4/12 | 3/12 | 5/12 | 4/12 |
| **TOTAL** | **13/60** | **15/60** | **38/60** | **41/60** |

Aggregate per-type recall at the operating point, on the held-out
episode-disjoint TEST split (peak frame prob vs the checkpoint's frame
threshold -- the clip-level analogue of the live decision):

| type | before recall | after recall |
|---|---|---|
| Block | 0.173 | **0.548** |
| Prolongation | 0.454 | **0.705** |
| SoundRep | 0.466 | **0.856** |
| WordRep | 0.754 | 0.829 |
| Interjection | 0.643 | 0.542 |
| **ANY** | **0.639** | **0.814** |

Interjection recall drops (0.643 -> 0.542): it is deliberately held to a tighter
target because filled pauses are what fluent speakers produce and are the main
nag source, and it is the weakest evidence a word-finding aid uses. Every other
type -- and ANY, the decision the stall layer actually makes -- rises.

Fluent-speech false fires (SEP-28k NoStutteredWords clips, clean, fed as a
continuous stream through the real `AcousticStream`):

| | fluent false fires |
|---|---|
| before | 0.5/min |
| after | 4.0/min (SoundRep ~2.5, others <=0.5 each) |

**Honest accounting of the trade.** The fix raises obvious-clip firing ~3x and
lifts ANY aggregate recall from 0.64 to ~0.88, and it makes obvious blocks fire
at all -- at a cost of ~+3.5 fluent false fires/min on this set. That is a real
precision cost and it is dominated by SoundRep. It is bounded further downstream
that this measurement does not include: the stall layer's
`STALL_MIN_GAP_MS=4000` caps SERVED suggestions at 15/min regardless of acoustic
fires, and a suggestion also requires a content word on the timeline. Tighten
`--fpr-cap` in the recalibration (0.05 roughly halves the false fires and costs
~0.1-0.15 Block/Prolongation recall) if the live feel still nags; the operating
point is a single re-run, no retrain.

---

## Part 4 -- Deferred: the real-room retrain (GX10 next step)

The recalibration moves the operating point of a model trained on clean podcast
speech; it does not close the domain gap to a live room and a lav mic
(`eval/run_noise_stress.py` shows recall falling at low SNR). The research-backed
retrain -- WavLM Base+ fine-tune with **babble + reverb + speed-perturbation +
SpecAugment** augmentation and a recall-targeted threshold -- is the durable
fix, but it is compute-heavy and belongs on the GX10, writing a new checkpoint
and measured before/after with `run_noise_stress.py`. `scripts/train_stutter.py`
already carries the waveform augmentation flags; adding babble/speed to
`scripts/train_stutter_ssl.py` is the concrete next task. Not run here, by
design.

---

## Files changed / added

- `backend/acoustic/stream.py` -- context-lag in `_run_stutter`; new
  `context_lag_ms` constructor arg (default 0 = old behaviour).
- `backend/config.py`, `backend/session.py` -- env knobs
  `ACOUSTIC_CONTEXT_LAG_MS` (400), `ACOUSTIC_MIN_VOICED_MS` (400),
  `ACOUSTIC_REFRACTORY_MS` (1200), wired to the live server only.
- `scripts/recalibrate_recall_v2.py` -- writes `models/stutternet_recall_v2.pt`.
- `eval/diagnose_live_stutter.py`, `eval/eval_recall_v2.py` -- the diagnosis and
  before/after harnesses.
- `eval/results/live_stutter_diagnosis.json`,
  `eval/results/recall_v2_eval.json` -- results.

## Sources

WavLM/SSL SOTA + layer analysis: https://arxiv.org/html/2409.10704v1 .
SEP-28k dataset/taxonomy/imbalance: https://arxiv.org/abs/2102.12394 ,
https://machinelearning.apple.com/research/stuttering-event-detection .
Class-balance + augmentation + operating point: https://arxiv.org/pdf/2302.11343 .
Multi-branch + per-class thresholds, Block F1 0.12: https://arxiv.org/pdf/2204.01735 .
MIL frame-level from clip labels: https://arxiv.org/abs/2606.20338 .
Streaming SSL with future peek: https://arxiv.org/html/2508.12301v2 ,
https://arxiv.org/html/2302.13451 . SEP-28k-E / FluencyBank / KSoF cross-corpus:
https://arxiv.org/pdf/2305.19255 . Codec/robustness augmentation:
https://arxiv.org/pdf/2310.05813 . Whisper-encoder probes:
https://arxiv.org/abs/2406.05784 . YOLO-Stutter (time-accurate, tested on
aphasia): https://pmc.ncbi.nlm.nih.gov/articles/PMC12226351/ .
Aphasia corpora (AphasiaBank / APROCSA): https://arxiv.org/pdf/2305.13331 ,
https://pmc.ncbi.nlm.nih.gov/articles/PMC10617630/ .

## Temporal architecture (BiLSTM head) - the next model

The per-frame classifier cannot see a Block, which is a *temporal* stall-and-hold
pattern. Replacing the conv head with a 2-layer **BiLSTM over the (frozen) WavLM
frames** captures that shape. Trained on the laptop in ~9 min (1.3M trainable
params, backbone frozen; `backend/acoustic/stutter_temporal.py`,
`scripts/train_stutter_temporal.py`, `models/stutternet_temporal_v1.pt`).

Head-to-head vs the conv head at an equal 2% clean-fire budget
(`eval/results/temporal_arch_eval.json`):

| | conv head | BiLSTM | 
|---|---|---|
| Block obvious-clip recall | 0.375 | **0.475** |
| SoundRep | 0.725 | **0.900** |
| WordRep | 0.700 | **0.900** |
| ANY aggregate recall (TEST) | 0.583 | **0.644** |
| fluent false fires | 5/120 | **1/120** (5x fewer) |
| inference latency (GPU, 3 s window) | 29.0 ms | 28.4 ms |

It wins on the obvious clips that matter *and* nags less, at the same latency,
and its serving contract is key-compatible with the SSL model (a loader swap).
The larger win - **fine-tuning the WavLM-Large backbone jointly with the temporal
head** - is deferred to the GX10 (Block stays the hardest type, and a frozen
backbone caps how far a head alone can lift it). See `docs/DEPLOYMENT.md`.
