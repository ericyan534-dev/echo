# Echo — self-recorded eval set protocol

The clip-level PFSD numbers (`docs/EVAL.md`) are an optimistic proxy: podcast
speech, not aphasic word-finding speech. This protocol defines a small
**self-recorded, aphasia-style utterance set** that exercises the *whole*
pipeline (STT + acoustic + fusion + serving), so we can report an
utterance-level number instead of only clip classification.

`eval/run_stall_eval.py --wav-dir <dir>` already consumes this set; it is the
only thing missing is the recordings themselves.

## What to record

**40 utterances total — 2 speakers × 5 categories × 4 takes.** Each utterance
is scripted (so STT can't sink the eval) and acted to contain exactly one
target phenomenon.

| Category | What the speaker does | Expected Echo behaviour |
|---|---|---|
| `pause` | speak, then go silent mid-sentence for > 1.5 s | pause trigger fires |
| `filler` | insert a clear "um" / "uh" mid-sentence | `filler_acoustic` fires (and `filler` text if STT keeps it) |
| `prolongation` | stretch a vowel: "I'll take theee…" (hold ≥ 1 s) | `prolongation` fires; transcript shows clean "the" |
| `hedge` | circumlocute: "the thing you put bread in…" | `hedge` trigger fires |
| `fluent` | speak a full sentence smoothly, no stall | **nothing fires** (this is the negative control) |

The `fluent` takes are the most important: a word-finding aid that nags a
fluent speaker is worse than no aid. Half the value of this set is proving the
false-positive rate on real connected speech.

## Recording requirements

- **16 kHz, mono, 16-bit PCM WAV.** (Other rates/channels are rejected by the
  loader with a `SKIP` message — re-record, don't resample after the fact.)
- Quiet room, consistent mic distance (~30 cm), normal speaking volume.
- One utterance per file. ~3–8 s each.
- Two speakers (`spk1`, `spk2`) for a minimal generalisation check.

## File naming (parsed by `run_stall_eval.py`)

```
<speaker>_<category>_<idx>.wav
```

- `speaker`: lowercase alphanumeric, e.g. `spk1`, `spk2`
- `category`: one of `pause | filler | prolongation | hedge | fluent`
- `idx`: zero-padded take number, e.g. `01`

Examples:
```
spk1_filler_01.wav
spk1_prolongation_03.wav
spk2_fluent_04.wav
```

Files that don't match the pattern, or aren't 16 kHz, are skipped with a
printed reason — check the console after running.

## Running the eval

```bash
python eval/run_stall_eval.py --wav-dir recordings/
```

It feeds each file through `backend.acoustic.stream.AcousticStream` in 20 ms
chunks (the live socket's framing) and reports, per category, whether the
expected `AcousticEvent` fired. Current scope (honest): `filler` and
`prolongation` are scored on the acoustic channel directly; `pause`, `hedge`,
and `fluent` require a transcript replay through the full pipeline and are
reported `UNSCORED` until that replay path is wired up. Results merge into
`eval/results/stall_eval.json` under `self_recorded`.

## Why this matters for judging

- Turns "FillerNet scores 0.93 on podcasts" into "Echo fires on N/N of our own
  staged stalls and stays silent on M/M fluent sentences" — a claim grounded
  in the demo domain, not a benchmark set.
- The `fluent` negative control is the credible answer to "doesn't it just fire
  constantly?" — the question every judge asks about an always-listening aid.
