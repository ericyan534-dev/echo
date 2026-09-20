# Data provenance — what the acoustic models are trained and evaluated on

Every detection number depends on which corpus produced it. This records where
each corpus came from, what is missing, and what that forbids claiming.

## SEP-28k (stuttering) — reconstructed 2026-08-18, **74% recovered**

Labels from `apple/ml-stuttering-events-dataset` (`SEP-28k_labels.csv`): 28,177
clips of 3.00 s, each rated by 3 annotators across five dysfluency types.
**Multi-label with vote counts, not single-class.** At the ≥2/3 agreement
threshold Echo uses:

| type | full release | recovered | kept |
|---|---|---|---|
| Block | 3,370 | 2,468 | 73% |
| Interjection | 5,973 | 4,679 | 78% |
| Prolongation | 2,812 | 2,001 | 71% |
| WordRep | 2,770 | 2,308 | 83% |
| SoundRep | 2,342 | 1,796 | 77% |
| any dysfluency | 14,022 | 10,721 | 76% |

Usable clips after excluding `Unsure`/`PoorAudioQuality`/`Music`/`NoSpeech`:
**20,127**, 53.3% positive.

**What is missing, and why it's not random.** The release ships URLs to 385
podcast episodes, not audio; **258 of 385 resolve** (Aug 2026). Three entire shows
are permanently gone:

| show | episodes | cause |
|---|---|---|
| StutteringIsCool | 85 (0 recovered) | `feedproxy.google.com` FeedBurner discontinued; 404 |
| StrongVoices | 35 (0) | SoundCloud stream URLs; 404 |
| IStutterSoWhat | 5 (0) | `istuttersowhat.com` origin down; 522 |

Internet Archive recovery failed on evidence: `wayback/available` returns 429
(rate-limited, discarded); the authoritative CDX endpoint reports **zero** archived
`.mp3` for StutteringIsCool/StrongVoices and only 9 HTML captures (no audio) for
IStutterSoWhat. Surviving pools: HeStutters 3,529 · HVSA 652 · MyStutteringLife
2,215 · StutterTalk 4,910 · WomenWhoStutter 8,821.

**A second, silent loss (found + fixed 2026-08-18).** `fetch_sep28k.py` accepted
any download >10 KB. **27 of the 258 are M4A/AAC served under an `.mp3` URL**
(`ftypM4A`), which libsndfile can't open — every consumer skipped them inside a
bare `except: continue`, so the loss never surfaced. Not spread evenly (HVSA 0→4
readable; MyStutteringLife 15→38; the other 216 unaffected).
`scripts/sep28k_audio.py` transcodes them to 16 kHz mono WAV via one
`episode_path()`. Clips actually cut: **20,124 of 20,127 usable** (was 18,135).

**Forbids:** don't compare to published SEP-28k numbers (those use three shows we
can't obtain — ours is a smaller corpus sharing a name); don't claim
speaker-independent generalisation (three communities absent); **do** state
"SEP-28k (74% subset, 5 of 8 shows)" wherever a derived number is reported. The
per-show table is written into `data/sep28k/manifest.json` so the bias travels
with the data.

*Correction worth remembering:* the first version used
`isabelarvelo/sep28k-*-4-second-clips`, a third-party mirror (8,142 clips, single
`int64` label), and reported its schema as the dataset's — wrong: 8,142 vs 28,177
was a visible discrepancy that should have prompted a source check.
**Convenience mirrors are not sources.**

## SEP-28k via the `saeedzou` mirror — recovered 2026-08-18, **verified**

`saeedzou/sep28k-fluencybank-stutter-dataset` (31,908 clips, 2.87 GB, revision
`e8f1b699b9c05feedbdd617a1bceb27ee5ed1eab`). Not taken on trust — three checks:

1. **Labels.** All 31,908 rows joined to Apple's `SEP-28k_labels.csv` +
   `fluencybank_labels.csv` on `(Show,EpId,ClipId)`: **31,908/31,908 keys found,
   all 12 annotator-count columns and Start/Stop equal on every row**, zero
   mismatches; a strict subset (413 Apple rows missing), never a contradiction.
2. **Audio vs ours.** 98 HeStutters clips cross-correlated: mean best-lag
   correlation **0.996**, best lag **0 samples for all 98** (not bit-identical only
   because of different MP3 decoders).
3. **Audio for shows we can't check.** SEP-28k clip windows overlap in time, so two
   clips share samples; **20/20 overlapping pairs are sample-exact** across
   StutteringIsCool, StrongVoices, IStutterSoWhat, FluencyBank.

Recovers all three lost shows + FluencyBank:

| | before | after |
|---|---|---|
| usable clips | 20,124 | **30,962** |
| Block positives (≥2/3) | 2,468 | **3,690** |
| speaker pools | 5 | **9** |

**Forbids:** don't redistribute or lightly cite it — Apple shipped SEP-28k as URLs
and TalkBank gates FluencyBank; this mirror republishes both and declares **no
licence** (`cardData` has no `license:`). OK for a local build, a hazard for a
model card / dataset release. **Deduplicate before training** — ~20k rows
duplicate clips we hold; concatenating naively puts byte-identical clips in
train+test (`scripts/train_stutter.py` dedupes on `(show,ep,clip)`; fetch offers
`--only-new`). **FluencyBank speaker identity is not established** — episodes named
age+gender+letter (`24fa`, `24fb`); the letter most plausibly disambiguates
subjects but the confirming metadata is gated; the splitter keeps each
age+gender group wholly on one side (if letters are the same person, splitting
would inflate every FluencyBank number).

## Corpora checked and rejected

| corpus | why not |
|---|---|
| `tong0/LLM_Dys` | synthetic TTS; **no annotations** (label is the dir name); no Block; ~2.8–3.4 TB |
| `nyralabs/disfluency_speech_english` | real, apache-2.0, but **one speaker** re-reading Switchboard; labels only fillers/cutoffs (classes Echo already over-detects) |
| **TORGO** | open (9.58 GB) and dysarthric, but labels are transcription + healthy/dysarthric only — **no per-event annotation** — and largely isolated-word reading; no word to fail to find |
| Speech Accessibility Project | signed institutional DUA |
| UASpeech | paid IEEE DataPort + PI approval; the ~46 HF re-uploads are provenance-uncertain |
| Project Euphonia / Relate | Google-internal, not released |
| `hugging-science/arc-aphasia-bids` | MRI, no audio |
| `Aphasia500`, `Aphasiabench` | text/CSV only |

## PodcastFillers (current FillerNet training data) — the wrong corpus

`models/fillernet.pt` trains on PodcastFillers with `CLASSES =
["uh","um","speech","other"]` — fluent podcast hosts. It contains no blocks, no
sound/word repetitions. Consequences: Echo **cannot detect a block** (no class for
it), and it detects **interjections** fluent speakers produce constantly — the
false-positive source (the model is correct on the task it was trained for, which
isn't Echo's). PodcastFillers stays valid for filler classification and its EVAL
numbers are not retracted; it's simply the wrong corpus for stutter detection.

## APROCSA — real aphasic speech, held since 2026-08-18

**Casilio M, Rising K, Beeson PM, Bunton K, Wilson SM (2022). An Open Dataset of
Connected Speech in Aphasia with Consensus Ratings of Auditory-Perceptual
Features. *Data* 7(11):148.** [doi:10.3390/data7110148](https://doi.org/10.3390/data7110148)
· source <https://langneurosci.org/aprocsa-dataset> · licence: unrestricted for
research, education, clinical use.

Six people with chronic post-stroke aphasia; full elicitation protocol,
audiovisual, CHAT transcripts, consensus ratings on 27 auditory-perceptual
features. **The only aphasic speech Echo has ever been evaluated on.**

*Redaction:* the licence covers *using*, not republishing. Five tracked result
files (`aphasia_prediction.json`, `aphasia_eval.json`, their `_pre_sc700_`
snapshots, `asr_context_prompt.json`) had ~125 KB of verbatim participant speech
keyed to participant ID (CHAT source line, ASR fragment, context turns); those
strings are now a redaction marker. **Every metric, count, span, threshold,
timestamp and participant ID is unchanged** — verified by regenerating
`docs/EVAL.md` before/after (byte-identical). The full record regenerates locally
from the gitignored `eval/results/cache/aprocsa/`. Participant IDs (1554, 1713,
1731, 1738, 1833, 1944) are the corpus's own pseudonyms, kept so per-participant
results are auditable. *Not redacted, deliberately:* transcripts in
`asr_model_bench.json` / `asr_filler_recall.json` — those come from SEP-28k and
PodcastFillers (public podcasts), not clinical data.

The CHAT codes mark word finding **directly** — 1,486 participant utterances,
1,324 media-aligned, **686 (46%) coded as word searches:**

| code | meaning | count |
|---|---|---|
| `&-um` | filled pause | 899 |
| `[/]` | repetition | 381 |
| `&+lo` | phonological fragment (cut-off) | 346 |
| `[//]` | retracing (revises the attempt) | 291 |
| `(.)`/`(..)`/`(...)` | timed pauses | 319 |
| `[* s:ur]` | error codes (paraphasias) | 200 |
| `+...` | trailing off — utterance abandoned | 113 |

Example (participant 1554): `and &-um I have speech &-um (.) &-um (...) spring [//]
Christmas` — three filled pauses, a long pause, the wrong word, a retracing to the
right one. No stuttering corpus contains that pattern.

**Must not be used for:** *training* (six speakers can't train a detector, and it
must never be split train/test — one speaker's idiosyncrasies would leak; enforced
by keeping every consumer under `eval/`); *a population estimate* (six people;
between-participant spread reported alongside every pooled number); *a
threshold-fitting set for Echo's own metric* (`fit_aphasia_pause.py` chooses
against a property of the *speech* — silence duration — never against Echo's
recall/false-alarm curve).

## AphasiaBank / FluencyBank audio — not held, verified gated

Both are TalkBank; every media URL under `media.talkbank.org` returns an auth
modal, not audio (checked 2026-08-18: `200 text/html` with an
`initAuthModals(...)` body, not the requested `.mp4`/`.cha`). Apple ships
**FluencyBank labels** (4,144 clips, same five-type schema) in the SEP-28k repo,
unusable here because the indexed audio can't be fetched. The labels are not the
corpus.

## Speaker gate

Synthetic two-speaker mixes built from PodcastFillers clips of different episodes.
**No real two-speaker recording of the hardware exists.** See `docs/EVAL.md` §9;
those numbers are labelled "unvalidated in real rooms" and must stay so.
