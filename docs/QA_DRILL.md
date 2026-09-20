# Echo — Judge Q&A Drill

27 of the hardest questions, grouped by attack surface, each with a 15–30 s spoken
answer. Every factual claim traces to a doc or eval artifact, cited after the
answer. Where the honest answer is a disclosed limitation, the answer **leads with
the limitation** and lands on the bounding fact — that pattern is the brand. No
answer invents a user study, clinical validation, or a number not in the repo.
Fallback for anything uncovered: state the limitation first, then the narrowest
true claim that bounds it, and cite `docs/EVAL.md` rather than memory.

**Spot-check these five first:** "GPT wrapper?" (#12) · "self-written 60-item
set?" (#1) · "wrong word — harm?" (#9) · "Apple/Google will ship this — moat?"
(#19) · "why not Deepgram `filler_words=true`?" (#18).

## Evidence

**1. [LIKELY] "Your eval set is self-written and only 60 items — why trust the
accuracy?"** We disclose it up front. The 60-item set was hand-authored (20
concrete, 20 proper-noun, 20 abstract/verb), scored by a fixed mechanical rule
(lowercase, strip punctuation, strip one trailing "'s"/plural "s") — no
embeddings, no LLM judge — under a freeze protocol. It proves the shipped prompt
pipeline works on constructed cases that mirror real stall shapes; it does NOT
prove performance on real aphasic speech — the explicit open item, pending
AphasiaBank access. *(EVAL §5; README gaps)*

**2. "n=60 is tiny."** True, and hand-built — we say so in the Devpost draft.
Sixty items split three ways shows a mechanism (context ablation collapsing
proper-noun recall 19/20→2/20 is the real finding, not the 96.7% top-line), not a
population claim. *(EVAL §5; DEVPOST)*

**3. "5/40 prolongation hit rate is terrible."** It's a deliberately conservative
lower bound, and the report says so: palindrome-looped conversational um/uh clips
contain internal phone transitions a held vowel doesn't, so the tracker is tested
on a harder case than it's designed for. The number that matters for restraint is
zero false fires over 120 s of running speech. The held-vowel ground-truth set
isn't recorded yet. *(EVAL §3, §8; eval/record_protocol.md)*

**4. "Misses two-thirds of fillers in noise."** At 5 dB SNR, recall drops to 0.33,
yes — we measured and published it. But precision holds at 1.00: the failure mode
is silence, not false alarms (miss-over-nag). That's the classifier in isolation;
the live pipeline's VAD + voiced-time gates aren't included, so it's conservative,
and the lav mic centimeters from the mouth (DJI Mic 2S) is the mitigation. *(EVAL §7)*

**5. "Who graded it — your own homework?"** A fixed mechanical rule, not human
judgment — exact match against a pre-written gold list per item. Removes grader
bias but isn't a substitute for third-party/clinician grading, which we haven't
done. *(EVAL §5)*

**6. "You missed your own 600 ms latency gate."** We publish the miss because the
number that matters survives it: 755 ms median is still ~1.7× earlier than the
1300 ms pause-timeout baseline — the comparison a speaker actually experiences.
Better to show a missed internal gate than move the goalpost. *(EVAL Table 2)*

## Clinical

**7. "No patient with aphasia has used this — so what does it prove?"** Correct,
and we lead with it — validated on podcast speech and a hand-built text eval, not
on people with aphasia. What we honestly claim: each design choice (1300 ms pause
tolerance, multimodal delivery, verify-don't-correct) traces to a citable SCA
principle. That's "designed against a documented principle," explicitly not
"validated by a study of Echo." *(PITCH design-principles; README gaps)*

**8. "Cognitive load of reading a card while struggling for a word?"** Fair, and
we have no user study. Delivery is deliberately multimodal and low-friction (a
large word on screen + an optional spoken cue), mirroring aphasia-friendly-formatting
research. Whether that reduces or adds load in the actual moment is an open
question needing real users, not our reasoning. *(PITCH table; Rose 2003/2011)*

**9. [LIKELY] "What if it suggests the wrong word — could it harm someone, put
words in their mouth?"** The design defends against exactly that: Echo never
speaks unprompted (autospeak off, confirm-to-speak) and shows ranked candidates,
not one forced answer. A miss costs one tap ("not it" promotes the next, excluding
rejected). The speaker always finishes in their own words; Echo never corrects or
overrides. *(PITCH Judge Q&A; frontend/app.js rejectCard)*

**10. "Why not co-design with an SLP or people with aphasia?"** Honestly —
hackathon timeline, no clinical access yet. We deliberately didn't claim
co-design: every choice is checked against published SCA principles rather than
invented, a narrower checkable claim than "validated by users." An SLP-guided
pilot is the explicit next step. *(PITCH; ROADMAP)*

**11. "Is this clinically validated?"** No. Every row of the principles table is
"designed against a documented principle," not "validated by an outcome study of
Echo" — no SLP co-design, no patient outcome study yet. We say so rather than let
the citations imply otherwise. *(PITCH clinical framing)*

## Technical

**12. [LIKELY] "Isn't this just a GPT wrapper?"** The prediction step calls an
LLM, but that's a fraction of the engineering. The core is the dual-channel stall
detector: a deterministic pure-Python state machine fusing five triggers, a
from-scratch acoustic model (FillerNet, ~136k params on PodcastFillers), a
rule-based prolongation detector, and speculative prefetch that serves cached
words at near-zero latency so the LLM's ~1.5 s round-trip is usually hidden. Swap
the LLM and the rest is untouched. *(ARCHITECTURE; predictor/base.py)*

**13. "Why not fine-tune your own model?"** That's the explicit next step —
EchoLM, a QLoRA-fine-tuned Qwen2.5-1.5B on reverse-dictionary data (3D-EX,
WordNet) + synthetic circumlocutions, served locally under 300 ms via llama.cpp.
Deferred because training/eval is a multi-day GPU job that would crowd out the
sensing + live-loop work that is the demo. The provider interface exists so EchoLM
drops in with zero refactor. *(ROADMAP Phase 3)*

**14. "You're streaming private conversation to Google's cloud — privacy?"** Only
text ever leaves the machine to the LLM (a few words of fragment + recent turns,
never raw audio). The acoustic channel (VAD, FillerNet, prolongation) runs
entirely locally against raw PCM. The fully-local path (on-device STT + EchoLM,
and the GX10 appliance) closes even the text-to-cloud gap, but isn't the demo path
today. *(predictor/gemini.py; ARCHITECTURE; DEPLOYMENT)*

**15. "Why Gemini and not Claude/GPT/a smaller model?"** gemini-3.5-flash was fast
and cheap enough for a sub-2 s loop once we set thinking_budget=0 (otherwise it
burns the output budget on hidden reasoning and returns nothing). Not
load-bearing: Claude (claude-haiku-4-5) and DeepSeek are implemented behind the
same interface, `PREDICTOR_PROVIDER` is a one-env-var swap, the seam EchoLM will
use. *(predictor/gemini.py; config.py; README)*

**16. "What breaks first at a thousand concurrent users?"** The architecture is
one `EchoSession` per server process — a documented hackathon-scale decision, not
a hidden limitation. Multi-tenant session management is a named roadmap item. The
per-request pieces (LLM call, acoustic model) are stateless and horizontally
scale; the shared-session assumption is what needs rework first. *(session.py
docstring; ROADMAP)*

**17. "What if it hallucinates an unrelated word?"** It's one ranked candidate
among up to three, never spoken unless tapped — a hallucination costs a glance,
not a spoken error. "not it" gets a replacement excluding it at similar latency.
We haven't measured a "wildly unrelated" rate — our eval labels top-1/top-3
correctness, not how bad the misses are. *(PITCH reject path; EVAL §5)*

## Comparative

**18. [LIKELY] "Why not just Deepgram `filler_words=true`?"** Restores filler text
only — nothing for prolongations, normalized by design even with filler words on
("uhhhh"→"uh" in Deepgram's own docs). It adds a cloud dependency on the
privacy-critical audio path, and detection is still gated on ASR finalization
latency vs our 125 ms analysis hop deciding directly from raw audio. *(PITCH
Pre-empt; ARCHITECTURE thesis)*

**19. [LIKELY] "Apple/Google will ship this in a year — your moat?"** We don't
have a moat in the startup sense and aren't claiming one — this is a hackathon
prototype. Narrow claim: to our knowledge this is the first real-time,
speaker-side co-pilot for anomia; the prediction idea was proposed but only tested
offline on transcripts. If a big platform ships the real-time version, that's the
outcome we'd want — 2 million people getting the tool matters more than who built
it first. *(PITCH novelty claim)*

**20. "How is this different from an AAC app like Proloquo?"** AAC apps make you
type or tap symbol grids — built for people who can't speak reliably at all. Echo
is for people who CAN speak and know the word but can't retrieve it in the moment
— it listens passively and offers the word inside their own spoken sentence, no
typing, no menus. *(PITCH; positioning)*

**21. "How is this different from Broca AI Speech?"** Broca listens to the OTHER
person and suggests whole reply sentences — partner support. Echo listens to YOU
and recovers YOUR word mid-sentence. Different mechanism, arguably different
dignity: you're still constructing and speaking your own sentence. *(PITCH
positioning)*

**22. "Purohit already showed ChatGPT can do circumlocution recovery — what's
new?"** Purohit 2023 is exactly our precedent, cited directly — 11/12 AphasiaBank
items recovered offline, on transcripts, after the fact. Nobody built the
real-time loop: live stall detection from raw audio in under a second, sub-2 s
prediction, delivery timed to help mid-sentence. That's what we built. *(PITCH
prior art; EVAL §5)*

## Curveballs

**23. "What would falsify your thesis?"** If a consumer ASR shipped a
disfluency-preserving mode that also un-normalizes prolongations by default, our
dual-channel argument for THAT gap would be moot (though prefetch latency and
context memory still stand). More concretely: if the self-recorded held-vowel set
showed the prolongation rule barely beats chance on real held vowels rather than
being a lower bound, that would falsify the "0.94/600 ms rule works" claim. We
built that eval precisely so it could say that. *(ARCHITECTURE thesis;
eval/record_protocol.md)*

**24. "Weakest part, honestly?"** The prediction accuracy number — 96.7% top-1 —
because it's on a set we wrote, not real aphasic speech, and it's the number most
likely not to generalize. We say so in our own "what we haven't proven yet."
*(DEVPOST)*

**25. "What did you cut for the deadline?"** Server-side streaming STT (Deepgram)
is an interface skeleton — browser STT is the demo path. On-device EchoLM is
speced but not trained. The self-recorded held-vowel eval is a written protocol,
not yet recorded. All three are listed as explicit known gaps, not silent
omissions. *(README gaps)*

**26. "Six more months — next thing?"** AphasiaBank consortium access and
re-running every eval against real aphasic speech — the single change that most
changes what we're allowed to claim. In parallel, training EchoLM, since the
recipe and integration seam are built. *(ROADMAP Phase 3, AphasiaBank; README)*

**27. "What's most likely to go wrong on stage now?"** A drifted prefetch cache
serving "live" instead of "prefetch" on the timed beat, or the venue network
dropping the live call. Both have rehearsed recoveries: the live path is the
~1.5 s fallback and we can show a real prior prefetch hit from history;
`PREDICTOR_PROVIDER=demo_fallback` serves a disclosed local backup on a genuine
network failure only. *(DEMO_SCRIPT drift recovery, Tier 3)*
