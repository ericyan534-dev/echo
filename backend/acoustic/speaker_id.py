"""Who is talking -- speaker attribution by voice, not by loudness.

WHY THIS EXISTS, AND WHY IT REVERSES AN EARLIER DECISION
--------------------------------------------------------
Echo's first attempt at "is this the wearer" was proximity only: level and
spectral tilt, no voice model, chosen deliberately to avoid asking anyone to
enrol their voice. That approach was then measured, and it does not work.
`docs/EVAL.md` section 9: bystander suppression **0.000** at 0, 3 and 6 dB,
and 0.448 at 12 dB only by muting the wearer 18-26% of the time.

Then the whole stack was run on real two-speaker audio for the first time
(APROCSA -- a person with aphasia and a clinician, one room, one microphone),
and the cost showed up as a number: Echo fired during the CLINICIAN's speech on
0.206-0.465 of their utterances, and the clinician's words landed in the
fragment that gets sent to the predictor. Echo was suggesting words for the
wrong person's sentences.

Proximity cannot fix that. Two people at conversational distance in one room
are not separable by level, which is exactly what the 0.000 measured. Voice is
separable. So this uses a speaker embedding (ECAPA-TDNN, the standard
VoxCeleb-trained model) and costs one short enrolment at setup.

That is a real change to the product, not just to the code: the wearer now has
to say a few sentences once. It is recorded here rather than buried because the
earlier scope decision explicitly ruled enrolment out, and the evidence for
revisiting it is the measurement above.

TWO MODES, AND THE HONEST DIFFERENCE BETWEEN THEM
-------------------------------------------------
`score_live` is what the live path can actually do: one enrolment centroid,
cosine similarity per segment, no lookahead. It is deployable today.

`attribute_session` additionally clusters the whole session and asks which
cluster the enrolment sits in. It is better -- an absolute cosine threshold is
sensitive to microphone, room and how much the speaker's voice has changed
since enrolment, whereas "closer to A than to B" is not -- but it needs the
recording to exist first. It is what the offline evaluation uses, and the
difference is stated wherever a number from it is reported.

FAIL OPEN, ALWAYS
-----------------
Every path returns None when it does not know: no enrolment, too little audio,
or two clusters that are not actually distinguishable. None means unknown, and
unknown must never suppress -- an aid that goes silent because it is unsure who
is speaking is worse than one that occasionally answers the wrong person.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger("echo.speaker")

SR = 16_000
MIN_SEG_S = 0.6          # ECAPA is unreliable below this; return None instead
_MODEL = None


def get_encoder(device: str = "cpu"):
    """Load-once ECAPA-TDNN encoder.

    local_strategy=COPY is required on Windows: speechbrain's default is to
    symlink out of the HF cache, and a non-elevated Windows account cannot
    create symlinks (WinError 1314).
    """
    global _MODEL
    if _MODEL is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        _MODEL = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="models/ecapa",
            local_strategy=LocalStrategy.COPY,
            run_opts={"device": device})
    return _MODEL


def embed(segments: list[np.ndarray], device: str = "cpu") -> np.ndarray | None:
    """(N, 192) L2-normalised embeddings, or None if nothing was long enough."""
    import torch

    usable = [s for s in segments if len(s) >= int(MIN_SEG_S * SR)]
    if not usable:
        return None
    enc = get_encoder(device)
    out = []
    for s in usable:
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(s[None, :].astype("float32")))
        out.append(e.squeeze().cpu().numpy())
    m = np.stack(out)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


class SpeakerID:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.centroid: np.ndarray | None = None

    # --- enrolment -------------------------------------------------------
    def enroll(self, segments: list[np.ndarray]) -> bool:
        """Build the wearer's centroid from a handful of their utterances."""
        m = embed(segments, self.device)
        if m is None or len(m) == 0:
            return False
        c = m.mean(axis=0)
        self.centroid = c / max(np.linalg.norm(c), 1e-9)
        return True

    # --- live path -------------------------------------------------------
    def score_live(self, segment: np.ndarray, margin: float = 0.25) -> float | None:
        """Wearer confidence for one segment, using only the enrolment.

        `margin` is where cosine similarity is treated as 50/50. ECAPA on
        VoxCeleb puts same-speaker pairs well above it and different-speaker
        pairs below, but the exact value moves with microphone and room -- so
        this returns a graded confidence and lets the caller's threshold do the
        deciding, rather than pretending to a hard verdict.
        """
        if self.centroid is None:
            return None
        m = embed([segment], self.device)
        if m is None:
            return None
        sim = float(m[0] @ self.centroid)
        return float(np.clip(0.5 + (sim - margin) * 1.6, 0.0, 1.0))

    # --- offline path ----------------------------------------------------
    def attribute_session(self, segments: list[np.ndarray],
                          n_speakers: int = 2) -> list[float | None]:
        """Cluster the session, then ask which cluster the enrolment is in.

        Returns one confidence per input segment (None for segments too short
        to embed, preserving input order).

        This is more robust than an absolute threshold because it asks a
        relative question -- closer to the wearer's cluster or the other one --
        which survives a change of room or microphone that would move every
        absolute similarity at once.
        """
        idx = [i for i, s in enumerate(segments) if len(s) >= int(MIN_SEG_S * SR)]
        out: list[float | None] = [None] * len(segments)
        if self.centroid is None or len(idx) < n_speakers * 2:
            return out
        m = embed([segments[i] for i in idx], self.device)
        if m is None:
            return out

        cents = _kmeans_cosine(m, n_speakers)
        if cents is None:
            return out
        sims_to_enrol = cents @ self.centroid
        wearer = int(np.argmax(sims_to_enrol))
        # If the clusters are not actually distinguishable there is nothing to
        # gate on, and inventing a verdict here is how a real speaker gets
        # muted. Say unknown.
        if float(np.max(sims_to_enrol) - np.min(sims_to_enrol)) < 0.05:
            log.info("speaker clusters indistinguishable; attribution withheld")
            return out

        sims = m @ cents.T
        for k, i in enumerate(idx):
            s_w = sims[k, wearer]
            s_o = np.max(np.delete(sims[k], wearer))
            out[i] = float(np.clip(0.5 + (s_w - s_o) * 2.0, 0.0, 1.0))
        return out


def _kmeans_cosine(m: np.ndarray, k: int, iters: int = 40,
                   seed: int = 13) -> np.ndarray | None:
    """Spherical k-means. Not sklearn: this environment's sklearn import chain
    pulls a pandas built against NumPy 1.x and raises under NumPy 2.4."""
    if len(m) < k:
        return None
    rng = np.random.RandomState(seed)
    # k-means++ style seeding on cosine distance -- a random init on 192-dim
    # embeddings collapses to one cluster often enough to matter.
    cents = [m[rng.randint(len(m))]]
    for _ in range(1, k):
        d = 1.0 - np.max(m @ np.stack(cents).T, axis=1)
        p = np.maximum(d, 0) ** 2
        if p.sum() <= 0:
            cents.append(m[rng.randint(len(m))])
        else:
            cents.append(m[rng.choice(len(m), p=p / p.sum())])
    c = np.stack(cents)
    for _ in range(iters):
        lab = np.argmax(m @ c.T, axis=1)
        new = []
        for j in range(k):
            sel = m[lab == j]
            if len(sel) == 0:
                new.append(c[j])
            else:
                v = sel.mean(axis=0)
                new.append(v / max(np.linalg.norm(v), 1e-9))
        new = np.stack(new)
        if np.allclose(new, c, atol=1e-6):
            break
        c = new
    return c
