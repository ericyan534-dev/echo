"""Provider-agnostic predictor interface.

Every backend (Gemini, Claude, mock, local) implements `predict`. The pipeline
depends only on this ABC, so switching providers never touches pipeline code.

THE `predict_request` SHIM AND WHY IT EXISTS
-------------------------------------------
`docs/ROADMAP.md` (the EchoLM row of the Phase-3 recipe table) promises that a
local model drops in behind `WordPredictor` with **zero refactor** -- that the
provider abstraction was built for exactly this. A local model needs strictly
more information than the cloud providers do: the rolling summary as its own
field, and a declared context budget it can assemble against (see
`backend/context.py`, whose `ContextPayload` this mirrors field for field).

The naive way to give it that is to widen `predict(...)` again, which would
mean editing every provider -- i.e. exactly the refactor we promised was
unnecessary. So instead there is a second, richer entry point,
`predict_request(PredictionRequest)`, whose BASE implementation unpacks the
request back onto the existing `predict(...)` call. Gemini, Claude, mock and
demo_fallback therefore answer `predict_request` correctly with no edit at all;
a provider that wants the extra structure (`LocalPredictor`) may override it.

`predict_request` is deliberately NOT abstract, for the same reason.
"""
from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..schemas import Candidate


@dataclass(frozen=True)
class PredictorCapabilities:
    """What a provider can do, so callers stop guessing.

    `max_context_tokens` is what the context assembler budgets against (see
    `backend.context.ContextBuilder`): a 4k local model must not be handed a
    prompt sized for a cloud model. `supports_grammar` defaults to False
    because a provider that ignores a grammar it was handed would silently
    produce unconstrained output; only a provider that really constrains
    decoding may claim it.
    """

    max_context_tokens: int = 8192
    typical_latency_ms: int = 1500
    supports_grammar: bool = False


@dataclass(frozen=True)
class PredictionRequest:
    """One prediction ask, in assembled form.

    Mirrors `backend.context.ContextPayload` field for field on purpose: the
    context builder produces the layers, this carries them to the provider.
    """

    utterance: str
    recent_turns: list[str]
    summary: str = ""
    entities: list[str] = field(default_factory=list)
    excluded: list[str] = field(default_factory=list)
    already_served: list[str] = field(default_factory=list)


# The rolling summary has no slot in the legacy `predict(...)` signature, so the
# shim folds it into the context list as one labelled line rather than dropping
# it. Placed FIRST because it summarizes the oldest material -- the context list
# is rendered to the model in order (see backend.prompts.build_user_text).
SUMMARY_PREFIX = "Earlier in this conversation (summary): "

_DEFAULT_CAPABILITIES = PredictorCapabilities()

_OPTIONAL_HINTS = ("excluded", "entities", "already_served")
_hint_cache: dict[object, frozenset[str]] = {}


def _accepted_hints(predict_fn) -> frozenset[str]:
    """Which optional hints a concrete `predict` will actually accept.

    Nothing in the codebase forces an implementer to take the optional hints,
    and the original interface was two arguments. Passing a keyword such a
    predictor never declared would raise TypeError at the stall moment, so the
    shim asks first and drops what cannot be delivered. A `**kwargs` provider
    (demo_fallback) accepts everything.
    """
    cached = _hint_cache.get(predict_fn)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(predict_fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables
        accepted = frozenset(_OPTIONAL_HINTS)
    else:
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
            accepted = frozenset(_OPTIONAL_HINTS)
        else:
            accepted = frozenset(h for h in _OPTIONAL_HINTS if h in params)
    _hint_cache[predict_fn] = accepted
    return accepted


class WordPredictor(ABC):
    @abstractmethod
    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        """Return ranked candidate words for the speaker's intended word.

        *excluded*: words already rejected by the speaker this stall (reject
        path) -- implementations should avoid re-suggesting them. Optional and
        default-None so this is a backward-compatible signature extension:
        every existing 2-arg call site (incl. the prefetch shadow path) still
        works unchanged.

        *entities*: salient names/places/things mentioned earlier in the
        conversation but outside the recent-turns context window (see
        backend.entities.EntityTracker) -- implementations should surface
        them as a hint (e.g. via backend.prompts.build_user_text). Optional
        and default-None for the same backward-compatibility reason.

        *already_served*: words offered for EARLIER gaps in this same utterance.
        Distinct from *excluded* -- those were rejected as wrong; these were
        fine for a previous gap the speaker has moved past. Needed because v3
        no longer truncates the fragment, so the earlier gap is still visible
        in it. Optional and default-None, same reason.
        """
        raise NotImplementedError

    @property
    def capabilities(self) -> PredictorCapabilities:
        """Conservative defaults. Providers that know better override this."""
        return _DEFAULT_CAPABILITIES

    async def predict_request(self, req: PredictionRequest) -> list[Candidate]:
        """Richer entry point; unpacks onto `predict` so nothing else changes.

        See the module docstring: this shim is what makes the ROADMAP's
        "zero refactor" claim true. Override it only in a provider that can
        genuinely use the extra structure, and keep the behaviour identical to
        what this default would produce.
        """
        context = list(req.recent_turns)
        if req.summary:
            context.insert(0, f"{SUMMARY_PREFIX}{req.summary}")
        hints = _accepted_hints(type(self).predict)
        kwargs = {}
        if "excluded" in hints:
            kwargs["excluded"] = list(req.excluded)
        if "entities" in hints:
            kwargs["entities"] = list(req.entities)
        if "already_served" in hints:
            kwargs["already_served"] = list(req.already_served)
        return await self.predict(context, req.utterance, **kwargs)
