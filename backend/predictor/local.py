"""Local predictor -- llama.cpp `llama-server`, OpenAI-compatible endpoint.

Why HTTP and not an in-process binding: `llama-cpp-python` means compiling a
CUDA wheel on Windows, and all we need is inference. A prebuilt `llama-server`
plus `aiohttp` (already a dependency) adds NO new runtime Python dependency,
and it keeps a 17 GB model load OUT of the web-server process -- restarting
Echo does not mean reloading the model.

Two things make this provider different from the cloud ones:

  * GRAMMAR. llama-server accepts a GBNF grammar (`backend.prompts
    .candidates_gbnf`) that constrains decoding, so an invalid JSON shape
    cannot be sampled at all. A small quantized model left unconstrained is
    the likeliest cause of an empty local result.
  * FAILURE IS ROUTINE, NOT EXCEPTIONAL. The server is a separate process on
    the same laptop: it may not be running, it may still be loading 17 GB, it
    may be swapping because Q4_K_M does not fit 16 GB of VRAM. So `predict`
    NEVER raises. It is called at the stall moment from inside the WebSocket
    loop (`backend/pipeline.py::_predict`); an exception there kills the
    socket and Echo goes silent exactly when the speaker needs it. Every
    failure path -- refused connection, timeout, HTTP error, non-JSON body,
    prose instead of JSON -- degrades to `[]` plus one ASCII log line.
    (`asyncio.CancelledError` is deliberately NOT swallowed: that is the
    server shutting the task down, not a model failure.)

Defaults point at the plan's local stack (`Qwen3.8-27B-Q4_K_M` on
127.0.0.1:8080) and live HERE, in the constructor, not in
`backend/config.py` -- the integrator wires config last.
"""
from __future__ import annotations

import logging

import aiohttp

from ..prompts import FEW_SHOTS, SYSTEM_PROMPT, build_user_text, candidates_gbnf
from ..schemas import Candidate
from .base import PredictorCapabilities, WordPredictor
from .gemini import _parse, _shot_answer  # same tolerant parser / few-shot format

log = logging.getLogger("echo.predictor.local")


class LocalPredictor(WordPredictor):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        model: str = "Qwen3.8-27B-Q4_K_M",
        max_candidates: int = 3,
        ctx_size: int = 4096,
        timeout_s: float = 8.0,
        temperature: float = 0.2,
        max_tokens: int = 256,
        use_grammar: bool = True,
        typical_latency_ms: int = 3000,
        reasoning_effort: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_candidates = max_candidates
        self.ctx_size = ctx_size
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.use_grammar = use_grammar
        # Honest default: the cloud path measures ~1.4-2.1 s end to end, and a
        # 27B Q4_K_M partially offloaded on 16 GB of VRAM is not expected to
        # beat that. Measured: ~3.9 s median at -ngl 56 with ~1 GB of VRAM
        # headroom. Beware -- at -ngl 60 on a busy desktop the driver spills
        # over PCIe and the SAME model takes ~21 s while reporting itself idle
        # (see scripts/setup_local_llm.py DEFAULT_NGL).
        self.typical_latency_ms = typical_latency_ms
        # Thinking depth. None (the shipped default) DISABLES thinking, because
        # a thinking model spends the token budget on thoughts this loop cannot
        # wait for. Set to "low"/"medium"/"high"/"xhigh" to enable a level --
        # measured in eval/run_local_ablation.py. Either way the answer arrives
        # in reasoning_content when thinking is on, which _text_of handles.
        self.reasoning_effort = reasoning_effort

    @property
    def endpoint(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    @property
    def capabilities(self) -> PredictorCapabilities:
        # max_context_tokens is the SERVER's --ctx-size: the context assembler
        # budgets against this, and overshooting a local window silently drops
        # the oldest turns (or errors) rather than degrading gracefully.
        return PredictorCapabilities(
            max_context_tokens=self.ctx_size,
            typical_latency_ms=self.typical_latency_ms,
            supports_grammar=True,
        )

    def _build_messages(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for inp, out in FEW_SHOTS:
            messages.append({"role": "user",
                             "content": build_user_text(inp["context"], inp["fragment"])})
            messages.append({"role": "assistant", "content": _shot_answer(out)})
        messages.append({"role": "user",
                         "content": build_user_text(context, fragment, excluded,
                                                    entities, already_served)})
        return messages

    def _build_payload(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": self._build_messages(context, fragment, excluded, entities,
                                             already_served),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            # Qwen3.8's chat template defaults to reasoning_effort='xhigh'.
            # Left on, llama-server routes the ENTIRE generation into
            # message.reasoning_content and returns an empty message.content --
            # measured: the model answered "toaster" correctly and Echo saw
            # nothing at all. It also spends the token budget on thoughts, which
            # this loop cannot afford. Same failure the Gemini provider avoids
            # with thinking_budget=0 (see backend/predictor/gemini.py).
            # Sent two ways because llama-server has accepted both spellings
            # across releases, and an unknown key is ignored rather than fatal.
            "enable_thinking": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if self.reasoning_effort:
            # Opt in to thinking at a level. Both disable-keys must go, or the
            # template suppresses thinking and reasoning_effort does nothing.
            payload.pop("enable_thinking", None)
            payload.pop("chat_template_kwargs", None)
            payload["reasoning_effort"] = self.reasoning_effort
        if self.use_grammar:
            # llama.cpp extension to the OpenAI schema. Omit it (use_grammar=
            # False) for a server build that rejects unknown fields.
            payload["grammar"] = candidates_gbnf()
        return payload

    async def predict(
        self,
        context: list[str],
        fragment: str,
        excluded: list[str] | None = None,
        entities: list[str] | None = None,
        already_served: list[str] | None = None,
    ) -> list[Candidate]:
        payload = self._build_payload(context, fragment, excluded, entities,
                                      already_served)
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            # One session per call: the alternative is a long-lived session
            # bound to whichever event loop created it, which is a subtle
            # crash waiting to happen across reconnects. A loopback TCP
            # handshake is noise next to local token generation.
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.endpoint, json=payload) as resp:
                    if resp.status != 200:
                        log.warning("[local] llama-server HTTP %s from %s",
                                    resp.status, self.endpoint)
                        return []
                    # content_type=None: a server that mislabels the body must
                    # not turn into an exception at the stall moment.
                    data = await resp.json(content_type=None)
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            log.warning("[local] llama-server call failed (%s: %s); serving no "
                        "candidates. Is llama-server running at %s?",
                        type(exc).__name__, exc, self.base_url)
            return []
        return _parse(_text_of(data), self.max_candidates)


def _text_of(data) -> str | None:
    """Pull the assistant text out of an OpenAI-shaped completion.

    Tolerates every deviation seen from local servers: a top-level list, an
    error object with no `choices`, an empty `choices`, a null content, or the
    older `text` field used by non-chat completions.
    """
    if not isinstance(data, dict):
        return None
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first = choices[0]
    if not isinstance(first, dict):
        return None
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        # Belt and braces for thinking models. The payload asks for thinking to
        # be disabled, but a template that ignores the flag puts the whole
        # (grammar-constrained, perfectly valid) answer in reasoning_content and
        # leaves content empty. Reading it back costs nothing and turns a total
        # blackout into a working prediction.
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning
    text = first.get("text")
    return text if isinstance(text, str) and text.strip() else None
