"""DeepSeek predictor -- OpenAI-compatible cloud provider.

DeepSeek exposes an OpenAI-compatible `chat/completions` endpoint, so this
provider is a drop-in sibling of the Gemini/Claude cloud path: it reuses the
SAME prompt assets (SYSTEM_PROMPT, FEW_SHOTS, build_user_text) and the SAME
tolerant candidate parser as gemini.py, so its outputs match those providers
field for field. The transport is `aiohttp` (already a dependency; the local
provider uses it too), not an SDK, so importing this module never requires an
extra package or an API key -- only a live `predict` call touches the network.

Two disciplines are inherited from the two reference providers:

  * PROMPT/PARSE PARITY WITH GEMINI. `_build_messages` mirrors claude.py /
    local.py (system + few-shot user/assistant pairs + the real user turn), and
    parsing goes through gemini._parse via local._text_of -- the exact tolerant
    path the other OpenAI-shaped provider (local) uses. Structured output is
    requested with response_format={"type":"json_object"} (DeepSeek's JSON
    mode), the OpenAI-compatible analogue of Gemini's response_schema.

  * FAILURE IS ROUTINE, NOT EXCEPTIONAL (from local.py). `predict` is called at
    the stall moment from inside the WebSocket loop (backend/pipeline.py); an
    exception there kills the socket and Echo goes silent exactly when the
    speaker needs it. So `predict` NEVER raises: refused connection, timeout,
    HTTP error, non-JSON body, or prose instead of JSON all degrade to `[]`
    plus one ASCII log line. `asyncio.CancelledError` is deliberately NOT
    swallowed -- that is the server shutting the task down, not a model failure.
    The request timeout is kept tight because this runs on the ~1-2s
    stall->word loop.
"""
from __future__ import annotations

import asyncio
import logging

import aiohttp

from ..prompts import FEW_SHOTS, SYSTEM_PROMPT, build_user_text
from ..schemas import Candidate
from .base import WordPredictor
from .gemini import _parse, _shot_answer  # same tolerant parser / few-shot format
from .local import _text_of  # same OpenAI-shaped response extractor

log = logging.getLogger("echo.predictor.deepseek")


class DeepSeekPredictor(WordPredictor):
    def __init__(
        self,
        api_key: str | None,
        model: str = "deepseek-flash",
        base_url: str = "https://api.deepseek.com",
        max_candidates: int = 3,
        timeout: float = 3.0,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required for the DeepSeek predictor.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_candidates = max_candidates
        self.timeout = timeout

    @property
    def endpoint(self) -> str:
        # DeepSeek's OpenAI-compatible completions URL. The verified live path is
        # https://api.deepseek.com/chat/completions -- the base already carries
        # (or omits) any version segment, so nothing is appended but the route.
        return f"{self.base_url}/chat/completions"

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
        return {
            "model": self.model,
            "messages": self._build_messages(context, fragment, excluded, entities,
                                             already_served),
            "temperature": 0.2,
            "max_tokens": 256,
            "stream": False,
            # DeepSeek JSON mode -- the OpenAI-compatible analogue of Gemini's
            # response_schema. The shared prompt already instructs the JSON
            # shape; this pins the response to a JSON object.
            "response_format": {"type": "json_object"},
            # deepseek-flash is a REASONING model: left on, it spends the whole
            # max_tokens budget on reasoning_content and returns empty message
            # content (finish_reason="length"), so the parser sees no JSON and
            # serves nothing. Measured on the frozen set: 10/60 items came back
            # empty this way, and the reasoning added ~0.6-0.8s of latency the
            # ~1-2s stall->word loop cannot afford. "none" disables it -- the
            # same fix gemini.py applies with thinking_budget=0 and local.py
            # with enable_thinking=False. With it off: 9 of those 10 items
            # answer correctly and per-item latency drops to ~0.8s mean.
            "reasoning_effort": "none",
        }

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
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json"}
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            # One session per call: a long-lived session is bound to the event
            # loop that created it, a subtle crash across reconnects. The TLS
            # handshake cost is noise next to cloud token generation.
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(self.endpoint, json=payload,
                                        headers=headers) as resp:
                    if resp.status != 200:
                        log.warning("[deepseek] HTTP %s from %s",
                                    resp.status, self.endpoint)
                        return []
                    # content_type=None: a mislabeled body must not become an
                    # exception at the stall moment.
                    data = await resp.json(content_type=None)
        except asyncio.CancelledError:
            # The server is shutting the task down -- not a model failure. Never
            # swallow this (mirrors local.py).
            raise
        except Exception as exc:  # noqa: BLE001 - see the module docstring
            log.warning("[deepseek] call failed (%s: %s); serving no candidates.",
                        type(exc).__name__, exc)
            return []
        return _parse(_text_of(data), self.max_candidates)
