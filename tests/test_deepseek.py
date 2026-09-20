"""DeepSeekPredictor -- OpenAI-compatible cloud provider.

Everything here runs against a STUB aiohttp server on an ephemeral loopback
port. No network beyond loopback, no API key, no DeepSeek account -- so these
tests are meaningful on CI and offline.

THE LOAD-BEARING PROPERTY (shared with the local provider): `predict` must
never raise. It runs at the stall moment from inside the WebSocket loop; an
exception there takes the socket down mid-conversation and Echo goes silent
exactly when the speaker needs it. Every failure path -- refused connection,
timeout, HTTP 500, non-JSON body, prose instead of JSON, missing keys --
degrades to `[]`. Each has a test below.
"""
import asyncio
import contextlib
import json
import socket

from aiohttp import web

from backend.prompts import FEW_SHOTS, SYSTEM_PROMPT
from backend.predictor.base import PredictionRequest, WordPredictor
from backend.predictor.deepseek import DeepSeekPredictor

# --- stub DeepSeek server ----------------------------------------------------


@contextlib.asynccontextmanager
async def stub_server(handler, path="/chat/completions"):
    """Serve `handler` on an ephemeral loopback port; yield its base URL."""
    app = web.Application()
    app.router.add_post(path, handler)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        await runner.cleanup()


def _completion(content: str) -> dict:
    """A minimal OpenAI-compatible chat completion, as DeepSeek returns."""
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "model": "deepseek-flash",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
    }


def _candidates_json(*pairs) -> str:
    return json.dumps({"candidates": [{"word": w, "confidence": c} for w, c in pairs]})


def _free_port() -> int:
    """A port that is bound and then released -- nothing is listening on it."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _predict_against(handler, *, predict_kwargs=None, **ctor):
    """Run one predict() call against a stub server. Returns (candidates, seen)
    where `seen` collects (json_body, headers) the stub received."""
    seen: list[tuple] = []

    async def _wrapped(request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        seen.append((body, dict(request.headers)))
        return await handler(request)

    async def _main():
        async with stub_server(_wrapped) as base_url:
            p = DeepSeekPredictor(api_key="test-key", base_url=base_url, **ctor)
            kwargs = predict_kwargs or {}
            return await p.predict(
                kwargs.pop("context", ["What did you have for breakfast?"]),
                kwargs.pop("fragment", "I made some toast in the, um"),
                **kwargs,
            )

    return asyncio.run(_main()), seen


# --- interface / construction ------------------------------------------------

def test_deepseek_is_a_word_predictor():
    assert issubclass(DeepSeekPredictor, WordPredictor)


def test_constructor_requires_api_key():
    import pytest
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        DeepSeekPredictor(api_key=None)


def test_constructor_defaults_match_the_verified_deepseek_stack():
    p = DeepSeekPredictor(api_key="k")
    assert p.model == "deepseek-flash"
    assert p.base_url == "https://api.deepseek.com"
    assert p.endpoint == "https://api.deepseek.com/chat/completions"


def test_base_url_trailing_slash_is_tolerated():
    p = DeepSeekPredictor(api_key="k", base_url="https://api.deepseek.com/")
    assert p.base_url == "https://api.deepseek.com"
    assert p.endpoint == "https://api.deepseek.com/chat/completions"


def test_predict_signature_matches_the_other_providers():
    import inspect
    params = inspect.signature(DeepSeekPredictor.predict).parameters
    for name in ("excluded", "entities", "already_served"):
        assert name in params


def test_payload_requests_json_mode_and_disables_reasoning():
    """deepseek-flash is a reasoning model: left on, it spends the whole token
    budget on reasoning_content and returns empty message content, so the parser
    serves nothing (measured: 10/60 frozen-set items came back empty). "none"
    disables it -- the same fix gemini.py applies with thinking_budget=0."""
    p = DeepSeekPredictor(api_key="k")
    payload = p._build_payload(["ctx"], "I need the um")
    assert payload["response_format"] == {"type": "json_object"}
    assert payload["reasoning_effort"] == "none"
    assert payload["stream"] is False


# --- happy path --------------------------------------------------------------

async def _ok_handler(request):
    return web.json_response(_completion(
        _candidates_json(("toaster", 0.92), ("oven", 0.4), ("grill", 0.1))))


def test_happy_path_parses_ranked_candidates():
    out, _ = _predict_against(_ok_handler)
    assert [c.word for c in out] == ["toaster", "oven", "grill"]
    assert out[0].confidence == 0.92


def test_max_candidates_truncates():
    out, _ = _predict_against(_ok_handler, max_candidates=2)
    assert [c.word for c in out] == ["toaster", "oven"]


def test_request_payload_carries_prompt_json_mode_model_and_auth():
    out, seen = _predict_against(
        _ok_handler,
        model="deepseek-flash",
        predict_kwargs={
            "fragment": "I made some toast in the, um",
            "excluded": ["spoon"],
            "entities": ["Frank"],
            "already_served": ["fork"],
        },
    )
    assert out
    body, headers = seen[0]
    assert body["model"] == "deepseek-flash"
    assert body.get("stream") is False
    # DeepSeek JSON mode requested
    assert body["response_format"] == {"type": "json_object"}
    # Bearer auth carried the key
    assert headers.get("Authorization") == "Bearer test-key"
    messages = body["messages"]
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    # same few-shot priming as the cloud providers
    assert len(messages) == 1 + 2 * len(FEW_SHOTS) + 1
    assert messages[1]["role"] == "user" and messages[2]["role"] == "assistant"
    user = messages[-1]["content"]
    assert "I made some toast in the, um" in user
    assert "Do NOT suggest: spoon" in user
    assert "Frank" in user
    assert "fork" in user and "already offered" in user.lower()


def test_predict_request_reaches_deepseek_via_the_base_shim():
    seen: list[dict] = []

    async def handler(request):
        seen.append(await request.json())
        return await _ok_handler(request)

    async def _main():
        async with stub_server(handler) as base_url:
            p = DeepSeekPredictor(api_key="k", base_url=base_url)
            return await p.predict_request(PredictionRequest(
                utterance="I made some toast in the, um",
                recent_turns=["What did you have for breakfast?"],
                summary="They were discussing the kitchen.",
                entities=["Frank"],
            ))

    out = asyncio.run(_main())
    assert [c.word for c in out][:1] == ["toaster"]
    user = seen[0]["messages"][-1]["content"]
    assert "kitchen" in user            # the summary is not dropped
    assert "Frank" in user


# --- failure / garbage paths: EVERY one must return [], never raise ----------

def test_empty_response_returns_empty():
    async def handler(request):
        return web.json_response(_completion(""))

    assert _predict_against(handler)[0] == []


def test_prose_instead_of_json_returns_empty():
    async def handler(request):
        return web.json_response(_completion("I think they mean the toaster."))

    assert _predict_against(handler)[0] == []


def test_truncated_json_returns_empty():
    async def handler(request):
        return web.json_response(_completion('{"candidates":[{"word":"toas'))

    assert _predict_against(handler)[0] == []


def test_garbage_top_level_json_returns_empty():
    async def handler(request):
        return web.json_response(_completion('["a","b",1,2]'))

    assert _predict_against(handler)[0] == []


def test_http_500_returns_empty():
    async def handler(request):
        return web.Response(status=500, text="internal error")

    assert _predict_against(handler)[0] == []


def test_http_401_bad_key_returns_empty():
    async def handler(request):
        return web.json_response({"error": {"message": "auth"}}, status=401)

    assert _predict_against(handler)[0] == []


def test_non_json_response_body_returns_empty():
    async def handler(request):
        return web.Response(status=200, text="<html>not json</html>",
                            content_type="text/html")

    assert _predict_against(handler)[0] == []


def test_missing_choices_returns_empty():
    async def handler(request):
        return web.json_response({"error": {"message": "rate limit"}})

    assert _predict_against(handler)[0] == []


def test_empty_choices_returns_empty():
    async def handler(request):
        return web.json_response({"choices": []})

    assert _predict_against(handler)[0] == []


def test_null_message_content_returns_empty():
    async def handler(request):
        return web.json_response({"choices": [{"message": {"content": None}}]})

    assert _predict_against(handler)[0] == []


def test_schema_deviant_candidates_degrade_item_by_item():
    async def handler(request):
        return web.json_response(_completion(json.dumps({"candidates": [
            "toaster",                                  # bare string: skipped
            {"word": "", "confidence": 0.9},            # empty word: skipped
            {"word": "oven", "confidence": "high"},     # bad conf -> 0.0
            {"word": "grill"},                          # missing conf -> 0.0
        ]})))

    out, _ = _predict_against(handler)
    assert [c.word for c in out] == ["oven", "grill"]
    assert out[0].confidence == 0.0


def test_timeout_returns_empty_and_does_not_hang():
    async def handler(request):
        await asyncio.sleep(5.0)
        return web.json_response(_completion(_candidates_json(("toaster", 1.0))))

    out, _ = _predict_against(handler, timeout=0.15)
    assert out == []


def test_connection_refused_returns_empty():
    """The likely cloud failure at demo time: no route / server down."""
    async def _main():
        p = DeepSeekPredictor(api_key="k",
                              base_url=f"http://127.0.0.1:{_free_port()}",
                              timeout=2.0)
        return await p.predict(["ctx"], "I made some toast in the, um")

    assert asyncio.run(_main()) == []


def test_cancelled_error_is_not_swallowed():
    """asyncio.CancelledError is the server shutting the task down, not a model
    failure -- it must propagate, never degrade to []."""
    import pytest

    async def handler(request):
        await asyncio.sleep(5.0)
        return web.json_response(_completion(_candidates_json(("toaster", 1.0))))

    async def _main():
        async with stub_server(handler) as base_url:
            p = DeepSeekPredictor(api_key="k", base_url=base_url, timeout=10.0)
            task = asyncio.ensure_future(p.predict(["ctx"], "the um"))
            await asyncio.sleep(0.1)
            task.cancel()
            await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(_main())


def test_repeated_calls_do_not_leak_or_break_after_a_failure():
    """A failed stall must not poison the next one -- the session lives for the
    whole conversation."""
    state = {"n": 0}

    async def handler(request):
        state["n"] += 1
        if state["n"] == 1:
            return web.Response(status=500, text="cold")
        return web.json_response(_completion(_candidates_json(("toaster", 0.9))))

    async def _main():
        async with stub_server(handler) as base_url:
            p = DeepSeekPredictor(api_key="k", base_url=base_url)
            first = await p.predict([], "the um")
            second = await p.predict([], "the um")
            return first, second

    first, second = asyncio.run(_main())
    assert first == []
    assert [c.word for c in second] == ["toaster"]


# --- provider factory --------------------------------------------------------

def _settings(**over):
    from backend.config import Settings
    base = dict(
        predictor_provider="deepseek",
        gemini_model="gemini-3.5-flash", gemini_api_key=None,
        claude_model="claude-haiku-4-5", claude_api_key=None,
        deepseek_model="deepseek-flash", deepseek_api_key="dummy",
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="gemini", demo_fallback_timeout_s=1.5,
        stt_provider="mock", deepgram_api_key=None,
        max_candidates=3, pause_ms=1300, context_turns=6,
        acoustic_model="", acoustic_conf=0.7,
        prefetch=False, prefetch_every=3, entity_memory=False,
    )
    base.update(over)
    return Settings(**base)


def test_factory_returns_deepseek_predictor():
    from backend.predictor import get_predictor
    p = get_predictor(_settings(max_candidates=2))
    assert isinstance(p, DeepSeekPredictor)
    assert p.max_candidates == 2
    assert p.model == "deepseek-flash"
    assert p.base_url == "https://api.deepseek.com"


def test_factory_passes_configured_model_and_base_url():
    from backend.predictor import get_predictor
    p = get_predictor(_settings(deepseek_model="deepseek-v4-pro",
                                deepseek_base_url="https://example.test/v1"))
    assert isinstance(p, DeepSeekPredictor)
    assert p.model == "deepseek-v4-pro"
    assert p.endpoint == "https://example.test/v1/chat/completions"


def test_deepseek_appears_in_the_unknown_provider_error_message():
    import pytest
    from backend.predictor import get_predictor
    with pytest.raises(ValueError, match="deepseek"):
        get_predictor(_settings(predictor_provider="nonsense"))
