"""LocalPredictor (llama.cpp / llama-server) + the GBNF grammar.

Everything here runs against a STUB aiohttp server on an ephemeral port. No
model, no GPU, no network beyond loopback -- so these tests are meaningful on
CI and on a laptop with llama-server switched off.

THE LOAD-BEARING PROPERTY: `LocalPredictor.predict` must never raise. It is
called at the stall moment from inside the WebSocket loop
(`backend/pipeline.py::_predict` -> `backend/session.py`); an exception there
takes the socket down mid-conversation, which for this user means Echo goes
silent exactly when they need it. Every failure path -- refused connection,
timeout, HTTP 500, non-JSON body, prose instead of JSON, missing keys -- must
degrade to `[]`. Each one has a test below.
"""
import asyncio
import contextlib
import copy
import json
import re
import socket

from aiohttp import web

from backend.prompts import (
    CANDIDATES_SCHEMA,
    FEW_SHOTS,
    SYSTEM_PROMPT,
    candidates_gbnf,
)
from backend.predictor.base import (
    PredictionRequest,
    PredictorCapabilities,
    WordPredictor,
)
from backend.predictor.local import LocalPredictor

# --- stub llama-server -------------------------------------------------------


@contextlib.asynccontextmanager
async def stub_server(handler, path="/v1/chat/completions"):
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
    """A minimal OpenAI-compatible chat completion, as llama-server returns."""
    return {
        "id": "chatcmpl-stub",
        "object": "chat.completion",
        "model": "stub",
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
    where `seen` collects the JSON request bodies the stub received."""
    seen: list[dict] = []

    async def _wrapped(request):
        try:
            seen.append(await request.json())
        except Exception:  # a body we cannot parse is still worth recording
            seen.append({})
        return await handler(request)

    async def _main():
        async with stub_server(_wrapped) as base_url:
            p = LocalPredictor(base_url=base_url, **ctor)
            kwargs = predict_kwargs or {}
            return await p.predict(
                kwargs.pop("context", ["What did you have for breakfast?"]),
                kwargs.pop("fragment", "I made some toast in the, um"),
                **kwargs,
            )

    return asyncio.run(_main()), seen


# --- GBNF grammar (plan step B3) --------------------------------------------

def test_gbnf_is_a_non_empty_grammar_with_a_root_rule():
    g = candidates_gbnf()
    assert g.strip()
    assert re.search(r"^root\s*::=", g, re.M), "GBNF requires a rule named root"


def test_gbnf_mentions_every_schema_field():
    g = candidates_gbnf()
    for name in ("candidates", "word", "confidence"):
        assert f'\\"{name}\\"' in g, f"{name} is not constrained by the grammar"


def test_gbnf_is_derived_from_the_schema_not_a_literal():
    """If the grammar were pasted in as a literal string, a schema change would
    silently stop being enforced and the local model could emit a shape the
    parser drops."""
    mutated = copy.deepcopy(CANDIDATES_SCHEMA)
    item = mutated["properties"]["candidates"]["items"]
    item["properties"]["pos"] = {"type": "string"}
    item["required"] = ["word", "confidence", "pos"]

    base = candidates_gbnf()
    changed = candidates_gbnf(mutated)
    assert changed != base
    assert '\\"pos\\"' in changed
    assert '\\"pos\\"' not in base


def test_gbnf_omits_properties_that_are_not_required():
    mutated = copy.deepcopy(CANDIDATES_SCHEMA)
    mutated["properties"]["candidates"]["items"]["properties"]["notes"] = {"type": "string"}
    assert '\\"notes\\"' not in candidates_gbnf(mutated)


def _rule_refs(rhs: str) -> list[str]:
    """Bare identifiers in a GBNF right-hand side, scanning the way llama.cpp
    does: quoted literals and [char classes] are opaque."""
    out, i = [], 0
    while i < len(rhs):
        c = rhs[i]
        if c == '"':
            i += 1
            while i < len(rhs) and rhs[i] != '"':
                i += 2 if rhs[i] == "\\" else 1
            i += 1
        elif c == "[":
            i += 1
            while i < len(rhs) and rhs[i] != "]":
                i += 2 if rhs[i] == "\\" else 1
            i += 1
        elif c.isalpha():
            j = i
            while j < len(rhs) and (rhs[j].isalnum() or rhs[j] in "-_"):
                j += 1
            out.append(rhs[i:j])
            i = j
        else:
            i += 1
    return out


def test_gbnf_every_referenced_rule_is_defined():
    """A grammar referencing an undefined rule is rejected by llama.cpp at
    request time -- which would look exactly like 'the local model is broken'."""
    g = candidates_gbnf()
    defined, referenced = set(), set()
    for line in g.splitlines():
        if "::=" not in line:
            continue
        lhs, rhs = line.split("::=", 1)
        defined.add(lhs.strip())
        referenced.update(_rule_refs(rhs))
    assert "root" in defined
    assert referenced, "no rule references found -- the scanner is broken"
    assert referenced <= defined, f"undefined rules: {sorted(referenced - defined)}"


def test_gbnf_has_no_unreachable_rules():
    g = candidates_gbnf()
    rules = {}
    for line in g.splitlines():
        if "::=" in line:
            lhs, rhs = line.split("::=", 1)
            rules[lhs.strip()] = rhs
    reachable, stack = {"root"}, ["root"]
    while stack:
        for ref in _rule_refs(rules[stack.pop()]):
            if ref not in reachable:
                reachable.add(ref)
                stack.append(ref)
    assert set(rules) == reachable, f"dead rules: {sorted(set(rules) - reachable)}"


def test_gbnf_is_stable_across_calls():
    assert candidates_gbnf() == candidates_gbnf()


# --- capabilities (plan step B5) --------------------------------------------

def test_local_predictor_is_a_word_predictor():
    assert issubclass(LocalPredictor, WordPredictor)


def test_capabilities_declare_grammar_support_and_the_configured_window():
    p = LocalPredictor(ctx_size=4096)
    caps = p.capabilities
    assert isinstance(caps, PredictorCapabilities)
    assert caps.supports_grammar is True
    assert caps.max_context_tokens == 4096
    assert LocalPredictor(ctx_size=8192).capabilities.max_context_tokens == 8192


def test_constructor_defaults_match_the_planned_local_stack():
    """Defaults live here, NOT in backend/config.py (the integrator owns that
    file). A wrong default means the demo silently talks to nothing."""
    p = LocalPredictor()
    assert p.base_url == "http://127.0.0.1:8080"
    assert p.model == "Qwen3.8-27B-Q4_K_M"


def test_base_url_trailing_slash_is_tolerated():
    assert LocalPredictor(base_url="http://127.0.0.1:8080/").base_url == \
        "http://127.0.0.1:8080"


def test_predict_signature_matches_the_other_providers():
    import inspect
    params = inspect.signature(LocalPredictor.predict).parameters
    for name in ("excluded", "entities", "already_served"):
        assert name in params


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


def test_request_payload_carries_prompt_grammar_and_model():
    out, seen = _predict_against(
        _ok_handler,
        model="Qwen3.8-27B-Q4_K_M",
        predict_kwargs={
            "fragment": "I made some toast in the, um",
            "excluded": ["spoon"],
            "entities": ["Frank"],
            "already_served": ["fork"],
        },
    )
    assert out
    body = seen[0]
    assert body["model"] == "Qwen3.8-27B-Q4_K_M"
    assert body.get("stream") is False
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
    # grammar-constrained decoding, so the output is structurally valid JSON
    assert "candidates" in body["grammar"]
    assert body["grammar"] == candidates_gbnf()


def test_grammar_can_be_disabled_for_a_server_that_rejects_it():
    out, seen = _predict_against(_ok_handler, use_grammar=False)
    assert out and "grammar" not in seen[0]


def test_predict_request_reaches_the_local_provider_via_the_base_shim():
    seen: list[dict] = []

    async def handler(request):
        seen.append(await request.json())
        return await _ok_handler(request)

    async def _main():
        async with stub_server(handler) as base_url:
            p = LocalPredictor(base_url=base_url)
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


# --- failure paths: EVERY one must return [], never raise -------------------

def test_prose_instead_of_json_returns_empty():
    async def handler(request):
        return web.json_response(_completion("I think they mean the toaster."))

    assert _predict_against(handler)[0] == []


def test_truncated_json_returns_empty():
    async def handler(request):
        return web.json_response(_completion('{"candidates":[{"word":"toas'))

    assert _predict_against(handler)[0] == []


def test_http_500_returns_empty():
    async def handler(request):
        return web.Response(status=500, text="internal error")

    assert _predict_against(handler)[0] == []


def test_http_404_wrong_endpoint_returns_empty():
    async def handler(request):  # pragma: no cover - never reached
        return web.json_response(_completion(_candidates_json(("toaster", 1.0))))

    async def _main():
        async with stub_server(handler, path="/some/other/path") as base_url:
            return await LocalPredictor(base_url=base_url).predict([], "the um")

    assert asyncio.run(_main()) == []


def test_non_json_response_body_returns_empty():
    async def handler(request):
        return web.Response(status=200, text="<html>not json</html>",
                            content_type="text/html")

    assert _predict_against(handler)[0] == []


def test_missing_choices_returns_empty():
    async def handler(request):
        return web.json_response({"error": {"message": "no slot available"}})

    assert _predict_against(handler)[0] == []


def test_empty_choices_returns_empty():
    async def handler(request):
        return web.json_response({"choices": []})

    assert _predict_against(handler)[0] == []


def test_null_message_content_returns_empty():
    async def handler(request):
        return web.json_response({"choices": [{"message": {"content": None}}]})

    assert _predict_against(handler)[0] == []


def test_top_level_json_array_returns_empty():
    async def handler(request):
        return web.json_response([1, 2, 3])

    assert _predict_against(handler)[0] == []


def test_timeout_returns_empty_and_does_not_hang():
    """llama.cpp on a 16 GB card runs Q4_K_M partially offloaded, so a slow or
    wedged generation is a REAL scenario, not a hypothetical."""
    async def handler(request):
        await asyncio.sleep(5.0)
        return web.json_response(_completion(_candidates_json(("toaster", 1.0))))

    out, _ = _predict_against(handler, timeout_s=0.15)
    assert out == []


def test_connection_refused_returns_empty():
    """The overwhelmingly likely local failure: llama-server is not running."""
    async def _main():
        p = LocalPredictor(base_url=f"http://127.0.0.1:{_free_port()}",
                           timeout_s=2.0)
        return await p.predict(["ctx"], "I made some toast in the, um")

    assert asyncio.run(_main()) == []


def test_unroutable_host_returns_empty():
    async def _main():
        p = LocalPredictor(base_url="http://127.0.0.1:1/", timeout_s=2.0)
        return await p.predict([], "the um")

    assert asyncio.run(_main()) == []


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
            p = LocalPredictor(base_url=base_url)
            first = await p.predict([], "the um")
            second = await p.predict([], "the um")
            return first, second

    first, second = asyncio.run(_main())
    assert first == []
    assert [c.word for c in second] == ["toaster"]


# --- provider factory (plan step B7) ---------------------------------------

def test_factory_returns_local_predictor():
    from backend.config import Settings
    from backend.predictor import get_predictor

    s = Settings(
        predictor_provider="local",
        gemini_model="gemini-3.5-flash", gemini_api_key=None,
        claude_model="claude-haiku-4-5", claude_api_key=None,
        deepseek_model="deepseek-flash", deepseek_api_key=None,
        deepseek_base_url="https://api.deepseek.com",
        demo_fallback_inner="gemini", demo_fallback_timeout_s=1.5,
        stt_provider="mock", deepgram_api_key=None,
        max_candidates=2, pause_ms=1300, context_turns=6,
        acoustic_model="", acoustic_conf=0.7,
        prefetch=False, prefetch_every=3, entity_memory=False,
    )
    p = get_predictor(s)
    assert isinstance(p, LocalPredictor)
    assert p.max_candidates == 2
    # No local_* fields exist in Settings yet (the integrator adds them in I1),
    # so the factory must fall back to the constructor defaults rather than
    # raising AttributeError.
    assert p.base_url == "http://127.0.0.1:8080"
    assert p.model == "Qwen3.8-27B-Q4_K_M"


def test_factory_reads_local_settings_when_the_integrator_adds_them():
    """Forward-compatible with plan step I1 (local_llm_url / _model / _ctx)."""
    from dataclasses import dataclass

    from backend.predictor import get_predictor

    @dataclass(frozen=True)
    class FakeSettings:
        predictor_provider: str = "local"
        max_candidates: int = 3
        local_llm_url: str = "http://192.168.1.50:9090"
        local_llm_model: str = "some-other.gguf"
        local_llm_ctx: int = 2048

    p = get_predictor(FakeSettings())
    assert isinstance(p, LocalPredictor)
    assert p.base_url == "http://192.168.1.50:9090"
    assert p.model == "some-other.gguf"
    assert p.capabilities.max_context_tokens == 2048


def test_local_appears_in_the_unknown_provider_error_message():
    import pytest

    from dataclasses import dataclass

    from backend.predictor import get_predictor

    @dataclass(frozen=True)
    class FakeSettings:
        predictor_provider: str = "nonsense"
        max_candidates: int = 3

    with pytest.raises(ValueError, match="local"):
        get_predictor(FakeSettings())


# --- thinking-model regression ----------------------------------------------
# Found only by running against the real Qwen3.8-27B: its chat template defaults
# to reasoning_effort='xhigh', so llama-server put the ENTIRE grammar-constrained
# answer in message.reasoning_content and left message.content empty. Echo saw
# nothing at all while the model was answering correctly. 34 unit tests missed it
# because the stub server never emitted reasoning_content.

def test_payload_disables_thinking():
    """A thinking model spends the whole token budget on thoughts and returns an
    empty content -- the same failure gemini.py avoids with thinking_budget=0."""
    p = LocalPredictor()
    payload = p._build_payload(["ctx"], "I need the um")
    assert payload["enable_thinking"] is False
    assert payload["chat_template_kwargs"]["enable_thinking"] is False


def test_reasoning_content_is_read_when_content_is_empty():
    """Verbatim shape of the real llama-server reply for Qwen3.8-27B."""
    from backend.predictor.local import _text_of
    data = {"choices": [{"finish_reason": "stop", "index": 0, "message": {
        "role": "assistant",
        "content": "",
        "reasoning_content": '{"candidates": [{"word": "toaster", "confidence": 0.95}]}',
    }}]}
    assert _text_of(data) == '{"candidates": [{"word": "toaster", "confidence": 0.95}]}'


def test_content_still_wins_over_reasoning_content():
    from backend.predictor.local import _text_of
    data = {"choices": [{"message": {"content": "REAL", "reasoning_content": "THOUGHTS"}}]}
    assert _text_of(data) == "REAL"


def test_blank_reasoning_content_does_not_become_an_answer():
    from backend.predictor.local import _text_of
    assert _text_of({"choices": [{"message": {"content": "", "reasoning_content": "   "}}]}) is None


# --- reasoning_effort knob ---------------------------------------------------

def test_reasoning_effort_defaults_to_none_so_thinking_stays_off():
    """The shipped configuration must not enable thinking: it costs 1.3-2.8x
    latency on a loop that already runs ~2x the cloud path."""
    p = LocalPredictor()
    assert p.reasoning_effort is None
    payload = p._build_payload(["ctx"], "I need the um")
    assert payload["enable_thinking"] is False
    assert "reasoning_effort" not in payload


def test_reasoning_effort_removes_both_disable_keys():
    """Leaving either disable key in place makes the template suppress thinking,
    so reasoning_effort would silently do nothing and the arm would be a
    duplicate of the baseline rather than a measurement."""
    p = LocalPredictor(reasoning_effort="high")
    payload = p._build_payload(["ctx"], "I need the um")
    assert payload["reasoning_effort"] == "high"
    assert "enable_thinking" not in payload
    assert "chat_template_kwargs" not in payload


def test_every_documented_effort_level_is_accepted():
    for level in ("low", "medium", "high", "xhigh"):
        p = LocalPredictor(reasoning_effort=level)
        assert p._build_payload([], "x")["reasoning_effort"] == level


def test_empty_string_effort_is_treated_as_off():
    p = LocalPredictor(reasoning_effort="")
    payload = p._build_payload([], "x")
    assert "reasoning_effort" not in payload
    assert payload["enable_thinking"] is False


def test_grammar_can_be_disabled_independently_of_thinking():
    p = LocalPredictor(use_grammar=False, reasoning_effort="high")
    payload = p._build_payload([], "x")
    assert "grammar" not in payload
    assert payload["reasoning_effort"] == "high"
