"""Prompt assets shared by every predictor provider.

Keeping the system prompt, few-shots, and JSON schema here (provider-agnostic)
is what makes Gemini <-> Claude a drop-in swap: each provider just adapts these
to its own request format.
"""
from __future__ import annotations

import re

SYSTEM_PROMPT = (
    "You assist a person with aphasia (anomia) who is mid-sentence and cannot "
    "retrieve a word. Given the recent conversation and their unfinished "
    "utterance (which may contain filler words, pauses, or a description of the "
    "word they are grasping for), output the 1-3 most likely words they intend, "
    "ranked most-likely first.\n"
    "Rules:\n"
    "- Output ONLY the candidate word(s) the speaker is trying to say — never a "
    "full sentence, an explanation, or a question.\n"
    "- Prefer the single concrete word the description points to "
    '(e.g. "the thing you put bread in that gets hot" -> "toaster").\n'
    "- If a name or place mentioned earlier in the conversation fits, use it.\n"
    "- Be fast and decisive."
)

# (input, expected_words) pairs used as few-shot priming for the LLM providers.
FEW_SHOTS: list[tuple[dict, list[str]]] = [
    (
        {"context": ["So what did you have for breakfast?"],
         "fragment": "I made some toast in the, um, the thing, you know"},
        ["toaster", "oven"],
    ),
    (
        {"context": ["Where are you traveling next month?"],
         "fragment": "We're flying to, uh, the big city in, in Japan"},
        ["Tokyo", "Osaka", "Kyoto"],
    ),
    (
        {"context": ["Did the doctor say anything?"],
         "fragment": "She told me to take my, my, the pills for the, the pressure"},
        ["medication", "blood pressure pills", "prescription"],
    ),
]

# JSON schema for the structured response (used by Claude output_config and as
# documentation for Gemini's response_mime_type=application/json output).
CANDIDATES_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "word": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["word", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["candidates"],
    "additionalProperties": False,
}


# --- GBNF grammar (llama.cpp / local provider) ------------------------------
# The cloud providers get structural validity from a server-side schema
# (Gemini response_schema, Claude output_config). llama-server has no such
# feature; what it has is GBNF, a grammar that constrains DECODING, so an
# invalid shape is not merely rejected afterwards -- it cannot be sampled at
# all. That matters more locally than in the cloud: a small quantized model
# left unconstrained is the likeliest source of "the local path returns
# nothing", and a retry costs a second the speaker does not have.
#
# The grammar is DERIVED from CANDIDATES_SCHEMA rather than pasted in as a
# literal, so a schema change cannot silently stop being enforced.

_GBNF_TERMINALS: dict[str, str] = {
    "string": 'string ::= "\\"" ( [^"\\\\] | "\\\\" ["\\\\/bfnrt] )* "\\""',
    "number": 'number ::= "-"? ( [0-9] | [1-9] [0-9]* ) ( "." [0-9]+ )? ( [eE] [-+]? [0-9]+ )?',
    "boolean": 'boolean ::= "true" | "false"',
    "ws": "ws ::= [ \\t\\n]*",
}


def _gbnf_rule_name(prefix: str, key: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", key).strip("-").lower() or "field"
    return f"{prefix}-{slug}"


def _gbnf_key_literal(key: str) -> str:
    """A JSON object key as a GBNF string literal: word -> "\\"word\\""."""
    escaped = key.replace("\\", "\\\\").replace('"', '\\"')
    return '"\\"' + escaped + '\\""'


def _gbnf_emit(node: dict, name: str, rules: dict[str, str], used: set[str]) -> str:
    """Emit rules for `node` and return the rule/terminal it is referenced by.

    Only the schema subset CANDIDATES_SCHEMA uses is handled (object, array,
    string, number, boolean); anything else falls back to `string`, which is
    permissive rather than wrong.
    """
    kind = node.get("type")
    if kind == "object":
        rules[name] = ""  # reserve the slot so `root` stays the FIRST rule
        used.add("ws")
        props = node.get("properties") or {}
        # Only REQUIRED properties are generated, in `required` order: the
        # grammar's job is to pin the shape we parse, and an optional key the
        # parser ignores is just an invitation to spend tokens on it.
        required = [k for k in (node.get("required") or []) if k in props]
        tokens = ['"{"', "ws"]
        for i, key in enumerate(required):
            if i:
                tokens += ['","', "ws"]
            ref = _gbnf_emit(props[key], _gbnf_rule_name(name, key), rules, used)
            tokens += [_gbnf_key_literal(key), "ws", '":"', "ws", ref, "ws"]
        tokens.append('"}"')
        rules[name] = " ".join(tokens)
        return name
    if kind == "array":
        rules[name] = ""
        used.add("ws")
        item = _gbnf_emit(node.get("items") or {}, f"{name}-item", rules, used)
        rules[name] = " ".join(
            ['"["', "ws", f'( {item} ( ws "," ws {item} )* )?', "ws", '"]"'])
        return name
    if kind in ("number", "integer"):
        used.add("number")
        return "number"
    if kind == "boolean":
        used.add("boolean")
        return "boolean"
    used.add("string")
    return "string"


def candidates_gbnf(schema: dict | None = None) -> str:
    """GBNF grammar for the candidate JSON, derived from *schema*.

    Defaults to CANDIDATES_SCHEMA -- the same schema the cloud providers hand
    to their servers -- so all three providers are constrained to one shape.
    """
    rules: dict[str, str] = {}
    used: set[str] = set()
    _gbnf_emit(CANDIDATES_SCHEMA if schema is None else schema, "root", rules, used)
    lines = [f"{name} ::= {body}" for name, body in rules.items()]
    lines += [_GBNF_TERMINALS[t] for t in _GBNF_TERMINALS if t in used]
    return "\n".join(lines) + "\n"


def build_user_text(
    context: list[str],
    fragment: str,
    excluded: list[str] | None = None,
    entities: list[str] | None = None,
    already_served: list[str] | None = None,
) -> str:
    """Render the per-request user message shared by all providers.

    *excluded* carries words the speaker already rejected this stall (via the
    accept/reject card UI or a device button press) -- when present, the model
    is told not to re-suggest them on the next round-trip.

    *entities* carries salient names/places/things mentioned earlier in the
    conversation whose last mention has already scrolled OUTSIDE the recent
    context window above (see backend.entities.EntityTracker) -- when
    present, the model gets a hint about them without the token cost of
    replaying the full conversation. Entities still inside the context window
    are deliberately not passed here (they're already visible above).

    *already_served* carries words offered for EARLIER gaps in this same
    utterance. Not the same thing as *excluded*: a rejected word was judged
    wrong, whereas an already-offered word was fine for a previous gap and the
    speaker has moved on. Since v3 stopped truncating the fragment, the earlier
    gap is still present in the text, so without this hint the model answers it
    again instead of the current one.
    """
    ctx = "\n".join(f"- {c}" for c in context) if context else "(none)"
    entities_line = (
        f"\nNames and things mentioned earlier in this conversation (oldest "
        f"may be many turns back): {', '.join(entities)}\n"
        if entities else ""
    )
    reject_line = (
        f"\nDo NOT suggest: {', '.join(excluded)} (already rejected by the speaker).\n"
        if excluded else ""
    )
    # Distinct from reject_line on purpose. A rejected word was judged WRONG by
    # the speaker. An already-offered word was judged right (or at least not
    # rejected) for an EARLIER gap in this same utterance -- the speaker has
    # moved past it and is now hunting a different word. Since the fragment is
    # no longer truncated, the earlier gap is still visible in it, and without
    # this line the model tends to answer the first gap again.
    served_line = (
        f"\nAlready offered earlier in this same utterance: "
        f"{', '.join(already_served)}. The speaker moved past those; they are "
        f"now searching for a LATER word.\n"
        if already_served else ""
    )
    return (
        f"Recent conversation:\n{ctx}\n"
        f"{entities_line}\n"
        f'Unfinished utterance:\n"{fragment}"\n'
        f"{served_line}"
        f"{reject_line}\n"
        "Return the ranked candidate words as JSON matching "
        '{"candidates":[{"word":string,"confidence":number}]}.'
    )
