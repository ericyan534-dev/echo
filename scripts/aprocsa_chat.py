"""CHAT transcript parsing for APROCSA -- turning clinician codes into labels.

WHY THIS IS THE ONLY REAL GROUND TRUTH ECHO HAS
-----------------------------------------------
Every other corpus in this repo is stuttered or fluent podcast speech. APROCSA
is six people with chronic post-stroke aphasia, transcribed in CHAT by
clinicians, with media-aligned time bullets. The CHAT codes mark word-finding
difficulty DIRECTLY, which is the event Echo exists to detect:

    &-um            filled pause
    &+lo            phonological fragment (a cut-off word attempt)
    [/]             repetition of the preceding word/phrase
    [//]            retracing -- the speaker revises the attempt
    +...            trailing off -- the utterance is ABANDONED
    (.) (..) (...)  timed pauses, short / medium / long
    [* s:ur]        error codes, e.g. a semantic paraphasia

Example, verbatim from participant 1554:

    and &-um I have speech &-um (.) &-um (...) spring [//] Christmas

That is a word search: three filled pauses, a long pause, a wrong word, and a
retracing to the right one. No stuttering corpus contains that pattern, and no
amount of SEP-28k training produces a detector that was tested on it.

WHAT COUNTS AS A WORD-SEARCH EVENT
----------------------------------
Not every marker is equal evidence, so they are not pooled blindly. A single
`(.)` short pause is ordinary speech timing; a retracing or an abandoned
utterance is not. `STRONG` markers stand alone; `WEAK` ones need company. That
threshold is a judgement call and is stated here rather than buried, so the
metric can be recomputed under a different one.
"""
from __future__ import annotations

import re
from pathlib import Path

# Media-alignment bullet: \x15<start>_<end>\x15, milliseconds into the media.
_BULLET = re.compile(r"\x15(\d+)_(\d+)\x15")

MARKERS = {
    # --- strong: each one alone marks a word search -----------------------
    "retracing": (r"\[//\]", "strong"),          # revised the attempt
    "trailing_off": (r"\+\.\.\.", "strong"),     # gave up on the utterance
    "fragment": (r"&\+\S+", "strong"),           # cut-off word attempt
    "long_pause": (r"\(\.\.\.\)", "strong"),
    "error_code": (r"\[\* [^\]]+\]", "strong"),  # paraphasia coded by the clinician
    # --- weak: ordinary in fluent speech too, needs corroboration ---------
    "filled_pause": (r"&-\w+", "weak"),
    "repetition": (r"\[/\]", "weak"),
    "medium_pause": (r"\(\.\.\)", "weak"),
    "short_pause": (r"\(\.\)", "weak"),
}
_COMPILED = {k: (re.compile(p), w) for k, (p, w) in MARKERS.items()}

# Tokens that are annotation, not speech. Stripped to recover what was said.
_STRIP = [
    re.compile(r"&=\S+"),            # gesture / nonverbal
    re.compile(r"\[[^\]]*\]"),       # all bracketed codes
    re.compile(r"&[-+]"),            # marker prefixes, keeping the word
    re.compile(r"\(\.+\)"),          # pause codes
    re.compile(r"[<>]"),             # scope markers
    re.compile(r"\+\.\.\."),
    re.compile(r"\x15[^\x15]*\x15"),
]


def clean_text(chat: str) -> str:
    """The words the speaker actually produced, with CHAT codes removed.

    Parenthesised letters are CHAT's notation for unpronounced sounds --
    "(a)n(d)" is the speaker saying "n". Keeping the parens would put a word in
    the reference the speaker never said.
    """
    s = chat
    for pat in _STRIP:
        s = pat.sub(" ", s)
    s = re.sub(r"\(([a-z']+)\)", "", s)      # unpronounced material
    s = s.replace("xxx", " ")                 # unintelligible
    s = re.sub(r"[.?!,]", " ", s)
    return " ".join(s.split())


def parse(path: Path) -> dict:
    """One .cha file -> participant utterances with times and marker counts."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")

    # CHAT continuation lines start with a tab and belong to the tier above.
    lines: list[str] = []
    for raw in text.splitlines():
        if raw[:1] in ("*", "@", "%"):
            lines.append(raw)
        elif raw.startswith("\t") and lines:
            lines[-1] += " " + raw.strip()

    header = {}
    for ln in lines:
        if ln.startswith("@ID:") and "|PAR|" in ln:
            parts = ln[4:].split("|")
            header["age"] = parts[3] if len(parts) > 3 else ""
            header["sex"] = parts[4] if len(parts) > 4 else ""

    utterances = []
    for ln in lines:
        if not ln.startswith("*"):
            continue
        speaker = ln[1:ln.index(":")] if ":" in ln else "?"
        body = ln[ln.index(":") + 1:] if ":" in ln else ln
        m = _BULLET.search(body)
        counts = {k: len(rx.findall(body)) for k, (rx, _) in _COMPILED.items()}
        strong = sum(counts[k] for k, (_, w) in MARKERS.items() if w == "strong")
        weak = sum(counts[k] for k, (_, w) in MARKERS.items() if w == "weak")
        utterances.append({
            "speaker": speaker,
            "is_participant": speaker.startswith("PAR"),
            "start_ms": int(m.group(1)) if m else None,
            "end_ms": int(m.group(2)) if m else None,
            "chat": _BULLET.sub("", body).strip(),
            "text": clean_text(body),
            "markers": counts,
            "n_strong": strong,
            "n_weak": weak,
            # One strong marker is enough; weak markers need corroboration,
            # because a single short pause or "um" is ordinary speech and
            # counting it would make almost every utterance a positive.
            "word_search": bool(strong >= 1 or weak >= 2),
        })
    return {"header": header, "utterances": utterances}


def load_all(transcript_dir: Path) -> dict[str, dict]:
    out = {}
    for p in sorted(Path(transcript_dir).glob("aprocsa*.cha")):
        pid = re.sub(r"\D", "", p.stem)[:4]
        out[pid] = parse(p)
    return out


def summarize(all_parsed: dict[str, dict]) -> dict:
    total = pos = timed = 0
    per_marker: dict[str, int] = {k: 0 for k in MARKERS}
    for pid, d in all_parsed.items():
        for u in d["utterances"]:
            if not u["is_participant"]:
                continue
            total += 1
            timed += int(u["start_ms"] is not None)
            pos += int(u["word_search"])
            for k, v in u["markers"].items():
                per_marker[k] += v
    return {"participants": len(all_parsed), "par_utterances": total,
            "media_aligned": timed, "word_search_positive": pos,
            "positive_rate": round(pos / total, 4) if total else None,
            "markers": per_marker}


if __name__ == "__main__":
    import json
    import sys

    d = load_all(Path(sys.argv[1] if len(sys.argv) > 1
                      else "data/aprocsa/transcripts"))
    print(json.dumps(summarize(d), indent=2))
    for pid, parsed in d.items():
        ex = [u for u in parsed["utterances"]
              if u["is_participant"] and u["n_strong"] >= 2][:2]
        for u in ex:
            print("  %s [%s] %s" % (pid, "search" if u["word_search"] else "fluent",
                                    u["chat"][:110].encode("ascii", "replace").decode()))
