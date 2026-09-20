"""ExtractiveSummarizer: a RETENTION rule, not abstractive compression.

The defining property is that output is a literal subset of what was said, so
it cannot invent a fact. See backend/summarizer.py for why the default is not
an LLM.
"""
import asyncio

from backend.summarizer import ExtractiveSummarizer


def fold(turns, existing=""):
    return asyncio.run(ExtractiveSummarizer().fold(turns, existing))


def test_retains_a_turn_holding_the_only_mention_of_an_entity():
    turns = ["My sister Maria visited yesterday.", "It was raining all day."]
    out = fold(turns)
    assert "Maria" in out


def test_drops_turns_with_no_unique_entity():
    turns = ["It was raining all day.", "I stayed inside."]
    assert fold(turns) == ""


def test_output_is_a_subset_of_what_was_said_never_invented():
    """The DEFINING property: every retained line is a verbatim turn.

    The `for line in ...` loop is only load-bearing if it actually iterates.
    The earlier version had no non-empty guard, so a regression to returning
    "" (retaining nothing -- exactly the failure this module exists to rule
    out) ran the loop zero times and passed. Both halves are now pinned:
    something IS retained, and what is retained is verbatim.
    """
    turns = ["My sister Maria visited yesterday.", "We had tea."]
    out = fold(turns)
    lines = [l for l in out.split("\n") if l.strip()]
    assert lines, "retained nothing -- the sole mention of Maria was dropped"
    for line in lines:
        assert line in turns    # retention, not abstraction -- cannot hallucinate
    # and it kept the at-risk turn specifically, not an arbitrary one
    assert "My sister Maria visited yesterday." in lines


def test_existing_summary_lines_are_also_verbatim():
    """Same property across a fold that already carries an `existing` summary:
    every output line came from `existing` or from `turns`. No new sentence
    may appear, and the output must not be empty."""
    existing = "Earlier: Frank called about the roof."
    turns = ["My sister Maria visited yesterday.", "We had tea."]
    lines = [l for l in fold(turns, existing=existing).split("\n") if l.strip()]
    assert len(lines) >= 2, f"expected the existing summary plus the kept turn, got {lines}"
    allowed = set(turns) | {existing}
    for line in lines:
        assert line in allowed, f"synthesized line absent from the source text: {line!r}"


def test_is_deterministic():
    turns = ["Maria came to Boston.", "Then we drove to Salem."]
    assert fold(turns) == fold(turns)


def test_existing_summary_is_preserved_and_extended():
    out = fold(["Maria came to Boston."], existing="Earlier: Frank called.")
    assert "Frank called." in out
    assert "Maria came to Boston." in out


def test_a_repeated_entity_does_not_pin_every_turn_that_mentions_it():
    """Retention is for entities at risk of being LOST. A name repeated across
    several turns is not at risk, so those turns are not all kept -- otherwise
    the summary would grow linearly with the conversation."""
    turns = [
        "Maria called me this morning.",
        "Maria said she was running late.",
        "Maria finally arrived at noon.",
    ]
    assert fold(turns) == ""
