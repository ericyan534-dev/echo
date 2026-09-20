"""ContextBuilder: four layers assembled under an approximate token budget.

Layer priority (highest first) -- under pressure, lower layers are dropped:
  1. current utterance          INVIOLABLE
  2. verbatim recent turns
  3. rolling summary
  4. salient entities
"""
from backend.context import ContextBuilder, approx_tokens
from backend.summarizer import ExtractiveSummarizer


def build(**kw):
    cb = ContextBuilder(ExtractiveSummarizer(),
                        budget_tokens=kw.pop("budget", 2048),
                        verbatim_turns=kw.pop("verbatim_turns", 6))
    return cb.build(
        conversation_turns=kw.get("turns", []),
        utterance=kw.get("utterance", "I need the"),
        already_served=kw.get("already_served", []),
        excluded=kw.get("excluded", []),
        summary=kw.get("summary", ""),
    )


def test_approx_tokens_is_documented_char_heuristic():
    assert approx_tokens("abcd") == 1
    assert approx_tokens("") == 0


def test_utterance_is_never_dropped_even_at_a_tiny_budget():
    p = build(budget=1, utterance="I really need the", turns=["a"] * 50)
    assert p.utterance == "I really need the"


def test_layers_are_trimmed_under_budget_pressure():
    turns = [f"turn number {i} with some filler words in it" for i in range(40)]
    p = build(budget=40, turns=turns, summary="Maria came to Boston.")
    assert p.utterance
    assert len(p.recent_turns) < 6      # trimmed under pressure


def test_generous_budget_keeps_all_layers():
    turns = ["Maria came to Boston.", "We had tea.", "It rained."]
    p = build(budget=4096, turns=turns, summary="Frank called earlier.")
    assert p.recent_turns == turns
    assert p.summary == "Frank called earlier."


def test_verbatim_turns_capped_by_setting_and_keeps_the_newest():
    turns = [f"turn {i}" for i in range(20)]
    p = build(budget=4096, turns=turns, verbatim_turns=6)
    assert len(p.recent_turns) == 6
    assert p.recent_turns[-1] == "turn 19"


def test_payload_total_respects_budget():
    turns = [f"turn number {i}" for i in range(30)]
    p = build(budget=200, turns=turns, summary="Maria came to Boston.")
    assert p.approx_total_tokens() <= 200


def test_recent_turns_stay_in_chronological_order_after_trimming():
    """Trimming walks backwards from the newest, but the payload must read
    oldest-first or the model sees the conversation reversed."""
    turns = [f"turn {i}" for i in range(10)]
    p = build(budget=30, turns=turns, verbatim_turns=6)
    assert p.recent_turns == sorted(p.recent_turns, key=lambda t: int(t.split()[1]))


def test_entities_only_include_names_that_scrolled_out_of_the_verbatim_window():
    turns = ["Maria came to Boston."] + [f"we talked about topic {i}." for i in range(10)]
    p = build(budget=4096, turns=turns, verbatim_turns=3)
    assert any("Boston" in e for e in p.entities)


def test_known_gap_sentence_initial_single_mention_name_is_not_tracked():
    """Documents a PRE-EXISTING limit of the capitalization heuristic in
    backend/entities.py, surfaced here because it bounds long-conversation
    recall: a capitalized word seen only at sentence start, in only one turn,
    is treated as grammar rather than a name.

    'Maria came to Boston.' yields Boston but not Maria. Mid-sentence
    ('My sister Maria came to Boston.') yields both. Not fixed by this plan --
    it needs a real NER, which would add a dependency.
    """
    from backend.entities import EntityTracker
    tracker = EntityTracker()
    filler = [f"we talked about topic {i}." for i in range(10)]
    assert tracker.out_of_window(["Maria came to Boston."] + filler, 3) == ["Boston"]
    both = tracker.out_of_window(["My sister Maria came to Boston."] + filler, 3)
    assert "Maria" in both and "Boston" in both


def test_retention_carries_untracked_names_along_with_their_turn():
    """Mitigation for the gap above: the summarizer retains the whole TURN, not
    just the entity, so a name the tracker missed still survives as long as
    something else in that turn was unique."""
    import asyncio

    from backend.summarizer import ExtractiveSummarizer
    turns = ["Maria came to Boston.", "It rained all afternoon."]
    out = asyncio.run(ExtractiveSummarizer().fold(turns))
    assert "Maria" in out      # carried by Boston's uniqueness, not its own
