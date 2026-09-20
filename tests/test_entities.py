"""Tests for backend/entities.py: EntityTracker's capitalization heuristic
(extraction rules, recency, cap) and the out-of-window filter used to inject
salient entities into the predictor prompt (ENTITY_MEMORY)."""
from backend.entities import EntityTracker, _split_sentences


# --- sentence splitting / abbreviation handling -----------------------------
def test_split_sentences_basic():
    assert _split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_split_sentences_does_not_split_on_title_abbreviation():
    # "Dr." must not end the sentence -- else "Patel" would land at a false
    # sentence-start and never be captured (see test_dr_abbreviation_extracted).
    assert _split_sentences("My therapist is Dr. Patel. She is great.") == [
        "My therapist is Dr. Patel.", "She is great.",
    ]


def test_split_sentences_abbreviation_tradeoff_disclosed():
    # Documented limitation: "St." at the END of a sentence (meaning
    # "Street", not a title) is indistinguishable from the title case and
    # still merges with what follows -- an accepted, disclosed tradeoff.
    assert _split_sentences("I live on Main St. Two blocks over is the park.") == [
        "I live on Main St. Two blocks over is the park.",
    ]


# --- rule 1: capitalized token NOT at sentence start ------------------------
def test_extracts_mid_sentence_capitalized_name():
    t = EntityTracker()
    assert t.extract(["I met Frank at the store."]) == [("Frank", 0)]


def test_sentence_start_single_word_not_captured_on_first_mention():
    t = EntityTracker()
    # "Frank" opens the sentence here -- a single occurrence is not enough.
    assert t.extract(["Frank is my neighbor."]) == []


# --- rule 2: multi-word capitalized runs (position-independent) ------------
def test_multiword_run_captured_regardless_of_position():
    t = EntityTracker()
    assert t.extract(["We ate at Pete's Diner last night."]) == [("Pete's Diner", 0)]


def test_multiword_run_at_sentence_start_still_captured():
    t = EntityTracker()
    assert t.extract(["Lake Winnipesaukee is beautiful in the fall."]) == [
        ("Lake Winnipesaukee", 0)
    ]


def test_dr_abbreviation_extracted_as_one_entity():
    t = EntityTracker()
    assert t.extract(["My therapist is Dr. Patel."]) == [("Dr Patel", 0)]


# --- rule 3: sentence-start word promoted after recurring across turns -----
def test_sentence_start_word_promoted_after_second_turn():
    t = EntityTracker()
    result = t.extract(["Frank is my neighbor.", "Frank called again today."])
    assert result == [("Frank", 1)]


def test_single_letter_pronoun_never_captured():
    t = EntityTracker()
    # "I" recurs at sentence-start in nearly every turn -- must never be
    # promoted (single-letter tokens are excluded regardless of recurrence).
    turns = ["I went to the store.", "I bought some milk.", "I came home."]
    assert t.extract(turns) == []


# --- recency: re-mention refreshes the last-seen turn index ----------------
def test_remention_refreshes_recency():
    t = EntityTracker()
    turns = [
        "I met Frank at the store.",       # turn 0: Frank
        "The weather was nice.",           # turn 1
        "I saw Elena downtown.",           # turn 2: Elena
        "I ran into Frank again.",         # turn 3: Frank re-mentioned
    ]
    result = t.extract(turns)
    # Frank's last mention (turn 3) is more recent than Elena's (turn 2).
    assert result[0] == ("Frank", 3)
    assert ("Elena", 2) in result


# --- cap ---------------------------------------------------------------------
def test_cap_keeps_only_most_recent_entries():
    t = EntityTracker(cap=3)
    names = ["Frank", "Elena", "Gus", "Nora", "Theo", "Mabel"]
    turns = [f"I met {name} today." for name in names]
    result = t.extract(turns)
    assert len(result) == 3
    # most-recent-first: turns 5, 4, 3 survive; 0-2 are dropped.
    assert [last for _, last in result] == [5, 4, 3]
    assert [text for text, _ in result] == ["Mabel", "Theo", "Nora"]


# --- out_of_window -----------------------------------------------------------
def test_out_of_window_excludes_entities_inside_the_context_window():
    t = EntityTracker()
    turns = [
        "I met Frank at the store.",   # turn 0 -- Frank, will be out of window
        "t1", "t2", "t3", "t4",
        "I saw Elena downtown.",       # turn 5 -- Elena, inside a 6-turn window
    ]
    # window = turns[-6:] = turns[0:6] -- wait len(turns)=6, so with
    # context_turns=6 the whole conversation is in-window: nothing excluded.
    assert t.out_of_window(turns, 6) == []
    # with a narrower window, Frank (turn 0) falls outside; Elena (turn 5) stays in.
    assert t.out_of_window(turns, 3) == ["Frank"]


def test_out_of_window_matches_conversation_recent_edge_cases():
    t = EntityTracker()
    turns = ["I met Frank at the store."] + ["filler"] * 9
    # Conversation.recent(0) / negative -> [] (nothing in window) -> everything out.
    assert t.out_of_window(turns, 0) == ["Frank"]
    assert t.out_of_window(turns, -1) == ["Frank"]
    # A window bigger than the conversation covers everything -> nothing out.
    assert t.out_of_window(turns, 100) == []


def test_out_of_window_respects_cap():
    t = EntityTracker(cap=2)
    names = ["Frank", "Elena", "Gus", "Nora", "Theo"]
    turns = [f"I met {name} today." for name in names] + ["filler"] * 10
    # Only the 2 most recent survive extract()'s cap in the first place, so
    # out_of_window can never return more than that even though several
    # earlier mentions are also technically out-of-window.
    result = t.out_of_window(turns, 3)
    assert len(result) <= 2
