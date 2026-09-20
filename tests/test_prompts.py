from backend.prompts import (
    CANDIDATES_SCHEMA,
    FEW_SHOTS,
    SYSTEM_PROMPT,
    build_user_text,
)


def test_build_user_text_includes_fragment_and_context():
    text = build_user_text(["Where are you going?"], "the big city in Japan")
    assert "Japan" in text
    assert "Where are you going?" in text
    assert "candidates" in text  # asks for the JSON shape


def test_build_user_text_handles_empty_context():
    text = build_user_text([], "the thing")
    assert "(none)" in text


def test_build_user_text_renders_excluded_words():
    text = build_user_text(["ctx"], "pass me the", excluded=["toaster", "oven"])
    assert "Do NOT suggest: toaster, oven (already rejected by the speaker)." in text
    assert "pass me the" in text  # fragment still present alongside the exclusion


def test_build_user_text_no_excluded_line_by_default():
    assert "Do NOT suggest" not in build_user_text(["ctx"], "pass me the")
    assert "Do NOT suggest" not in build_user_text(["ctx"], "pass me the", excluded=[])
    assert "Do NOT suggest" not in build_user_text(["ctx"], "pass me the", excluded=None)


def test_build_user_text_renders_entities_line():
    text = build_user_text(["ctx"], "call, um, the guy", entities=["Frank", "Pete's Diner"])
    assert ("Names and things mentioned earlier in this conversation "
            "(oldest may be many turns back): Frank, Pete's Diner") in text
    assert "call, um, the guy" in text  # fragment still present alongside the hint


def test_build_user_text_no_entities_line_by_default():
    assert "mentioned earlier" not in build_user_text(["ctx"], "pass me the")
    assert "mentioned earlier" not in build_user_text(["ctx"], "pass me the", entities=[])
    assert "mentioned earlier" not in build_user_text(["ctx"], "pass me the", entities=None)


def test_build_user_text_identical_output_when_entities_absent():
    # entities is a pure additive extension: output must be byte-identical
    # to the pre-entity-memory signature when the arg is omitted/None/empty.
    base = build_user_text(["ctx"], "pass me the")
    assert base == build_user_text(["ctx"], "pass me the", entities=None)
    assert base == build_user_text(["ctx"], "pass me the", entities=[])
    assert base == build_user_text(["ctx"], "pass me the", None, None)


def test_build_user_text_entities_and_excluded_together():
    text = build_user_text(
        ["ctx"], "pass me the", excluded=["toaster"], entities=["Frank"])
    assert "Do NOT suggest: toaster" in text
    assert "Frank" in text and "mentioned earlier" in text


def test_schema_shape():
    assert CANDIDATES_SCHEMA["required"] == ["candidates"]
    items = CANDIDATES_SCHEMA["properties"]["candidates"]["items"]
    assert items["required"] == ["word", "confidence"]


def test_assets_present():
    assert SYSTEM_PROMPT.strip()
    assert len(FEW_SHOTS) >= 3
    for inp, out in FEW_SHOTS:
        assert "fragment" in inp and "context" in inp
        assert isinstance(out, list) and out


def test_already_served_words_are_rendered_and_distinct_from_rejections():
    """A word offered earlier in the same turn is spent, but it was not
    rejected -- the speaker simply moved past it. The prompt must say so
    without using the reject wording, which means something different."""
    text = build_user_text(
        context=["We were talking about lunch."],
        fragment="I need the um a sandwich um",
        already_served=["toaster"],
    )
    assert "toaster" in text
    assert "already offered" in text.lower()


def test_no_already_served_line_when_none():
    text = build_user_text(context=[], fragment="I need the")
    assert "already offered" not in text.lower()


def test_already_served_and_excluded_render_separately():
    text = build_user_text(
        ["ctx"], "pass me the", excluded=["spoon"], already_served=["fork"])
    assert "Do NOT suggest: spoon" in text
    assert "fork" in text and "already offered" in text.lower()
