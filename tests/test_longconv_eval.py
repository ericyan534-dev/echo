"""The long-conversation harness must be able to FAIL.

A gate that cannot fail proves nothing, so these tests pin the scoring and the
dataset shape rather than the result.
"""
from eval.run_longconv_eval import bucket, load_set, score_probe


def test_dataset_probes_span_shallow_and_deep():
    convs = load_set()
    assert len(convs) >= 12
    for c in convs:
        assert len(c["turns"]) >= 40
        depths = [p["depth"] for p in c["probes"]]
        assert min(depths) <= 5 and max(depths) >= 35


def test_target_appears_only_in_the_opening_turn():
    """If the target recurred in filler, a deep probe would be answerable from
    the verbatim window and the test would measure nothing."""
    for c in load_set():
        target = c["target"]
        later = [t for t in c["turns"][1:] if target.lower() in t.lower()]
        assert later == [], (c["id"], later)


def test_score_probe_rejects_a_wrong_answer():
    assert score_probe(["oven"], ["Maria"]) is False


def test_score_probe_accepts_a_correct_top1():
    assert score_probe(["Maria", "sister"], ["Maria"]) is True


def test_score_probe_is_case_insensitive():
    assert score_probe(["maria"], ["Maria"]) is True


def test_score_probe_handles_empty_candidates():
    assert score_probe([], ["Maria"]) is False


def test_score_probe_requires_top1_not_merely_present():
    """Top-3 containment is a weaker claim; this harness scores top-1 so a
    lucky third-place hit is not counted as a success."""
    assert score_probe(["oven", "toaster", "Maria"], ["Maria"]) is False


def test_bucket_boundaries():
    assert bucket(3) == "early(<=10)"
    assert bucket(10) == "early(<=10)"
    assert bucket(11) == "mid(11-25)"
    assert bucket(25) == "mid(11-25)"
    assert bucket(26) == "late(>25)"
    assert bucket(41) == "late(>25)"
