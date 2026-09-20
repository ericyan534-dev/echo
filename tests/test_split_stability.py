"""A split must not move an episode across the train/test line when the
corpus grows.

This is not hypothetical. `make_splits(split_version="legacy")` draws one
permutation per show from a shared RandomState, so a show's assignment depends
on every show processed before it. When the corpus grew 20,124 -> 30,962
clips, 18 episodes crossed from TRAIN into TEST -- 1,380 clips, 31% of the new
test set -- and the old checkpoint scored Block 0.410 on them against 0.304 on
clips it had genuinely never seen. An "old model on new test set" row read as
partly memorisation.

The legacy behaviour is kept, and tested, because every published number was
produced under it. The point of these tests is that the two versions differ in
exactly the way claimed: legacy is reproducible but not stable, "stable" is
both.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_stutter import make_splits, speaker_group  # noqa: E402


def _rows(shows: dict[str, int], tag: str = "a") -> list[dict]:
    return [{"show": show, "ep": "%s%03d" % (tag, e), "clip": c}
            for show, n_eps in shows.items()
            for e in range(n_eps) for c in range(3)]


def _eps(rows, split, name):
    return {(rows[i]["show"], speaker_group(rows[i]["show"], rows[i]["ep"]))
            for i in split[name]}


SMALL = {"showA": 20, "showB": 20, "showC": 20}
GROWN = {"showA": 20, "showB": 40, "showC": 20}   # showB gains episodes


@pytest.mark.parametrize("version", ["legacy", "stable"])
def test_deterministic_across_calls(version):
    rows = _rows(SMALL)
    a = make_splits(rows, None, split_version=version)
    b = make_splits(rows, None, split_version=version)
    assert a == b


@pytest.mark.parametrize("version", ["legacy", "stable"])
def test_splits_are_episode_disjoint(version):
    rows = _rows(SMALL)
    sp = make_splits(rows, None, split_version=version)
    tr, va, te = (_eps(rows, sp, k) for k in ("train", "val", "test"))
    assert not (tr & te) and not (tr & va) and not (va & te)
    assert tr and va and te


def test_legacy_reshuffles_when_the_corpus_grows():
    """The bug, pinned. If this ever stops failing to be stable, the legacy
    numbers were produced by something else and must be re-derived."""
    small, grown = _rows(SMALL), _rows(GROWN)
    s = make_splits(small, None, split_version="legacy")
    g = make_splits(grown, None, split_version="legacy")
    moved = _eps(grown, g, "test") & _eps(small, s, "train")
    assert moved, "legacy is expected to leak; it no longer does"
    # and the leak lands on shows that did not change at all
    assert any(show != "showB" for show, _ in moved)


def test_stable_never_moves_an_episode_when_the_corpus_grows():
    small, grown = _rows(SMALL), _rows(GROWN)
    s = make_splits(small, None, split_version="stable")
    g = make_splits(grown, None, split_version="stable")
    for name in ("train", "val", "test"):
        before = _eps(small, s, name)
        for other in ("train", "val", "test"):
            if other == name:
                continue
            assert not (before & _eps(grown, g, other)), (
                "%s episode moved to %s when the corpus grew" % (name, other))


def test_stable_gives_every_show_a_test_and_val_episode():
    rows = _rows({"tiny": 3, "showA": 20})
    sp = make_splits(rows, None, split_version="stable")
    for name in ("train", "val", "test"):
        shows = {rows[i]["show"] for i in sp[name]}
        assert "tiny" in shows, "tiny show missing from %s" % name


def test_holdout_show_is_entirely_test_and_others_never_are():
    rows = _rows(SMALL)
    for version in ("legacy", "stable"):
        sp = make_splits(rows, "showB", split_version=version)
        assert {s for s, _ in _eps(rows, sp, "test")} == {"showB"}
        assert "showB" not in {s for s, _ in _eps(rows, sp, "train")}


def test_unknown_version_is_rejected():
    with pytest.raises(ValueError):
        make_splits(_rows(SMALL), None, split_version="v3")


def test_tiny_show_fallback_fires_rather_than_dropping_the_show():
    """A show too small to hash-stratify can land entirely in the train band.
    It gets episodes forced into val/test instead of vanishing from them -- and
    that forced choice is NOT growth-invariant, which is stated in make_splits'
    docstring. This pins the behaviour so it cannot drift into being silent."""
    from scripts.train_stutter import _stable_unit

    # Find a show name whose every episode hashes into train, so the test
    # exercises the fallback rather than assuming it does.
    name = next(n for n in ("s%d" % i for i in range(200))
                if all(_stable_unit(13, n, "a%03d" % e) >= 0.25 for e in range(4)))
    rows = _rows({name: 4, "showA": 20})
    sp = make_splits(rows, None, split_version="stable")
    for split in ("train", "val", "test"):
        assert name in {rows[i]["show"] for i in sp[split]}, (
            "%s dropped out of %s" % (name, split))
