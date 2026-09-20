"""Tests for eval/run_prediction_eval.py: normalization, JSONL loader
validation, deterministic MockPredictor end-to-end scoring, the
--ablate-context behavior, and integrity of the frozen 60-item dataset."""
import asyncio
import json
import re
from collections import Counter
from pathlib import Path

import pytest

from backend.entities import EntityTracker
from backend.predictor.base import WordPredictor
from backend.predictor.mock import MockPredictor
from backend.prompts import FEW_SHOTS
from backend.schemas import Candidate
from eval.run_prediction_eval import (
    DATASET,
    FREEZE_PROTOCOL,
    load_items,
    normalize,
    run_eval,
    score_candidates,
)

LONGCTX_DATASET = Path(__file__).resolve().parent.parent / "eval" / "data" / "prediction_eval_longctx_v1.jsonl"


# --- normalization ---------------------------------------------------------
def test_normalize_lowercases_and_strips_punctuation():
    assert normalize("Toaster!") == "toaster"
    assert normalize('"umbrella,"') == "umbrella"
    assert normalize("  Remote   Control ") == "remote control"


def test_normalize_strips_trailing_possessive():
    assert normalize("Pete's") == "pete"
    assert normalize("Marino's") == "marino"
    # possessive and bare form converge
    assert normalize("Frank's") == normalize("Frank")


def test_normalize_strips_one_trailing_plural_s():
    assert normalize("fridges") == "fridge"
    # symmetric: plural candidate matches singular gold and vice versa
    assert normalize("clothespins") == normalize("clothespin")
    # only ONE trailing s is stripped, and single letters are left alone
    assert normalize("s") == "s"


def test_normalize_multiword_phrases():
    assert normalize("St. Bridget's Church") == normalize("St Bridgets Church")
    assert normalize("meet in the middle") == "meet in the middle"


# --- scoring ---------------------------------------------------------------
def test_score_candidates_top1_and_top3():
    assert score_candidates(["fridge", "refrigerator"], ["Fridge", "oven"]) == (True, True)
    assert score_candidates(["fridge"], ["oven", "fridges", "sink"]) == (False, True)
    assert score_candidates(["fridge"], ["oven", "sink", "stove"]) == (False, False)
    assert score_candidates(["fridge"], []) == (False, False)


def test_score_candidates_only_first_three_count():
    assert score_candidates(["target"], ["a", "b", "c", "target"]) == (False, False)


# --- JSONL loader validation ----------------------------------------------
def _item(**over) -> str:
    base = {"id": "x1", "category": "concrete", "context": ["hi there"],
            "fragment": "the, um, the thing", "gold": ["thing"], "notes": "easy"}
    base.update(over)
    return json.dumps(base)


def _write(tmp_path, lines):
    p = tmp_path / "set.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_loader_accepts_valid_items(tmp_path):
    p = _write(tmp_path, [_item(), _item(id="x2", category="abstract_verb")])
    items = load_items(p)
    assert [i["id"] for i in items] == ["x1", "x2"]


def test_loader_rejects_missing_keys(tmp_path):
    line = json.dumps({"id": "x1", "category": "concrete"})
    with pytest.raises(ValueError, match="missing keys"):
        load_items(_write(tmp_path, [line]))


def test_loader_rejects_bad_category(tmp_path):
    with pytest.raises(ValueError, match="bad category"):
        load_items(_write(tmp_path, [_item(category="objects")]))


def test_loader_rejects_empty_or_nonlist_gold(tmp_path):
    with pytest.raises(ValueError, match="gold"):
        load_items(_write(tmp_path, [_item(gold=[])]))
    with pytest.raises(ValueError, match="gold"):
        load_items(_write(tmp_path, [_item(gold="fridge")]))


def test_loader_rejects_nonlist_context(tmp_path):
    with pytest.raises(ValueError, match="context"):
        load_items(_write(tmp_path, [_item(context="hi there")]))


def test_loader_rejects_duplicate_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate id"):
        load_items(_write(tmp_path, [_item(), _item()]))


def test_loader_rejects_invalid_json_with_line_number(tmp_path):
    with pytest.raises(ValueError, match=":2:"):
        load_items(_write(tmp_path, [_item(), "{not json"]))


def test_loader_rejects_missing_or_empty_file(tmp_path):
    with pytest.raises(ValueError, match="not found"):
        load_items(tmp_path / "nope.jsonl")
    with pytest.raises(ValueError, match="no items"):
        load_items(_write(tmp_path, [""]))


# --- end-to-end scoring with MockPredictor (deterministic) ------------------
def _mock_items() -> list[dict]:
    # MockPredictor's default table keys on "bread" -> ["toaster", "oven"].
    return [
        {"id": "a", "category": "concrete", "context": ["breakfast?"],
         "fragment": "I put the bread in the, um, the", "gold": ["toaster"], "notes": ""},
        {"id": "b", "category": "concrete", "context": ["breakfast?"],
         "fragment": "I put the bread in the, um, the", "gold": ["oven"], "notes": ""},
        {"id": "c", "category": "abstract_verb", "context": [],
         "fragment": "totally unrelated stall, um", "gold": ["zilch"], "notes": ""},
    ]


def test_run_eval_scores_mock_deterministically():
    results = asyncio.run(run_eval(
        MockPredictor(), _mock_items(), sleep_s=0.0, provider="mock", model="(mock)"))
    assert results["overall"]["top1"] == {"hits": 1, "n": 3, "rate": round(1 / 3, 4)}
    assert results["overall"]["top3"] == {"hits": 2, "n": 3, "rate": round(2 / 3, 4)}
    assert results["per_category"]["concrete"]["top3"]["hits"] == 2
    assert results["per_category"]["abstract_verb"]["top1"]["hits"] == 0
    by_id = {r["id"]: r for r in results["items"]}
    assert by_id["a"]["top1_hit"] and by_id["a"]["top3_hit"]
    assert not by_id["b"]["top1_hit"] and by_id["b"]["top3_hit"]
    assert not by_id["c"]["top1_hit"] and by_id["c"]["candidates"] == []
    # confidence split: item a correct (top conf 1.0), item b wrong, item c excluded
    conf = results["confidence"]
    assert conf["n_correct"] == 1 and conf["n_wrong"] == 1
    assert conf["mean_when_correct"] == 1.0


def test_run_eval_echoes_freeze_protocol_and_provenance():
    results = asyncio.run(run_eval(
        MockPredictor(), _mock_items(), sleep_s=0.0, provider="mock", model="(mock)",
        dataset_path=DATASET))
    from eval import run_prediction_eval as rpe

    prov = results["provenance"]
    assert prov["freeze_protocol"] == FREEZE_PROTOCOL
    assert "FREEZE PROTOCOL" in rpe.__doc__
    assert "frozen at its first commit" in " ".join(rpe.__doc__.split())
    assert prov["provider"] == "mock" and prov["n_items"] == 3
    assert re.fullmatch(r"[0-9a-f]{64}", prov["dataset_sha256"])


# --- ablation flag behavior --------------------------------------------------
class _RecordingPredictor(WordPredictor):
    def __init__(self):
        self.calls: list[tuple[list[str], str]] = []

    async def predict(self, context, fragment):
        self.calls.append((list(context), fragment))
        return [Candidate(word="x", confidence=0.5)]


def test_ablate_context_sends_empty_context():
    items = _mock_items()
    rec = _RecordingPredictor()
    results = asyncio.run(run_eval(rec, items, ablate_context=True, sleep_s=0.0))
    assert [ctx for ctx, _ in rec.calls] == [[], [], []]
    assert [frag for _, frag in rec.calls] == [i["fragment"] for i in items]
    assert results["provenance"]["ablate_context"] is True


def test_full_run_sends_item_context():
    items = _mock_items()
    rec = _RecordingPredictor()
    results = asyncio.run(run_eval(rec, items, ablate_context=False, sleep_s=0.0))
    assert [ctx for ctx, _ in rec.calls] == [i["context"] for i in items]
    assert results["provenance"]["ablate_context"] is False


# --- frozen dataset integrity ------------------------------------------------
# Theme anchors of the three FEW_SHOTS in backend/prompts.py (toaster/breakfast,
# Tokyo/Japan travel, blood-pressure medication). The frozen set must not lean
# on any of them, in gold OR in item text.
_FEW_SHOT_THEME_PATTERNS = [
    r"\btoast\w*", r"\boven\b", r"\btokyo\b", r"\bosaka\b", r"\bkyoto\b",
    r"\bjapan\b", r"\bmedication\b", r"\bblood pressure\b", r"\bprescription\b",
]


def test_dataset_integrity():
    items = load_items(DATASET)  # loader enforces schema + unique ids
    assert len(items) == 60
    assert Counter(i["category"] for i in items) == {
        "concrete": 20, "proper_context": 20, "abstract_verb": 20}
    assert len({i["id"] for i in items}) == 60

    # gold lists are tight (1-4 enumerated surface forms)
    assert all(1 <= len(i["gold"]) <= 4 for i in items)

    # no gold overlap with the FEW_SHOTS answers, under the scoring normalization
    few_shot_norm = {normalize(w) for _, out in FEW_SHOTS for w in out}
    for it in items:
        for g in it["gold"]:
            assert normalize(g) not in few_shot_norm, (it["id"], g)

    # no FEW_SHOTS theme leakage anywhere in the item text
    for it in items:
        text = " ".join(it["context"] + [it["fragment"]] + it["gold"]).lower()
        for pat in _FEW_SHOT_THEME_PATTERNS:
            assert not re.search(pat, text), (it["id"], pat)

    # frozen file stays ASCII (GBK Windows console safety)
    assert DATASET.read_text(encoding="utf-8").isascii()


# --- longctx_v1 dataset integrity (context-window extension / entity memory) -
def _mention_indices(context: list[str], gold_norm: set[str]) -> list[int]:
    """Indices of the context turns that NAME one of the gold surface forms.

    Matches a gold entry as a contiguous run of normalized tokens, so
    multi-word entities ("Copper Kettle Cafe") are found -- a per-word set
    membership test never matches those and reports zero mentions.
    """
    out: list[int] = []
    for idx, turn in enumerate(context):
        words = [normalize(w) for w in re.findall(r"[A-Za-z']+", turn)]
        for gold in gold_norm:
            tokens = gold.split()
            n = len(tokens)
            if n and any(words[i:i + n] == tokens for i in range(len(words) - n + 1)):
                out.append(idx)
                break
    return out


def test_longctx_dataset_integrity():
    items = load_items(LONGCTX_DATASET)  # loader enforces schema + unique ids
    assert len(items) == 20
    assert all(i["category"] == "proper_context" for i in items)
    assert len({i["id"] for i in items}) == 20

    # every item's context is 10-16 turns long (long-context by construction)
    assert all(10 <= len(i["context"]) <= 16 for i in items)

    # the target entity is named ONCE, only within the first 4 context turns
    # (index 0-3), and never repeated afterward -- the whole point of the set
    # is that it has scrolled out of a 6-turn window by the time the fragment
    # stalls on it.
    #
    # _mention_indices matches multi-word gold entries as a contiguous run of
    # normalized tokens. A per-word `in gold_norm` test (the earlier version)
    # silently found ZERO mentions for the multi-word items ("Copper Kettle
    # Cafe", "Gilded Spoon"), so the invariant below was skipped for them.
    # Asserting the count makes the coverage explicit instead of conditional.
    for it in items:
        gold_norm = {normalize(g) for g in it["gold"]}
        idxs = _mention_indices(it["context"], gold_norm)
        assert len(idxs) == 1, (it["id"], "entity must be named exactly once", idxs)
        assert idxs[0] <= 3, (it["id"], "entity mentioned outside first 4 turns", idxs[0])

    # at least 6 items use invented names/businesses (see notes convention)
    n_invented = sum(1 for it in items if "invented name" in it["notes"])
    assert n_invented >= 6

    # ASCII-only (GBK Windows console safety)
    assert LONGCTX_DATASET.read_text(encoding="utf-8").isascii()


def test_longctx_no_overlap_with_frozen_set_or_few_shots():
    frozen_items = load_items(DATASET)
    longctx_items = load_items(LONGCTX_DATASET)

    frozen_gold_norm = {normalize(g) for it in frozen_items for g in it["gold"]}
    for it in longctx_items:
        for g in it["gold"]:
            assert normalize(g) not in frozen_gold_norm, (it["id"], g)

    few_shot_norm = {normalize(w) for _, out in FEW_SHOTS for w in out}
    for it in longctx_items:
        for g in it["gold"]:
            assert normalize(g) not in few_shot_norm, (it["id"], g)
        text = " ".join(it["context"] + [it["fragment"]] + it["gold"]).lower()
        for pat in _FEW_SHOT_THEME_PATTERNS:
            assert not re.search(pat, text), (it["id"], pat)


def test_longctx_mechanism_fairness_gate():
    """Every item's target entity must actually be extractable by
    EntityTracker AND out-of-window at context_turns=6 -- otherwise the eval
    would measure a tracker-extraction miss instead of the injection
    benefit. (The set is hand-verified clean of stray phantom entities from
    filler turns; see eval/data generation notes.)"""
    tracker = EntityTracker()
    for it in load_items(LONGCTX_DATASET):
        gold_norm = {normalize(g) for g in it["gold"]}

        extracted_norm = {normalize(t) for t, _ in tracker.extract(it["context"])}
        assert gold_norm & extracted_norm, (it["id"], "not extracted by EntityTracker")

        ool_norm = {normalize(t) for t in tracker.out_of_window(it["context"], 6)}
        assert gold_norm & ool_norm, (it["id"], "not out-of-window at context_turns=6")


# --- --dataset / --entity-memory flag behavior (context-window extension) ----
class _EntityRecordingPredictor(WordPredictor):
    def __init__(self):
        self.calls: list[tuple[list[str], str, object]] = []  # (context, fragment, entities)

    async def predict(self, context, fragment, excluded=None, entities="__unset__"):
        self.calls.append((list(context), fragment, entities))
        return [Candidate(word="x", confidence=0.5)]


def _longctx_style_items() -> list[dict]:
    return [{
        "id": "lc1", "category": "proper_context", "notes": "",
        "context": [
            "My neighbor Frank fixed the fence yesterday.",
            "It's supposed to rain again this weekend.",
            "I need to get the car's oil changed soon.",
            "I still haven't finished that book I started.",
            "I keep forgetting to water the plants.",
            "I should really get more sleep this week.",
            "I've got a bunch of laundry to catch up on.",
            "I'm thinking about repainting the kitchen.",
        ],
        "fragment": "I should call, um, the guy who fixed the fence, what's his",
        "gold": ["Frank"],
    }]


def test_run_eval_windows_context_to_context_turns():
    items = _longctx_style_items()
    rec = _EntityRecordingPredictor()
    asyncio.run(run_eval(rec, items, sleep_s=0.0, context_turns=3))
    context_sent, fragment, entities = rec.calls[0]
    assert context_sent == items[0]["context"][-3:]
    assert fragment == items[0]["fragment"]
    assert entities == "__unset__"  # entity_memory defaults off -- kwarg omitted entirely


def test_run_eval_entity_memory_off_never_passes_entities_kwarg():
    items = _longctx_style_items()
    rec = _EntityRecordingPredictor()
    asyncio.run(run_eval(rec, items, sleep_s=0.0, context_turns=3, entity_memory=False))
    assert rec.calls[0][2] == "__unset__"


def test_run_eval_entity_memory_on_injects_out_of_window_entities():
    items = _longctx_style_items()
    rec = _EntityRecordingPredictor()
    results = asyncio.run(run_eval(rec, items, sleep_s=0.0, context_turns=3, entity_memory=True))
    context_sent, fragment, entities = rec.calls[0]
    assert context_sent == items[0]["context"][-3:]  # still windowed, same as off
    assert entities == ["Frank"]
    assert results["provenance"]["entity_memory"] is True
    assert results["provenance"]["context_turns"] == 3


def test_run_eval_entity_memory_on_with_nothing_out_of_window_omits_kwarg():
    # context_turns large enough that Frank's mention (turn 0) is INSIDE the
    # window -- nothing to inject, kwarg must be omitted (same as off).
    items = _longctx_style_items()
    rec = _EntityRecordingPredictor()
    asyncio.run(run_eval(rec, items, sleep_s=0.0, context_turns=20, entity_memory=True))
    assert rec.calls[0][2] == "__unset__"


def test_run_eval_entity_memory_ignored_under_ablate_context():
    # ablate_context strips context entirely; injecting entities on top of an
    # otherwise-empty-context ablation isn't a combination the harness needs
    # to support, so entity_memory is a no-op there.
    items = _longctx_style_items()
    rec = _EntityRecordingPredictor()
    asyncio.run(run_eval(
        rec, items, sleep_s=0.0, context_turns=3, entity_memory=True, ablate_context=True))
    context_sent, _, entities = rec.calls[0]
    assert context_sent == []
    assert entities == "__unset__"


def test_dataset_flag_default_is_the_frozen_set():
    """--dataset must DEFAULT to the frozen set.

    The previous version re-imported DATASET and asserted `_D == DATASET`,
    i.e. `x == x` -- it could not fail. This reaches into main()'s real
    argparse parser and reads the registered default, so repointing the flag
    at another file (or renaming/dropping it) turns this red.
    """
    import argparse

    from eval import run_prediction_eval as rpe

    seen: dict[str, argparse.ArgumentParser] = {}

    class _StopBeforeRunning(Exception):
        pass

    def _capture(self, *a, **kw):
        seen["parser"] = self
        raise _StopBeforeRunning

    real = argparse.ArgumentParser.parse_args
    argparse.ArgumentParser.parse_args = _capture
    try:
        with pytest.raises(_StopBeforeRunning):
            rpe.main()
    finally:
        argparse.ArgumentParser.parse_args = real

    parser = seen["parser"]
    assert "--dataset" in parser.format_usage() or parser.get_default("dataset") is not None
    assert parser.get_default("dataset") == DATASET
    assert DATASET.name == "prediction_eval_set.jsonl"
    # and the longctx set is a genuinely different file, not an alias
    assert LONGCTX_DATASET != DATASET
    assert LONGCTX_DATASET.read_bytes() != DATASET.read_bytes()
