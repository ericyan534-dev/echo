"""Generates eval/data/longconv_eval_set.jsonl deterministically.

Kept in the repo so the frozen dataset is reproducible and auditable rather
than a mystery blob. Run once; the OUTPUT is the frozen artifact (see the
freeze protocol in the harness docstring), not this generator.

Design of a probe: each conversation introduces ONE salient target (a person,
a place, or an object) in turn 1, then never mentions it again. Probes fire at
increasing depth with a fragment that can only be completed from that early
turn. So a shallow probe is answerable from the verbatim window, and a deep
probe is answerable ONLY if long-horizon memory works. That contrast is the
measurement.

Filler turns deliberately avoid the target and avoid introducing competing
entities of the same type, so a miss means "context was lost", not "the model
picked a plausible rival".
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data" / "longconv_eval_set.jsonl"

# (id, opening turn introducing the target, stall fragment, accepted answers)
CASES = [
    ("conv-01", "My sister Maria visited me at the house yesterday.",
     "I really need to call, um", ["Maria"]),
    ("conv-02", "We are flying to Tokyo for the whole of next month.",
     "I keep thinking about our trip to, uh", ["Tokyo"]),
    ("conv-03", "The doctor put me on a new blood pressure medication.",
     "I forgot to take my, um", ["medication", "medicine", "pills"]),
    ("conv-04", "My neighbour Frank came over to fix the fence.",
     "I should really thank, uh", ["Frank"]),
    ("conv-05", "I bought a secondhand piano for the front room.",
     "I have not had time to play the, um", ["piano"]),
    ("conv-06", "We adopted a greyhound called Rufus in the spring.",
     "I still need to walk, uh", ["Rufus"]),
    ("conv-07", "My physical therapist Sarah gave me a set of exercises.",
     "I have my appointment with, um", ["Sarah"]),
    ("conv-08", "We spent the summer at a cottage in Maine.",
     "I want to go back to, uh", ["Maine"]),
    ("conv-09", "My grandson gave me a tablet for reading the news.",
     "I cannot find my, um", ["tablet"]),
    ("conv-10", "The bakery on Chestnut Street makes sourdough on Fridays.",
     "I want to stop at the, uh", ["bakery"]),
    ("conv-11", "My old friend Eleanor wrote me a long letter.",
     "I owe a reply to, um", ["Eleanor"]),
    ("conv-12", "I left my reading glasses somewhere in the kitchen.",
     "I still cannot find my, uh", ["glasses"]),
]

# Neutral filler turns: no proper nouns, no objects that could be mistaken for
# a target. Enough of them to push the opening turn far out of any window.
FILLER = [
    "It has been a quiet week otherwise.",
    "The weather turned colder than I expected.",
    "I did not sleep very well last night.",
    "There was not much on the television.",
    "I made a pot of soup in the afternoon.",
    "The post arrived later than usual.",
    "I sat outside for a little while.",
    "Someone was doing building work down the road.",
    "I finished the crossword eventually.",
    "The heating has been making an odd noise.",
    "I tidied out one of the cupboards.",
    "There was a programme about gardening on.",
    "I had toast and jam for breakfast.",
    "The bins were collected in the morning.",
    "I watered the plants in the window.",
    "It rained hard for about an hour.",
    "I listened to the radio for a bit.",
    "The kettle takes forever to boil now.",
    "I wrote out a shopping list.",
    "There was a queue at the counter.",
    "I walked as far as the corner and back.",
    "The afternoon went by quite quickly.",
    "I dozed off in the chair after lunch.",
    "Nothing much happened in the evening.",
    "I put the washing on before bed.",
    "The light in the hall keeps flickering.",
    "I read a few pages and gave up.",
    "It was too cold to sit outside.",
    "I had a cup of tea around four.",
    "The road outside was unusually busy.",
    "I found an old photograph in a drawer.",
    "There was a delivery for next door.",
    "I swept the step in the morning.",
    "The clock in the kitchen is running slow.",
    "I had a long telephone conversation.",
    "It was dark by five in the afternoon.",
    "I defrosted something for dinner.",
    "The neighbours were out early.",
    "I sorted through a pile of paperwork.",
    "It has been a slow sort of week.",
]

PROBE_DEPTHS = [3, 20, 41]


def build() -> list[dict]:
    convs = []
    for cid, opening, fragment, expect in CASES:
        turns = [opening] + list(FILLER)
        assert len(turns) >= 41, len(turns)
        convs.append({
            "id": cid,
            "target": expect[0],
            "turns": turns,
            "probes": [{"depth": d, "fragment": fragment, "expect": expect}
                       for d in PROBE_DEPTHS],
        })
    return convs


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    convs = build()
    with OUT.open("w", encoding="utf-8", newline="\n") as fh:
        for c in convs:
            fh.write(json.dumps(c, ensure_ascii=True) + "\n")
    print("wrote %d conversations, %d turns each, probes at %s"
          % (len(convs), len(convs[0]["turns"]), PROBE_DEPTHS))


if __name__ == "__main__":
    main()
