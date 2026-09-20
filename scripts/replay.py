"""Hour-1 tuning harness: feed the sample utterances through the StallDetector
(+ a predictor) and print where stalls fire and what gets predicted.

    python -m scripts.replay              # uses MockPredictor (no key needed)
    PREDICTOR_PROVIDER=gemini GEMINI_API_KEY=... python -m scripts.replay --live

Use this to tune STALL_PAUSE_MS / triggers before wiring the UI.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from backend.config import get_settings
from backend.pipeline import EchoPipeline
from backend.predictor import get_predictor
from backend.predictor.mock import MockPredictor
from backend.stall_detector import StallDetector
from backend.stt.mock import MockSTT, script_from_spec
from backend.transcript import Conversation

SAMPLES = Path(__file__).resolve().parent.parent / "samples" / "utterances.json"


async def run_case(case: dict, live: bool) -> bool:
    s = get_settings()
    predictor = get_predictor(s) if live else MockPredictor(max_candidates=s.max_candidates)
    convo = Conversation()
    for c in case["context"]:
        convo.add_turn(c)
    pipe = EchoPipeline(StallDetector(pause_ms=s.pause_ms), predictor, conversation=convo)
    script = script_from_spec(case["words"], stall_after=case.get("stall_after"))
    await pipe.run(MockSTT(script).stream())

    fired = pipe.predictions[0] if pipe.predictions else None
    exp_trigger = case.get("expected_trigger")
    ok_trigger = (fired.trigger if fired else None) == exp_trigger
    print(f"\n=== {case['name']} ===")
    print(f"  utterance : {' '.join(case['words'])}")
    if fired:
        words = ", ".join(f"{c.word} ({c.confidence})" for c in fired.candidates)
        print(f"  STALL     : trigger={fired.trigger}  fragment=\"{fired.fragment}\"")
        print(f"  PREDICTED : {words}")
    else:
        print("  STALL     : (none)")
    print(f"  expected  : trigger={exp_trigger}  word={case.get('expected_word')}")
    print(f"  trigger OK: {ok_trigger}")
    return ok_trigger


async def main(live: bool) -> int:
    cases = json.loads(SAMPLES.read_text())
    results = [await run_case(c, live) for c in cases]
    passed = sum(results)
    print(f"\n{passed}/{len(results)} cases triggered as expected.")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main(live="--live" in sys.argv)))
