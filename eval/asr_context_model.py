"""Moved to `backend/stt/context_prompt.py`.

The mechanism is now used by the live streaming path
(`VerbatimASR(context_prompt=True)`), so it cannot live under `eval/` -- the
backend must not import from the evaluation tree. This shim keeps the old
import path working; the WHY, and the one place it mismatches training, are
documented at the new home.

The measurement is `eval/bench_asr_context_prompt.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.stt.context_prompt import (  # noqa: E402,F401
    ContextModel,
    ctx_prompt_builder,
)

_ctx_prompt_builder = ctx_prompt_builder

__all__ = ["ContextModel", "ctx_prompt_builder"]
