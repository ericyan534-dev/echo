"""Render the fixed-chunk grid next to the baselines it has to beat.

Pure reporting -- reads eval/results/asr_fixed_chunk_stream.json and
eval/results/asr_offline_vs_stream.json and prints ASCII tables. No GPU.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FCS = ROOT / "eval" / "results" / "asr_fixed_chunk_stream.json"
BASE = ROOT / "eval" / "results" / "asr_offline_vs_stream.json"
PIDS = ["1554", "1713", "1731", "1738", "1833", "1944"]
BUDGET_MS = 1300

REF = [
    ("offline (one pass) -- the ceiling", "offline"),
    ("offline chunk=10 stride=8", "offline@chunk_duration=10@stride=8"),
    ("offline longform=chunked_lcs", "offline@longform_strategy=chunked_lcs"),
    ("BEST STREAMING (shipped family)",
     "stream@silence_commit_ms=700@word_time_policy=incremental"),
    ("stream, shipped defaults", "stream@silence_commit_ms=280"),
]


def main() -> int:
    fcs = json.loads(FCS.read_text(encoding="utf-8"))["results"]
    base = json.loads(BASE.read_text(encoding="utf-8"))["results"]

    print("")
    print("=" * 96)
    print("BASELINES (eval/results/asr_offline_vs_stream.json, same audio, same scorer)")
    print("=" * 96)
    print("  %-38s %7s %7s %9s" % ("config", "wer", "concat", "lag_ms"))
    for label, key in REF:
        v = base.get(key)
        if not v:
            continue
        print("  %-38s %7.3f %7.3f %9s"
              % (label, v["wer"], v["wer_concat"], v.get("commit_lag_ms_median")))

    print("")
    print("=" * 96)
    print("FIXED ROLLING BUFFER -- FULL GRID (buffer x stride x commit policy)")
    print("  lag_ms = audio-clock wait + measured decode wall time")
    print("  ready  = fraction of committed words on the timeline within %d ms of"
          % BUDGET_MS)
    print("           the first silence onset after they were spoken")
    print("=" * 96)
    print("  %-5s %-6s %-7s %7s %7s %8s %8s %9s %6s %6s"
          % ("buf", "stride", "commit", "wer", "concat", "lag_med", "lag_p90",
             "lag_audio", "ready", "rtf"))
    rows = sorted(fcs.values(), key=lambda v: (v["buffer_s"], v["stride_s"],
                                               ["scroll", "agree", "now"].index(v["commit"])))
    for v in rows:
        print("  %-5g %-6g %-7s %7.3f %7.3f %8s %8s %9s %6.2f %6.2f"
              % (v["buffer_s"], v["stride_s"], v["commit"], v["wer"],
                 v["wer_concat"], v["lag_ms_median"], v["lag_ms_p90"],
                 v.get("lag_audio_ms_median"), v["frac_ready"] or 0.0,
                 v["rtf"] or 0.0))

    print("")
    print("=" * 96)
    print("PER-PARTICIPANT wer (concat in parentheses)")
    print("=" * 96)
    hdr = "  %-34s" % "config" + "".join("%14s" % p for p in PIDS)
    print(hdr)
    for label, key in REF:
        v = base.get(key)
        if not v:
            continue
        line = "  %-34s" % label[:34]
        for p in PIDS:
            r = v["per_participant"].get(p)
            line += "%14s" % ("%.3f(%.3f)" % (r["wer"], r["wer_concat"]) if r else "-")
        print(line)
    print("  " + "-" * 92)
    for v in rows:
        label = "b%g s%g %s" % (v["buffer_s"], v["stride_s"], v["commit"])
        line = "  %-34s" % label
        for p in PIDS:
            r = v["per_participant"].get(p)
            line += "%14s" % ("%.3f(%.3f)" % (r["wer"], r["wer_concat"]) if r else "-")
        print(line)

    print("")
    print("=" * 96)
    print("GAP CLOSED vs the best shipped streaming policy (0.3752 wer / 0.3215 concat)")
    print("  offline ceiling is 0.2876 wer / 0.2707 concat")
    print("=" * 96)
    s_wer, s_cat = 0.3752, 0.3215
    o_wer, o_cat = 0.2876, 0.2707
    print("  %-34s %8s %8s %8s %8s %7s"
          % ("config", "d_wer", "gap%", "d_concat", "gap%", "ready"))
    for v in rows:
        label = "b%g s%g %s" % (v["buffer_s"], v["stride_s"], v["commit"])
        g1 = (s_wer - v["wer"]) / (s_wer - o_wer) * 100
        g2 = (s_cat - v["wer_concat"]) / (s_cat - o_cat) * 100
        print("  %-34s %+8.4f %7.0f%% %+8.4f %7.0f%% %7.2f"
              % (label, v["wer"] - s_wer, g1, v["wer_concat"] - s_cat, g2,
                 v["frac_ready"] or 0.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
