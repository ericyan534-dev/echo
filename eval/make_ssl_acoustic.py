"""Regenerate the APROCSA acoustic streams using the WavLM StutterNet.

The SSL model is a large step up on SEP-28k -- on the expanded corpus,
episode-disjoint TEST (n=4411): ANY AP 0.882 and Block 0.325, against the
log-mel CNN's 0.786 / 0.256 -- but that is a clip-level number on stuttered
podcast speech. Whether it changes what Echo actually fires on, in aphasic
speech, is a separate question and this produces the streams that answer it.

The default checkpoint is stutternet_ssl_v2.pt, NOT stutternet_ssl.pt: the
latter was overwritten by a `--limit 200` smoke run (its own embedded metrics
say n_train=200, epochs=1, n_unfreeze=0) and is not the model any published
number refers to. `*.pt` is gitignored, so the original could not be restored.

PROVENANCE: this SEP-28k is a partial reconstruction extended with a
HuggingFace mirror that declares no licence. It is NOT comparable to published
SEP-28k figures. Stuttering is not aphasia.

Cached under a distinct key so the log-mel CNN's streams remain on disk and the
two can be compared without re-running either.

    python eval/make_ssl_acoustic.py --region-s 300 --skip-s 120
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from eval.run_aphasia_eval import CACHE, _encode, load_region, run_acoustic  # noqa: E402

CKPT = ROOT / "models" / "stutternet_ssl_v2.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region-s", type=float, default=300.0)
    ap.add_argument("--skip-s", type=float, default=120.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not CKPT.exists():
        print("SKIPPED -- no %s" % CKPT)
        return 0

    for pid in ("1554", "1713", "1731", "1738", "1833", "1944"):
        audio, _ = load_region(pid, args.region_s, args.skip_s)
        if audio is None:
            print("  %s: no audio" % pid)
            continue
        tag = "%s_%d_%d" % (pid, int(args.skip_s), int(args.region_s))
        out = CACHE / ("ac_%s_sslstutter.json" % tag)
        if out.exists():
            print("  %s: cached" % pid)
            continue
        items = run_acoustic(audio, stutter_model=str(CKPT),
                             stutter_backend="ssl", device=args.device)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([_encode(t, it) for t, it in items]),
                       encoding="utf-8")
        kinds = {}
        for _, ev in items:
            kinds[ev.kind] = kinds.get(ev.kind, 0) + 1
        print("  %s: %d events %s" % (pid, len(items), kinds), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
