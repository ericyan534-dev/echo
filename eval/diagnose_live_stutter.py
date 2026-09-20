"""Diagnose why OBVIOUS dysfluencies do not fire in the LIVE acoustic path.

This does NOT sweep thresholds. It takes clearly-labelled, unambiguous SEP-28k
clips (all 3 annotators agree on one dysfluency type, none of the others, clean
audio) and runs each one through the REAL backend.acoustic.stream.AcousticStream
exactly as backend/session.py drives it -- 20 ms / 640-byte PCM16 frames, Silero
VAD gating, the min_voiced_ms / refractory gates, and the per-kind frame
thresholds carried in the checkpoint -- then instruments every stage so the exact
point that drops an obvious event is visible per clip.

For each clip and backend it reports, stage by stage:
  vad_voiced_ms   how much voiced time the VAD accumulated (gate needs >=800)
  gate_ever_open  did _voiced_ms ever reach min_voiced_ms during the clip
  raw_pmax        max per-frame model probability over the WHOLE clip window
                  for the clip's own type (the model ceiling, no gate)
  frame_thresh    the per-kind threshold the live stream compares against
  model_clears    raw_pmax >= frame_thresh (would the model+threshold fire if
                  the gate were open and the frame were the recent one)
  fired           did the real stream actually emit an event of that kind

Two feed modes per clip:
  bare      the 3 s clip alone (an obvious stutter at the very start of speech)
  leadin    ~1.5 s of fluent voiced speech prepended, so the min_voiced gate is
            already satisfied when the stutter arrives (mimics mid-utterance)

Run:
  python eval/diagnose_live_stutter.py --backend cnn --device cpu
  python eval/diagnose_live_stutter.py --backend ssl --device cuda
  python eval/diagnose_live_stutter.py --both --per-type 8 --out eval/results/live_stutter_diagnosis.json

ASCII only. Re-runnable. Writes JSON if --out is given; never overwrites weights.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.acoustic import stream as stream_mod  # noqa: E402
from backend.acoustic.stream import AcousticStream, SR  # noqa: E402

DATA = ROOT / "data" / "sep28k"
CSV = DATA / "SEP-28k_labels.csv"
NPY = DATA / "clips_16k.npy"
IDX = DATA / "clips_index.json"

STUTTER_COLS = ["Prolongation", "Block", "SoundRep", "WordRep", "Interjection"]
# map csv column -> the AcousticEvent kind the stream emits
COL_TO_KIND = {
    "Block": "block",
    "Prolongation": "prolongation",
    "SoundRep": "sound_rep",
    "WordRep": "word_rep",
    "Interjection": "filler",
}
FRAME_BYTES = 640   # 320 samples @ 16 kHz PCM16 = 20 ms, as the client sends


def _key(show, ep, clip):
    return (show.strip(), str(ep).strip(), str(clip).strip())


def load_pool():
    """Return {key: row_index_in_npy} and the counts row for every clip."""
    idx = json.loads(IDX.read_text(encoding="utf-8"))
    key_to_row = {}
    for r in idx["rows"]:
        key_to_row[_key(r["show"], r["ep"], r["clip"])] = r["row"]
    counts = {}
    with open(CSV, newline="") as fh:
        for r in csv.DictReader(fh):
            k = _key(r["Show"], r["EpId"], r["ClipId"])
            counts[k] = r
    return key_to_row, counts


def select_obvious(key_to_row, counts, per_type, agree=3):
    """Unambiguous clips: target type == `agree` annotators, every other
    stutter type 0, clean audio, actually present in the npy."""
    out = {}
    for target in STUTTER_COLS:
        picks = []
        for k, row in key_to_row.items():
            c = counts.get(k)
            if c is None:
                continue
            if int(c[target]) < agree:
                continue
            if any(int(c[o]) > 0 for o in STUTTER_COLS if o != target):
                continue
            if int(c["PoorAudioQuality"]) or int(c["DifficultToUnderstand"]):
                continue
            if int(c["Unsure"]) or int(c["Music"]) or int(c["NoSpeech"]):
                continue
            picks.append((k, row))
            if len(picks) >= per_type:
                break
        out[target] = picks
    return out


def select_fluent(key_to_row, counts, n):
    """Clean clips with NO stuttered words -- lead-in and false-fire material."""
    out = []
    for k, row in key_to_row.items():
        c = counts.get(k)
        if c is None:
            continue
        if any(int(c[o]) > 0 for o in STUTTER_COLS):
            continue
        if int(c["NoStutteredWords"]) < 2:
            continue
        if int(c["PoorAudioQuality"]) or int(c["NoSpeech"]) or int(c["Music"]):
            continue
        out.append((k, row))
        if len(out) >= n:
            break
    return out


def pcm_of(arr, row):
    return torch.from_numpy(arr[row].astype("float32") / 32768.0)


def float_to_pcm16_bytes(x: torch.Tensor) -> bytes:
    i16 = (x.clamp(-1, 1) * 32768.0).to(torch.int16).numpy()
    return i16.tobytes()


def make_stream(backend, device, ckpt, **over):
    kw = dict(
        model_path=str(ROOT / "models" / "fillernet.pt"),
        stutter_model=str(ckpt),
        stutter_backend=backend,
        device=device,
    )
    kw.update(over)
    return AcousticStream(**kw)


def run_clip(st: AcousticStream, wav: torch.Tensor, target_kind: str):
    """Feed wav as 20 ms frames; capture gate state, raw model probs, fires.

    We replace the stream's own _run_stutter with a copy that does ONE model
    forward (the original would forward again), captures the raw frame probs,
    and then runs the IDENTICAL emission logic from backend/acoustic/stream.py
    so the fire decision is exactly the live one.
    """
    from backend.acoustic.features import logmel
    from backend.schemas import AcousticEvent
    types = st.stutter_types
    raw_whole = {t: 0.0 for t in types}   # max frame prob over whole clip
    raw_recent = {t: 0.0 for t in types}  # max of the 'recent' slice actually used
    gate_open_seen = {"v": False}

    def patched(end_off=0):
        if st._voiced_ms >= st.min_voiced_ms:
            gate_open_seen["v"] = True
        if st._buf.numel() - end_off < st._model_window // 2:
            return []
        window = st._window_ending_at(end_off, st._model_window)
        with torch.no_grad():
            if st.stutter_backend == "ssl":
                fp = torch.sigmoid(st.stutter(window.unsqueeze(0).to(st.device)))[0]
            else:
                feats = logmel(window).unsqueeze(0).unsqueeze(0).to(st.device)
                fp = torch.sigmoid(st.stutter(feats))[0]
        n_recent = max(1, int(round(st.hop * 1000 / SR / st.stutter_frame_ms)))
        rec = fp[:, -n_recent:].max(dim=1).values
        whole = fp.max(dim=1).values
        for i, t in enumerate(types):
            raw_whole[t] = max(raw_whole[t], float(whole[i]))
            raw_recent[t] = max(raw_recent[t], float(rec[i]))
        # identical emission logic to AcousticStream._run_stutter
        evs = []
        at = st._at_ms(end_off)
        for i, t in enumerate(types):
            kind = st.STUTTER_KIND.get(t)
            if kind is None:
                continue
            thresh = st.stutter_thresholds.get(t, 0.5) * st.stutter_scale
            p = float(rec[i])
            if p >= thresh and st._gate(kind, at):
                evs.append(AcousticEvent(kind, at, round(p, 3)))
        return evs

    orig = st._run_stutter
    st._run_stutter = patched
    fired = {t: 0 for t in COL_TO_KIND.values()}
    max_voiced = 0
    pcm_all = float_to_pcm16_bytes(wav)
    for off in range(0, len(pcm_all), FRAME_BYTES):
        frame = pcm_all[off:off + FRAME_BYTES]
        evs = st.feed(frame)
        max_voiced = max(max_voiced, st._voiced_ms)
        for e in evs:
            if e.kind in fired:
                fired[e.kind] += 1
    st._run_stutter = orig
    return {
        "vad_voiced_ms": int(max_voiced),
        "gate_ever_open": bool(gate_open_seen["v"]),
        "raw_whole": raw_whole,
        "raw_recent": raw_recent,
        "fired": fired,
        "fired_target": fired.get(target_kind, 0),
    }


def diagnose(backend, device, ckpt, obvious, fluent, arr, min_voiced_ms,
             refractory_ms, stutter_scale):
    # thresholds carried in this checkpoint (what the live path uses)
    probe = make_stream(backend, device, ckpt, min_voiced_ms=min_voiced_ms,
                        refractory_ms=refractory_ms, stutter_scale=stutter_scale)
    thresh = {t: probe.stutter_thresholds.get(t, 0.5) * stutter_scale
              for t in probe.stutter_types}
    lead = pcm_of(arr, fluent[0][1]) if fluent else torch.zeros(int(1.5 * SR))
    lead = lead[-int(1.5 * SR):]

    rows = []
    for target, clips in obvious.items():
        kind = COL_TO_KIND[target]
        print("    [%s] %s x%d" % (backend, target, len(clips)), flush=True)
        for (k, row) in clips:
            wav = pcm_of(arr, row)
            for mode, w in (("bare", wav), ("leadin", torch.cat([lead, wav]))):
                st = make_stream(backend, device, ckpt, min_voiced_ms=min_voiced_ms,
                                 refractory_ms=refractory_ms, stutter_scale=stutter_scale)
                res = run_clip(st, w, kind)
                rpm = res["raw_whole"][target]
                rows.append({
                    "type": target, "kind": kind, "clip": "/".join(k),
                    "mode": mode,
                    "vad_voiced_ms": res["vad_voiced_ms"],
                    "gate_ever_open": res["gate_ever_open"],
                    "raw_pmax": round(rpm, 4),
                    "raw_recent_pmax": round(res["raw_recent"][target], 4),
                    "frame_thresh": round(thresh[target], 4),
                    "model_clears": bool(rpm >= thresh[target]),
                    "fired": res["fired_target"],
                })
    return {"thresholds": {t: round(v, 4) for t, v in thresh.items()}, "rows": rows}


def summarize(tag, diag):
    rows = diag["rows"]
    print("\n==== %s ====" % tag)
    print("frame thresholds: %s" % diag["thresholds"])
    hdr = ("%-12s %-6s %-26s %8s %6s %9s %8s %6s %6s"
           % ("type", "mode", "clip", "voiced", "gate", "raw_pmax", "thresh",
              "clrs", "fired"))
    print(hdr)
    for r in rows:
        print("%-12s %-6s %-26s %8d %6s %9.3f %8.3f %6s %6d"
              % (r["type"], r["mode"], r["clip"][:26], r["vad_voiced_ms"],
                 "Y" if r["gate_ever_open"] else "n", r["raw_pmax"],
                 r["frame_thresh"], "Y" if r["model_clears"] else "n",
                 r["fired"]))
    # aggregate per type/mode
    print("\n  per type/mode: fired / n   (leadin isolates the model from the gate)")
    for target in STUTTER_COLS:
        for mode in ("bare", "leadin"):
            sub = [r for r in rows if r["type"] == target and r["mode"] == mode]
            if not sub:
                continue
            fired = sum(1 for r in sub if r["fired"] > 0)
            clears = sum(1 for r in sub if r["model_clears"])
            gate = sum(1 for r in sub if r["gate_ever_open"])
            print("    %-12s %-6s fired %d/%d  model_clears %d/%d  gate_open %d/%d"
                  % (target, mode, fired, len(sub), clears, len(sub), gate, len(sub)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["cnn", "ssl"], default="cnn")
    ap.add_argument("--both", action="store_true", help="run cnn (cpu) and ssl (cuda)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--per-type", type=int, default=8)
    ap.add_argument("--min-voiced-ms", type=int, default=800)
    ap.add_argument("--refractory-ms", type=int, default=1200)
    ap.add_argument("--stutter-scale", type=float, default=1.0)
    ap.add_argument("--cnn-ckpt", default=str(ROOT / "models" / "stutternet.pt"))
    ap.add_argument("--ssl-ckpt", default=str(ROOT / "models" / "stutternet_ssl_v2.pt"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key_to_row, counts = load_pool()
    obvious = select_obvious(key_to_row, counts, args.per_type)
    fluent = select_fluent(key_to_row, counts, 8)
    arr = np.load(NPY, mmap_mode="r")
    for t, c in obvious.items():
        print("  obvious %-12s selected %d (3/3 agreement, clean)" % (t, len(c)))
    print("  fluent lead-in/false-fire clips: %d" % len(fluent))

    jobs = []
    if args.both:
        jobs = [("cnn", "cpu", args.cnn_ckpt), ("ssl", "cuda", args.ssl_ckpt)]
    else:
        ckpt = args.ssl_ckpt if args.backend == "ssl" else args.cnn_ckpt
        jobs = [(args.backend, args.device, ckpt)]

    out = {"config": {"min_voiced_ms": args.min_voiced_ms,
                      "refractory_ms": args.refractory_ms,
                      "stutter_scale": args.stutter_scale,
                      "per_type": args.per_type, "frame_bytes": FRAME_BYTES},
           "backends": {}}
    for backend, device, ckpt in jobs:
        tag = "%s @ %s (%s)" % (backend, device, Path(ckpt).name)
        diag = diagnose(backend, device, ckpt, obvious, fluent, arr,
                        args.min_voiced_ms, args.refractory_ms, args.stutter_scale)
        summarize(tag, diag)
        out["backends"][backend] = {"device": device, "ckpt": Path(ckpt).name, **diag}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print("\n  wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
