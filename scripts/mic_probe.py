# -*- coding: utf-8 -*-
"""Microphone probe: does this capture device feed Echo's acoustic channel?

One committed diagnostic in place of the ad-hoc scripts used to bring up the
DJI Mic 2S receiver. It answers, with numbers rather than a green light:

  1. Which input devices exist, and which is the DJI Mic 2S
     (backend/audio_sources.py: the DJI receiver enumerates on Windows as
     "Wireless Mic Rx", USB VID_2CA3 PID_4015; the laptop array as Realtek).
  2. What the device actually delivers: per-channel level / DC / clipping /
     dead-zero counts, inter-channel correlation (a receiver that duplicates
     one transmitter across both channels is safe for the frontend worklet,
     which reads channel 0 only), and the 20-100 Hz vs 2-4 kHz spectral tilt
     that separates a live capsule in a room from a muted one.
  3. Whether the shipped acoustic channel accepts it: the capture is
     resampled to the 16 kHz mono PCM16 /ws/audio carries and fed, in 20 ms
     frames, to a real AcousticStream built exactly the way
     backend/session.py builds one. feed() timing and event counts by kind
     are reported.

--compare opens a second device SIMULTANEOUSLY (two PortAudio streams) so
two microphones hear the same program at the same moment; that is how the
numbers in docs/MICROPHONE.md ("DJI Mic 2S -- measured compatibility") were
produced. --wav-in analyses an existing 16 kHz capture instead of recording.

Usage (run from the repo root so models/stutternet.pt resolves):
  python scripts/mic_probe.py --list
  python scripts/mic_probe.py --device 11 --seconds 20 --wav-out capture.wav
  python scripts/mic_probe.py --device 11 --compare 12 --seconds 20 --floor-dbfs -61
  python scripts/mic_probe.py --wav-in capture.wav

Exit codes: 0 ok, 1 usage / device error, 2 sounddevice or soxr missing.
Output is ASCII only: device names on a localised Windows contain non-ASCII
prefixes and are folded before printing.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import subprocess
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import numpy as np
except ImportError:  # numpy is a runtime dep of the acoustic channel
    print("numpy is missing: pip install -r requirements.txt")
    sys.exit(2)

try:
    import sounddevice as sd
    import soxr
except ImportError as exc:
    print("missing capture dependency (%s)." % exc)
    print("install: pip install -r requirements-dev.txt   (sounddevice, soxr)")
    sys.exit(2)

try:
    from backend.audio_sources import match_profile
except Exception:  # pragma: no cover - run outside the repo
    match_profile = None  # type: ignore[assignment]

SR_OUT = 16000
FRAME_SAMPLES = 320          # 20 ms @ 16 kHz, what the browser path posts
FRAME_BYTES = FRAME_SAMPLES * 2
EPS = 1e-12


# ------------------------------------------------------------------ helpers
def ascii_(s: object) -> str:
    return str(s).encode("ascii", "replace").decode("ascii")


def dbfs(v: float) -> float:
    return 20.0 * float(np.log10(max(float(v), EPS)))


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(v.astype(np.float64) ** 2))) if len(v) else 0.0


def profile_id_for(name: str) -> str:
    if match_profile is None:
        return "-"
    p = match_profile(name)
    return p.id if p else "-"


# ------------------------------------------------------------- enumeration
def input_devices() -> list[dict]:
    apis = sd.query_hostapis()
    out = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        out.append({
            "index": i,
            "hostapi": apis[d["hostapi"]]["name"],
            "name": ascii_(d["name"]),
            "channels": int(d["max_input_channels"]),
            "samplerate": float(d["default_samplerate"]),
            "profile": profile_id_for(d["name"]),
        })
    return out


def windows_pnp_dji() -> list[str]:
    """USB PnP ids carrying DJI's vendor id (VID_2CA3). Windows only; empty on
    any failure, which is not an error: the endpoint-name match is enough."""
    if platform.system() != "Windows":
        return []
    cmd = ("Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like "
           "'USB\\VID_2CA3*' -and $_.Class -eq 'MEDIA' } | "
           "ForEach-Object { $_.InstanceId + ' | ' + $_.FriendlyName }")
    try:
        r = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                           capture_output=True, text=True, timeout=15)
    except Exception:
        return []
    return [ascii_(line.strip()) for line in r.stdout.splitlines() if line.strip()]


def print_device_list() -> None:
    devs = input_devices()
    print("%-4s %-20s %-48s %3s %7s  %s" % ("idx", "hostapi", "name", "ch", "sr", "echo-profile"))
    for d in devs:
        print("%-4d %-20s %-48s %3d %7.0f  %s" % (
            d["index"], d["hostapi"][:20], d["name"][:48], d["channels"],
            d["samplerate"], d["profile"]))
    try:
        default_in = sd.query_devices(kind="input")
        print("default input: %s" % ascii_(default_in["name"]))
    except Exception:
        print("default input: (none)")
    pnp = windows_pnp_dji()
    if pnp:
        print("USB PnP (VID_2CA3 = DJI):")
        for line in pnp:
            print("  " + line)


def resolve_device(arg: str | None) -> int:
    """--device as an index or a case-insensitive name fragment. With no
    argument: the first DJI-profile device on WASAPI (its native 48 kHz
    endpoint), else the system default input."""
    devs = input_devices()
    if arg is None:
        dji = [d for d in devs if d["profile"] == "dji-mic-2s"]
        wasapi = [d for d in dji if "WASAPI" in d["hostapi"]]
        if wasapi:
            return wasapi[0]["index"]
        if dji:
            return dji[0]["index"]
        try:
            return int(sd.query_devices(kind="input")["index"])
        except Exception:
            print("no input device found; use --list")
            sys.exit(1)
    if arg.strip().lstrip("-").isdigit():
        idx = int(arg)
        if not any(d["index"] == idx for d in devs):
            print("device %d is not an input device; use --list" % idx)
            sys.exit(1)
        return idx
    frag = arg.lower()
    hits = [d for d in devs if frag in d["name"].lower()]
    if not hits:
        print("no input device matches %r; use --list" % arg)
        sys.exit(1)
    hits.sort(key=lambda d: (0 if "WASAPI" in d["hostapi"] else 1, -d["samplerate"]))
    return hits[0]["index"]


# ----------------------------------------------------------------- capture
def _open_stream(dev: int, q: "queue.Queue") -> tuple[sd.InputStream, int, int]:
    info = sd.query_devices(dev)
    sr = int(info["default_samplerate"])
    ch = int(info["max_input_channels"])
    status_count = {"n": 0}

    def cb(indata, frames, tinfo, status):
        if status:
            status_count["n"] += 1
        q.put(indata.copy())

    stream = sd.InputStream(device=dev, samplerate=sr, channels=ch, dtype="float32",
                            blocksize=sr // 10, callback=cb)
    stream._echo_status = status_count  # type: ignore[attr-defined]
    return stream, sr, ch


def capture(devices: list[int], seconds: float) -> list[dict]:
    """Record every device in `devices` at once (one PortAudio stream each) so
    they hear the same program. Returns [{index, name, sr, ch, x, xruns}]."""
    qs = [queue.Queue() for _ in devices]
    streams = [_open_stream(dev, q) for dev, q in zip(devices, qs)]
    for s, _, _ in streams:
        s.start()
    t0 = time.time()
    last = 0
    while time.time() - t0 < seconds:
        time.sleep(0.25)
        el = int(time.time() - t0)
        if el >= last + 5:
            last = el
            print("  ... %2d s" % el, flush=True)
    for s, _, _ in streams:
        s.stop()
        s.close()
    out = []
    for dev, q, (s, sr, ch) in zip(devices, qs, streams):
        chunks = []
        while not q.empty():
            chunks.append(q.get_nowait())
        x = np.concatenate(chunks) if chunks else np.zeros((1, ch), np.float32)
        out.append({"index": dev, "name": ascii_(sd.query_devices(dev)["name"]),
                    "sr": sr, "ch": ch, "x": x, "xruns": s._echo_status["n"]})  # type: ignore[attr-defined]
    return out


def read_wav(path: str) -> dict:
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        print("only 16-bit PCM wav is supported (%s is %d-bit)" % (path, sw * 8))
        sys.exit(1)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    x = x.reshape(-1, ch)
    return {"index": -1, "name": ascii_(os.path.basename(path)), "sr": sr, "ch": ch,
            "x": x, "xruns": 0}


def write_wav(path: str, y16k: np.ndarray) -> None:
    pcm = (np.clip(y16k, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR_OUT)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------- analysis
def channel_stats(x: np.ndarray) -> list[dict]:
    rows = []
    for c in range(x.shape[1]):
        v = x[:, c]
        rows.append({
            "chan": c,
            "rms_dbfs": dbfs(rms(v)),
            "peak_dbfs": dbfs(float(np.max(np.abs(v))) if len(v) else 0.0),
            "dc": float(np.mean(v)),
            "clipped": int(np.sum(np.abs(v) >= 0.999)),
            "zeros": int(np.sum(v == 0.0)),
        })
    return rows


def interchannel(x: np.ndarray) -> dict | None:
    if x.shape[1] < 2:
        return None
    a, b = x[:, 0].astype(np.float64), x[:, 1].astype(np.float64)
    den = float(np.std(a) * np.std(b))
    r = float(np.mean((a - a.mean()) * (b - b.mean())) / den) if den > 1e-20 else float("nan")
    maxdiff = float(np.max(np.abs(a - b)))
    if maxdiff < 1e-6:
        verdict = "bit-identical: one mono source duplicated (channel-0 capture is lossless)"
    elif r > 0.99:
        verdict = "near-identical: same source with a tiny gain/dither difference"
    else:
        verdict = "different signals: channels carry separate sources; the worklet keeps ch 0 only"
    return {"pearson_r": r, "max_abs_diff": maxdiff, "verdict": verdict}


def to_mono16k(x: np.ndarray, sr: int) -> np.ndarray:
    # Channel 0, not a downmix: frontend/pcm-worklet.js reads inputs[0][0],
    # so this is the signal Echo would actually receive from this device.
    mono = np.ascontiguousarray(x[:, 0], dtype=np.float32)
    if sr == SR_OUT:
        return mono
    return soxr.resample(mono, sr, SR_OUT).astype(np.float32)


BANDS = ((20, 100), (100, 300), (300, 800), (800, 2000), (2000, 4000), (4000, 8000))


def band_powers(y: np.ndarray, sr: int = SR_OUT, n: int = 4096) -> list[float] | None:
    win = np.hanning(n)
    segs = [np.abs(np.fft.rfft(y[i:i + n] * win)) ** 2 for i in range(0, len(y) - n, n)]
    if not segs:
        return None
    P = np.mean(segs, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        out.append(10.0 * float(np.log10(max(float(P[m].mean()), 1e-30))))
    return out


def frame_levels(y: np.ndarray, hop: int = SR_OUT // 5) -> np.ndarray:
    """RMS dBFS of consecutive 200 ms frames."""
    if len(y) < hop:
        return np.array([dbfs(rms(y))])
    return np.array([dbfs(rms(y[i:i + hop])) for i in range(0, len(y) - hop + 1, hop)])


def run_acoustic_stream(y16k: np.ndarray) -> dict:
    """Feed the capture to a real AcousticStream, constructed as
    backend/session.py constructs one (same settings, same gate flags)."""
    from backend.acoustic.stream import AcousticStream
    from backend.config import get_settings

    s = get_settings()
    t0 = time.time()
    st = AcousticStream(
        model_path=s.acoustic_model or None,
        conf_thresh=s.acoustic_conf,
        wearer_gate=s.wearer_gate,
        wearer_conf_min=s.wearer_conf_min,
        stutter_model=s.stutter_model or None,
        stutter_backend=s.stutter_backend,
        stutter_scale=s.stutter_scale,
        stutter_required=s.stutter_required,
        device=s.acoustic_device,
    )
    build_s = time.time() - t0
    pcm = (np.clip(y16k, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    lat, events = [], []
    t0 = time.time()
    for i in range(0, len(pcm) - FRAME_BYTES + 1, FRAME_BYTES):
        a = time.perf_counter()
        events.extend(st.feed(pcm[i:i + FRAME_BYTES]))
        lat.append((time.perf_counter() - a) * 1000.0)
    wall = time.time() - t0
    lat_a = np.array(lat) if lat else np.zeros(1)
    audio_s = len(pcm) / 2.0 / SR_OUT
    by_kind: dict[str, int] = {}
    for e in events:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    return {
        "build_s": build_s,
        "backend": s.stutter_backend,
        "device": s.acoustic_device,
        "stutternet": st.stutter is not None,
        "fillernet": st.model is not None,
        "wearer_gate": s.wearer_gate,
        "frames": len(lat),
        "audio_s": audio_s,
        "wall_s": wall,
        "realtime_factor": (audio_s / wall) if wall > 0 else float("inf"),
        "feed_mean_ms": float(lat_a.mean()),
        "feed_p50_ms": float(np.percentile(lat_a, 50)),
        "feed_p95_ms": float(np.percentile(lat_a, 95)),
        "feed_max_ms": float(lat_a.max()),
        "events": len(events),
        "events_per_min": (len(events) / audio_s * 60.0) if audio_s > 0 else 0.0,
        "events_by_kind": by_kind,
        "first_events": [{"kind": e.kind, "at_ms": e.at_ms, "confidence": round(e.confidence, 3)}
                         for e in events[:8]],
    }


def analyse(rec: dict, floor_dbfs: float | None, run_stream: bool) -> dict:
    x, sr = rec["x"], rec["sr"]
    y = to_mono16k(x, sr)
    lv_all = frame_levels(y)
    # WASAPI can hand over a frame or two of exact digital zeros before the
    # device is live; at -186 dBFS they are not a level, they are "not yet
    # started", and on a short capture they drag the p10 to nonsense.
    digital_silence = lv_all < -150.0
    lv = lv_all[~digital_silence] if (~digital_silence).any() else lv_all
    bands = band_powers(y)
    res = {
        "device": rec["index"],
        "name": rec["name"],
        "profile": profile_id_for(rec["name"]),
        "native_sr": sr,
        "native_ch": rec["ch"],
        "seconds": len(x) / float(sr),
        "xruns": rec["xruns"],
        "channels": channel_stats(x),
        "interchannel": interchannel(x),
        "mono16k_rms_dbfs": dbfs(rms(y)),
        "mono16k_peak_dbfs": dbfs(float(np.max(np.abs(y))) if len(y) else 0.0),
        "frame_p10_dbfs": float(np.percentile(lv, 10)),
        "frame_p90_dbfs": float(np.percentile(lv, 90)),
        "digital_silence_frames": int(digital_silence.sum()),
        "band_db": dict(zip(["%d-%d" % b for b in BANDS], bands)) if bands else None,
        "tilt_db": (bands[0] - bands[4]) if bands else None,
        "floor_dbfs": floor_dbfs,
        "snr_proxy_db": (dbfs(rms(y)) - floor_dbfs) if floor_dbfs is not None else None,
        "stream": run_acoustic_stream(y) if run_stream else None,
        "_y16k": y,
    }
    return res


# ------------------------------------------------------------------ report
def print_report(r: dict) -> None:
    print()
    print("=== device %s: %s  [profile: %s]" % (r["device"], r["name"], r["profile"]))
    print("native %d Hz x %d ch, %.2f s captured, portaudio status flags: %d"
          % (r["native_sr"], r["native_ch"], r["seconds"], r["xruns"]))
    print("%-5s %10s %10s %10s %8s %8s" % ("chan", "rms dBFS", "peak dBFS", "dc", "clipped", "zeros"))
    for c in r["channels"]:
        print("%-5d %10.1f %10.1f %10.2e %8d %8d" % (
            c["chan"], c["rms_dbfs"], c["peak_dbfs"], c["dc"], c["clipped"], c["zeros"]))
    ic = r["interchannel"]
    if ic:
        print("ch0 vs ch1: pearson r = %.6f, max |diff| = %.2e" % (ic["pearson_r"], ic["max_abs_diff"]))
        print("  -> %s" % ic["verdict"])
    print("16 kHz mono (ch 0): rms %.1f dBFS, peak %.1f dBFS; 200 ms frames p10 %.1f / p90 %.1f dBFS"
          % (r["mono16k_rms_dbfs"], r["mono16k_peak_dbfs"], r["frame_p10_dbfs"], r["frame_p90_dbfs"]))
    if r["digital_silence_frames"]:
        print("  (%d x 200 ms frame(s) of exact digital zeros excluded from the percentiles: "
              "device not yet live)" % r["digital_silence_frames"])
    if r["band_db"]:
        print("band power dB (relative): " + "  ".join(
            "%s:%.1f" % (k, v) for k, v in r["band_db"].items()))
        print("tilt (20-100 Hz minus 2-4 kHz): %+.1f dB" % r["tilt_db"])
    if r["snr_proxy_db"] is not None:
        print("SNR proxy vs supplied floor %.1f dBFS: %+.1f dB" % (r["floor_dbfs"], r["snr_proxy_db"]))
    st = r["stream"]
    if st:
        print("AcousticStream: backend=%s device=%s stutternet=%s fillernet=%s wearer_gate=%s (built in %.2f s)"
              % (st["backend"], st["device"], st["stutternet"], st["fillernet"],
                 st["wearer_gate"], st["build_s"]))
        print("  fed %.2f s in %.2f s wall (%.1fx realtime); feed() over %d x 20 ms frames: "
              "mean %.2f  p50 %.2f  p95 %.2f  max %.2f ms"
              % (st["audio_s"], st["wall_s"], st["realtime_factor"], st["frames"],
                 st["feed_mean_ms"], st["feed_p50_ms"], st["feed_p95_ms"], st["feed_max_ms"]))
        print("  events: %d (%.1f/min) %s" % (st["events"], st["events_per_min"],
                                              json.dumps(st["events_by_kind"], sort_keys=True)))
        for e in st["first_events"]:
            print("    %s at %d ms conf %.3f" % (e["kind"], e["at_ms"], e["confidence"]))


def print_comparison(results: list[dict]) -> None:
    print()
    print("=== side by side (same program, same moment)")
    cols = ["device", "profile", "rms dBFS", "p10", "p90", "tilt dB", "snr proxy",
            "feed mean", "feed p95", "feed max", "rt x", "events/min"]
    print("  ".join("%-10s" % c for c in cols))
    for r in results:
        st = r["stream"] or {}
        row = [
            str(r["device"]), r["profile"][:10],
            "%.1f" % r["mono16k_rms_dbfs"], "%.1f" % r["frame_p10_dbfs"], "%.1f" % r["frame_p90_dbfs"],
            "%+.1f" % r["tilt_db"] if r["tilt_db"] is not None else "-",
            "%+.1f" % r["snr_proxy_db"] if r["snr_proxy_db"] is not None else "-",
            "%.2f" % st.get("feed_mean_ms", float("nan")) if st else "-",
            "%.2f" % st.get("feed_p95_ms", float("nan")) if st else "-",
            "%.2f" % st.get("feed_max_ms", float("nan")) if st else "-",
            "%.1f" % st.get("realtime_factor", float("nan")) if st else "-",
            "%.1f" % st.get("events_per_min", float("nan")) if st else "-",
        ]
        print("  ".join("%-10s" % c for c in row))


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="enumerate input devices and exit")
    ap.add_argument("--device", help="input device index or name fragment (default: DJI if present, else system default)")
    ap.add_argument("--compare", help="second device to capture SIMULTANEOUSLY, for A/B")
    ap.add_argument("--seconds", type=float, default=10.0, help="capture length (default 10)")
    ap.add_argument("--wav-out", help="write the 16 kHz mono PCM16 capture here (primary device)")
    ap.add_argument("--wav-in", help="analyse this 16-bit wav instead of recording")
    ap.add_argument("--floor-dbfs", type=float, default=None,
                    help="room-tone RMS to compute the SNR proxy against (e.g. -61)")
    ap.add_argument("--no-stream", action="store_true", help="skip the AcousticStream pass")
    ap.add_argument("--json", help="also write every number to this JSON file")
    a = ap.parse_args(argv)

    if a.list:
        print_device_list()
        return 0

    if a.wav_in:
        recs = [read_wav(a.wav_in)]
    else:
        devices = [resolve_device(a.device)]
        if a.compare:
            devices.append(resolve_device(a.compare))
        for dev in devices:
            print("capturing device %d: %s" % (dev, ascii_(sd.query_devices(dev)["name"])))
        print("recording %.1f s%s ..." % (a.seconds, " from %d devices at once" % len(devices)
                                          if len(devices) > 1 else ""), flush=True)
        recs = capture(devices, a.seconds)

    results = [analyse(rec, a.floor_dbfs, not a.no_stream) for rec in recs]
    for r in results:
        print_report(r)
    if len(results) > 1:
        print_comparison(results)

    if a.wav_out:
        write_wav(a.wav_out, results[0]["_y16k"])
        print()
        print("wrote %s (%.2f s @ 16 kHz mono PCM16)" % (a.wav_out, len(results[0]["_y16k"]) / SR_OUT))
    if a.json:
        payload = []
        for r in results:
            d = {k: v for k, v in r.items() if not k.startswith("_")}
            payload.append(d)
        with open(a.json, "w", encoding="ascii") as f:
            json.dump({"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "argv": [ascii_(s) for s in (argv if argv is not None else sys.argv[1:])],
                       "results": payload}, f, indent=2, ensure_ascii=True)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
