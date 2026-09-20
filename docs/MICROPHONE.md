# Microphone — the DJI Mic 2S as Echo's input

Echo's audio input is a microphone captured by the browser: the laptop's own
array, or — the recognised, preferred option — a **DJI Mic 2S** wireless
lavalier worn by the speaker. The transmitter clips to the speaker; its USB
receiver plugs into the laptop and enumerates as an ordinary audio input. Either
way the bytes are the same: `getUserMedia` → `AudioWorklet` @ 16 kHz → PCM16 over
`/ws/audio` (see [`PROTOCOL.md`](PROTOCOL.md)); nothing in the backend is
DJI-specific.

Why a lav on the speaker matters: a capsule 5 cm from the mouth hears the
speaker louder than the room, which is exactly what the level-based speaker gate
(`wearer_conf`) needs and a table mic cannot give.

## How the console treats it

- **Recognition.** `backend/audio_sources.py` (served as `GET /api/audio/sources`)
  and the fallback `PROFILES` table in `frontend/console.js` carry a
  `dji-mic-2s` profile matched by label regex `wireless\s*mic\s*rx|\bdji\b`.
  Windows enumerates the receiver as `Wireless Mic Rx`; macOS/Chrome label it
  `DJI MIC` / `DJI Mic 2S`. The mic picker auto-prefers this profile when the
  receiver is plugged in; the page reports the chosen device with
  `POST /api/audio/source`, and `/healthz` shows it as `audio_source`.
- **Browser constraints for the DJI profile** (`dji-mic-2s`):
  `echoCancellation:false`, `noiseSuppression:false`, `autoGainControl:false`,
  `channelCount:1` — the receiver already applies its own DSP, and Chrome's AGC
  would normalise away the level evidence the speaker gate reads. TTS bleed is
  handled by the page dropping PCM while it speaks (`flushPCM`), not by AEC.
- **Speaker gate.** The profile carries `wearer_gate: true`; the laptop-array
  profile does not, because a room mic has no physical basis for the gate.

## Audio tiers (rehearse them)

1. **DJI Mic 2S** on the speaker, via the browser AudioWorklet — full demo.
2. DJI receiver absent → **laptop mic** feeds `/ws/audio` via the same
   AudioWorklet. Identical pipeline; the speaker gate loses its physical basis.
3. No mic → **Simulate** tab; same engine, keyboard-driven.

## Measured compatibility

Regenerate with `scripts/mic_probe.py` (`--list`; then `--device "Wireless Mic
Rx" --compare Realtek --seconds 20 --floor-dbfs -61 --json mic_probe.json`).
`--compare` opens both devices simultaneously so the two microphones hear the
same program at the same moment.

**Enumeration.** USB `VID_2CA3&PID_4015&MI_01`, endpoint `Wireless Mic Rx`, UAC
via Intel SST, 48 kHz / 2 ch / 24-bit, present on every host API and the default
input. With one transmitter both channels are bit-identical (Pearson r =
1.000000), so the worklet's channel-0 capture is lossless. Room tone (60 s): RMS
−61 dBFS, +28 dB tilt 20-100 Hz→2-4 kHz (HVAC), 0 xruns, 0 acoustic events;
`AcousticStream` ingests at ~7× realtime on CPU.

Two 20 s music captures (A, B) + one 2 s paired quiet check, both devices open at
once, StutterNet CNN on CPU, `WEARER_GATE` off:

| metric (16 kHz mono, ch 0) | DJI Mic 2S | laptop array (Realtek) |
|---|---|---|
| clipped / dead-zero (A, of 960k) | 0 / 112 | 0 / 18 |
| DC offset (A) | 6e-8 | 4e-8 |
| ch0 vs ch1 Pearson r (A / B) | 1.000000 / 1.000000 (bit-identical) | 0.215 / 0.158 |
| RMS music A / B | −40.9 / −42.8 dBFS | −45.7 / −48.3 dBFS |
| peak music A / B | −25.3 / −23.7 | −27.9 / −28.3 |
| tilt (20-100 minus 2-4 kHz) A / B | +3.1 / +6.5 dB | −12.2 / −6.9 dB |
| paired quiet floor (2 s RMS) | −63.6 dBFS | −75.6 dBFS |
| **level at capsule, DJI − array** A / B | **+4.8 / +5.5 dB** | — |
| SNR proxy = music − paired floor A / B | +22.7 / +20.8 dB | **+29.9 / +27.3 dB** |
| `feed()` per 20 ms mean/p95/max (A) | 3.13 / 19.01 / 48.04 ms | 3.00 / 18.80 / 21.72 ms |
| realtime factor (CPU) A / B | 6.4× / 6.3× | 6.7× / 6.6× |
| acoustic events on 2×20 s music | 0 | 0 |

(The Realtek path hands over one or two 200 ms frames of exact zeros before it's
live — excluded from percentiles, otherwise p10 reads −186 dBFS.)

**What this shows.** *Measured:* the DJI is a clean, format-compatible source —
no clipping, negligible DC, level in the trained speech range, consumed at 6–7×
realtime with `feed()` p95 under the 20 ms frame budget (the max is the 125 ms
CNN hop). *Measured:* the program arrives **4.8–5.5 dB hotter** at the DJI capsule
and keeps the low end (+3 to +6 dB tilt vs −7 to −12 dB) — geometry: a mic on the
speaker hears the speaker louder than the room, which is exactly what the
level-based speaker gate (`wearer_conf`) needs and a table mic can't give. *Does
NOT favour the DJI:* "RMS minus idle floor" is higher for the array (+27 to +30
vs +21 to +23 dB) because the array's idle floor is 12 dB lower — don't quote it
as a DJI advantage; the DJI's real advantage is unprocessed level at the capsule,
not a lower noise floor. *Not measured / not claimed:* any change in
stall-detection accuracy — the program here was music, not a wearer, and nothing
in `eval/` scores unlabeled program audio. The honest claim is "higher SNR at the
capsule and a physically meaningful speaker gate," not "more accurate"; a
DJI-vs-array accuracy comparison needs the held-vowel protocol
(`eval/record_protocol.md`) through both mics.

## What not to claim

- Never quote a DJI accuracy gain: none has been measured.
- Do not present "RMS minus idle floor" as a DJI advantage (see above).
- Pairing a **second** DJI transmitter would need the stereo check redone —
  the two channels are only known to be identical with one transmitter
  (WDM-KS exposes separate Mic 1 / Mic 2 pins).
- Every number above came from music program and room tone, not from a person
  stalling; re-run `scripts/mic_probe.py` for any new figure rather than
  extrapolating.

## Demo-day check

Receiver plugged in → backend up → open the console → the mic picker shows
**DJI Mic 2S (wireless lav)** selected → `/healthz` reports it under
`audio_source` → then walk to the judging table. If the receiver is missing,
the laptop array is selected automatically and the demo still runs.
