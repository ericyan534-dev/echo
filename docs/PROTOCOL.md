# Echo — WebSocket Protocol Reference

Complete, code-verified reference for Echo's two WebSocket endpoints and the
REST endpoints a client needs. Every message is read from the source that
implements it (`backend/app.py`, `session.py`, `schemas.py`,
`frontend/app.js`). For the *why* (dual-channel thesis,
prefetch, triggers) see [`ARCHITECTURE.md`](ARCHITECTURE.md); this covers the
*wire format*.

| Endpoint | Transport | Direction | Purpose |
|---|---|---|---|
| `GET /healthz` | REST JSON | server→client | one-shot status snapshot |
| `/ws` | WS, JSON text | bidirectional | transcript events in, predictions + viz out |
| `/ws/audio` | WS, binary | client→server only | raw mic PCM in, no reply frames |

Both are joined by one shared `EchoSession` per server process
(`session.py:37`) — a prediction born on `/ws/audio` or `/ws` fans out to every
`/ws` UI client. One session per process (hackathon-scale; see `ROADMAP.md`).

**Error-handling convention** (`/ws` and `/ws/audio`): every message after the
handshake is processed inside a `try/except Exception` that logs and continues
(`app.py:120-123`, `141-143`) — malformed JSON, a missing field, or a bad
coercion is logged (`"ws message error (ignored): %s"`) and **never kills the
socket**. Only `WebSocketDisconnect` ends the connection.

## `GET /healthz` — status snapshot

`app.py:62-78`; every client (`app.js:63`) and the runbook poll it first.

```json
{ "status":"ok", "provider":"gemini", "model":"gemini-3.5-flash",
  "active_predictor":"GeminiPredictor", "acoustic":"fillernet+prolongation",
  "prefetch":true,
  "audio_source":{"label":"Microphone (Wireless Mic Rx)","deviceId":"3f9c...",
    "profile_id":"dji-mic-2s","kind":"lav-wireless",
    "display_name":"DJI Mic 2S (wireless lav)","wearer_gate_recommended":true,
    "floor_dbfs":-61.0,"set_at":1757250000.0} }
```

| Field | Meaning |
|---|---|
| `provider` | `PREDICTOR_PROVIDER` as configured (`gemini`/`claude`/`mock`/`deepseek`/`local`) |
| `model` | configured model id, or `"(mock)"` |
| `active_predictor` | the class actually constructed — `"MockPredictor"` if the real provider failed to init (missing/invalid key), even if `provider` says otherwise |
| `acoustic` | `"fillernet+prolongation"` if `ACOUSTIC_MODEL` exists on disk, else `"prolongation-only"` |
| `prefetch` | whether speculative prefetch is enabled (`PREFETCH`) |
| `audio_source` | the mic the page last reported via `POST /api/audio/source`, or `null`. `profile_id`/`kind`/`display_name`/`wearer_gate_recommended` are `null` for unknown hardware. Informational only — no threshold reads it |

## `GET /api/audio/sources` · `POST /api/audio/source` — which microphone

`backend/audio_sources.py` is a registry of known capture hardware. The bytes on
`/ws/audio` are identical whichever mic produced them; what differs is what the
mic points at, which decides which browser processing to request and whether the
level-based speaker gate means anything.

`GET` returns the table:

```json
{"profiles":[
   {"id":"dji-mic-2s","display_name":"DJI Mic 2S (wireless lav)",
    "kind":"lav-wireless","match":["wireless\\s*mic\\s*rx","\\bdji\\b"],
    "constraints":{"echoCancellation":false,"noiseSuppression":false,"autoGainControl":false},
    "channel_count":1,"wearer_gate":true,"notes":"..."},
   {"id":"laptop-array","kind":"onboard-array","...":"..."}],
 "kinds":["lav-wireless","onboard-array"]}
```

`match` is case-insensitive regexes run over `MediaDeviceInfo.label`; the first
profile (table order) with a hit wins, so the specific lav sits above the generic
patterns. `constraints` are the `getUserMedia` constraints to request.

`POST` is sent by the page when it starts/restarts capture. Body: `{"label":str,
"deviceId":str|null, "profile_id":str|null, "floor_dbfs":number|null}`. A `null`
`profile_id` is resolved from the label server-side; an unknown `profile_id` or an
empty `label` is a `400` and does not replace the stored source. The reply is the
resolved record, byte-for-byte what `/healthz` shows. `floor_dbfs` is the
room-tone RMS the page measured, or `null` (non-finite stored as `null`).

## `GET /api/config` — who owns the transcript

```json
{"asr_provider":"crisper","asr_mode":"verbatim","asr_model":"turbo",
 "predictor":"gemini","stall_pause_ms":1300,"stall_min_gap_ms":4000,
 "wearer_gate":false,"audio_source":null}
```

`wearer_gate` = whether the level-based speaker gate is on (`WEARER_GATE`, default
off). When `asr_provider` is `"crisper"` the browser MUST NOT start
`SpeechRecognition` — the server already transcribes the same PCM, and two
transcript sources feeding one `StallDetector` would put every word on the
timeline twice (only one carrying fillers/cut-offs). It's HTTP, not a WS greeting,
so an unsolicited first frame can't be misread as a reply.

## `/ws` — UI protocol (JSON)

`app.py:82-127`. One connection per UI tab; on connect `sess.attach_ui`, on
disconnect `sess.detach_ui`.

### Client → server

| `type` | Fields | Parsing / defaults | Effect |
|---|---|---|---|
| `word` | `text`, `start_ms`, `end_ms`, `is_final`, `wearer_conf` (float\|null, **optional**) | `text` `""`; `start/end_ms` `0` (via `int()`); `is_final` `True`; `wearer_conf` `None`, non-numeric ignored | Builds a `Word`, feeds `EchoPipeline.handle()` — may fire a trigger |
| `silence` | `at_ms` (int) | `0` | `SilenceTick(at_ms)` — lets the detector notice a mid-utterance pause |
| `turn_end` | — | — | Closes the utterance: the fragment (or last punctuation-completed sentence) becomes context; detector + prefetch cache reset |
| `context` | `lines` (list[str]) | missing/non-list → no-op | Each line appended via `conversation.add_turn(str(line))` — seeds prior turns on connect |
| `reject` | `rejected` (list[str]) | non-list/missing → `[]` (no-op) | Re-predicts for the **current active stall** excluding every word rejected so far; arrives as a `prediction` with `served:"reject"` |
| `ping` | — | — | Liveness; server replies `pong` |

Example — a stalled utterance, then rejecting the served word:

```json
{"type":"word","text":"made","start_ms":400,"end_ms":680,"is_final":true}
{"type":"silence","at_ms":3200}
{"type":"reject","rejected":["toaster"]}
```

The frontend sends the **full accumulated rejected list** each time, but the
pipeline also unions rejections per stall (`_rejected_this_stall`), so a delta or
the full list both work (`tests/test_ws_reject.py`). `reject` is a no-op when
there's no active stall or a reject is already in flight.

**`word.wearer_conf`** — optional speaker-gate evidence, a float `0.0..1.0`: the
sender's confidence that **this word was spoken by the wearer**. Computed in the
browser (`app.js` `wearerConfForWord`) where transcript words and PCM share a
clock: the word's 90th-percentile dBFS (padded 120 ms) vs a rolling 75th-pct
baseline over 12 s (adaptive, so moving the mic doesn't mute the wearer
permanently); `baseline−3 dB`→`1.0`, `baseline−15 dB`→`0.0`, linear between.
**Fail-open — absent/`null` means "unknown" and must never suppress.** The field
is *omitted entirely* (never sent as `0.0`) when the browser can't answer
honestly: no mic stream, <~2 s of voiced audio (cold baseline), no level samples
in the word's span, or Echo's own TTS speaking (those blocks excluded from the
baseline). A receiver must treat a missing field exactly as before the field
existed. **Honest limitation:** this is a level argument, not speaker ID — strong
for a lav ~5 cm from the mouth (wearer 10–20 dB louder), materially weaker for
a laptop mic on a table. Validated on synthetic
two-speaker mixes only (`eval/run_speaker_gate_eval.py`); **unvalidated in real
rooms.**

### Server → client

| `type` | Fields | Sent when |
|---|---|---|
| `prediction` | `fragment`, `trigger`, `served`, `latency_ms` (float, 1 dp), `candidates` (`[{word,confidence}]`, may be `[]`) | a stall fired and the predictor or cache produced a result |
| `acoustic_event` | `kind`, `at_ms` (int), `confidence` (float) | mirrored from `/ws/audio` for every detected event (dual-channel viz) |
| `transcript_word` | `text`, `start_ms`, `end_ms`, `source:"asr"` | **crisper only** — one committed verbatim word (`"[UM]"`, `"f-"`, a repeat) |
| `interim` | `text` | **crisper only** — the uncommitted tail, display only; never fed to the detector (a revised filler would fire a false stall) |
| `pong` | — | reply to `ping` |

**`acoustic_event.kind`**: with FillerNet (v2 default) `"filler"` or
`"prolongation"`; with StutterNet (`STUTTER_MODEL` set, v3 default) one of
`"block"`, `"prolongation"`, `"sound_rep"`, `"word_rep"`, `"filler"` — `"block"`
is the important new one (a silent struggle, the strongest evidence a speaker is
stuck). **`prediction.trigger`**: `"pause"`/`"filler"`/`"hedge"`/
`"filler_acoustic"`/`"prolongation"`. **`prediction.served`** (`schemas.py:25`):

| Value | Meaning |
|---|---|
| `"live"` | a real predictor round-trip (or an empty prefetch/live result) |
| `"prefetch"` | instantly from the prefetch cache (~0 ms) |
| `"prefetch-stale"` | the live call failed/timed out; the (possibly drifted) cache was served rather than crashing the stall |
| `"reject"` | re-predicted after the speaker rejected the previous word(s) |

`candidates` is ranked most-likely-first; may be `[]` (UI renders "no
suggestion"). Example:

```json
{"type":"prediction","fragment":"I made some toast in the","trigger":"pause",
 "served":"prefetch","latency_ms":0.3,
 "candidates":[{"word":"toaster","confidence":0.95},{"word":"oven","confidence":0.8}]}
```

## `/ws/audio` — raw audio (binary)

`app.py:130-145`. No JSON, no handshake: **every binary message is raw audio
appended to that connection's `AcousticStream`** (`session.py:74-84`). The server
sends nothing here — results mirror to `/ws` as `acoustic_event` (and feed the
shared pipeline, so they can also produce a `prediction`).

| Property | Value |
|---|---|
| Format | PCM16, little-endian, 16 kHz, mono |
| Framing | none — any frame size, concatenated into a rolling buffer |
| Odd-length frames | trailing unpaired byte dropped (`features.py pcm16_to_float`), never crashes |
| Streams per connection | each gets its own `AcousticStream` (independent VAD + gating) |

The browser `AudioWorklet` speaks it (`frontend/pcm-worklet.js` + `app.js
flushPCM`, flushing ≥1600 samples / 100 ms as `Int16Array`) whichever host input
is selected — the DJI Mic 2S receiver or the laptop array (`MICROPHONE.md`); any
other client sending the same format is accepted identically. Internally (not
the wire contract): Silero VAD
on 512-sample chunks, prolongation on 800-sample (50 ms) frames, FillerNet on the
trailing 1.0 s every `HOP_MS`=125 ms — all independent of the client's frame size.

**Clock:** `acoustic_event.at_ms` is sample-accurate — ms of audio consumed by
that stream since creation (`AcousticStream.now_ms`), **plus the connection's
offset from the session audio epoch**. The epoch is set by the process's first
`/ws/audio` connection and never resets; every consumer (`AcousticStream` and the
verbatim ASR's `Word`/`SilenceTick` timestamps) is lifted by the same offset,
because the session owns one `Timeline` — otherwise a socket joining 40 s late
would stamp its first word at 100 ms and interleave it 40 s into the past
(observed: fragment `"remember hello the there"`, computed pause 40100 ms).
**Only one connection at a time transcribes** under `crisper` — the first live
connection owns the transcript; later ones run acoustic-only; ownership releases
when the owner drops.

## Cross-references

- Design rationale, trigger definitions, dual-channel thesis: `ARCHITECTURE.md`
- Demo-day operational notes (chip/badge meanings, fallbacks): `DEMO_SCRIPT.md`
- The microphone profiles and the measured DJI Mic 2S behaviour: `MICROPHONE.md`
- Metric definitions and numbers (latency, F1): `EVAL.md`
