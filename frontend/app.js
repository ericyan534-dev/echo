// Echo frontend v3 — Live (mic) + Simulate modes over one session.
//
// Live mode runs TWO channels in parallel. Which one carries the transcript
// depends on the server's ASR_PROVIDER, fetched from /api/config at load:
//
//   browser  1. SpeechRecognition (Chrome) -> words -> /ws   (what ASR admits to)
//            2. AudioWorklet @16kHz -> PCM16 -> /ws/audio    (what the mic hears)
//
//   crisper  1. AudioWorklet @16kHz -> PCM16 -> /ws/audio, and the SERVER
//               transcribes those same samples verbatim, sending words back
//               as `transcript_word`. SpeechRecognition is switched off
//               entirely -- two transcript sources feeding one detector would
//               double every word, and only one of them keeps the "[UM]",
//               the "f- Facebook" and the "you you recently".
//
// The server fuses transcript and acoustic events in one StallDetector; the
// dual-channel timeline shows the difference live.
//
// Mic note: SpeechRecognition always uses the OS-default input. The device
// dropdown selects the input for the ACOUSTIC channel + VU meter; when the
// DJI Mic 2S receiver is plugged in, that is the input it prefers.

"use strict";

// ---------- elements ----------
const $ = (id) => document.getElementById(id);
const wsChip = $("ws-chip"), modelChip = $("model-chip"), micChip = $("mic-chip");
const acousticChip = $("acoustic-chip");
const candidatesEl = $("candidates"), metaEl = $("meta"), historyEl = $("history");
const liveTranscriptEl = $("live-transcript"), simTranscriptEl = $("sim-transcript");
const laneText = $("lane-text"), laneAcoustic = $("lane-acoustic");

// ---------- websockets ----------
let ws, wsAudio;
let contextSent = false;  // send context only on the first open; reset by explicit user action
let pingInterval = null;  // track ping interval so it can be cleared on reconnect

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    setChip(wsChip, "connected", "ok");
    window.EchoConsole?.onSocket("open");
    if (!contextSent) { sendContext(); contextSent = true; }
    startPing();
  };
  ws.onclose = () => {
    setChip(wsChip, "reconnecting...", "warn");
    window.EchoConsole?.onSocket("closed");
    if (pingInterval) { clearInterval(pingInterval); pingInterval = null; }
    setTimeout(connect, 1000);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    window.EchoConsole?.onMessage(msg);
    if (msg.type === "pong" && _pingSentAt) { window.EchoConsole?.onRtt(performance.now() - _pingSentAt); _pingSentAt = 0; }
    if (msg.type === "prediction") renderPrediction(msg);
    else if (msg.type === "transcript_word") onServerWord(msg);
    else if (msg.type === "interim") onServerInterim(msg);
    else if (msg.type === "acoustic_event") handleAcousticEvent(msg);
  };
}

function connectAudio() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  wsAudio = new WebSocket(`${proto}://${location.host}/ws/audio`);
  wsAudio.binaryType = "arraybuffer";
  wsAudio.onclose = () => { if (listening) setTimeout(connectAudio, 800); };
}

const send = (obj) => { if (ws && ws.readyState === 1) ws.send(JSON.stringify(obj)); };

// Chips carry a persistent inline-SVG icon, so only the label text is
// rewritten. State is always spelled out in that label ("connected" /
// "reconnecting" / "listening") -- color is a glance aid, never the sole
// carrier of the information.
function setChip(el, text, cls) {
  const label = el.querySelector(".chip-label");
  if (label) label.textContent = text; else el.textContent = text;
  el.className = "chip" + (cls ? " " + cls : "");
}

const setIcon = (el, id) => {
  const use = el && el.querySelector("use");
  if (use) use.setAttribute("href", id);
};

fetch("/healthz").then((r) => r.json()).then((h) => {
  const live = h.active_predictor !== "MockPredictor";
  setChip(modelChip, `${h.model}${live ? "" : " (mock fallback)"}${h.prefetch ? " · prefetch" : ""}`,
          live ? "ok" : "warn");
  // "full" means a learned classifier is loaded, whichever one. StutterNet is
  // named explicitly because it carries five dysfluency types including Block,
  // and "full" would hide the difference from the only screen anyone watches.
  const stutter = h.acoustic === "stutternet+prolongation";
  const full = stutter || h.acoustic === "fillernet+prolongation";
  setChip(acousticChip,
          stutter ? "acoustic: stutternet (5 types)"
                  : full ? "acoustic: full" : "acoustic: prolongation-only",
          full ? "ok" : "warn");
}).catch(() => setChip(modelChip, "predictor offline", "warn"));

// A ping every 5 s; the pong's round trip feeds the rail's RTT readout.
let _pingSentAt = 0;  // performance.now() of the last ping
function startPing() {
  if (pingInterval) clearInterval(pingInterval);
  pingInterval = setInterval(() => { _pingSentAt = performance.now(); send({ type: "ping" }); }, 5000);
}

// ---------- tabs ----------
let mode = "live";
// Mode is an attribute on <body>; CSS hides every [data-when] branch that
// doesn't match. That lets one mode own content in more than one region of
// the console (control bar + stage) without a second toggle to keep in sync.
function setMode(m) {
  mode = m;
  document.body.dataset.mode = m;
  window.EchoConsole?.onMode(m);
  resetLatencyCounter();  // the previous mode's stamp does not describe this one
  for (const [id, want] of [["tab-live", "live"], ["tab-sim", "sim"]]) {
    const el = $(id);
    el.classList.toggle("active", m === want);
    el.setAttribute("aria-pressed", String(m === want));
  }
}
$("tab-live").onclick = () => setMode("live");
$("tab-sim").onclick = () => { stopListening(); setMode("sim"); };

// =====================================================================
// LIVE MODE — SpeechRecognition transcript channel
// =====================================================================
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null, listening = false;
let sessionT0 = 0;
let committedCount = 0;
let lastWordEndMs = 0;
let lastSuggestionAt = -Infinity;  // last time a prediction/candidates message was rendered
let silenceTimer = null, turnOpen = false;

const nowMs = () => Math.round(performance.now() - sessionT0);

function startListening() {
  if (SERVER_ASR) {
    // The server owns the transcript, the turn boundaries and the silence
    // ticks -- all three derived from the audio itself rather than from
    // browser event arrival times, which is the point: a transcript that
    // arrives late is indistinguishable from a speaker who paused.
    sessionT0 = performance.now();
    serverWords = []; serverInterim = "";
    lastWordEndMs = 0; lastSuggestionAt = -Infinity; turnOpen = false;
    resetWearerLevels();
    liveTranscriptEl.innerHTML = "";
    laneText.innerHTML = "";
    laneAcoustic.innerHTML = "";
    onListenStart();
    connectAudio();
    startAudioChannel();
    listening = true;
    setMicButton(true);
    silenceTimer = setInterval(() => {
      setIdleBreathing(!turnOpen && nowMs() - lastSuggestionAt > 4000
                       && nowMs() - _lastAcousticAt > 4000);
    }, 250);
    return;
  }
  if (!SR) {
    alert("SpeechRecognition is not supported in this browser. Use Chrome/Edge, or the Simulate tab.");
    return;
  }
  rec = new SR();
  rec.continuous = true;
  rec.interimResults = true;
  rec.lang = "en-US";

  sessionT0 = performance.now();
  committedCount = 0; lastWordEndMs = 0; lastSuggestionAt = -Infinity; turnOpen = false;
  resetWearerLevels();  // nowMs() restarts with sessionT0; stale samples would mis-span
  liveTranscriptEl.innerHTML = "";
  laneText.innerHTML = "";
  laneAcoustic.innerHTML = "";
  onListenStart();

  rec.onresult = (event) => {
    let finalText = "", interimText = "";
    for (let i = 0; i < event.results.length; i++) {
      const r = event.results[i];
      if (r.isFinal) finalText += r[0].transcript + " ";
      else interimText += r[0].transcript + " ";
    }
    const finalTokens = tokenize(finalText);
    const allTokens = finalTokens.concat(tokenize(interimText));
    const stableCount = Math.max(finalTokens.length, allTokens.length - 1);
    for (let i = committedCount; i < stableCount; i++) commitWord(allTokens[i]);
    committedCount = Math.max(committedCount, stableCount);
    renderLiveTranscript(allTokens, committedCount);
  };

  rec.onerror = (e) => {
    if (e.error === "not-allowed") { setChip(micChip, "mic blocked", "err"); stopListening(); }
  };
  rec.onend = () => {
    if (listening) { committedCount = 0; try { rec.start(); } catch (_) {} }
  };

  rec.start();
  listening = true;
  setMicButton(true);
  setChip(micChip, "listening", "ok");
  connectAudio();
  startAudioChannel();

  silenceTimer = setInterval(() => {
    setIdleBreathing(!turnOpen && nowMs() - lastSuggestionAt > 4000 && nowMs() - _lastAcousticAt > 4000);
    if (!turnOpen) return;
    const t = nowMs();
    send({ type: "silence", at_ms: t });
    // fluent turns commit conversation context 1.5s sooner; after a served suggestion the speaker gets the full 4s runway to read the card and finish the sentence
    const windowMs = (t - lastSuggestionAt < 6000) ? 4000 : 2500;
    if (t - lastWordEndMs > windowMs) { send({ type: "turn_end" }); turnOpen = false; onTurnEnd(); }
  }, 250);
}

function stopListening() {
  listening = false;
  if (rec) { try { rec.stop(); } catch (_) {} rec = null; }
  if (silenceTimer) { clearInterval(silenceTimer); silenceTimer = null; }
  if (turnOpen) { send({ type: "turn_end" }); turnOpen = false; }
  onTurnEnd();
  setIdleBreathing(false);
  setMicButton(false);
  setChip(micChip, "idle");
  stopAudioChannel();
  if (wsAudio) { try { wsAudio.close(); } catch (_) {} wsAudio = null; }
}

function setMicButton(on) {
  const btn = $("mic-btn");
  $("mic-btn-label").textContent = on ? "Stop listening" : "Start listening";
  btn.classList.toggle("live", on);
  btn.setAttribute("aria-pressed", String(on));
  setIcon(btn, on ? "#i-stop" : "#i-mic");
  window.EchoConsole?.onListening(on);
}

$("mic-btn").onclick = () => (listening ? stopListening() : startListening());

const tokenize = (text) => text.trim().split(/\s+/).filter(Boolean);

function commitWord(text) {
  const end = nowMs();
  const start = Math.max(0, end - 250);
  const msg = { type: "word", text, start_ms: start, end_ms: end, is_final: true };
  // Optional field: attached ONLY when the level history can actually answer.
  // Omitted means unknown, and unknown must never suppress (see
  // wearerConfForWord and docs/PROTOCOL.md).
  const conf = wearerConfForWord(start, end);
  if (conf !== null) msg.wearer_conf = Math.round(conf * 1000) / 1000;
  send(msg);
  lastWordEndMs = end;
  turnOpen = true;
  noteWordActivity();
  addLaneItem(laneText, text, "word");
  noteFinalWord();  // implicit-reject counter for the active card
}

// =====================================================================
// SERVER-SIDE VERBATIM ASR
// =====================================================================
// When ASR_PROVIDER=crisper the server transcribes the same PCM this page
// already streams to /ws/audio, and it keeps what Chrome deletes: "[UM]",
// "f- Facebook", "you you recently". The browser recognizer is then switched
// OFF entirely -- not merged. Two transcript sources feeding one detector
// would double every word on the timeline, and only one of them contains the
// evidence the detector exists to find.
let SERVER_ASR = false;
let serverWords = [];
let serverInterim = "";

// Fetched over HTTP before the socket opens, not pushed as a websocket
// greeting: /ws has a documented request/response shape and an unsolicited
// first frame breaks every client that reads the next message as its answer.
async function loadServerConfig() {
  try {
    const cfg = await (await fetch("/api/config")).json();
    SERVER_ASR = cfg.asr_provider === "crisper";
    if (SERVER_ASR) setChip(micChip, `server ASR (${cfg.asr_model} ${cfg.asr_mode})`, "ok");
  } catch (_) { SERVER_ASR = false; }
}
loadServerConfig();

function onServerWord(msg) {
  serverWords.push(msg.text);
  if (serverWords.length > 60) serverWords = serverWords.slice(-60);
  lastWordEndMs = nowMs();
  turnOpen = true;
  noteWordActivity();
  addLaneItem(laneText, msg.text, "word");
  noteFinalWord();
  renderServerTranscript();
}

function onServerInterim(msg) {
  serverInterim = msg.text || "";
  renderServerTranscript();
}

function renderServerTranscript() {
  const interimTokens = serverInterim ? serverInterim.split(/\s+/).filter(Boolean) : [];
  renderLiveTranscript(serverWords.concat(interimTokens), serverWords.length);
}

function renderLiveTranscript(tokens, committed) {
  liveTranscriptEl.innerHTML = "";
  tokens.forEach((t, i) => {
    const span = document.createElement("span");
    span.textContent = t + " ";
    if (i >= committed) span.className = "interim";
    liveTranscriptEl.appendChild(span);
  });
}

// ---------- acoustic channel: mic -> 16kHz PCM16 -> /ws/audio ----------
let audioCtx = null, vuStream = null, workletNode = null, vuRaf = null;
let pcmPending = [];
let pcmPendingLen = 0;

// TTS self-hearing suppression: while the synthesiser is speaking we drop PCM
// frames so the mic doesn't feed TTS audio back through the acoustic pipeline.
let ttsActive = false;
let _ttsTimer = null;
let _ttsGen = 0; // per-utterance token: a stale utterance's late onend/onerror
                 // must not stomp the current utterance's suppression window

function _clearTts() { ttsActive = false; _ttsTimer = null; }
function _armTtsClear(delayMs) {
  if (_ttsTimer) clearTimeout(_ttsTimer);
  _ttsTimer = setTimeout(_clearTts, delayMs);
}

async function startAudioChannel() {
  try {
    const sel = $("mic-select");
    const deviceId = sel.value || undefined;
    // Capture constraints follow the hardware profile of the chosen input
    // (console.js): a lav on the speaker wants no echo cancellation, the
    // laptop array wants it. Unknown inputs keep the historical defaults.
    const chosenLabel = sel.selectedOptions[0] ? sel.selectedOptions[0].textContent : "";
    const cons = window.EchoConsole ? window.EchoConsole.constraintsFor(deviceId ? chosenLabel : "")
                                    : { echoCancellation: true, noiseSuppression: false };
    vuStream = await navigator.mediaDevices.getUserMedia({
      audio: deviceId ? Object.assign({ deviceId: { exact: deviceId } }, cons) : cons,
    });
    window.EchoConsole?.onSource(vuStream.getAudioTracks()[0], cons);
    audioCtx = new AudioContext({ sampleRate: 16000 });
    await audioCtx.audioWorklet.addModule("/pcm-worklet.js");
    const src = audioCtx.createMediaStreamSource(vuStream);

    // VU meter
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    src.connect(analyser);
    const buf = new Uint8Array(analyser.frequencyBinCount);
    const bar = $("vu-bar");
    const loop = () => {
      analyser.getByteTimeDomainData(buf);
      let peak = 0;
      for (const v of buf) peak = Math.max(peak, Math.abs(v - 128));
      // scaleX, not width: this runs every animation frame for the whole
      // session, and animating width relayouts the control bar each time.
      bar.style.transform = "scaleX(" + Math.min(1, peak / 64).toFixed(3) + ")";
      vuRaf = requestAnimationFrame(loop);
    };
    loop();

    // PCM forwarder -> ~100ms binary frames
    workletNode = new AudioWorkletNode(audioCtx, "pcm-forwarder");
    src.connect(workletNode);
    workletNode.port.onmessage = (e) => {
      // Two accepted shapes: {pcm, rms} from the current worklet, and a bare
      // Float32Array from a STALE CACHED /pcm-worklet.js. The browser can
      // serve the old worklet from cache after a deploy; the acoustic channel
      // (the thing the demo depends on) must not die for a level number.
      const d = e.data;
      const pcm = d && d.pcm ? d.pcm : d;
      if (!pcm || !pcm.length) return;
      noteMicLevel(pcm, d && typeof d.rms === "number" ? d.rms : null);
      pcmPending.push(pcm);
      pcmPendingLen += pcm.length;
      if (pcmPendingLen >= 1600) flushPCM();
    };
    populateMics();
  } catch (err) {
    // The transcript channel (SpeechRecognition) can be perfectly healthy
    // while this fails -- unplugged USB mic (OverconstrainedError), device
    // held by another app (NotReadableError), worklet load failure. Without
    // this the mic chip stays green and the acoustic lane just silently
    // never fires, which is indistinguishable from "no stalls happened".
    console.warn("audio channel unavailable:", err);
    setChip(micChip, "transcript only - no mic stream", "warn");
  }
}

function flushPCM() {
  if (!wsAudio || wsAudio.readyState !== 1 || ttsActive) { pcmPending = []; pcmPendingLen = 0; return; }
  const total = pcmPendingLen;
  const f32 = new Float32Array(total);
  let off = 0;
  for (const chunk of pcmPending) { f32.set(chunk, off); off += chunk.length; }
  pcmPending = []; pcmPendingLen = 0;
  const i16 = new Int16Array(total);
  for (let i = 0; i < total; i++) {
    const v = Math.max(-1, Math.min(1, f32[i]));
    i16[i] = v < 0 ? v * 32768 : v * 32767;
  }
  wsAudio.send(i16.buffer);
}

function stopAudioChannel() {
  if (vuRaf) cancelAnimationFrame(vuRaf), (vuRaf = null);
  if (workletNode) { try { workletNode.disconnect(); } catch (_) {} workletNode = null; }
  if (vuStream) vuStream.getTracks().forEach((t) => t.stop()), (vuStream = null);
  if (audioCtx) { audioCtx.close(); audioCtx = null; }
  pcmPending = []; pcmPendingLen = 0;
  resetWearerLevels();  // a new device/stream is a new level regime
  $("vu-bar").style.transform = "scaleX(0)";
}

// =====================================================================
// WEARER CONFIDENCE (speaker gate, browser half)
//
// Transcript words and mic PCM share ONE clock only here in the browser, so
// this is where a word can be tied to the level at which it was spoken. Per
// finalized word we compare its level against a rolling baseline of the
// wearer's own recent voiced level, and ship the result as the optional
// `wearer_conf` field on the /ws `word` message (docs/PROTOCOL.md).
//
// FAILS OPEN, and that asymmetry is the whole design: wrongly muting the
// wearer is far worse than admitting a bystander. `wearer_conf` is OMITTED
// (never sent as 0, never guessed) whenever we cannot honestly answer --
// no mic stream, less than BASELINE_MIN_VOICED_MS of observed voiced audio,
// or no level samples inside the word's span. Absent == unknown == the
// server must not suppress.
//
// Honest limitation: this is a LEVEL argument. It is strong for the DJI Mic
// 2S lav worn a few cm from the speaker's mouth, and materially weaker
// for a laptop mic on a table where wearer and bystander sit at similar
// distance.
// =====================================================================
const LEVEL_HISTORY_MS = 12000;        // rolling level window (also the baseline window)
const BASELINE_MIN_VOICED_MS = 2000;   // below this: unknown, send nothing
const VOICED_FLOOR_DBFS = -55;         // below this a block is room noise, not speech
const WORD_PAD_MS = 120;               // ASR word boundaries are approximate
const CONF_FULL_DB = -3;               // at/above baseline-3dB  -> conf 1.0
const CONF_ZERO_DB = -15;              // at/below baseline-15dB -> conf 0.0
const MIN_BASELINE_BLOCKS = 24;        // ~200 ms of voiced blocks before any percentile

let levelHistory = [];  // [{t: sessionMs, db: dBFS}], oldest first
let voicedMsSeen = 0;   // accumulated voiced audio time this stream

let _floorAcc = 0, _floorN = 0;  // 100 ms block accumulator for the source-floor readout
function resetWearerLevels() { levelHistory = []; voicedMsSeen = 0; _floorAcc = 0; _floorN = 0; }

// One mic block -> one level sample. Called for every worklet block.
function noteMicLevel(pcm, rms) {
  // ttsActive is Echo's own TTS suppression window (+300 ms tail, 10 s safety
  // cap). Those blocks are the synthesiser, not the wearer: letting them into
  // the baseline would raise it and start suppressing the real wearer.
  if (ttsActive) return;
  if (rms === null) {
    let sum = 0;
    for (let i = 0; i < pcm.length; i++) sum += pcm[i] * pcm[i];
    rms = Math.sqrt(sum / pcm.length);
  }
  const t = nowMs();
  const db = rms > 1e-6 ? 20 * Math.log10(rms) : -120;
  levelHistory.push({ t, db });
  _floorAcc += rms * rms; _floorN += 1;
  if (_floorN >= 12) {  // ~100 ms at 16 kHz / 128-sample blocks
    const blockRms = Math.sqrt(_floorAcc / _floorN);
    window.EchoConsole?.onLevel(blockRms > 1e-6 ? 20 * Math.log10(blockRms) : -120);
    _floorAcc = 0; _floorN = 0;
  }
  if (db > VOICED_FLOOR_DBFS) {
    const sr = audioCtx ? audioCtx.sampleRate : 16000;
    voicedMsSeen += (pcm.length / sr) * 1000;
  }
  const cutoff = t - LEVEL_HISTORY_MS;
  while (levelHistory.length && levelHistory[0].t < cutoff) levelHistory.shift();
}

function percentile(sorted, q) {
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round(q * (sorted.length - 1))));
  return sorted[i];
}

// Returns 0..1, or null for "unknown" (fail open -- caller must OMIT the field).
function wearerConfForWord(startMs, endMs) {
  if (!audioCtx || voicedMsSeen < BASELINE_MIN_VOICED_MS) return null;
  const voiced = [], inWord = [];
  for (const s of levelHistory) {
    if (s.db <= VOICED_FLOOR_DBFS) continue;
    voiced.push(s.db);
    if (s.t >= startMs - WORD_PAD_MS && s.t <= endMs + WORD_PAD_MS) inWord.push(s.db);
  }
  if (voiced.length < MIN_BASELINE_BLOCKS || !inWord.length) return null;
  voiced.sort((a, b) => a - b);
  inWord.sort((a, b) => a - b);
  // Baseline = 75th percentile of recent voiced level (rolling, so the wearer
  // moving the mic or leaning back becomes the new normal instead of a
  // permanent mute). Word level = 90th percentile of its own blocks: a word
  // is judged by its loud part, not by the closing consonant.
  const delta = percentile(inWord, 0.9) - percentile(voiced, 0.75);
  const conf = (delta - CONF_ZERO_DB) / (CONF_FULL_DB - CONF_ZERO_DB);
  return Math.max(0, Math.min(1, conf));
}

let _micAutoPicked = false;
async function populateMics() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const sel = $("mic-select");
    const cur = sel.value;
    sel.innerHTML = '<option value="">System default</option>';
    const inputs = devices.filter((d) => d.kind === "audioinput");
    inputs.forEach((d) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Microphone ${sel.length}`;
      sel.appendChild(o);
    });
    sel.value = cur;
    window.EchoConsole?.onHostInputs(inputs);
    // First sight of labelled inputs: prefer the recognised DJI Mic 2S over
    // "System default" so the acoustic channel and the level meter follow
    // the capsule on the speaker without a manual pick.
    let manual = false;
    try { manual = !!localStorage.getItem("echo.micChoice"); } catch (_) {}
    if (!cur && !manual && !_micAutoPicked && window.EchoConsole) {
      const pref = window.EchoConsole.preferredDeviceId(inputs);
      if (pref) {
        _micAutoPicked = true;
        sel.value = pref;
        window.EchoConsole.log("source", "auto-selected " + (sel.selectedOptions[0]?.textContent || "input"), "event");
        if (listening) { stopAudioChannel(); startAudioChannel(); }
      }
    }
  } catch (_) {}
}
populateMics();
$("mic-select").onchange = () => {
  // an operator's explicit pick is remembered so auto-preference never
  // overrides it on a later enumerate
  try { localStorage.setItem("echo.micChoice", "1"); } catch (_) {}
  if (listening) { stopAudioChannel(); startAudioChannel(); }
};

// ---------- dual-channel timeline ----------
function addLaneItem(lane, text, cls) {
  const el = document.createElement("span");
  el.className = "lane-item " + cls;
  el.textContent = text;
  lane.appendChild(el);
  while (lane.children.length > 24) lane.firstChild.remove();
  lane.scrollLeft = lane.scrollWidth;
}

function renderAcousticEvent(msg) {
  const label = msg.kind === "prolongation" ? "PROLONGATION" : "FILLER";
  const conf = msg.confidence < 1 ? ` ${(msg.confidence * 100).toFixed(0)}%` : "";
  addLaneItem(laneAcoustic, `${label}${conf}`, "acoustic " + msg.kind);
}

// =====================================================================
// SIMULATE MODE (venue-proof fallback)
// =====================================================================
let simClock = 0;

function sendContext() {
  const box = $("context");
  if (!box) return;
  const lines = box.value.split("\n").map((s) => s.trim()).filter(Boolean);
  if (lines.length) send({ type: "context", lines });
}

$("speak").onclick = () => {
  const words = tokenize($("utterance").value);
  simTranscriptEl.textContent = "";
  let i = 0;
  const step = () => {
    if (i >= words.length) return;
    const w = words[i++];
    const start = simClock, end = simClock + 280;
    simClock = end + 120;
    simTranscriptEl.textContent += (simTranscriptEl.textContent ? " " : "") + w;
    send({ type: "word", text: w, start_ms: start, end_ms: end, is_final: true });
    noteWordActivity();
    noteFinalWord();  // implicit-reject counter for the active card
    setTimeout(step, 200);
  };
  step();
};

$("stall").onclick = () => { simClock += 1600; send({ type: "silence", at_ms: simClock }); };

$("reset").onclick = () => {
  send({ type: "turn_end" });
  onTurnEnd();
  sendContext();
  simTranscriptEl.innerHTML = '<span class="muted">Press Speak it word by word.</span>';
  clearCard();
  metaEl.textContent = "";
  resetLatencyCounter();  // a frozen stamp next to an emptied card is a lie
  simClock = 0;
};

// =====================================================================
// SHARED OUTPUT
// =====================================================================
function renderPrediction(msg) {
  lastSuggestionAt = nowMs();
  const cands = (msg.candidates || []).filter(
    (c) => c.word && !c.word.startsWith("(") && c.confidence > 0
  );
  if (msg.served === "reject") {
    refreshCard(cands);  // replacement for the current stall's card
    // refreshCard only emits card-rendered when the TOP word changed, and
    // early-returns outright when nothing is usable -- in either case the
    // counter would otherwise tick on to the 8s safety hide. No-ops when
    // nothing is pending, so it never fabricates a stamp.
    freezeLatencyCounter();
  } else if (!cands.length) {
    // A legitimate empty answer is still an answer: freeze the perceived-
    // latency counter (it no-ops when none is ticking) instead of letting
    // it run unfrozen to the 8s safety hide during exactly the honest
    // "no suggestion" scenario the Q&A script demos confidently.
    freezeLatencyCounter();
    clearCard('<p class="placeholder">No suggestion for this stall.' +
              '<span>Echo returns nothing rather than guessing.</span></p>');
  } else {
    showCard(cands);
    if ($("autospeak").checked) acceptCard(cands[0].word, "auto");
    addHistory(msg, cands);
    onCardRendered();
  }
  const servedBadge = msg.served === "prefetch"
    ? `prefetch hit ${msg.latency_ms} ms`
    : msg.served === "prefetch-stale"
    ? `live (stale cache fallback) ${msg.latency_ms} ms`
    : msg.served === "reject"
    ? `reject → replacement ${msg.latency_ms} ms`
    : `live ${msg.latency_ms} ms`;
  metaEl.innerHTML = `trigger: <b>${esc(msg.trigger)}</b> · ` +
    `<span class="${msg.served === "prefetch" ? "served-fast" : ""}">${servedBadge}</span> · ` +
    `"${esc(msg.fragment)}"`;
}

function addHistory(msg, cands) {
  if (historyEl.firstElementChild?.classList.contains("placeholder-row")) historyEl.innerHTML = "";
  const li = document.createElement("li");
  const served = msg.served === "prefetch" ? "prefetch · " : "";
  li.innerHTML = `<span class="h-frag">"${esc(msg.fragment)}"</span> → <b>${esc(cands[0].word)}</b>` +
    `<span class="h-meta">${esc(msg.trigger)} · ${served}${Math.round(msg.latency_ms)} ms</span>`;
  historyEl.prepend(li);
  while (historyEl.children.length > 8) historyEl.lastChild.remove();
}

// =====================================================================
// CANDIDATE CARD — accept / reject interaction model
// One large dominant word + small secondary chips. Tap a word = accept
// (speak via the existing TTS path). "not it" = reject: chip #2 is
// promoted instantly and the server re-predicts with all rejected words
// excluded (arrives as a served="reject" prediction -> refreshCard).
// If >= 4 new final transcript words arrive with no accept, the card
// fades out (implicit reject — a wrong word must never linger).
// Lifecycle CustomEvents on document for other teams to hook:
//   echo:card-rendered / echo:card-accepted / echo:card-rejected
//   (detail: {word, at}; accepted adds how: "tap"|"auto")
// =====================================================================
let card = null;  // active card for the current stall (null = none showing)
const cardStats = { taps: 0, autos: 0, rejects: 0, implicit: 0, tapMsTotal: 0 };

function emitCardEvent(name, word, extra) {
  document.dispatchEvent(new CustomEvent(name, {
    detail: Object.assign({ word, at: Date.now() }, extra || {}),
  }));
}

const RESTING_CARD =
  '<p class="placeholder">Silent while speech is fluent.' +
  '<span>A word appears only when Echo detects a stall.</span></p>';

// Destroying the card removes whatever the keyboard user was standing on.
// Park focus on the container (tabindex=-1) instead of letting it fall to
// <body>, which would cost them a full re-tab through the whole console --
// and renderCardDom() maps the container back onto the new dominant word if
// a replacement arrives.
function clearCard(html) {
  const hadFocus = candidatesEl.contains(document.activeElement);
  card = null;
  candidatesEl.innerHTML = html || RESTING_CARD;
  if (hadFocus) candidatesEl.focus();
}

function showCard(cands) {
  card = {
    words: cands.slice(),   // ranked [{word, confidence}], words[0] = dominant
    rejected: [],           // everything rejected this stall (sent to server)
    renderedAt: performance.now(),
    wordsSinceRender: 0,    // final transcript words since (re)render
    accepted: false,
    done: false,            // implicitly dismissed; drop late replacements
  };
  renderCardDom();
  emitCardEvent("echo:card-rendered", card.words[0].word);
  console.log("[card] rendered top=" + card.words[0].word);
}

// A served="reject" replacement for the current stall's card.
function refreshCard(cands) {
  if (!card || card.done) {
    // A served="reject" replacement can land AFTER the browser already
    // implicitly dismissed (or cleared) the card. The speaker actively asked
    // for a replacement -- show it as a fresh card instead of silently
    // dropping it (the server has already filtered rejected words).
    if (cands.length) showCard(cands);
    return;
  }
  const words = cands.filter((c) => !card.rejected.includes(c.word));
  if (!words.length) return;       // nothing usable: keep the local promotion
  const topChanged = !card.words.length || card.words[0].word !== words[0].word;
  card.words = words;
  card.wordsSinceRender = 0;       // fresh content gets a fresh implicit window
  renderCardDom();
  if (topChanged) emitCardEvent("echo:card-rendered", words[0].word);
  console.log("[card] refreshed top=" + words[0].word);
}

// Which control inside the card currently has focus, if any. Rebuilding the
// card destroys the focused node, and a keyboard user who accepts or rejects
// would otherwise be dumped back to <body> and have to re-tab the whole page.
function focusedCardRole() {
  const el = document.activeElement;
  if (!el || !candidatesEl.contains(el)) return null;
  if (el.classList.contains("not-it")) return ".not-it";
  if (el.classList.contains("cand-chip")) return ".cand-chip";
  if (el.classList.contains("dominant-word")) return ".dominant-word";
  // focus was parked on the container by clearCard(): a replacement card
  // should hand it back to the word, not strand it on a generic div
  if (el === candidatesEl) return ".dominant-word";
  return null;
}

function renderCardDom() {
  const refocus = focusedCardRole();
  candidatesEl.innerHTML = "";
  const wrap = document.createElement("div");
  wrap.className = "word-card" + (card.accepted ? " accepted" : "");
  const top = card.words[0];

  const dom = document.createElement("button");
  dom.className = "dominant-word";
  dom.setAttribute("aria-label",
    (card.accepted ? "Accepted: " : "Accept suggested word: ") + top.word);
  dom.innerHTML = `<span class="word">${esc(top.word)}</span>` +
    `<span class="conf">${card.accepted
      ? '<svg class="ic" aria-hidden="true"><use href="#i-check"/></svg>accepted'
      : Math.round(top.confidence * 100) + "%"}</span>`;
  dom.onclick = () => acceptCard(top.word, "tap");
  wrap.appendChild(dom);

  const row = document.createElement("div");
  row.className = "card-row";
  card.words.slice(1, 3).forEach((c) => {
    const chip = document.createElement("button");
    chip.className = "cand-chip";
    chip.textContent = c.word;
    chip.setAttribute("aria-label", "Accept alternative word: " + c.word);
    chip.onclick = () => acceptCard(c.word, "tap");
    row.appendChild(chip);
  });
  const notIt = document.createElement("button");
  notIt.className = "not-it";
  notIt.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-x"/></svg>not it';
  notIt.setAttribute("aria-label", "Reject " + top.word + " and predict again");
  notIt.onclick = rejectCard;
  row.appendChild(notIt);
  wrap.appendChild(row);
  candidatesEl.appendChild(wrap);

  if (refocus) {
    // the equivalent control in the rebuilt card, or the dominant word when
    // the previous one no longer exists (a rejected chip, say)
    const next = candidatesEl.querySelector(refocus) ||
                 candidatesEl.querySelector(".dominant-word");
    if (next) next.focus();
  }
}

function acceptCard(word, how) {
  if (!card || card.done || !card.words.length) { speak(word); return; }
  // Re-entry guard: a double-tap (touch bounce / eager presenter) on the
  // already-accepted word must not re-speak it or inflate the accept stats.
  // A TAP after an AUTO accept is still allowed through: that deliberate
  // upgrade is part of the design (auto-spoken cards stay interactive).
  if (card.accepted && card.acceptedHow === how && card.words[0].word === word) return;
  const at = performance.now();
  if (how === "tap") {
    // Only tap-accepts feed time-to-accept: an autospeak "accept" happens at
    // ~0 ms by construction and would poison the recovery metric.
    cardStats.taps += 1;
    cardStats.tapMsTotal += at - card.renderedAt;
  } else {
    cardStats.autos += 1;
  }
  card.accepted = true;
  card.acceptedHow = how;
  card.acceptedAt = at;
  const i = card.words.findIndex((c) => c.word === word);
  if (i > 0) card.words.unshift(card.words.splice(i, 1)[0]);  // chosen word -> dominant slot
  speak(word);  // the one TTS path — suppression machinery stays intact
  renderCardDom();  // accepted styling; chips + "not it" stay usable
  emitCardEvent("echo:card-accepted", word, { how });
  console.log("[card] accepted word=" + word + " how=" + how +
              " ms=" + Math.round(at - card.renderedAt));
  updateCardStats();
}

function rejectCard() {
  if (!card || card.done || !card.words.length) return;
  const word = card.words[0].word;
  card.rejected.push(word);
  card.words = card.words.slice(1);
  card.accepted = false;        // rejecting revokes an earlier (auto) accept
  card.rejectedAt = performance.now();
  card.wordsSinceRender = 0;    // speaker is engaged: reset the implicit window
  cardStats.rejects += 1;
  // Instant local promotion; the server replacement arrives as served="reject".
  send({ type: "reject", rejected: card.rejected.slice() });
  if (card.words.length) {
    renderCardDom();
  } else {
    const hadFocus = candidatesEl.contains(document.activeElement);
    candidatesEl.innerHTML =
      '<p class="placeholder">Finding another word…<span>Re-predicting with rejected words excluded.</span></p>';
    if (hadFocus) candidatesEl.focus();  // don't strand the keyboard user
  }
  emitCardEvent("echo:card-rejected", word);
  console.log("[card] rejected word=" + word + " total_rejected=" + card.rejected.length);
  updateCardStats();
}

// Called for every committed final transcript word (live + simulate).
function noteFinalWord() {
  if (!card || card.done || card.accepted) return;
  card.wordsSinceRender += 1;
  if (card.wordsSinceRender >= 4) implicitRejectCard();
}

function implicitRejectCard() {
  card.done = true;
  card.implicitAt = performance.now();
  cardStats.implicit += 1;
  const n = card.wordsSinceRender;  // clearCard() nulls card below
  const el = candidatesEl.firstElementChild;
  if (el && el.classList) {
    el.classList.add("fading");
    setTimeout(() => {
      if (el.parentNode === candidatesEl) clearCard();  // not replaced meanwhile
    }, 650);
  } else {
    clearCard();
  }
  console.log("[card] implicit_reject after " + n + " new words");
  updateCardStats();
}

function updateCardStats() {
  const el = $("card-stats");
  if (!el) return;
  const avg = cardStats.taps
    ? (cardStats.tapMsTotal / cardStats.taps / 1000).toFixed(1) + "s"
    : "—";
  el.textContent = `accept(tap) ${cardStats.taps} · time-to-accept avg ${avg} · ` +
    `reject ${cardStats.rejects} · implicit ${cardStats.implicit} · auto-speak ${cardStats.autos}`;
}

function speak(word) {
  if ("speechSynthesis" in window) {
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(word);
    u.rate = 1.05;
    const gen = ++_ttsGen;
    u.onstart = () => { if (gen === _ttsGen) { ttsActive = true; _armTtsClear(10000); } }; // 10 s safety cap
    u.onend = () => { if (gen === _ttsGen) _armTtsClear(300); };   // 300 ms tail after speech ends
    u.onerror = () => { if (gen === _ttsGen) _armTtsClear(300); };
    speechSynthesis.speak(u);
  }
}

const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// =====================================================================
// DEMO THEATER (additive presentation layer)
//   1. live perceived-latency counter (real timestamps, never canned)
//   2. "ASR heard it wrong" reveal sequencing (visual-only delay)
//   3. stalls-recovered session tally
//   4. idle breathing on the timeline lanes
// Everything below is new code; hooks into the existing flow are the
// single-line calls inserted above (handleAcousticEvent dispatch,
// onListenStart, noteWordActivity, onTurnEnd, onCardRendered).
// =====================================================================

// ---- (1) latency counter: acoustic detection arrival -> card render.
// Starts at the true UI arrival of an acoustic_event message
// (performance.now() in handleAcousticEvent), ticks via rAF, freezes when
// the card renders. This is wall-clock UI-perceived latency, so the frozen
// stamp is labeled "perceived" to distinguish it from the backend
// latency_ms badge in the meta line. Transcript-only stalls (pause/hedge)
// have no pre-card UI message, so they honestly get no stamp at all --
// freezeLatencyCounter() no-ops when nothing is pending.
let _latT0 = null;      // performance.now() at detection arrival, or null
let _latRaf = null;
let _latSafety = null;
const LAT_SAFETY_MS = 8000; // > backend 4 s predict timeout; hide if no card ever comes

function startLatencyCounter() {
  if (_latT0 !== null) return; // already ticking: keep the FIRST detection's t0
  _latT0 = performance.now();
  const el = $("latency-counter");
  if (el) {
    el.hidden = false;
    el.classList.remove("frozen");
    el.classList.add("ticking");
    const tick = () => {
      if (_latT0 === null) return;
      el.textContent = Math.round(performance.now() - _latT0) + " ms";
      _latRaf = requestAnimationFrame(tick);
    };
    tick();
  }
  if (_latSafety) clearTimeout(_latSafety);
  _latSafety = setTimeout(() => { if (_latT0 !== null) resetLatencyCounter(); }, LAT_SAFETY_MS);
}

function freezeLatencyCounter() {
  if (_latT0 === null) return; // no pending detection -> no stamp (never fake one)
  const ms = Math.round(performance.now() - _latT0);
  _latT0 = null;
  if (_latRaf) { cancelAnimationFrame(_latRaf); _latRaf = null; }
  if (_latSafety) { clearTimeout(_latSafety); _latSafety = null; }
  const el = $("latency-counter");
  if (el) {
    el.textContent = ms + " ms perceived";
    el.classList.remove("ticking");
    el.classList.add("frozen");
  }
}

function resetLatencyCounter() { // full reset: new listening session or safety timeout
  _latT0 = null;
  if (_latRaf) { cancelAnimationFrame(_latRaf); _latRaf = null; }
  if (_latSafety) { clearTimeout(_latSafety); _latSafety = null; }
  const el = $("latency-counter");
  if (el) { el.hidden = true; el.classList.remove("ticking", "frozen"); }
}

// ---- (2) reveal sequencing: on an acoustic event, strike the transcript
// lane's most recent word ('ASR heard: "the"') immediately, and let the
// acoustic lane flash pop only after --reveal-ms (styles.css). The delay is
// purely presentational: the lane item is appended NOW and hidden by a CSS
// animation-delay; the latency counter and all backend traffic run at true
// arrival time.
let _lastAcousticAt = -Infinity;

function handleAcousticEvent(msg) {
  startLatencyCounter();  // real arrival timestamp -- never delayed by the reveal
  _lastAcousticAt = nowMs();
  setIdleBreathing(false);
  const revealed = revealAsrHeard();
  renderAcousticEvent(msg);
  if (revealed && laneAcoustic.lastElementChild) {
    laneAcoustic.lastElementChild.classList.add("reveal-delayed");
  }
}

function revealAsrHeard() {
  const last = laneText.lastElementChild;
  if (!last || !last.classList.contains("word")) return false;
  if (last.classList.contains("asr-missed")) return false;
  last.classList.add("asr-missed");
  setTimeout(() => last.classList.remove("asr-missed"), revealMs() + 250);
  return true;
}

function revealMs() {
  const v = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--reveal-ms"), 10);
  return Number.isFinite(v) ? v : 500;
}

// ---- (1+3) card-render hook. Called directly from renderPrediction in
// this branch AND via the 'echo:card-rendered' CustomEvent that the
// card-interaction branch emits from its showCard(); the dedupe window
// makes a double fire a no-op whichever render path survives the merge.
let _lastCardRenderAt = -Infinity;
let _awaitingContinuation = false;
let _tallyCount = 0;

function onCardRendered() {
  const now = performance.now();
  if (now - _lastCardRenderAt < 60) return; // direct call + CustomEvent dedupe
  _lastCardRenderAt = now;
  freezeLatencyCounter();
  _awaitingContinuation = true; // armed: next committed word = turn continued
}
document.addEventListener("echo:card-rendered", onCardRendered);
// A rejected card is not a recovery even if speech continues afterwards.
document.addEventListener("echo:card-rejected", () => { _awaitingContinuation = false; });

// ---- (3) session tally: a stall produced a card AND the speaker then kept
// going (next committed word in live or sim mode before the turn ended).
// Session-local behavioral count -- the badge carries a "this session"
// qualifier and makes no efficacy claim.
function noteWordActivity() {
  setIdleBreathing(false);
  if (!_awaitingContinuation) return;
  _awaitingContinuation = false;
  _tallyCount += 1;
  const el = $("tally-count");
  if (el) el.textContent = String(_tallyCount);
}

function onTurnEnd() { _awaitingContinuation = false; } // turn died unfinished: not a recovery

// ---- (4) idle breathing: subtle border pulse on the two lanes while
// listening and silent, so a quiet UI reads alive rather than frozen.
// State is owned by the 250 ms silence poll in startListening; word/acoustic
// activity clears it instantly.
let _breathing = false;
function setIdleBreathing(on) {
  on = !!on;
  if (on === _breathing) return;
  _breathing = on;
  laneText.classList.toggle("idle-breathing", on);
  laneAcoustic.classList.toggle("idle-breathing", on);
}

function onListenStart() { // fresh listening session: reset theater state
  _lastAcousticAt = -Infinity;
  setIdleBreathing(false);
  resetLatencyCounter();
}

connect();
setMode("live");
updateCardStats();  // show the zeroed accept/reject HUD from the start
