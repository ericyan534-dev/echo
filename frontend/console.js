// Echo Console shell -- pages, inputs tree, audio-source profile, event log,
// session table, settings sheet. Loaded BEFORE app.js; app.js calls the
// hooks on window.EchoConsole at the few points where it learns something
// (host inputs, socket state, source track, ping RTT, incoming messages).
// Every hook is optional-chained on the app.js side so a stale cached copy
// of either file cannot break the other.
//
// Nothing here fabricates: the input list, RTT, levels, latency and the
// runtime sheet are read from the browser, the server or the live stream.

"use strict";

(function () {
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const T0 = performance.now();
  const clock = () => {
    const s = Math.floor((performance.now() - T0) / 1000);
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0");
  };
  const stamp = () => {
    const ms = Math.round(performance.now() - T0);
    const s = Math.floor(ms / 1000);
    return String(Math.floor(s / 60)).padStart(2, "0") + ":" + String(s % 60).padStart(2, "0") +
      "." + String(Math.floor((ms % 1000) / 100));
  };

  // ------------------------------------------------------------------
  // known input hardware. The server's /api/audio/sources overrides this
  // table when present; this copy exists so the console still recognises
  // the DJI on a server build that predates the endpoint.
  // ------------------------------------------------------------------
  let PROFILES = [
    {
      id: "dji-mic-2s", name: "DJI Mic 2S (wireless lav)", kind: "lav-wireless", wearer_gate: true,
      match: "wireless\\s*mic\\s*rx|\\bdji\\b",
      constraints: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      notes: "capsule on the speaker; receiver is a 48 kHz UAC device whose two channels are identical",
    },
    {
      id: "laptop-array", name: "Laptop microphone array", kind: "onboard-array", wearer_gate: false,
      match: "realtek|mic(rophone)?\\s*array|built-?in|internal\\s*mic|macbook|\\bsst\\b",
      constraints: { echoCancellation: true, noiseSuppression: false, autoGainControl: true, channelCount: 1 },
      notes: "hears the room, not the speaker",
    },
  ];
  const DEFAULT_CONSTRAINTS = { echoCancellation: true, noiseSuppression: false };

  function profileFor(label) {
    if (!label) return null;
    for (const p of PROFILES) {
      try { if (new RegExp(p.match, "i").test(label)) return p; } catch (_) {}
    }
    return null;
  }

  fetch("/api/audio/sources").then((r) => (r.ok ? r.json() : null)).then((j) => {
    const list = Array.isArray(j) ? j : j && Array.isArray(j.profiles) ? j.profiles : null;
    if (!list || !list.length) return;
    PROFILES = list.map((p) => ({
      id: p.id, name: p.display_name || p.name || p.id, kind: p.kind || "",
      match: Array.isArray(p.match) ? p.match.join("|") : (p.match || ""),
      constraints: Object.assign({}, p.constraints || DEFAULT_CONSTRAINTS,
                                 p.channel_count ? { channelCount: p.channel_count } : {}),
      notes: p.notes || "",
      wearer_gate: !!p.wearer_gate,
    }));
    renderProfiles();
  }).catch(() => {});

  function renderProfiles() {
    const tb = $("profile-table") && $("profile-table").tBodies[0];
    if (!tb) return;
    tb.innerHTML = PROFILES.map((p) =>
      `<tr><td class="word">${esc(p.name)}</td><td class="mono nowrap">${esc(p.kind)}</td>` +
      `<td class="mono dimtext regex">${esc(p.match)}</td><td class="mono">${esc(fmtConstraints(p.constraints))}</td></tr>`
    ).join("");
  }
  function fmtConstraints(c) {
    return Object.entries(c || {}).map(([k, v]) =>
      (k === "echoCancellation" ? "aec" : k === "noiseSuppression" ? "ns" : k === "autoGainControl" ? "agc" : k) +
      (typeof v === "boolean" ? (v ? " on" : " off") : " " + v)).join(" · ");
  }

  // ------------------------------------------------------------------
  // inputs tree: the host audio inputs the browser can capture
  // ------------------------------------------------------------------
  let hostInputs = [];    // [{deviceId, label}] from enumerateDevices
  let activeSource = null;

  const shortLabel = (l) => String(l || "").replace(/\s*\([0-9a-f]{4}:[0-9a-f]{4}\)\s*$/i, "").slice(0, 40);
  const profileShort = (p) => String(p.name).replace(/\s*\(.*\)\s*$/, "");
  const isAlias = (d) => d.deviceId === "default" || d.deviceId === "communications";

  function renderTree() {
    const ul = $("tree-inputs");
    if (!ul) return;
    // Chrome lists "Default - X" / "Communications - X" aliases next to the
    // concrete entries; show the aliases only when they are all there is.
    const concrete = hostInputs.filter((d) => !isAlias(d));
    const list = concrete.length ? concrete : hostInputs;
    const labelled = list.filter((d) => d.label);
    const isLive = (d) => !!activeSource && activeSource.label === d.label;
    const rows = labelled.map((d) => {
      const p = profileFor(d.label);
      const live = isLive(d);
      const cls = live ? " source" : p ? " ready" : "";
      const state = live ? "live" : p ? "detected" : "idle";
      const title = shortLabel(d.label) + (p ? " -- " + p.name : " -- generic input, no profile") +
        (live ? " (capturing)" : "");
      return `<li class="unit-row${cls}" title="${esc(title)}"><span class="dot" aria-hidden="true"></span>` +
        `<span class="u-name">${esc(p ? profileShort(p) : shortLabel(d.label))}</span>` +
        `<span class="u-state">${state}</span></li>`;
    });
    // the active track is always shown, even when enumerateDevices has not
    // caught up (a default-input capture reports the OS device label)
    if (activeSource && !labelled.some(isLive)) {
      const p = activeSource.profile;
      rows.unshift(`<li class="unit-row source" title="host input in use"><span class="dot" aria-hidden="true"></span>` +
        `<span class="u-name">${esc(p ? profileShort(p) : shortLabel(activeSource.label) || "default input")}</span>` +
        `<span class="u-state">live</span></li>`);
    }
    ul.innerHTML = rows.length ? rows.join("")
      : `<li class="unit-row" title="input names are hidden until the mic has been used once">` +
        `<span class="dot" aria-hidden="true"></span><span class="u-name">${list.length ? list.length + " unnamed" : "no inputs yet"}</span>` +
        `<span class="u-state">${list.length ? "start listening to name" : ""}</span></li>`;
    const n = labelled.length || list.length;
    set("tree-summary", (activeSource ? "1 live · " : "") + n + (n === 1 ? " input" : " inputs"));
  }

  // ------------------------------------------------------------------
  // audio source (the DJI question, answered on screen)
  // ------------------------------------------------------------------
  let floorSamples = [];   // dBFS of 100 ms blocks, rolling
  let serverCfg = null;    // /api/config, for the speaker-gate hint
  let floorPosted = false; // second POST carries the measured floor once
  function gateText(profile) {
    if (!profile) return "n/a (unknown input)";
    if (!profile.wearer_gate) return "n/a for this input (mic hears the room)";
    if (!serverCfg || typeof serverCfg.wearer_gate !== "boolean") return "recommended on for this mic";
    return serverCfg.wearer_gate ? "on" : "off on server; recommended on for this mic";
  }
  function postSource(floor) {
    if (!activeSource) return;
    const a = activeSource;
    try {
      fetch("/api/audio/source", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: a.label, deviceId: a.settings.deviceId || null,
                               profile_id: a.profile ? a.profile.id : null,
                               floor_dbfs: floor == null ? null : Math.round(floor * 10) / 10 }),
      }).then((r) => { if (!r.ok && r.status !== 404) log("source", "server rejected source report (" + r.status + ")", "warn"); }).catch(() => {});
    } catch (_) {}
  }
  function onSource(track, chosen) {
    const label = track ? track.label : "";
    const profile = profileFor(label);
    const settings = track && track.getSettings ? track.getSettings() : {};
    activeSource = { label, profile, settings, constraints: chosen };
    floorSamples = []; floorPosted = false;
    set("src-name", shortLabel(label) || "default input");
    set("src-profile", profile ? profile.name + " · " + profile.kind : "generic (no profile)");
    set("src-format", [settings.sampleRate ? settings.sampleRate + " Hz" : null,
                       settings.channelCount ? settings.channelCount + " ch" : null,
                       "→ 16 kHz mono to /ws/audio"].filter(Boolean).join(" · "));
    set("src-constraints", fmtConstraints(chosen));
    const gate = $("src-gate");
    if (gate) { gate.textContent = gateText(profile); gate.className = profile && profile.wearer_gate && serverCfg && serverCfg.wearer_gate === false ? "warn" : ""; }
    const badge = $("source-badge");
    if (badge) {
      badge.hidden = !profile;
      if (profile) badge.innerHTML = '<svg class="ic" aria-hidden="true"><use href="#i-radio"/></svg>' + esc(profile.name);
    }
    log(profile ? "source" : "source", (profile ? profile.name + " recognised: " : "input: ") + (shortLabel(label) || "default"), profile ? "event" : "");
    renderAll();
    // tell the server which hardware is feeding it (additive endpoint; a
    // server without it answers 404 and nothing changes). A second report
    // follows once the room-tone floor has been measured.
    postSource(null);
  }
  function onSourceStop() {
    activeSource = null;
    set("src-name", "none (not listening)"); set("src-profile", "--"); set("src-format", "--"); set("src-constraints", "--");
    if ($("source-badge")) $("source-badge").hidden = true;
    if ($("source-floor")) $("source-floor").innerHTML = "floor <b>--</b> dBFS";
    set("src-gate", "--"); set("vu-db", "-- dBFS");
    renderAll();
  }
  // called from app.js per 100 ms block with dBFS; floor = 10th percentile
  function onLevel(db) {
    if (!Number.isFinite(db)) return;
    set("vu-db", (db <= -99 ? "-inf" : db.toFixed(0)) + " dBFS");
    floorSamples.push(db);
    if (floorSamples.length > 600) floorSamples = floorSamples.slice(-600);
    if (floorSamples.length % 10 === 0 && floorSamples.length >= 20) {
      const sorted = floorSamples.slice().sort((a, b) => a - b);
      const floor = sorted[Math.floor(sorted.length * 0.1)];
      if ($("source-floor")) $("source-floor").innerHTML = "floor <b>" + floor.toFixed(0) + "</b> dBFS";
      if (activeSource) activeSource.floor = floor;
      if (!floorPosted && floorSamples.length >= 30) { floorPosted = true; postSource(floor); }
    }
  }
  function constraintsFor(label) {
    const p = profileFor(label);
    return Object.assign({}, p ? p.constraints : DEFAULT_CONSTRAINTS);
  }
  // A recognised lav on the speaker (the DJI Mic 2S) wins over "System
  // default"; a concrete device entry wins over Chrome's "default" /
  // "communications" aliases.
  function preferredDeviceId(devices) {
    const hits = devices.filter((d) => { const p = profileFor(d.label); return p && /lav/i.test(p.kind); });
    const concrete = hits.find((d) => !isAlias(d));
    return (concrete || hits[0] || { deviceId: "" }).deviceId;
  }
  function onHostInputs(devices) { hostInputs = devices.slice(); renderAll(); }

  // ------------------------------------------------------------------
  // event log + session table
  // ------------------------------------------------------------------
  const LOG_MAX = 120;
  const logRows = [];
  function log(kind, text, cls) {
    const row = { t: stamp(), kind, text: String(text), cls: cls || "" };
    logRows.push(row);
    if (logRows.length > LOG_MAX) logRows.shift();
    const ol = $("event-log");
    if (!ol) return;
    if (ol.firstElementChild && ol.firstElementChild.classList.contains("placeholder-row")) ol.innerHTML = "";
    const li = document.createElement("li");
    li.className = row.cls;
    li.innerHTML = `<span class="t">${row.t}</span><span class="k">${esc(kind)}</span><span class="m">${esc(row.text)}</span>`;
    ol.appendChild(li);
    while (ol.children.length > LOG_MAX) ol.firstChild.remove();
    ol.scrollTop = ol.scrollHeight;
  }
  if ($("log-clear")) $("log-clear").onclick = () => { logRows.length = 0; $("event-log").innerHTML = '<li class="placeholder-row">Cleared.</li>'; };

  const sessionRows = [];   // one per prediction served
  let verbose = false;
  function onMessage(msg) {
    if (!msg || !msg.type) return;
    if (msg.type === "acoustic_event") {
      log("acoustic", (msg.kind || msg.event || "event") + (msg.confidence != null ? " " + Math.round(msg.confidence * 100) + "%" : "") +
        (msg.text ? ' "' + msg.text + '"' : ""), "event");
    } else if (msg.type === "prediction") {
      const top = (msg.candidates || [])[0];
      const word = top && top.word && !top.word.startsWith("(") ? top.word : "";
      log("predict", (word ? '"' + word + '" ' : "no suggestion ") + "via " + msg.trigger + " · " + msg.served + " " + Math.round(msg.latency_ms) + " ms", word ? "ok" : "warn");
      if (word) {
        sessionRows.push({ t: stamp(), fragment: msg.fragment || "", word, trigger: msg.trigger, served: msg.served, latency_ms: Math.round(msg.latency_ms), outcome: "shown" });
        renderSessions();
      }
    } else if (msg.type === "transcript_word") {
      if (verbose) log("word", msg.text);
    }
  }
  function outcome(word, what) {
    for (let i = sessionRows.length - 1; i >= 0; i--) {
      if (sessionRows[i].word === word || what === "rejected") { sessionRows[i].outcome = what; break; }
    }
    renderSessions();
  }
  document.addEventListener("echo:card-accepted", (e) => { log("accept", '"' + e.detail.word + '" (' + e.detail.how + ")", "ok"); outcome(e.detail.word, "accepted · " + e.detail.how); });
  document.addEventListener("echo:card-rejected", (e) => { log("reject", '"' + e.detail.word + '"', "warn"); outcome(e.detail.word, "rejected"); });

  function renderSessions() {
    const tb = $("session-table") && $("session-table").tBodies[0];
    if (tb) {
      tb.innerHTML = sessionRows.length ? sessionRows.slice().reverse().map((r) =>
        `<tr><td class="mono dimtext">${r.t}</td><td class="dimtext">"${esc(r.fragment)}"</td><td class="word">${esc(r.word)}</td>` +
        `<td class="mono">${esc(r.trigger)}</td><td class="mono${r.served === "prefetch" ? " served-fast" : ""}">${esc(r.served)}</td>` +
        `<td class="mono">${r.latency_ms} ms</td><td class="mono">${esc(r.outcome)}</td></tr>`).join("")
        : '<tr class="placeholder-tr"><td colspan="7">No stalls yet.</td></tr>';
    }
    const lat = sessionRows.map((r) => r.latency_ms).sort((a, b) => a - b);
    set("ss-predictions", String(sessionRows.length));
    set("ss-prefetch", String(sessionRows.filter((r) => r.served === "prefetch").length));
    set("ss-taps", String(sessionRows.filter((r) => /^accepted/.test(r.outcome)).length));
    set("ss-rejects", String(sessionRows.filter((r) => r.outcome === "rejected").length));
    set("ss-latency", lat.length ? lat[Math.floor(lat.length / 2)] + " ms" : "--");
    const tally = $("tally-count");
    if (tally) set("ss-recovered", tally.textContent);
  }
  if ($("session-export")) $("session-export").onclick = () => {
    const blob = new Blob([JSON.stringify({ exported_at: new Date().toISOString(), console_uptime: clock(),
      source: activeSource ? { label: activeSource.label, profile: activeSource.profile && activeSource.profile.id, floor_dbfs: activeSource.floor } : null,
      predictions: sessionRows, log: logRows }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "echo-session-" + new Date().toISOString().replace(/[:.]/g, "-") + ".json";
    document.body.appendChild(a); a.click(); a.remove();
    log("session", "exported " + sessionRows.length + " predictions and " + logRows.length + " log rows");
  };
  if ($("session-clear")) $("session-clear").onclick = () => {
    sessionRows.length = 0; renderSessions();
    if ($("history")) $("history").innerHTML = '<li class="placeholder-row">No stalls yet.</li>';
    log("session", "console counters cleared (server context untouched)");
  };

  // ------------------------------------------------------------------
  // runtime sheet + prefs
  // ------------------------------------------------------------------
  function renderRuntime(h, c) {
    const dl = $("runtime-sheet");
    if (!dl) return;
    const rows = [];
    const add = (k, v, cls) => rows.push(`<div><dt>${esc(k)}</dt><dd${cls ? ` class="${cls}"` : ""}>${esc(v == null ? "--" : String(v))}</dd></div>`);
    if (h) {
      add("Server", h.status, h.status === "ok" ? "ok" : "warn");
      add("Predictor", h.active_predictor + " (" + h.provider + ")", h.active_predictor === "MockPredictor" ? "warn" : "");
      add("Model", h.model);
      add("Speculative prefetch", h.prefetch ? "on" : "off");
      add("Acoustic channel", h.acoustic);
      add("Acoustic backend", (h.acoustic_backend || "--") + " on " + (h.acoustic_device || "--"));
      add("ASR", h.asr);
      const src = h.audio_source;
      add("Audio source (server)", !src ? "none reported yet"
        : (src.display_name || src.label || "input") + (src.floor_dbfs != null ? " / floor " + src.floor_dbfs + " dBFS" : ""));
    }
    if (c) {
      serverCfg = c;
      if (typeof c.wearer_gate === "boolean") add("Speaker gate", c.wearer_gate ? "on" : "off", c.wearer_gate ? "ok" : "");
      add("ASR provider", c.asr_provider + (c.asr_model ? " · " + c.asr_model : "") + (c.asr_mode ? " · " + c.asr_mode : ""));
      add("Pause trigger", c.stall_pause_ms + " ms");
      add("Min gap between stalls", c.stall_min_gap_ms + " ms");
    }
    add("Console origin", location.host);
    dl.innerHTML = rows.join("");
  }
  Promise.all([
    fetch("/healthz").then((r) => r.json()).catch(() => null),
    fetch("/api/config").then((r) => r.json()).catch(() => null),
  ]).then(([h, c]) => { renderRuntime(h, c); if (h) log("server", "healthz ok · " + h.model + " · " + h.acoustic, "ok"); });

  const pa = $("pref-autospeak"), as = $("autospeak");
  if (pa && as) {
    pa.onchange = () => { as.checked = pa.checked; log("prefs", "speak top suggestion " + (as.checked ? "on" : "off")); };
    as.addEventListener("change", () => { pa.checked = as.checked; });
  }
  const pr = $("pref-reveal");
  if (pr) pr.oninput = () => {
    document.documentElement.style.setProperty("--reveal-ms", pr.value + "ms");
    set("pref-reveal-val", pr.value + " ms");
  };
  const pv = $("pref-log-verbose");
  if (pv) pv.onchange = () => { verbose = pv.checked; };
  renderProfiles();

  // ------------------------------------------------------------------
  // pages
  // ------------------------------------------------------------------
  const TITLES = { console: "Console", sessions: "Sessions", settings: "Settings" };
  function goto(page) {
    if (!TITLES[page]) page = "console";
    document.body.dataset.page = page;
    document.querySelectorAll(".page").forEach((p) => { p.hidden = p.dataset.pageWhen !== page; });
    document.querySelectorAll(".nav-item").forEach((b) => {
      if (b.dataset.nav === page) b.setAttribute("aria-current", "page"); else b.removeAttribute("aria-current");
    });
    set("page-title", TITLES[page]);
    if (page === "sessions") renderSessions();
    try { history.replaceState(null, "", "#" + page); } catch (_) {}
  }
  document.querySelectorAll(".nav-item").forEach((b) => { b.onclick = (e) => { e.preventDefault(); goto(b.dataset.nav); }; });
  document.addEventListener("keydown", (e) => {
    if (e.altKey && !e.ctrlKey && !e.metaKey && /^[1-3]$/.test(e.key)) { goto(Object.keys(TITLES)[Number(e.key) - 1]); e.preventDefault(); }
  });
  goto((location.hash || "#console").slice(1));

  // ------------------------------------------------------------------
  // link
  // ------------------------------------------------------------------
  function onSocket(state) {
    log("link", state === "open" ? "/ws connected" : "/ws closed, reconnecting", state === "open" ? "ok" : "warn");
    if (state !== "open" && $("server-rtt")) { $("server-rtt").textContent = "-- ms"; $("server-rtt").className = "server-rtt"; }
  }
  function onRtt(ms) {
    const el = $("server-rtt");
    if (!el) return;
    el.textContent = Math.round(ms) + " ms";
    el.className = "server-rtt ok";
  }
  function onMode(m) { log("mode", m === "sim" ? "Simulate (typed input, same pipeline)" : "Live (microphone)"); }
  function onListening(on) { log("mic", on ? "listening" : "stopped", on ? "ok" : ""); if (!on) onSourceStop(); }

  function set(id, text) { const el = $(id); if (el) el.textContent = text; }
  function renderAll() { renderTree(); }

  if ($("server-host")) $("server-host").textContent = location.host || "file";
  setInterval(() => {
    set("session-clock", clock());
    const tc = $("tally-count");
    if (tc) tc.classList.toggle("ok", tc.textContent.trim() !== "0");
  }, 1000);
  renderAll();

  window.EchoConsole = { onSocket, onRtt, onMessage, onSource, onSourceStop, onLevel, onHostInputs,
                         constraintsFor, preferredDeviceId, profileFor, onMode, onListening, log, goto };
})();
