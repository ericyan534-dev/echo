---
name: Echo Console
description: A dark instrument console that makes a two-channel speech mechanism, and the microphone feeding it, legible from four feet away.
colors:
  bg: "#0a0c10"
  bg-bar: "#0c0f14"
  panel: "#12151b"
  well: "#0d0f14"
  line: "#262b35"
  line-soft: "#1c212b"
  line-ctl: "#5f6878"
  ink: "#f2f4f7"
  muted: "#9aa3b3"
  muted-dim: "#838d9d"
  top: "#38d39f"
  accent: "#6c8eff"
  cyan: "#4fd6e8"
  warn: "#f0b429"
  err: "#ef6a6a"
  top-line: "#327a61"
  warn-line: "#7d6528"
  err-line: "#a34747"
typography:
  display:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "clamp(46px, 6.2vw, 84px)"
    fontWeight: 700
    lineHeight: 1.05
    letterSpacing: "-0.025em"
  headline:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "26px"
    fontWeight: 400
    lineHeight: 1.35
    letterSpacing: "normal"
  title:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "18px"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  body:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "16px"
    fontWeight: 400
    lineHeight: 1.5
    letterSpacing: "normal"
  data:
    fontFamily: "ui-monospace, Cascadia Code, SFMono-Regular, Consolas, Liberation Mono, monospace"
    fontSize: "14px"
    fontWeight: 600
    lineHeight: 1
    letterSpacing: "0.01em"
    fontFeature: "tabular-nums"
  control:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "normal"
  headline-compact:
    fontFamily: "-apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif"
    fontSize: "22px"
    fontWeight: 400
    lineHeight: 1.35
    letterSpacing: "normal"
  caption:
    fontFamily: "ui-monospace, Cascadia Code, SFMono-Regular, Consolas, Liberation Mono, monospace"
    fontSize: "13px"
    fontWeight: 500
    lineHeight: 1.4
    letterSpacing: "0.01em"
  label:
    fontFamily: "ui-monospace, Cascadia Code, SFMono-Regular, Consolas, Liberation Mono, monospace"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.2
    letterSpacing: "0.09em"
rounded:
  xs: "2px"
  sm: "3px"
  md: "4px"
  lg: "6px"
spacing:
  s1: "4px"
  s2: "8px"
  s3: "12px"
  s4: "16px"
  s5: "20px"
  s6: "24px"
  s7: "32px"
components:
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.bg}"
    rounded: "{rounded.sm}"
    padding: "10px 16px"
    height: "42px"
    typography: "{typography.body}"
  button-secondary:
    backgroundColor: "#232733"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "10px 16px"
    height: "42px"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "10px 16px"
    height: "42px"
  btn-sm:
    rounded: "{rounded.sm}"
    padding: "5px 10px"
    height: "32px"
    size: "13px"
  btn-sm-table:
    padding: "3px 8px"
    height: "28px"
  button-mic:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.bg}"
    rounded: "{rounded.sm}"
    padding: "12px 20px"
    height: "46px"
    width: "200px"
  button-mic-live:
    backgroundColor: "{colors.err}"
    textColor: "{colors.bg}"
  nav-item:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "7px 10px"
    height: "36px"
    size: "14px"
  nav-item-hover:
    backgroundColor: "#151a22"
    textColor: "{colors.ink}"
  nav-item-current:
    backgroundColor: "#1b2029"
    textColor: "{colors.ink}"
  nav-count:
    textColor: "{colors.muted-dim}"
    typography: "{typography.data}"
    size: "12px"
  nav-key:
    textColor: "{colors.muted-dim}"
    typography: "{typography.data}"
    size: "12px"
  unit-row:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "5px 8px"
    height: "30px"
    typography: "{typography.data}"
    size: "13px"
  unit-row-source:
    textColor: "{colors.ink}"
  tab:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.xs}"
    padding: "8px 14px"
    height: "38px"
  tab-active:
    backgroundColor: "#1b2029"
    textColor: "{colors.ink}"
  chip:
    backgroundColor: "transparent"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "6px 10px"
    typography: "{typography.data}"
  chip-ok:
    textColor: "{colors.top}"
  chip-warn:
    textColor: "{colors.warn}"
  chip-err:
    textColor: "{colors.err}"
  chip-dim:
    textColor: "{colors.muted-dim}"
  chip-tree:
    padding: "4px 8px"
    size: "13px"
  source-badge:
    backgroundColor: "transparent"
    textColor: "{colors.cyan}"
    rounded: "{rounded.sm}"
    padding: "5px 9px"
    typography: "{typography.data}"
    size: "12px"
  input-text:
    backgroundColor: "{colors.well}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "10px 12px"
    typography: "{typography.body}"
  deck:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "12px 16px"
  panel:
    backgroundColor: "{colors.panel}"
    textColor: "{colors.ink}"
    rounded: "{rounded.md}"
    padding: "12px 16px 16px"
  readout:
    backgroundColor: "{colors.well}"
    textColor: "{colors.warn}"
    rounded: "{rounded.sm}"
    padding: "3px 9px"
    typography: "{typography.data}"
  readout-frozen:
    textColor: "{colors.top}"
  readout-quiet:
    textColor: "{colors.muted-dim}"
  sheet-row:
    padding: "7px 0"
  sheet-term:
    textColor: "{colors.muted-dim}"
    typography: "{typography.label}"
  sheet-value:
    textColor: "{colors.ink}"
    typography: "{typography.data}"
  table-header:
    textColor: "{colors.muted-dim}"
    typography: "{typography.label}"
    padding: "0 10px 7px 0"
  table-cell:
    textColor: "{colors.ink}"
    padding: "8px 10px 8px 0"
    size: "14px"
  log:
    backgroundColor: "{colors.well}"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
    typography: "{typography.data}"
    size: "13px"
    height: "300px"
  dominant-word:
    backgroundColor: "{colors.well}"
    textColor: "{colors.ink}"
    rounded: "{rounded.lg}"
    padding: "20px 24px"
    typography: "{typography.display}"
  cand-chip:
    backgroundColor: "#171b22"
    textColor: "{colors.muted}"
    rounded: "{rounded.sm}"
    padding: "9px 16px"
  not-it:
    backgroundColor: "transparent"
    textColor: "{colors.err}"
    rounded: "{rounded.sm}"
    padding: "9px 14px"
    typography: "{typography.data}"
  lane-item:
    backgroundColor: "#1a1f28"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "6px 11px"
  lane-item-filler:
    backgroundColor: "#3d2e10"
    textColor: "{colors.warn}"
  lane-item-prolongation:
    backgroundColor: "#113a33"
    textColor: "{colors.cyan}"
  lane-item-asr-missed:
    backgroundColor: "#241a1c"
    textColor: "#d9bcbc"
---

# Design System: Echo Console

## Overview

**Creative North Star: "The Instrument Console"**

This surface is in Operate mode, not Marketing mode. It is the operator terminal of a small system — one server and the host's audio inputs, with a recognised wireless lav (the DJI Mic 2S) on the speaker — read at 2–4 ft by someone standing beside the machine, in a noisy room, under uncontrolled light, with no second chance and no time to lean in. Nothing here is arranged to persuade; everything is arranged to be *checked*. A judge reads presence first (which microphone is live, in the rail), then the mechanism (the word, the transcript, two sensing channels running side by side), then the ledger and the log in the inspector.

The whole system is built around a single asymmetry: the resting state is deliberately quiet, and emphasis is hoarded for event and state moments. Both timeline lanes share one well, one border, and one resting treatment, so the only thing the eye can read at rest is the label — which means the instant a lane fires, the contrast is unmissable. The same discipline runs through color (mint is not decoration, it is state: the server is up, a card was accepted, a page is current), through motion (one authored moment, everything else silent), and through size (anything carrying a claim is ≥14px, supporting labels are 12–13px, nothing anywhere is below 12px).

Materially the console is flat, dark, tight-cornered and hairline-ruled: recessed wells for anything the machine writes into (inputs, lanes, readouts, the event log), raised panels for anything the operator reads, and no ornament that is not carrying information. Corners are 3–6px, never pills; every label the console writes about itself is set small, uppercase and monospaced, like engraving on a front panel. Blurred shadow appears on exactly two objects — the word that gets photographed and the token that marks the rarest event — and everything else that reads as "lifted" or "selected" is a flat inset rule. The palette is cold near-black with a single mint anchor and three semantic signals; it is not a "dashboard aesthetic" so much as a lab bench that happens to be backlit.

**Key Characteristics:**
- Dark, cold-neutral, hairline-ruled, tight-cornered (3 / 4 / 6px); flat by default with two blurred exceptions
- Rail + main + inspector composition: presence on the left, evidence in the middle, ledger on the right
- Two typographic voices — humanist sans for the speaker's words and the operator's controls, monospace for the machine's measurements and the console's own engraving
- Mint (`#38d39f`) as a fixed anchor used for state, never as decoration
- Every ink token holds ≥4.5:1; every interactive boundary holds ≥3:1
- Silence at rest; emphasis spent entirely on event and state moments
- Dependency-free: no build step, no framework, no external font, no CDN

## Colors

A cold near-black substrate carrying one mint anchor and a three-signal semantic set, all measured rather than eyeballed.

### Primary
- **Echo Mint** (`{colors.top}`, 9.6:1 on panel): The fixed brand anchor. Applied to the accepted state, the frozen latency readout, the online status chip, the healthy server RTT, the current page's inset rule, the dominant word's border, the brand mark, the selected-tab underline, the `ok` log kind, the `ok` sheet value, the `ok` figure inside a quiet readout, the prefetch-served cell, and the checkbox and range `accent-color`. It is a *state* color: mint appearing anywhere means something succeeded, connected, landed, or is the one you are on. The value is exact and must not be shifted.

### Secondary
- **Signal Blue** (`{colors.accent}`, 6.1:1): The action color. Primary buttons and the 2px focus ring. It reads as "you can press this," distinct from mint's "this happened."

### Tertiary
- **Event Cyan** (`{colors.cyan}`): The hardware-and-event signal. The prolongation token in the acoustic lane, the recognised-hardware badge beside the mic picker, the live host-input row's dot and the hollow ring of a detected-but-idle input in the input tree, the profile tag on a host input, and the `event` log kind. It exists so a recognised input and the rarest acoustic event do not have to share mint's vocabulary; cyan says "the machine identified something," mint says "something succeeded." Its stroke partner is the literal `#1f5a52`, used on both the source badge and the prolongation token; it is not a `:root` token.
- **Instrument Amber** (`{colors.warn}`, 9.9:1): Waiting and measuring. The ticking latency readout, the filler token, the top of the level-meter gradient, the `warn` log kind, and a `warn` sheet value (a mock predictor, a degraded server).
- **Reject Red** (`{colors.err}`, 6.1:1): Recording (the live mic button), rejection ("not it"), the `err` log kind, and the strike-through that marks what the transcript got wrong.

### Neutral
- **Console Black** (`{colors.bg}`) / **Bar Black** (`{colors.bg-bar}`): Page ground, and the two sticky chrome surfaces — the navigation rail and the session bar — a half-step apart so chrome separates from content without a heavy rule.
- **Panel Slate** (`{colors.panel}`): Raised reading surfaces — every `.panel` and the deck.
- **Recessed Well** (`{colors.well}`): Anything the machine writes into or the operator types into — inputs, lanes, the timeline, readouts, the event log, the segmented track, the dominant word's plate.
- **Hairline** (`{colors.line}`) / **Hairline Soft** (`{colors.line-soft}`): Decorative rules only — panel edges, the rail and session-bar edges, table header rules, sheet and list row separators. 1.3:1, correct for decoration and wrong for anything clickable.
- **Control Stroke** (`{colors.line-ctl}`, 3.3:1 on panel / 3.4:1 on well): The boundary of an interactive control, and the resting fill of any presence dot. Inputs, selects, textareas, the level meter, candidate chips, ghost and small buttons, the offline input's dot.
- **Bright Ink** (`{colors.ink}`, 16.4:1): The speaker's words, any committed value, the current page, the live input's name, the figure inside a quiet readout.
- **Label Ink** (`{colors.muted}`, 7.3:1): Panel titles, lane names, nav items at rest, offline input names, a detected input's state word, log message text, secondary copy.
- **Quiet Ink** (`{colors.muted-dim}`, 5.5:1): Hints, offline states, table headers, sheet terms, log timestamps and kinds, tree group labels, the brand subtitle, the session clock, keyboard hints, small print. Quiet is not faded — it is still fully legible.
- **Semantic strokes** — `{colors.top-line}` (3.6:1), `{colors.warn-line}` (3.3:1), `{colors.err-line}` (3.1:1): the control-grade partners of mint, amber, and red, used wherever a colored boundary belongs to something interactive or state-bearing.

### Named Rules

**The Two Borders Rule.** There are exactly two border families and they are not interchangeable. `--line` / `--line-soft` are *decorative hairlines*: panel edges, dividers, list and table separators, timeline frame, the log's frame. `--line-ctl` and the semantic set (`--top-line`, `--warn-line`, `--err-line`) are *interactive-control boundaries* and every one of them clears WCAG 1.4.11's 3:1. Test: if a pointer can press it, hover it, focus it, or if it is the sole carrier of a state, its border comes from the control family. A decorative hairline on a control is a contrast failure, not a style choice.

**The Fixed Anchor Rule.** `#38d39f` is the exact Echo mint. Do not shift, tint, or "harmonize" it. Derivative mint surfaces (the accepted gradient, the lane-label tint) may vary; the anchor itself may not.

**The Never Color Alone Rule.** No state is signaled by hue alone. The offline chip says the word "offline" and stays at 5.5:1 rather than fading out. An input row says "offline", "detected" or "live" beside a dot whose fill or ring matches. The selected tab and the current page both carry a mint inset rule in addition to their raised fill, because the fill alone is 1.2:1 against the track. Every colored chip carries a glyph and a text label; every log line carries its kind as a word.

**The Measured Ink Rule.** Every ink token in `:root` carries its measured contrast ratio in a source comment, and every one holds ≥4.5:1 on both `--panel` and `--well`. Adding an ink token means measuring it against both surfaces and recording the number beside it. A token without a measured ratio is not part of this system.

## Typography

**Body Font:** system humanist sans (`-apple-system`, Segoe UI, Roboto, Helvetica, Arial)
**Data Font:** system monospace (`ui-monospace`, Cascadia Code, SF Mono, Consolas, Liberation Mono)

**Character:** Two voices, deliberately unmixed. The sans is neutral and unstyled — it exists to get out of the way of the word a person was trying to say, and it carries the operator's controls: nav items, tabs, buttons, form labels, hints. The monospace is tabular and tracked-out; it reads as an instrument label, and in this build it carries both the machine's measurements (chips, the clock, the RTT, input rows, sheet values, table figures, the log) and the console's own *engraving* — panel titles, sheet terms, table headers, tree group labels, the brand subtitle — set small, uppercase and letterspaced like text silk-screened on a front panel. Nothing here is a "display face": the drama comes from scale, not from personality. No webfont is loaded at all, because the venue may have no usable network.

### Hierarchy
- **Display** (700, `clamp(46px, 6.2vw, 84px)`, 1.05, −0.025em): The dominant suggested word. The single thing on the screen that gets photographed, and the only place the type is allowed to be loud.
- **Headline** (400, 26px, 1.35): The live transcript — the speaker's own words as they arrive. Drops to 22px below 720px. Its placeholder prompt is reset to 16px, because "Press start listening" is not speech and must not occupy the size reserved for it.
- **Title** (700, 18px, −0.01em): The brand name at the top of the rail. Its subtitle beside it is engraving (12px mono, uppercase, 0.1em, Quiet Ink; dropped below 720px). The page title `h1` in the session bar is a small uppercase sans (700, 15px, 0.02em): a page name is a location, not a brand.
- **Body** (400, 16px, 1.5): Base. Buttons run at 15px / 600, nav items and tabs at 14px / 600; hints run at 13px / 1.5 and cap at 78ch (52ch when beside the transport in the deck); the disclaimer is 13px centered and capped at 60ch.
- **Data** (600–700, 14px, tabular-nums, 0.01–0.02em): Status chips, the latency readout, the trigger/served/latency line, acoustic lane tokens, "not it", sheet values, table figures, history metadata. Everything the machine measured. Steps down to 13px for the input row, the session clock, the event log, the presence state cell and the regex column; to 12px for the keyboard hint, the tree summary, the input state figure, the server row, the host-input tag, the source badge and the log's kind column.
- **Label** (600, 12px, 0.09em, uppercase, mono): The engraving voice. Panel titles (`h2`) and tree group labels at 0.09em; table headers at 0.08em; lane names at 13px / 0.06em; sheet terms at 0.06em; the source badge at 0.05em; the brand subtitle at 0.1em. All of it Quiet Ink except panel titles and lane names, which sit one step up in Label Ink because they name what a judge is looking at.

### Named Rules

**The Two Voices Rule.** The humanist sans carries the speaker's own words — the transcript and the dominant suggested word — and the operator's controls: nav items, tabs, buttons, form labels, hints. The monospace carries everything the machine measured or identified — chips, latency, tallies, counts, acoustic tokens, "not it", input names, the clock, the RTT, sheet values, table figures, the log — and every label the console engraves on itself: section titles, sheet terms, table headers, tree groups. This pairing mirrors the product's two-channel thesis and is binding. A measured number set in the body face, or a human utterance set in mono, breaks the thesis the interface exists to demonstrate.

**The Two-Foot Rule.** Sizing is set for a judge reading from 2–4 ft. Anything carrying a claim someone reads from across the table — the word, the transcript, lane tokens, status chips, the trigger/latency line, sheet values, table figures — is ≥14px. Panel titles, lane names, input rows, the log, hints and engraving are supporting labels at 12–13px. Nothing anywhere goes below 12px.

## Layout

The app is a two-column grid, `248px` (`--rail-w`) / `minmax(0, 1fr)`, at `min-height: 100vh`: a **navigation rail** on the left and a **main column** on the right. Every page in the main column is the same shape — a **session bar**, then a page body that is a **deck** (Console only) followed by a **workspace** of **stage + inspector**.

**The rail** is sticky at `top: 0`, `100vh` tall and scrolls internally, on Bar Black with a 1px hairline on its right edge, padded 12px all round, a vertical flex stack gapped at 20px. Top to bottom: the brand (mark, name, subtitle, baseline-aligned, 8px gap), the page list (1px gaps between 36px items), the input tree (12px gaps; a head row and an Inputs group), and a foot row pushed to the bottom with `margin-top: auto`, separated by a soft hairline above 12px of padding, holding the server host and its RTT.

**The session bar** is sticky at `top: 0`, `z-index: 10`, Bar Black with a hairline beneath, padded `10px 20px`, a wrapping flex row: page title and session clock baseline-aligned on the left (12px gap), the four system status chips on the right (8px gaps). It is always visible because "is it actually listening?" is the first question every judge asks.

**The page body** caps at 1520px with `16px 20px 64px` padding; only one page is shown at a time via `data-page-when` and the `hidden` attribute, and every page is padded identically so switching never shifts the chrome.

**The deck** (Console page) is a Panel Slate strip, 4px radius, `12px 16px` padding, a wrapping flex row at 20px gap: the segmented source control pinned at the start, then the transport — the mic button and the level meter with its dBFS figure on one row (`vu-wrap`, 10px gap, `flex: 1 1 200px`), the input picker, source badge and autospeak toggle on the next — and the hint beside them at 52ch while there is room for it (`flex: 1 1 300px`).

**The workspace** is a two-column grid, `minmax(0, 1.75fr)` / `minmax(320px, 0.8fr)`, gapped at 16px with a 16px top margin: the **stage** on the left is evidence at full size (the suggested word; the transcript over the dual-channel timeline; on Sessions and Settings, the tables and the runtime sheet), the **inspector** on the right is the ledger (audio source sheet, history, tally, event log; host inputs; counts; console prefs). Both columns are flex stacks with a 16px gap and `align-items: start`, so a growing stage never stretches the inspector.

The spacing rhythm is a 4px base scale (4 / 8 / 12 / 16 / 20 / 24 / 32). Panel padding is `12px 16px 16px`; the deck `12px 16px`; the session bar `10px 20px`; the rail 12px. Gaps inside dense clusters (tree rows, nav items, chips) use 1–8px; between controls 8–12px; between structural regions 16–20px. Row-based lists (sheets, tables, history, host inputs) use 7–10px of vertical padding per row with a soft hairline between rows and none after the last.

Three breakpoints, all driven by content rather than device classes:

- **≤1180px** — the workspace collapses to a single column (inspector under stage) and the deck hint releases its 52ch cap to 78ch and takes the full row. Above this width the hint sits *beside* the transport, so a wide demo laptop does not render half a metre of empty deck.
- **≤900px** — the rail becomes a **top strip**: the app grid drops to one column, the rail loses its stickiness and right edge and takes a hairline beneath, its contents flow as a wrapping row padded `12px 16px`; nav items shed their full width and the current page's mint rule moves from the left edge to an underline; the tree head and groups are hidden and the two presence chips (13px, `4px 8px`) take over, because presence is the rail's job at any width; the server row is pushed to the far right. Session bar and page padding step down to `12px 16px` (page bottom 56px).
- **≤720px** — the brand subtitle is dropped, the transcript drops to 22px, timeline rows stack label-over-lane, sheets go single-column (term over value, 2px gap), the log drops its kind column and narrows its timestamp column to 58px, and the transport goes vertical with full-width button and meter. That last change exists because the transport's intrinsic minimums (a 200px button plus a 120px meter plus gap) exceed a 390px viewport and would otherwise widen the whole document.

**The Reserved Space Rule.** The suggestion region holds a 187px minimum height (150px below 720px) at rest, so a card landing mid-demo never shoves the timeline down the page. The event log caps at 300px and scrolls inside its well rather than growing the inspector. Any region that fills asynchronously reserves its space in advance.

## Elevation & Depth

The system is essentially flat and works by **tonal layering plus hairlines**, not by shadow. Depth reads as a four-step tonal stack: page ground → sticky chrome (rail, session bar) → raised panel → recessed well, each separated by a 1px rule rather than a blur. The evidence column is pushed one further step forward than the inspector beside it purely by lightening its border (`#2f3542` against the standard hairline) — a border shift, not a shadow. Selection and state are **inset rules**, never lifts: a 2px mint rule on the leading edge of the current nav item and the active tab, a 1px mint ring inside the accepted word plate, a 1.5px cyan ring inside a detected input's dot.

Blurred shadow appears in exactly two places.

### Shadow Vocabulary
- **Card seat** (`0 8px 20px -14px rgba(0,0,0,0.9)`): The dominant word only. A tight drop that seats the plate in its well; the mint border does the work of marking it.
- **Accepted ring** (`inset 0 0 0 1px var(--top), 0 8px 20px -14px rgba(0,0,0,0.9)`): The same plate after accept. A second mint line inside the first, plus the mint-tinted vertical gradient — the only place elevation changes as a response to state.
- **Segment lift** (`0 1px 2px rgba(0,0,0,0.4), inset 0 -2px 0 var(--top)`): The selected tab. The inset mint rule is doing the accessibility work; the drop is only there to seat the segment in its track.
- **Current-page rule** (`inset 2px 0 0 var(--top)`; `inset 0 -2px 0 var(--top)` below 900px): The current nav item. The same device as the tab's rule, with no drop at all — the rail is chrome, not a track.
- **Detected ring** (`inset 0 0 0 1.5px var(--cyan)` on a transparent 8px dot): A recognised host input that is plugged in but not capturing. Hollow because it is not yet doing anything.
- **Event glow** (`0 0 14px rgba(79,214,232,0.22)`): The prolongation token in the acoustic lane. It exists solely while that event is on screen.

### Named Rules

**The Event-Only Glow Rule.** Glow is legal in this system, but only as an event. The cyan prolongation halo and the pulsing red recording ring are earned by something happening; nothing at rest glows. A live input's dot is a plain cyan disc, an offline one a plain control-stroke disc, a resting panel a flat plate. A resting surface that glows has spent emphasis the event moment needed.

## Shapes

A tight three-step radius family. **3px** (`{rounded.sm}`) for anything pressed, listed or tokenized — inputs, buttons, nav items, input rows, chips, readouts, the source badge, candidate chips, "not it", lane tokens, lanes, the event log. **4px** (`{rounded.md}`) for containers — panels, the deck, the timeline, the segmented track. **6px** (`{rounded.lg}`) for the dominant word's plate alone, so the object that gets photographed is visibly softer than everything around it. **2px** (`{rounded.xs}`) for the two things that sit *inside* a track — the tab segment and the level meter's bar. There are no pills: a status chip is a small engraved plate, not a capsule. **Disc** (8px, `border-radius: 50%`) for the presence dot in input rows and table state cells — the one non-rectilinear shape, and it is always paired with a word.

Every surface is a 1px-stroked rectangle; nav items and input rows carry a transparent 1px border at rest so hover and selection never shift layout. There is no clipping, masking or non-rectilinear geometry except two functional cases — the lanes carry a 20px linear-gradient mask on their leading edge so overflowing tokens fade out rather than being guillotined, and hide their scrollbars entirely; long host labels are single-line ellipsized. The timeline well carries a 48px repeating vertical rule at 2.2% white: a graph-paper ground that reads as a time axis, not as texture.

Icons are a **local inline SVG sprite** (`<symbol>` + `<use>`) rendered at one grid (24px viewBox), one stroke weight (1.75), round caps and joins, `fill: none`, `stroke: currentColor`. Default box is 16px; 14px inside chips and small buttons; 13px in tree group labels; 22px for the brand mark.

**The Presentation Attribute Rule.** Only inherited properties (`fill`, `stroke`, `stroke-width`) cross into the shadow tree that `<use>` builds — a descendant selector never can. Any per-shape override therefore lives on the shape itself as a presentation attribute, not in the stylesheet. This is why the mark's filled dot, the presence dot glyph, and the pause, list and server icons' heavier strokes are written as attributes in the sprite.

## Components

### Buttons
- **Shape:** 3px radius, 42px minimum height, icon-and-label flex row with an 8px gap, 15px / 600 sans.
- **Primary:** Signal Blue plate with near-black ink (`{components.button-primary}`) — dark ink on the accent, never white.
- **Secondary / Ghost:** Secondary is a neutral slate plate with bright ink; ghost is transparent with a control-grade stroke and label ink, brightening to bright ink and a Quiet Ink border on hover.
- **Small** (`{components.btn-sm}`): the ledger-scale button — 32px minimum, `5px 10px`, 13px label, 14px icon. Always ghost in this build: "Clear" in the log head, "Export JSON" / "Clear console" in the session head (grouped in a `head-actions` row at 8px gap).
- **Hover / Active / Focus:** `filter: brightness(1.06)` on hover, a 1px `translateY` on press, and a 2px Signal Blue focus ring at 2px offset on every focusable element in the system.
- **Mic button:** the one oversized control — 46px tall, 200px minimum width, 15px label at 0.01em. In its recording state it turns Reject Red with near-black ink (white on red is only 3.0:1, and this is the primary control in the state that matters most: recording, on camera) and pulses an expanding red ring on a 2s loop.

### Chips
- **Status chips** (`{components.chip}`): 3px plate, transparent, hairline border, monospace 14px, `6px 10px`, 14px icon. Four states — ok (mint), warn (amber), err (red), dim (offline). Each swaps both text color and border to the matching control-grade stroke, and each carries a word.
- **Tree chips** (`{components.chip-tree}`): the same chip one notch smaller (13px, `4px 8px`) for the input presence chip. Hidden at desktop width, where the tree's own rows carry presence; shown when the rail collapses to a strip below 900px and the rows are gone.
- **Offline is a state, not an absence:** the dim chip stays at 5.5:1, keeps its glyph at 55% opacity, and says "offline" in words.
- **Source badge** (`{components.source-badge}`): the recognised-hardware badge beside the input picker. A 3px plate, transparent, Event Cyan engraving (12px / 600 mono, uppercase, 0.05em) with a radio glyph, on the `#1f5a52` cyan stroke. Hidden until a profile matches; it never shows a guess.
- **Candidate chips** (`{components.cand-chip}`): 3px plate, 15px semibold sans, control-grade stroke, dark neutral ground, `9px 16px`. Hover promotes both text and border to bright ink.

### Cards / Containers
- **Panels:** 4px radius, Panel Slate ground, decorative hairline border, `12px 16px 16px` padding. Stage panels take the lighter `#2f3542` border. The head is a flex row with the engraved title left and an optional readout or small-button group right, separated by a soft hairline with 8px of padding beneath and 12px of margin below that.
- **Deck** (`{components.deck}`): the transport strip above the workspace — a panel with no head and `12px 16px` padding, holding the segmented control and the mode's controls side by side.
- **Readouts** live in panel heads: monospace, tabular, 14px, well ground, 3px plate, `3px 9px`, semantic stroke. They are measured values and are never louder than the thing they measure. A `--quiet` variant (`{components.readout-quiet}`) drops to Quiet Ink at weight 500 with a decorative border for non-alarming tallies and annotations; a `<b>` inside it lifts the one figure to bright ink at 700 (mint when it carries `ok`), and a `<small>` drops a qualifier to 12px.

### Definition Sheets
The read-only config surface: a `<dl>` where each row is a two-column grid, `120px / minmax(0, 1fr)` (200px in the `--wide` runtime sheet), baseline-aligned at 12px gap, `7px 0` padding, soft hairline beneath every row but the last, no top padding on the first. Terms (`dt`) are engraving — 12px mono, uppercase, 0.06em, Quiet Ink; values (`dd`) are 14px bright monospace with tabular figures and `overflow-wrap: anywhere`, because a constraint string or a model name can be long. A value may carry `ok` (mint) or `warn` (amber) when the figure itself is a verdict — a healthy server, a mock predictor. Below 720px every row stacks term-over-value at a 2px gap. Sheets show what the server and the stream report; they never hold an editable control.

### Tables
Flat, ruled, and left-aligned throughout; wrapped in an `overflow-x: auto` container so a seven-column session table scrolls rather than widening the page. Headers are engraving — 12px mono, uppercase, 0.08em, Quiet Ink, `0 10px 7px 0`, no-wrap — over a standard hairline; cells are 14px with `8px 10px 8px 0` padding, tabular figures, a soft hairline beneath each row and none after the last. Column voices are assigned per cell, not per table: `mono` for anything measured (time, trigger, served, latency, outcome, kind), `word` (700, 15px, body face) for the recovered word or the profile name, `dimtext` (Quiet Ink) for the fragment, the last-seen figure and a match pattern, `nowrap` for a name that must not break, `regex` (13px, ≤280px, wraps anywhere) for a match pattern. A **presence state** cell is a 13px monospace inline-flex of an 8px dot plus the word "online" (mint dot, mint text) or "offline" (control-stroke dot, Quiet Ink text). Placeholder rows (`placeholder-tr`) span all columns in Quiet Ink.

### Event Log
The machine's own diary. An `<ol>` in a recessed well (3px radius, hairline border, `8px 12px`), 13px monospace at 1.5, capped at 300px and scrolling with a thin scrollbar. Each line is a three-column grid — `62px / 74px / 1fr` at 10px gap, `2px 0` — timestamp (Quiet Ink, tabular `mm:ss.t`), kind (Quiet Ink, uppercase, 12px, `0.04em`), message (Label Ink, wraps anywhere). The kind column takes the semantic color when a line is a verdict: `ok` mint, `warn` amber, `err` red, `event` cyan. `aria-live` is off — the log rewrites every few seconds and would otherwise talk over the card. Below 720px it drops the kind column (`58px / 1fr`) rather than shrinking the message. The log holds 120 lines and then forgets the oldest; nothing in it is ever fabricated.

### Inputs / Fields
- **Style:** recessed well ground, control-grade 1px stroke, 3px radius, `10px 12px` padding, 15px body face, full width (selects size to content at `9px 10px`, capped at 320px). Placeholders use Quiet Ink at full opacity.
- **Focus:** the global 2px Signal Blue ring; inputs get no separate treatment.
- **Checkboxes and ranges:** 17px checkboxes and full-width ranges both take a mint `accent-color`; their labels are 40px tall so the whole label is a comfortable target. A range sits in a `range-row` beside a quiet readout that echoes its value (12px gap). `color-scheme: dark` is set globally so native controls, selects and scrollbars render dark instead of punching a white hole through the console.

### Navigation
Two navigation controls, one language: a raised fill plus a mint inset rule, never a fill alone.

- **The page list** (`{components.nav-item}`): three full-width links in the rail — Console, Sessions, Settings — each a 36px flex row at `7px 10px`, 3px radius, 14px / 600 sans, 10px gap after a 16px glyph, Label Ink on a transparent plate with a transparent 1px border and no underline. Hover lifts to bright ink on `#151a22`. The current page (`aria-current="page"`) takes bright ink on the `#1b2029` raised fill plus a 2px mint rule inset on its left edge. Every item ends in a **keyboard hint** (`{components.nav-key}`, a `<kbd>` reading "alt 1" … "alt 3" in 12px mono Quiet Ink at 0.04em and 85% opacity, pushed right with `margin-left: auto`). Alt+1…3 switch pages; the hash mirrors the page. Below 900px the list runs as a wrapping row and the mint rule moves to an underline.
- **The segmented source control** (`{components.tab}`): the Live / Simulate switch at the start of the deck — a 4px-radius well track with 2px padding holding two 38px tab buttons at 2px radius, `8px 14px`, 14px / 600 sans. Inactive tabs are transparent with label ink; hover lifts to bright ink; the active tab takes the raised fill *plus* an inset mint rule. Mode switching is driven by `body[data-mode]` — both branches stay in the DOM and the inactive one is hidden by attribute selector. Below 720px the tabs stretch to full width and center their labels.

### Input Tree (signature)
The host's audio inputs, read at a glance, in the rail. A head row (engraved "Inputs" at 12px left, "n detected" in 12px monospace Quiet Ink right) over a soft hairline; then the Inputs group, led by engraving ("INPUTS": 12px mono, 0.09em, Quiet Ink) with a 13px glyph.

Each **input row** (`{components.unit-row}`) is a three-column grid — `8px / minmax(0, 1fr) / auto` at 9px gap — 30px tall at `5px 8px`, 3px radius, 13px monospace: a presence dot, an ellipsized input name, and a right-aligned 12px tabular state figure. Three states, every one worded:
- **offline** (a recognised input, such as the DJI Mic 2S receiver, that is not plugged in): control-stroke disc, Label Ink name, Quiet Ink "offline".
- **detected** (`ready`): a recognised host input that is plugged in but not the active source — a hollow cyan ring (1.5px inset on a transparent dot), Label Ink name and "detected".
- **live** (`source`): the host input actually feeding `/ws/audio` — bright name, cyan disc, cyan "live".

Offline inputs are never faded or hidden; the recognised profile is the console's expectation and an absent receiver is information. The order within Inputs is live, then detected, then the rest.

### Dual-Channel Timeline (signature)
The thesis, made watchable. Two rows — transcript and acoustic — inside one graph-ruled well (4px radius, 10px padding). Each row is a right-aligned 116px monospace uppercase label plus a horizontally scrolling lane (48px minimum height, 3px radius, hidden scrollbar, leading fade mask).

Rest state is deliberately undifferentiated: **both lanes share one well, one border, and one resting treatment.** The only resting hint of lane identity is a hairline mint tint on the acoustic label (`#8fb3a4`). Emphasis is spent entirely on event moments, so the fluent-vs-stall contrast is what the eye actually reads. When both lanes are idle they breathe on a 4.5s border-color loop that starts and ends on the lane's own resting border, so toggling the class never visibly snaps.

Tokens (`{components.lane-item}`) pop in at 0.28s on a decelerating ease. Transcript tokens are plain neutral; acoustic tokens switch to the monospace voice at 700 weight. Filler tokens go amber-on-brown; prolongation tokens go cyan-on-deep-green with the event glow.

**The One Authored Moment Rule.** The system has exactly one authored motion sequence: a transcript token is struck through in Reject Red and labelled "ASR heard", and the acoustic token that caught what the transcript erased lands `--reveal-ms` (500ms by default; the Settings page's reveal-window range rewrites the custom property live) later. Everything else — the recording pulse, the idle breathe, the latency tick, the token pop — is quiet, small, and looping. Do not author a second moment; it competes with the only one that carries the argument.

The strike-through's label is an `::before` set to `display: inline-block` rather than plain inline. Text decoration propagates into inline children and cannot be switched off there, but it never crosses into an atomic inline-level box. Without this the "ASR heard" label is struck through along with the word, which is exactly backwards: the label is the explanation, the word is the thing that was wrong.

**Reduced motion:** the recording pulse, the idle breathe, the latency tick and every transition (cards, meter, tabs, nav items, buttons) are switched off under `prefers-reduced-motion: reduce`. The lane token's animation is **overridden, not disabled** — the `pop` keyframe is redefined to opacity-only, so the scale (decorative, fires on every committed word) is dropped while the `--reveal-ms` delay on `.reveal-delayed` survives. That delay is the sequencing of the authored moment; it is information, not decoration.

### Word Card (signature)
Built entirely at runtime by `renderCardDom()` in `app.js`; it does not exist in the static HTML. A vertical stack, 12px gap, full width:

1. **The dominant word** — a full-width 6px-radius button on a recessed well plate with a mint 1px border and the card-seat drop. Inside, a baseline-aligned row: the word at display scale, and a monospace confidence percentage at 16px in label ink, 20px to its right. Accepting swaps the plate for a mint-tinted vertical gradient (`#142a21` → `#0e1a15`), adds a second 1px mint line inside the border, replaces the percentage with a check glyph and the word "accepted" in mint at 700, and keeps everything interactive.
2. **The card row** — up to two alternative candidate chips, with **"not it"** pushed to the far right by `margin-left: auto` (it drops back to the left edge below 720px). "not it" is a transparent 3px plate with a red control-grade stroke and red 14px monospace label at `9px 14px`; it darkens to a deep red plate on hover.

Dismissal is a 0.5s opacity fade, never a collapse.

**The Latency Stopwatch.** The readout in the suggestion panel head ticks amber (opacity 1 → 0.75 on a 1.1s loop) while the speaker waits, then freezes mint with a mint border and full opacity when the card lands. The numbers are real instrumentation — wall-clock from acoustic detection reaching the UI to the card rendering. Never render a canned value into an instrument readout.

## Do's and Don'ts

### Do:
- **Do** pick the border family by function: decorative hairline (`--line`, `--line-soft`) for edges and dividers, control stroke (`--line-ctl`, `--top-line`, `--warn-line`, `--err-line`) for anything pressable or state-bearing. Every control-family value clears 3:1.
- **Do** set the speaker's words and the operator's controls in the body face, and the machine's measurements and the console's own engraving — chips, figures, input rows, sheets, tables, the log, section titles, headers — in the mono face. The split is the thesis.
- **Do** set engraving small, uppercase and letterspaced (12px, 0.05–0.1em) in Quiet Ink; lift it to Label Ink only for panel titles and lane names.
- **Do** record a measured contrast ratio in a comment next to every new ink token, against both `--panel` and `--well`.
- **Do** keep everything carrying a claim at ≥14px and nothing anywhere below 12px.
- **Do** give every state a glyph or a word alongside its color — the offline chip says "offline"; the input row says "offline", "detected" or "live"; the selected tab and the current page carry a mint rule as well as a fill; every log line names its kind.
- **Do** mark selection and state with an inset rule on a raised fill or inside a border, on whichever edge the layout makes the "leading" one (left in the rail, bottom in a track or a strip, all four sides on the accepted plate).
- **Do** keep corners tight: 3px for anything pressed or tokenized, 4px for containers, 6px for the word plate, 2px inside a track. No pills.
- **Do** reserve space for asynchronous regions (the 187px suggestion well, the 300px-capped log) so nothing on screen jumps during a demo.
- **Do** put per-shape SVG overrides on the shape as presentation attributes; document CSS cannot reach inside a `<use>` shadow tree.
- **Do** use dark ink on the accent and error plates. White on `--err` is 3.0:1 and fails at the moment it matters most.
- **Do** animate transform rather than layout properties on anything that repaints continuously — the level meter uses `scaleX`, not `width`, for the whole session.
- **Do** keep every page the same shape — session bar, optional deck, stage + inspector workspace — so switching pages moves content, never chrome.
- **Do** treat element IDs, class names, `data-*` hooks and the `echo:card-rendered` / `-accepted` / `-rejected` CustomEvents as load-bearing API. `app.js` and `console.js` bind to them, so renaming one is a behavioral change, not a cosmetic one — **and nothing will catch it for you**: `tests/` is entirely Python backend and `scripts/e2e_live.py` drives the WebSocket protocol, not the DOM. There is no automated test that opens this page. Verify ID changes by loading the app.
- **Do** keep the frontend dependency-free — no build step, no framework, no external font, no CDN. The venue may have no usable network, and a system-font stack that always resolves beats a webfont that sometimes does.

### Don't:
- **Don't** shift `#38d39f`. Derivative mint surfaces may vary; the anchor may not.
- **Don't** put a decorative hairline on an interactive control, or a control stroke on a plain divider. This is the single easiest thing in the system to get wrong on a future edit.
- **Don't** differentiate the two timeline lanes at rest beyond the label and its hairline tint. Their sameness at rest is what makes an event legible.
- **Don't** author a second motion moment. The struck ASR token plus the `--reveal-ms` acoustic reveal is the only sequence that carries an argument; everything else stays quiet.
- **Don't** disable the lane token animation under reduced motion. Override the `pop` keyframe to opacity-only instead — the `--reveal-ms` delay is information and must survive.
- **Don't** set the placeholder prompt at transcript scale. The size reserved for the speaker's own words belongs to speech, not to instructions.
- **Don't** fade, hide, or drop an offline or degraded state. An offline input stays in the tree in words at 5.5:1; every fallback tier must look designed, not broken.
- **Don't** glow at rest. A presence dot is a flat disc or a hollow ring; halos belong to the prolongation token and the recording button only.
- **Don't** use cyan for success or mint for identification. Cyan is "the machine recognised something" (a hardware profile, a prolongation); mint is "something succeeded or is on."
- **Don't** put an editable control inside a definition sheet, or a fabricated row into a table or the log. Sheets, tables and the log show what the server and the stream reported.
- **Don't** render a canned or estimated number into a readout, a sheet value, a table cell or the meta line. Those slots are instruments.
