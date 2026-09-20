# Echo deployment — offline stack + the GX10 private appliance

Echo can run its whole stall-to-word loop with **no network**: a local LLM
predicts the word and (optionally) a local STT reads the audio, so no audio,
transcript, or predicted word ever leaves the machine. For a communication aid
this is the product claim, not a footnote — an aphasic conversation is clinical
data, and every competing "AI listens and suggests" tool ships that audio to a
third party. Echo does the detection, prediction, and transcription locally.

Two deployment targets:

- **Laptop** (RTX 4090 Laptop GPU, 16 GB VRAM) — dev + a real offline demo, but
  only one heavy model on the GPU at a time. All numbers below were measured on
  it (2026-09-19/20), reproducible with `python -m scripts.setup_local_llm`.
- **ASUS Ascent GX10** (128 GB unified) — the shipped appliance: local LLM + SSL
  acoustic + CrisperWhisper all resident at once, air-gapped.

## What "offline" means

| Component | Offline provider | Cloud it replaces |
|---|---|---|
| Word prediction | `PREDICTOR_PROVIDER=local` → `LocalPredictor` → local `llama-server` (Qwen3.8-27B Q4) on `127.0.0.1:8080` | `gemini` / `claude` |
| Speech-to-text | `ASR_PROVIDER=crisper` → CrisperWhisper (Whisper large-v3 turbo) on GPU | browser Web Speech / cloud STT |
| Acoustic channel | already local (FillerNet / StutterNet, CPU or CUDA) | — always local |

## Laptop bring-up

```
python -m scripts.setup_local_llm              # bring-up + honest bench, then stops
python -m scripts.setup_local_llm --keep-alive # same, but leave the server up for the app
```

The script fetches/verifies the prebuilt CUDA `llama-server`, gates on model
architecture support, confirms the GGUF is present, and launches with the
measured-best partial offload. The command it runs (also runnable by hand):

```
models/llama.cpp/llama-server.exe -m models/gguf/Qwen3.8-27B-Q4_K_M.gguf \
  --n-gpu-layers 56 --ctx-size 4096 --host 127.0.0.1 --port 8080 --jinja --no-webui
```

Wait for `{"status":"ok"}` from `http://127.0.0.1:8080/health`, then start Echo
with no cloud key:

```
PREDICTOR_PROVIDER=local python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

`get_settings()` reads `LOCAL_LLM_URL` (default `http://127.0.0.1:8080`),
`LOCAL_LLM_MODEL` (`Qwen3.8-27B-Q4_K_M`) and `LOCAL_LLM_CTX` (4096); the
defaults already match the server. Confirm with `curl .../healthz` →
`"provider":"local"`, `"active_predictor":"LocalPredictor"`, `"asr":"browser"`.
`LocalPredictor` needs no key or reachable server to construct (it degrades at
call time), so it never silently falls back to `MockPredictor`.

## Which model runs where

| | Laptop (16 GB VRAM) | GX10 (128 GB unified) |
|---|---|---|
| LLM predictor | Qwen3.8-27B Q4_K_M, **partial** offload (56/65 blocks), ~15.5 GB VRAM | same or larger, **full** offload |
| Acoustic detector | FillerNet / StutterNet CNN | **SSL StutterNet on GPU** (`STUTTER_BACKEND=ssl ACOUSTIC_DEVICE=cuda`) |
| STT | browser STT (CrisperWhisper needs the GPU too, ~1.6 GB + workspace) | **CrisperWhisper**, local |
| Both heavy models at once? | **No** — the LLM alone uses ~95% of VRAM | **Yes** — 128 GB fits LLM + STT + acoustic |
| Network | cloud reachable | **air-gapped** — console (and the DJI receiver's host) on the local router only |

There is one GGUF on disk — `Qwen3.8-27B-Q4_K_M.gguf` (17.11 GB), larger than
16 GB VRAM, so **partial offload is the expected laptop config, not a failure**.
For more VRAM margin the honest options are a lower `--n-gpu-layers` (slower) or
a smaller model (not on this machine).

## Measured numbers (laptop, RTX 4090 Laptop GPU, 16 GB)

**LLM server bring-up:** model `Qwen3.8-27B-Q4_K_M.gguf` (17.11 GB, Q4_K_M);
binary `llama-server.exe` build 10456 / f275595dd, CUDA 12.4; offload 56/65
(partial); **load ~21 s** (READY 20.6 s); **VRAM 15461 / 16376 MiB** (~915 MiB
headroom); **~12.3 tok/s** median, TTFT ~0.82 s; per-generation (64 tok) median
4.62 s.

`--n-gpu-layers 56` is measured, not guessed: the fastest rung that still leaves
~1 GB headroom. Pushing to 60+ can *look* like it loads but the driver silently
spills weights over PCIe and runs ~7× slower while reporting the GPU idle (see
`DEFAULT_NGL` in `scripts/setup_local_llm.py`). Re-tune with `--tune`.

**Offline prediction accuracy** (frozen 60-item set through
`PREDICTOR_PROVIDER=local`, scored by `eval/run_prediction_eval.py`, written to
`eval/results/prediction_eval_local.json`):

| Category | top-1 | top-3 |
|---|---|---|
| concrete (objects) | 20/20 (100%) | 20/20 (100%) |
| proper_context (names from context) | 19/20 (95%) | 19/20 (95%) |
| abstract_verb | 18/20 (90%) | 19/20 (95%) |
| **overall** | **57/60 (95.0%)** | **58/60 (96.7%)** |

Per-item latency, same run: p50 **4726 ms** / p95 **5215 ms** / min 2283 / max
5433 / mean 4099. The 27B Q4 local model is **as accurate as the cloud path on
this set** but not fast (cloud ~1.4–2.1 s, local ~4.7 s median). **Offline
trades latency for privacy, not accuracy.**

**End-to-end proof:** `uvicorn` on port 8011, `PREDICTOR_PROVIDER=local`, every
cloud key blanked. `/healthz` → local / LocalPredictor / browser, no cloud key.
One stall over the real `/ws` protocol returned **`toaster` (1.0), `oven`
(0.85)**, `served:"prefetch"`, llm 0.2 ms / round-trip 2 ms; the only traffic
was loopback to `127.0.0.1:8080`.

## The latency gotcha (offline live demo)

The pipeline's **cold-stall serve budget is 4.0 s** (`_PREDICT_TIMEOUT_S` in
`backend/pipeline.py`); the local model's median generation is **~4.7 s**. So a
**cold** stall (no fluent run-up) exceeds the budget and serves an empty result
— measured: a cold `/ws` stall returned `served:"live"` with zero candidates at
~4.0 s. **Speculative prefetch** (on by default) saves the offline path: during
fluent speech Echo shadow-predicts ahead, so the word is already cached and
served in **~0–2 ms** (`served:"prefetch"`), exactly as the E2E proof shows.

- **Offline live demo REQUIRES prefetch on** — it hides the 4.7 s generation
  behind fluent speech. Lead each stall with a few fluent words.
- A dead-cold stall (first words of a turn, a terse utterance) can still come
  back empty offline. On the cloud path the 4.7 s is 1.7 s and this rarely bites.
- The GX10, with full offload and more throughput, shrinks the gap.

## Offline STT (CrisperWhisper) — status and VRAM

**Loads fully offline: yes**, two ways: the HF cache holds
`nyralabs/CrisperWhisper2.0_turbo` (+large/medium/small), so `ASR_MODEL=turbo`
resolves with `HF_HUB_OFFLINE=1`; and a full snapshot sits in
`models/crisperwhisper/` (Whisper large-v3 turbo: d_model 1280, 32 encoder / 4
decoder layers, 1.62 GB fp16 `model.safetensors`). Verified offline
(`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`, from the local dir): loaded in ~32 s
and transcribed a 1 s clip. That test ran on CPU on purpose, and CPU decode is
far too slow for live use (~27 s for a 1 s clip). **CrisperWhisper needs CUDA to
run in real time**; `ASR_PROVIDER=crisper ASR_DEVICE=cuda` is the live config.

VRAM: turbo weights ~1.6 GB fp16; with the CUDA context, mel workspace, and
decode buffers, realistically ~**2–3 GB** (estimate — not measured on GPU here,
because the LLM held the card). With `llama-server` already at ~15.5 GB, that
leaves ~0.5 GB — **not enough to also load CrisperWhisper on the GPU**; both
would OOM. Hence the rule: **on the 16 GB laptop, run one heavy model on the GPU
at a time.** The GX10's 128 GB runs both.

Practical laptop offline configs:

1. **Local LLM + browser STT** — the proven offline demo path above. (Browser
   Web Speech is not itself local, so for a *strictly* no-network demo use
   option 2 or the GX10.)
2. **Local STT + cloud LLM** (`ASR_PROVIDER=crisper ASR_DEVICE=cuda`) — keeps
   the raw audio local (the privacy-sensitive part) but is not fully offline.
3. **Both local — only on the GX10.**

**Bottom line:** offline mode works and is accurate (95% top-1), and it is the
privacy story that differentiates Echo. Its honest cost on a 16 GB laptop is
latency (~4.7 s vs ~1.7 s cloud), which prefetch hides for any stall after
fluent speech, and a GPU that fits only one heavy model at a time.

---

## The GX10 appliance

**The decision: Echo runs entirely on one ASUS Ascent GX10 — no cloud, no
internet, nothing leaves the box.** The GX10 pairs an NVIDIA GB10
Grace-Blackwell superchip with **128 GB unified memory** and a Blackwell GPU
(5th-gen Tensor cores) on an Arm (aarch64) host running NVIDIA DGX OS. Three
reasons it is the right home:

- **Privacy is the feature.** Detection, prediction, and STT run locally, so a
  clinic's conversations stay in the room — a healthcare claim a cloud product
  cannot copy.
- **128 GB runs the whole stack at once** (LLM + SSL acoustic + CrisperWhisper),
  which the 16 GB laptop cannot.
- **Blackwell makes the best acoustic model the default.** The WavLM-based SSL
  StutterNet (best Block recall) is off by default on the laptop because it is
  too slow for the 125 ms hop on CPU (`backend/config.py` refuses
  `STUTTER_BACKEND=ssl` on CPU; on GPU it runs in single-digit ms/window,
  `V6_RESEARCH.md`). On the GX10 it is simply the default detector.

The GX10 *is* the Echo server; the operator's browser (with the DJI Mic 2S
receiver plugged into that machine) connects over the local router. The same
code that runs against the cloud on the laptop runs against the
local models — the predictor and STT are swapped by two environment variables.

```
        DJI Mic 2S (on the speaker)
          └─► browser AudioWorklet ─►  ASUS Ascent GX10  ──►  operator console (browser):
                                       (GB10, 128 GB)         word cards + TTS
                                       llama-server (local LLM)
                               SSL StutterNet  +  CrisperWhisper
        local router only — no internet path
```

### Bring-up on the GX10 (aarch64 + Blackwell)

The GX10 is Arm64, not x86 — use the Arm CUDA builds:

1. **Base:** NVIDIA DGX OS (Ubuntu aarch64); confirm driver + CUDA with `nvidia-smi`.
2. **Python deps:** `pip install -r requirements.txt` using the aarch64 CUDA
   wheels for `torch`/`torchaudio` from NVIDIA's index (Windows wheels do not apply).
3. **Local LLM:** build `llama.cpp` for Arm+CUDA (`cmake -DGGML_CUDA=on`) and run
   `llama-server`; with 128 GB you can load a much larger model than the laptop's
   Qwen 27B. `scripts/setup_local_llm.py` documents the stack.
4. **STT:** CrisperWhisper via `faster-whisper` / CTranslate2 (aarch64), model in
   `models/crisperwhisper/`.
5. **Run it offline:**
   ```
   STUTTER_BACKEND=ssl ACOUSTIC_DEVICE=cuda PREDICTOR_PROVIDER=local \
   ASR_PROVIDER=crisper uvicorn backend.app:app --host 0.0.0.0 --port 8000
   ```
   No API key, no internet; the console connects over the router.

**Push-button:** `bash scripts/gx10_bringup.sh` runs all five steps, verifying
aarch64 + CUDA + torch + a CUDA `llama-server` + a GGUF (stopping with the exact
fix if any is missing), then launching llama-server (all layers on GPU) and Echo.
`--check` verifies only; `--stop` tears it down. Written against DGX OS docs; it
needs a first run on the box to confirm the torch and llama.cpp build steps.

> Spec figures for the GX10 (GB10 superchip, 128 GB unified memory, Blackwell
> GPU, aarch64 DGX OS) are per NVIDIA/ASUS published specifications; per-model
> latencies are measured on this project's own hardware (see above and
> `V6_RESEARCH.md`).

### Bring-up status (in progress)

The GX10 (`gx10-eaaa`, NVIDIA GB10, aarch64, Ubuntu kernel 6.17-nvidia, 121 GB
RAM, CUDA 13.0 toolkit at `/usr/local/cuda-13.0`, cmake 3.28, gcc 13.3, internet
reachable, user `asus`) was reached over SSH and set up this far:

- **Done:** repo copied to `~/echo` (via `git archive` over SFTP — there is no
  git remote; re-sync the same way after local commits); `python3 -m venv
  ~/echo/.venv` created.
- **In flight when it disconnected:** `pip install torch torchaudio --index-url
  https://download.pytorch.org/whl/cu128` — the aarch64 CUDA wheel is a long
  download that had not finished. Verify with `python -c "import torch;
  print(torch.cuda.is_available())"`; if it is False on the GB10 (Blackwell),
  fall back to the CUDA-13 wheel index or the NGC PyTorch container.

Remaining ordered steps (all scripted in `scripts/gx10_bringup.sh`, which checks
each prerequisite and stops with the fix):

1. Finish `torch`/`torchaudio` (CUDA), then `pip install -r requirements.txt`.
2. Copy the gitignored SSL weight `models/stutternet_ssl_v2.pt` (~118 MB) over
   SFTP from the laptop (not in `git archive`). `models/fillernet.pt` came with
   the repo.
3. Local LLM: download a GGUF on the box (`scripts/setup_local_llm.py`) — with
   128 GB you can run a much larger model than Qwen 27B — or SFTP the laptop's
   `models/gguf/*.gguf`; build `llama-server` for CUDA (`cmake -S <llama.cpp> -B
   build -DGGML_CUDA=on && cmake --build build -t llama-server`; CUDA 13 toolkit
   is present, put `/usr/local/cuda-13.0/bin` on PATH).
4. CrisperWhisper STT: `pip`-install faster-whisper / CTranslate2 (aarch64);
   model downloads on first use, or SFTP `models/crisperwhisper/`.
5. Launch `bash scripts/gx10_bringup.sh`. Confirm `http://<gx10-ip>:8000/healthz`
   shows `"provider":"local"`, `"acoustic_backend":"ssl"`,
   `"acoustic_device":"cuda"`, and a CrisperWhisper `asr`. Open the console from
   a laptop on the same router.
6. Measure on-box latency (local-LLM p50/p95, SSL window time) and record it here
   beside the laptop figures.

The prize: the full stack — local LLM + SSL acoustic + CrisperWhisper — resident
at once on 128 GB, running the whole aphasia loop with nothing leaving the box.
