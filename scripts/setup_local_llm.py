"""Bring up a local llama.cpp server for the Echo `local` predictor provider.

Reproduces, idempotently, the whole Phase-4 environment: fetch a prebuilt
llama.cpp Windows CUDA release, verify the target GGUF's architecture is
actually supported BEFORE spending 17 GB of bandwidth on it, fetch the weights,
launch `llama-server` with partial GPU offload, and measure latency honestly
against the cloud predictor baseline.

    python -m scripts.setup_local_llm                 # full bring-up + bench
    python -m scripts.setup_local_llm --check         # verify only, no download
    python -m scripts.setup_local_llm --bench-only    # server already running
    python -m scripts.setup_local_llm --tune          # search for the best -ngl
    python -m scripts.setup_local_llm --ngl 36        # pin the offload depth

WHY a prebuilt binary instead of `llama-cpp-python`: compiling CUDA wheels on
Windows is slow and failure-prone, and Echo only needs HTTP inference -- the
`LocalPredictor` speaks to `/v1/chat/completions` over aiohttp, so an in-process
binding buys nothing and costs a toolchain.

WHY partial offload is the expected configuration, not a failure: the target is
Qwen3.8-27B at Q4_K_M, 17.1 GB of weights against 16 GB of VRAM on an RTX 4090
Laptop. No Q4 variant of a 27B fits fully. The tunable is how many of the 65
blocks live on the GPU; the rest stream from the 64 GB of system RAM.

Console output is deliberately ASCII-only (Windows GBK console).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

MODELS = REPO / "models"
LLAMA_DIR = MODELS / "llama.cpp"
GGUF_DIR = MODELS / "gguf"
DL_DIR = MODELS / "dl"

# --- target model -----------------------------------------------------------
HF_REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF_NAME = "Qwen3.8-27B-Q4_K_M.gguf"
GGUF_URL = "https://huggingface.co/%s/resolve/main/%s" % (HF_REPO, GGUF_NAME)
GGUF_BYTES = 17_106_775_008  # verified against the HF API, 2026-08-17

# The GGUF `general.architecture` string. NOTE: this is NOT the HF `model_type`.
# Hugging Face calls the architecture `qwen3_5`; the GGUF writer and llama.cpp
# both spell it `qwen35`. Checking for the wrong one makes a supported model
# look unsupported, which is exactly the mistake the A2 gate exists to prevent.
GGUF_ARCH = "qwen35"
# llama.cpp compiles its arch registry in as ASCII literals, so the shipped DLL
# is searched for this symbol -- empirical evidence about the binary we will
# actually run, rather than a guess from upstream source at some other commit.
ARCH_SYMBOL = "llama_model_qwen35"

# --- llama.cpp release ------------------------------------------------------
# Pinned so a rerun reproduces the measured numbers. `--release latest`
# re-resolves against the GitHub API.
LLAMA_RELEASE = "b10456"
GH_LATEST = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
GH_DL = "https://github.com/ggml-org/llama.cpp/releases/download"

# CUDA 12.4 rather than 13.x on purpose: the 13.x builds need a much newer
# driver than the r566 (CUDA 12.7) class driver this machine runs, and CUDA
# 12.x minor-version compatibility makes the 12.4 build safe on any 12.x driver.
CUDA_TAG = "cuda-12.4"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
BASE_URL = "http://%s:%d" % (SERVER_HOST, SERVER_PORT)
CTX_SIZE = 4096
# MEASURED optimum on a 16 GB RTX 4090 Laptop, not a guess. All 65 blocks DO
# load (-ngl 65, 15957 MiB), but "highest that loads" is NOT fastest: past 60
# the allocation oversubscribes VRAM and the driver spills to system memory,
# which halves decode throughput. Measured sweep, median tok/s over 3 runs:
#   -ngl   52     56     58     60     62     64     65    auto-fit
#   tok/s  10.2   12.4   13.9   15.7    9.1    8.9    6.8   11.3
#   VRAM  13997  14955  15406  15868  15904  15894  15957  14484  MiB
# 60 is the peak IN THAT SWEEP; 62 is where the cliff starts.
#
# BUT 60 IS NOT A SAFE DEFAULT, and shipping it was a mistake caught later the
# same day. That sweep ran with a near-idle desktop holding ~485 MiB of VRAM.
# A normal working desktop (browser, editor, chat apps) holds ~919 MiB, and the
# extra ~430 MiB pushes -ngl 60 to 15956 / 16376 MiB -- over the edge. Measured
# consequence, same machine, same model, same prompt:
#
#   -ngl 60, desktop 919 MiB:  1.7 tok/s   pstate P8   210 MHz    8 W
#   -ngl 56, desktop 919 MiB: 11.5 tok/s   pstate P0  2340 MHz   96 W
#
# A 6.8x collapse. The GPU reports itself IDLE while generating, because the
# driver is spilling weights over PCIe and the SMs sit waiting. Nothing warns
# you: the model loads, answers correctly, and is simply 7x slower.
#
# So the default leaves ~1 GB of headroom for whatever else wants the GPU. The
# peak is only worth chasing on a machine whose VRAM use you control, and
# --tune re-measures rather than trusting either number.
DEFAULT_NGL = 56
# Descending: the first rung that loads AND generates is taken. Starts at 58
# rather than 60+ because the rungs above load fine yet run slower, so probing
# them first would "succeed" straight into the slow configuration -- and because
# anything at or above 60 is one browser tab away from the cliff above.
TUNE_LADDER = [58, 56, 54, 52, 48, 44, 40, 32, 24, 16]
N_BLOCKS = 65


def say(msg: str = "") -> None:
    """Print ASCII-only (the Windows console here is GBK, not UTF-8)."""
    print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def hdr(title: str) -> None:
    say("")
    say("=" * 68)
    say(title)
    say("=" * 68)


def gb(n: float) -> str:
    return "%.2f GB" % (n / 1e9)


# ---------------------------------------------------------------------------
# A1 -- prebuilt llama.cpp CUDA binary
# ---------------------------------------------------------------------------
def resolve_release(tag: str) -> tuple[str, str, str]:
    """Return (tag, binary_asset, cudart_asset) for the Windows CUDA x64 build."""
    if tag != "latest":
        return (tag,
                "llama-%s-bin-win-%s-x64.zip" % (tag, CUDA_TAG),
                "cudart-llama-bin-win-%s-x64.zip" % CUDA_TAG)
    req = urllib.request.Request(GH_LATEST, headers={"User-Agent": "echo-setup"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        rel = json.load(resp)
    names = [a["name"] for a in rel.get("assets", [])]
    want_bin = "llama-%s-bin-win-%s-x64.zip" % (rel["tag_name"], CUDA_TAG)
    want_rt = "cudart-llama-bin-win-%s-x64.zip" % CUDA_TAG
    for n in (want_bin, want_rt):
        if n not in names:
            raise SystemExit("asset missing from release %s: %s" % (rel["tag_name"], n))
    return rel["tag_name"], want_bin, want_rt


def fetch(url: str, dest: Path, expect: int | None = None) -> None:
    """Download with resume. Idempotent: a complete file is left alone."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and expect and dest.stat().st_size == expect:
        say("  have %s (%s) -- skip" % (dest.name, gb(expect)))
        return
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl:
        # -C - resumes a partial file, which matters a great deal for 17 GB.
        cmd = [curl, "-L", "--fail", "--retry", "10", "--retry-delay", "5",
               "--retry-all-errors", "-C", "-", "-o", str(dest), url]
        say("  curl %s -> %s" % (url.rsplit("/", 1)[-1], dest.name))
        rc = subprocess.call(cmd)
        if rc != 0 and not (dest.exists() and expect and dest.stat().st_size == expect):
            raise SystemExit("download failed (curl rc=%d): %s" % (rc, url))
        return
    say("  urllib %s -> %s" % (url.rsplit("/", 1)[-1], dest.name))
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": "echo-setup"}),
            timeout=120) as resp, open(dest, "wb") as fh:
        shutil.copyfileobj(resp, fh, 1 << 20)


def step_a1(tag: str) -> Path:
    hdr("A1  prebuilt llama.cpp Windows CUDA binary")
    server = LLAMA_DIR / "llama-server.exe"
    if server.exists() and (LLAMA_DIR / "ggml-cuda.dll").exists():
        say("  already unpacked at %s" % LLAMA_DIR)
    else:
        tag, bin_asset, rt_asset = resolve_release(tag)
        say("  release: %s" % tag)
        for asset in (bin_asset, rt_asset):
            zpath = DL_DIR / asset
            fetch("%s/%s/%s" % (GH_DL, tag, asset), zpath)
            say("  unpack %s" % asset)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(LLAMA_DIR)
    if not server.exists():
        raise SystemExit("llama-server.exe missing after unpack")
    out = subprocess.run([str(server), "--version"], capture_output=True, text=True)
    say("  %s" % (out.stderr or out.stdout).strip().splitlines()[0])
    dev = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True)
    for line in (dev.stdout + dev.stderr).splitlines():
        if "CUDA" in line or "Available" in line:
            say("  %s" % line.strip())
    if "CUDA" not in dev.stdout + dev.stderr:
        say("  WARNING: no CUDA device reported -- offload will not work")
    return server


# ---------------------------------------------------------------------------
# A2 -- STOP GATE: architecture support, checked before the 17 GB download
# ---------------------------------------------------------------------------
def read_gguf_arch(path_or_url: str, local: bool) -> tuple[str | None, int | None]:
    """Parse `general.architecture` and block_count out of a GGUF header.

    All GGUF metadata sits at the head of the file, so for a remote model a
    Range request for the first few MB answers the question without downloading
    the weights. That is the whole point of the A2 gate.
    """
    want = 8 << 20
    if local:
        buf = Path(path_or_url).open("rb").read(want)
    else:
        req = urllib.request.Request(path_or_url, headers={
            "Range": "bytes=0-%d" % (want - 1), "User-Agent": "echo-setup"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            buf = resp.read()
    if buf[:4] != b"GGUF":
        return None, None

    pos = [4]

    def take(fmt: str, size: int):
        v = struct.unpack_from(fmt, buf, pos[0])[0]
        pos[0] += size
        return v

    def gstr() -> str:
        n = take("<Q", 8)
        s = buf[pos[0]:pos[0] + n]
        pos[0] += n
        return s.decode("utf-8", "replace")

    take("<I", 4)          # gguf version
    take("<Q", 8)          # tensor count
    n_kv = take("<Q", 8)
    scalars = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
               4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
               10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}

    def val(t: int):
        if t == 8:
            return gstr()
        if t == 9:
            et = take("<I", 4)
            n = take("<Q", 8)
            return [val(et) for _ in range(n)]
        return take(*scalars[t])

    arch, blocks = None, None
    for _ in range(n_kv):
        try:
            key = gstr()
            v = val(take("<I", 4))
        except (struct.error, IndexError, KeyError):
            break  # header longer than our window; we may already have what we need
        if key == "general.architecture":
            arch = v
        if key.endswith(".block_count"):
            blocks = v
        if arch and blocks:
            break
    return arch, blocks


def step_a2(server: Path, local_gguf: Path | None) -> tuple[str, int]:
    hdr("A2  STOP GATE -- is the model architecture supported?")
    blobs = list(LLAMA_DIR.glob("*.dll")) + list(LLAMA_DIR.glob("*.exe"))
    hits = []
    for p in blobs:
        try:
            if ARCH_SYMBOL.encode() in p.read_bytes():
                hits.append(p.name)
        except OSError:
            pass
    say("  binary scan for '%s': %s"
        % (ARCH_SYMBOL, ", ".join(hits) if hits else "NOT FOUND"))

    if local_gguf and local_gguf.exists() and local_gguf.stat().st_size > (16 << 20):
        arch, blocks = read_gguf_arch(str(local_gguf), local=True)
        say("  gguf header (local file): arch=%s block_count=%s" % (arch, blocks))
    else:
        arch, blocks = read_gguf_arch(GGUF_URL, local=False)
        say("  gguf header (HTTP Range, no full download): arch=%s block_count=%s"
            % (arch, blocks))

    if arch is None:
        raise SystemExit("STOP: could not read the GGUF header -- refusing to "
                         "download 17 GB on an unverified assumption")
    if arch != GGUF_ARCH:
        raise SystemExit("STOP: GGUF arch is '%s', expected '%s'. Re-check "
                         "support before downloading." % (arch, GGUF_ARCH))
    if not hits:
        raise SystemExit("STOP: llama.cpp build does not contain '%s' -- "
                         "architecture '%s' is UNSUPPORTED by this binary. "
                         "Not downloading the model." % (ARCH_SYMBOL, arch))
    say("  GATE PASSED: arch '%s' is supported by this llama.cpp build." % arch)
    return arch, (blocks or N_BLOCKS)


# ---------------------------------------------------------------------------
# A3 -- weights
# ---------------------------------------------------------------------------
def step_a3() -> Path:
    hdr("A3  model weights (%s, %s)" % (GGUF_NAME, gb(GGUF_BYTES)))
    dest = GGUF_DIR / GGUF_NAME
    if dest.exists() and dest.stat().st_size == GGUF_BYTES:
        say("  have complete %s -- skip" % GGUF_NAME)
        return dest
    if dest.exists():
        say("  resuming from %s / %s" % (gb(dest.stat().st_size), gb(GGUF_BYTES)))
    fetch(GGUF_URL, dest, expect=GGUF_BYTES)
    size = dest.stat().st_size
    if size != GGUF_BYTES:
        raise SystemExit("size mismatch: got %d expected %d" % (size, GGUF_BYTES))
    say("  complete: %s" % gb(size))
    return dest


# ---------------------------------------------------------------------------
# A5 -- launch with partial offload
# ---------------------------------------------------------------------------
def vram_used_mib() -> int | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    out = subprocess.run(
        [smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    try:
        return int(out.stdout.strip().splitlines()[0])
    except (ValueError, IndexError):
        return None


def health_ok(timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(BASE_URL + "/health", timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError):
        return False


@dataclass
class Server:
    proc: subprocess.Popen
    log: Path
    ngl: int
    load_s: float = 0.0
    vram_mib: int | None = None

    def stop(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def tail(self, n: int = 25) -> str:
        try:
            lines = self.log.read_text("utf-8", "replace").splitlines()
        except OSError:
            return "(no log)"
        return "\n".join(lines[-n:])


def launch(server: Path, gguf: Path, ngl: int, ctx: int,
           wait_s: int = 900) -> Server | None:
    """Start llama-server and wait for /health. None means it failed to load."""
    DL_DIR.mkdir(parents=True, exist_ok=True)
    log = DL_DIR / ("llama_server_ngl%d.log" % ngl)
    cmd = [str(server), "-m", str(gguf), "--n-gpu-layers", str(ngl),
           "--ctx-size", str(ctx), "--host", SERVER_HOST,
           "--port", str(SERVER_PORT), "--jinja", "--no-webui"]
    say("  launch: --n-gpu-layers %d --ctx-size %d" % (ngl, ctx))
    fh = log.open("w", encoding="utf-8")
    env = dict(os.environ)
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                            cwd=str(LLAMA_DIR), env=env)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if proc.poll() is not None:
            say("  FAILED: server exited rc=%s after %.0fs" % (proc.returncode,
                                                               time.time() - t0))
            srv = Server(proc, log, ngl)
            say("  --- server log tail ---")
            for line in srv.tail(20).splitlines():
                say("  | " + line)
            return None
        if health_ok():
            load = time.time() - t0
            vram = vram_used_mib()
            say("  READY in %.1fs   VRAM used: %s MiB" % (load, vram))
            return Server(proc, log, ngl, load, vram)
        time.sleep(2.0)
    say("  FAILED: /health never came up within %ds" % wait_s)
    srv = Server(proc, log, ngl)
    srv.stop()
    return None


def step_a5(server: Path, gguf: Path, ngl: int | None, tune: bool,
            ctx: int) -> Server:
    hdr("A5  llama-server with partial GPU offload")
    say("  %d blocks total; 16 GB VRAM vs %s of weights, so partial offload"
        % (N_BLOCKS, gb(GGUF_BYTES)))
    say("  is the EXPECTED configuration, not a failure.")
    ladder = TUNE_LADDER if tune else [ngl or DEFAULT_NGL]
    for cand in ladder:
        srv = launch(server, gguf, cand, ctx)
        if srv is None:
            say("  -ngl %d did not load; stepping down" % cand)
            continue
        if not smoke(srv):
            say("  -ngl %d loaded but failed a 1-token generation; stepping down"
                % cand)
            srv.stop()
            continue
        # Deliberately not "highest that loads": -ngl 65 loads too, and is
        # slower. This is the fastest rung that loads and generates.
        say("  SELECTED -ngl: %d (measured-best, see DEFAULT_NGL notes)" % cand)
        return srv
    raise SystemExit("no --n-gpu-layers value on the ladder loaded cleanly")


def smoke(srv: Server) -> bool:
    """A server can pass /health and still fail to generate; check for real."""
    try:
        out = chat([{"role": "user", "content": "Say OK."}], max_tokens=4)
        return bool(out.get("text"))
    except Exception as exc:  # noqa: BLE001 - any failure means "step down"
        say("  smoke test raised: %s" % exc)
        return False


# ---------------------------------------------------------------------------
# A6 -- honest latency measurement
# ---------------------------------------------------------------------------
# A realistic Echo stall: several turns of conversation context plus an
# unfinished utterance, answered with a short JSON candidate list. Built from
# backend/prompts.py so the benchmark measures the prompt Echo actually sends,
# not a synthetic stand-in.
# Sized so the SERVER-REPORTED prompt_tokens lands near 400, which is the
# realistic budget for a live Echo stall. Verified with llama-tokenize against
# this model's own vocab rather than a chars/4 guess.
BENCH_CONTEXT = [
    "Oh nice, the photographs from the trip to Lisbon?",
    "Yes, and the ones from the garden before we moved.",
    "How is Ruth doing these days? Still teaching?",
    "She retired in spring, but she still helps out on Thursdays.",
    "That sounds like her. Did she stay for dinner?",
    "She did. I tried to make the fish the way she likes it.",
]
BENCH_FRAGMENT = ("I wanted to cook it in the, um, the thing with the lid that "
                  "you put in the oven, the heavy one, you know, the")
BENCH_ENTITIES = ["Ruth", "Lisbon"]

# Each run uses a DIFFERENT live stall. Sending one identical prompt N times
# would let llama.cpp's prompt cache re-evaluate only ~4 tokens and report a
# time-to-first-token that no real stall would ever see. Echo's real traffic
# shares the system prompt and few-shots (cacheable prefix) but brings a fresh
# final turn every time, and that is what these variants reproduce.
BENCH_VARIANTS = [
    (BENCH_FRAGMENT, BENCH_ENTITIES),
    ("we left the tickets on the, uh, the flat thing by the door where I put "
     "the keys, the", ["Ruth", "Lisbon"]),
    ("I need to call the man who fixes the, um, the pipes, when the water goes "
     "everywhere, the", ["Ruth"]),
    ("she gave me a, a, the paper you get at the shop that says what you paid, "
     "the", ["Ruth", "Lisbon"]),
    ("can you pass me the, um, the thing you look through to see far away, the "
     "long round", ["Lisbon"]),
    ("we watched that programme about the big grey animal with the long, the, "
     "you know, the", ["Ruth"]),
]


def bench_messages(variant: int = 0) -> list[dict]:
    from backend.prompts import FEW_SHOTS, SYSTEM_PROMPT, build_user_text
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for shot_in, shot_out in FEW_SHOTS:
        msgs.append({"role": "user", "content": build_user_text(
            shot_in["context"], shot_in["fragment"])})
        msgs.append({"role": "assistant", "content": json.dumps(
            {"candidates": [{"word": w, "confidence": round(0.9 - 0.2 * i, 2)}
                            for i, w in enumerate(shot_out)]})})
    frag, ents = BENCH_VARIANTS[variant % len(BENCH_VARIANTS)]
    msgs.append({"role": "user", "content": build_user_text(
        BENCH_CONTEXT, frag, entities=ents)})
    return msgs


def chat(messages: list[dict], max_tokens: int = 64,
         stream: bool = False) -> dict:
    """POST /v1/chat/completions. Streaming gives a true time-to-first-token."""
    # Qwen3.x chat templates commonly default to THINKING mode, which would add
    # hundreds of reasoning tokens before the first useful character -- fatal for
    # a latency-critical stall and fatal for strict JSON. Echo wants the
    # non-thinking path, so ask for it explicitly. Harmless if the template
    # ignores the kwarg; `reasoning_content` in the stream tells us if it didn't.
    body = {"model": "local", "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_format": "none"}
    if stream:
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(
        BASE_URL + "/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    t0 = time.perf_counter()
    ttft = None
    text_parts: list[str] = []
    think_parts: list[str] = []
    usage: dict = {}
    with urllib.request.urlopen(req, timeout=600) as resp:
        if not stream:
            payload = json.load(resp)
            total = time.perf_counter() - t0
            return {"text": payload["choices"][0]["message"]["content"],
                    "ttft_s": None, "total_s": total,
                    "usage": payload.get("usage", {}), "reasoning_chars": 0}
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices", []):
                delta = ch.get("delta") or {}
                if delta.get("reasoning_content"):
                    think_parts.append(delta["reasoning_content"])
                piece = delta.get("content") or ""
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    text_parts.append(piece)
    return {"text": "".join(text_parts), "ttft_s": ttft,
            "total_s": time.perf_counter() - t0, "usage": usage,
            "reasoning_chars": len("".join(think_parts))}


@dataclass
class Bench:
    runs: list[dict] = field(default_factory=list)

    def add(self, r: dict) -> None:
        self.runs.append(r)

    def col(self, key: str) -> list[float]:
        return [r[key] for r in self.runs if r.get(key) is not None]

    @staticmethod
    def med(xs: list[float]) -> float:
        if not xs:
            return float("nan")
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


CLOUD_LO, CLOUD_HI = 1.4, 2.1  # measured cloud predictor baseline, seconds


def step_a6(runs: int, srv: Server | None, ctx: int) -> Bench:
    hdr("A6  latency, measured over %d runs" % runs)
    msgs = bench_messages(0)
    say("  prompt: %d messages (system + %d few-shot pairs + 1 live stall)"
        % (len(msgs), (len(msgs) - 2) // 2))
    say("  context turns: %d   distinct live stalls: %d"
        % (len(BENCH_CONTEXT), len(BENCH_VARIANTS)))
    say("  each run sends a DIFFERENT final turn, so only the shared")
    say("  system+few-shot prefix is served from the prompt cache.")

    b = Bench()
    # One warm-up so the measured runs are not all paying to build the shared
    # prefix cache. Disclosed rather than hidden, and excluded from the table:
    # it is the only truly COLD-prompt number here, so it is printed too.
    warm = chat(msgs, max_tokens=64, stream=True)
    say("  warm-up (excluded): %.2f s, %d prompt tok, %d out tok"
        % (warm["total_s"], warm["usage"].get("prompt_tokens", -1),
           warm["usage"].get("completion_tokens", -1)))
    if warm.get("reasoning_chars"):
        say("  NOTE: model emitted %d chars of reasoning despite "
            "enable_thinking=False -- latency below INCLUDES thinking."
            % warm["reasoning_chars"])
    say("")
    say("  run  ttft_s  total_s  out_tok  tok/s   text")
    say("  ---  ------  -------  -------  -----   ----")
    for i in range(runs):
        # +1 so no measured run repeats the warm-up's exact prompt.
        r = chat(bench_messages(i + 1), max_tokens=64, stream=True)
        u = r["usage"] or {}
        out_tok = u.get("completion_tokens") or 0
        gen_s = (r["total_s"] - (r["ttft_s"] or 0.0)) or 1e-9
        tps = (out_tok - 1) / gen_s if out_tok > 1 else float("nan")
        rec = {"ttft_s": r["ttft_s"], "total_s": r["total_s"],
               "prompt_tokens": u.get("prompt_tokens"),
               "completion_tokens": out_tok, "tok_per_s": tps,
               "text": (r["text"] or "").strip().replace("\n", " ")}
        b.add(rec)
        say("  %3d  %6.2f  %7.2f  %7d  %5.1f   %s"
            % (i + 1, rec["ttft_s"] or -1, rec["total_s"], out_tok, tps,
               rec["text"][:48]))

    ttfts, totals, tpss = b.col("ttft_s"), b.col("total_s"), b.col("tok_per_s")
    ptok = b.col("prompt_tokens")
    if not ttfts or not totals:
        # Degenerate result: say so instead of dying in min() on an empty list.
        say("")
        say("  NO MEASUREMENTS: the server returned no streamed content.")
        say("  Treat this as a FAILED bench, not a fast one.")
        return b
    say("")
    say("  prompt tokens (server-reported): %s" % (int(ptok[0]) if ptok else "?"))
    say("  median ttft   : %.2f s   (min %.2f / max %.2f)"
        % (Bench.med(ttfts), min(ttfts), max(ttfts)))
    say("  median total  : %.2f s   (min %.2f / max %.2f)"
        % (Bench.med(totals), min(totals), max(totals)))
    say("  median tok/s  : %.1f" % Bench.med(tpss))
    if srv:
        say("  -ngl %d, ctx %d, VRAM %s MiB, load %.0fs"
            % (srv.ngl, ctx, srv.vram_mib, srv.load_s))

    med_total = Bench.med(totals)
    say("")
    say("  cloud predictor baseline: %.1f-%.1f s end-to-end" % (CLOUD_LO, CLOUD_HI))
    if med_total <= CLOUD_HI:
        verdict = ("COMPETITIVE: median %.2f s is inside the cloud band."
                   % med_total)
    elif med_total <= CLOUD_HI * 2:
        verdict = ("SLOWER: median %.2f s is %.1fx the top of the cloud band "
                   "(%.1f s). Usable but a regression for live serving."
                   % (med_total, med_total / CLOUD_HI, CLOUD_HI))
    else:
        verdict = ("MUCH SLOWER: median %.2f s is %.1fx the top of the cloud "
                   "band (%.1f s). Not viable for live stall serving at this "
                   "quant/offload." % (med_total, med_total / CLOUD_HI, CLOUD_HI))
    say("  VERDICT: %s" % verdict)
    return b


# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--release", default=LLAMA_RELEASE,
                    help="llama.cpp release tag, or 'latest' (default %s)"
                         % LLAMA_RELEASE)
    ap.add_argument("--ngl", type=int, default=None,
                    help="GPU layers to offload (default %d)" % DEFAULT_NGL)
    ap.add_argument("--ctx", type=int, default=CTX_SIZE)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--tune", action="store_true",
                    help="search the -ngl ladder for the highest that loads")
    ap.add_argument("--check", action="store_true",
                    help="A1+A2 only: verify support, download no weights")
    ap.add_argument("--bench-only", action="store_true",
                    help="A6 only against an already-running server")
    ap.add_argument("--keep-alive", action="store_true",
                    help="leave llama-server running after the bench")
    args = ap.parse_args(argv)

    say("Echo local-LLM bring-up  (model %s)" % GGUF_NAME)

    if args.bench_only:
        if not health_ok():
            raise SystemExit("no server on %s -- start one first" % BASE_URL)
        say("using the already-running server on %s" % BASE_URL)
        step_a6(args.runs, None, args.ctx)
        return 0

    server = step_a1(args.release)
    local = GGUF_DIR / GGUF_NAME
    step_a2(server, local)
    if args.check:
        say("")
        say("--check: gate passed, stopping before the %s download." % gb(GGUF_BYTES))
        return 0

    gguf = step_a3()
    srv = step_a5(server, gguf, args.ngl, args.tune, args.ctx)
    try:
        step_a6(args.runs, srv, args.ctx)
    finally:
        if args.keep_alive:
            say("")
            say("leaving llama-server up on %s (pid %d)" % (BASE_URL, srv.proc.pid))
        else:
            srv.stop()
            say("")
            say("llama-server stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
