#!/usr/bin/env bash
# Echo bring-up on the ASUS Ascent GX10 (NVIDIA GB10, aarch64, DGX OS).
# Runs the whole stack LOCALLY: local LLM predictor + SSL acoustic on GPU +
# CrisperWhisper STT. Nothing leaves the box.
#
# This is a STEP-CHECKED script: every prerequisite is verified and, if it is
# missing, the script stops and prints the exact fix instead of guessing. It
# was written against DGX OS / NVIDIA documentation and has NOT been run on an
# aarch64 Blackwell box from the dev laptop -- read the notes it prints.
#
# Usage (on the GX10, from the repo root):
#   bash scripts/gx10_bringup.sh            # check + launch
#   bash scripts/gx10_bringup.sh --check    # verify prerequisites only
#   bash scripts/gx10_bringup.sh --stop     # stop llama-server + uvicorn
#
# Env overrides: GGUF=/path/model.gguf  NGL=999  PORT=8000  LLAMA_PORT=8080
set -euo pipefail

PORT="${PORT:-8000}"
LLAMA_PORT="${LLAMA_PORT:-8080}"
NGL="${NGL:-999}"                 # 128 GB unified memory: offload every layer
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
LOG="$REPO/logs"; mkdir -p "$LOG"

say()  { printf '\n== %s ==\n' "$*"; }
die()  { printf '\nFATAL: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

if [ "${1:-}" = "--stop" ]; then
  pkill -f "llama-server.*--port ${LLAMA_PORT}" 2>/dev/null && echo "llama-server stopped" || echo "no llama-server"
  pkill -f "uvicorn backend.app:app.*--port ${PORT}" 2>/dev/null && echo "uvicorn stopped" || echo "no uvicorn"
  exit 0
fi

# ---------------------------------------------------------------------------
say "1. Host: expect aarch64 + NVIDIA GB10"
ARCH="$(uname -m)"
echo "arch: $ARCH"
[ "$ARCH" = "aarch64" ] || echo "  NOTE: not aarch64 -- this script targets the GX10; continuing anyway."
have nvidia-smi || die "nvidia-smi not found. Install the NVIDIA driver (DGX OS ships it)."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || die "nvidia-smi failed."

# ---------------------------------------------------------------------------
say "2. Python venv + deps"
have python3 || die "python3 not found."
[ -d .venv ] || python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
# torch/torchaudio must be the aarch64 CUDA build. On DGX OS these come from
# NVIDIA's index or are preinstalled; do NOT let pip pull an x86/CPU wheel.
if ! python -c "import torch" 2>/dev/null; then
  echo "  torch is not installed. Install the aarch64 CUDA build FIRST, e.g.:"
  echo "    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124"
  echo "    (or use the torch that ships with DGX OS / the NGC container)"
  die "install torch (aarch64 CUDA), then re-run."
fi
python - <<'PY' || die "torch is installed but CUDA is not available -- fix the driver/build before serving."
import torch, sys
ok = torch.cuda.is_available()
print("torch", torch.__version__, "cuda", ok, torch.cuda.get_device_name(0) if ok else "")
sys.exit(0 if ok else 1)
PY
# the rest of the runtime deps (no torch pin is forced here)
pip install -q -r requirements.txt || die "pip install -r requirements.txt failed."

# ---------------------------------------------------------------------------
say "3. llama.cpp built for CUDA (aarch64)"
LLAMA_BIN=""
for c in "$REPO/models/llama.cpp/build/bin/llama-server" "$REPO/models/llama.cpp/llama-server" "$(command -v llama-server || true)"; do
  [ -n "$c" ] && [ -x "$c" ] && { LLAMA_BIN="$c"; break; }
done
if [ -z "$LLAMA_BIN" ]; then
  echo "  No CUDA llama-server found. Build it (a few minutes on the GB10):"
  echo "    git clone https://github.com/ggml-org/llama.cpp models/llama.cpp.src"
  echo "    cmake -S models/llama.cpp.src -B models/llama.cpp.src/build -DGGML_CUDA=on"
  echo "    cmake --build models/llama.cpp.src/build --config Release -j -t llama-server"
  echo "    then re-run, or set the binary path in this script."
  die "build llama-server (CUDA), then re-run."
fi
echo "llama-server: $LLAMA_BIN"

# ---------------------------------------------------------------------------
say "4. Model weights"
GGUF="${GGUF:-}"
if [ -z "$GGUF" ]; then
  GGUF="$(ls -S models/gguf/*.gguf 2>/dev/null | head -1 || true)"
fi
[ -n "$GGUF" ] && [ -f "$GGUF" ] || die "no GGUF found. Put one under models/gguf/ (see scripts/setup_local_llm.py) or pass GGUF=/path/model.gguf. On 128 GB you can run a much larger model than the laptop demo."
echo "GGUF: $GGUF"
[ -d models/crisperwhisper ] || echo "  NOTE: models/crisperwhisper/ absent -- CrisperWhisper STT will fall back; browser STT still works."

if [ "${1:-}" = "--check" ]; then say "checks passed"; exit 0; fi

# ---------------------------------------------------------------------------
say "5. Start llama-server (all layers on GPU)"
"$LLAMA_BIN" --model "$GGUF" --host 127.0.0.1 --port "$LLAMA_PORT" \
  --n-gpu-layers "$NGL" --ctx-size 4096 --parallel 2 \
  > "$LOG/llama_gx10.log" 2>&1 &
echo "llama-server pid $! -> $LOG/llama_gx10.log"
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:${LLAMA_PORT}/health" >/dev/null 2>&1 && { echo "llama-server ready"; break; }
  sleep 2
  [ "$i" = 120 ] && die "llama-server did not become healthy in 240s -- see $LOG/llama_gx10.log"
done

# ---------------------------------------------------------------------------
say "6. Start Echo, fully local"
export PREDICTOR_PROVIDER=local
export LOCAL_LLM_URL="http://127.0.0.1:${LLAMA_PORT}"
export STUTTER_BACKEND=ssl
export ACOUSTIC_DEVICE=cuda
export ASR_PROVIDER=crisper
export ASR_DEVICE=cuda
python -m uvicorn backend.app:app --host 0.0.0.0 --port "$PORT" > "$LOG/echo_gx10.log" 2>&1 &
echo "uvicorn pid $! -> $LOG/echo_gx10.log"
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/healthz" >/dev/null 2>&1 && break
  sleep 2
  [ "$i" = 60 ] && die "Echo did not become healthy -- see $LOG/echo_gx10.log"
done

say "Echo is up on the GX10"
curl -s "http://127.0.0.1:${PORT}/healthz"
echo
echo "Open http://<gx10-ip>:${PORT} in Chrome on a device on the same router."
echo "Nothing leaves the box: predictor=local, acoustic=ssl/cuda, stt=crisper."
echo "Stop with: bash scripts/gx10_bringup.sh --stop"
