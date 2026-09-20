# Echo -- start the live demo stack with the decided configuration.
#
# Every knob the demo depends on is pinned HERE, not typed by hand, so a
# restart can never silently drop one. Secrets (API keys) stay in .env.
#
#   .\scripts\run_live.ps1              # foreground, Ctrl+C to stop
#   .\scripts\run_live.ps1 -Port 8011   # second instance
#
# Stack (see docs/DETECTION_TUNING.md and docs/VERSIONS.md):
#   detection  : SSL StutterNet recall_v2 on CUDA, 400 ms context lag
#   prediction : DeepSeek primary, Gemini raced behind it (provider failover)
param([int]$Port = 8000)

Set-Location (Split-Path -Parent $PSScriptRoot)

# --- acoustic channel ---
$env:STUTTER_BACKEND         = "ssl"
$env:ACOUSTIC_DEVICE         = "cuda"
$env:STUTTER_MODEL           = "models/stutternet_recall_v2.pt"
$env:STUTTER_SCALE           = "1.0"
$env:ACOUSTIC_CONTEXT_LAG_MS = "400"
$env:ACOUSTIC_MIN_VOICED_MS  = "400"

# --- word prediction ---
# A provider outage must never be an empty card: if DeepSeek 503s, errors,
# or takes > 1.2 s, Gemini answers the same fragment (backend/predictor/failover.py).
$env:PREDICTOR_PROVIDER  = "deepseek"
$env:PREDICTOR_FALLBACKS = "gemini"

python -m uvicorn backend.app:app --host 0.0.0.0 --port $Port
