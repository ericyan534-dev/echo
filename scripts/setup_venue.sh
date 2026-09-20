#!/usr/bin/env bash
# One-command venue bootstrap for Echo -- bash mirror of scripts/setup_venue.ps1
# for a borrowed non-Windows laptop. Simpler than the PS1: same checks, same
# ASCII [PASS]/[FAIL] lines, same summary block. Idempotent -- safe to re-run.
#
# Usage:
#   ./scripts/setup_venue.sh [--with-dev] [--with-e2e]
#
#   --with-dev   also install requirements-dev.txt (pytest + eval extras;
#                needed for the pytest check below)
#   --with-e2e   after the offline suite passes, also run
#                `python -m scripts.e2e_live` (needs a real GEMINI_API_KEY;
#                makes real network calls; expects 5/5)

set -u
START_TS=$SECONDS

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

WITH_DEV=0
WITH_E2E=0
for arg in "$@"; do
    case "$arg" in
        --with-dev) WITH_DEV=1 ;;
        --with-e2e) WITH_E2E=1 ;;
        *) echo "unknown flag: $arg (expected --with-dev and/or --with-e2e)" ;;
    esac
done

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
FIXES=()

# check <name> <ok:0|1> [hint] [note]
check() {
    name="$1"; ok="$2"; hint="${3:-}"; note="${4:-}"
    if [ "$ok" -eq 0 ]; then
        echo "[PASS] $name"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "[FAIL] $name"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        if [ -n "$hint" ]; then
            echo "       fix: $hint"
            FIXES+=("$name -- $hint")
        else
            FIXES+=("$name")
        fi
    fi
    if [ -n "$note" ]; then
        echo "       $note"
    fi
}

skip() {
    name="$1"; note="${2:-}"
    echo "[SKIP] $name"
    if [ -n "$note" ]; then echo "       $note"; fi
    SKIP_COUNT=$((SKIP_COUNT + 1))
}

echo "=== Echo venue bootstrap ==="
echo "repo: $REPO_ROOT"
echo ""

# --- 1. python >= 3.12 -------------------------------------------------------
PY=""
for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver="$("$cand" --version 2>&1)"
        maj="$(echo "$ver" | sed -n 's/.*Python \([0-9]*\)\.\([0-9]*\).*/\1/p')"
        min="$(echo "$ver" | sed -n 's/.*Python \([0-9]*\)\.\([0-9]*\).*/\2/p')"
        if [ -n "$maj" ] && [ -n "$min" ]; then
            if [ "$maj" -gt 3 ] || { [ "$maj" -eq 3 ] && [ "$min" -ge 12 ]; }; then
                PY="$cand"
                break
            fi
        fi
    fi
done
if [ -n "$PY" ]; then
    check "python >= 3.12 present" 0 "" "using '$PY' ($ver)"
else
    check "python >= 3.12 present" 1 "install Python 3.12+ (https://www.python.org/downloads/) and ensure it's on PATH"
fi

# --- 2. venv created / activatable -------------------------------------------
VENV_DIR="$REPO_ROOT/.venv"
VENV_PY="$VENV_DIR/bin/python"
VENV_OK=1
if [ -n "$PY" ]; then
    if [ ! -x "$VENV_PY" ]; then
        "$PY" -m venv "$VENV_DIR" >/dev/null 2>&1
    fi
    if [ -x "$VENV_PY" ] && [ -f "$VENV_DIR/bin/activate" ]; then
        VENV_OK=0
        check ".venv created and activatable" 0 "" "activate with: source .venv/bin/activate (this script drives it directly via its python)"
    else
        check ".venv created and activatable" 1 "run manually: $PY -m venv .venv"
    fi
else
    check ".venv created and activatable" 1 "fix the python check above first"
fi

# --- 3. pip install -----------------------------------------------------
PIP_OK=1
if [ "$VENV_OK" -eq 0 ]; then
    "$VENV_PY" -m pip install --upgrade pip --quiet >/dev/null 2>&1
    req_out="$("$VENV_PY" -m pip install -r "$REPO_ROOT/requirements.txt" --quiet 2>&1)"
    req_exit=$?
    dev_note=""
    if [ "$req_exit" -eq 0 ]; then
        PIP_OK=0
        if [ "$WITH_DEV" -eq 1 ]; then
            dev_out="$("$VENV_PY" -m pip install -r "$REPO_ROOT/requirements-dev.txt" --quiet 2>&1)"
            if [ $? -eq 0 ]; then
                dev_note="requirements-dev.txt also installed (--with-dev)"
            else
                PIP_OK=1
                req_out="$dev_out"
            fi
        else
            dev_note="requirements-dev.txt NOT installed (pass --with-dev for pytest/eval extras -- needed for the pytest check below)"
        fi
    fi
    if [ "$PIP_OK" -eq 0 ]; then
        check "pip install -r requirements.txt" 0 "" "$dev_note"
    else
        last_line="$(echo "$req_out" | tail -n 1)"
        check "pip install -r requirements.txt" 1 "run manually and read the error: .venv/bin/pip install -r requirements.txt" "last pip line: $last_line"
    fi
else
    check "pip install -r requirements.txt" 1 "fix the venv check above first"
fi

# --- 4. models/fillernet.pt exists -----------------------------------------
if [ -f "$REPO_ROOT/models/fillernet.pt" ]; then
    check "models/fillernet.pt present" 0
else
    check "models/fillernet.pt present" 1 "this is a committed plain file (NOT git-lfs -- no 'git lfs pull' needed); re-clone or run 'git checkout -- models/fillernet.pt'"
fi

# --- 5. .env exists AND defines GEMINI_API_KEY (presence only) -------------
ENV_PATH="$REPO_ROOT/.env"
ENV_EXAMPLE="$REPO_ROOT/.env.example"
ENV_JUST_CREATED=0
if [ ! -f "$ENV_PATH" ] && [ -f "$ENV_EXAMPLE" ]; then
    cp "$ENV_EXAMPLE" "$ENV_PATH"
    ENV_JUST_CREATED=1
fi
HAS_KEY=1  # boolean presence test only -- the key's value is never printed
if [ -f "$ENV_PATH" ]; then
    # non-empty value after GEMINI_API_KEY= or GOOGLE_API_KEY=, value itself
    # is never captured into an echoed variable.
    if grep -Eq '^[[:space:]]*(GEMINI_API_KEY|GOOGLE_API_KEY)[[:space:]]*=[[:space:]]*[^[:space:]]' "$ENV_PATH"; then
        HAS_KEY=0
    fi
fi
if [ -f "$ENV_PATH" ] && [ "$HAS_KEY" -eq 0 ]; then
    check ".env exists and defines GEMINI_API_KEY" 0
else
    if [ ! -f "$ENV_PATH" ]; then
        check ".env exists and defines GEMINI_API_KEY" 1 "copy the template and fill in your key: cp .env.example .env, then edit .env and set GEMINI_API_KEY=<your key>"
    else
        created_note=""
        if [ "$ENV_JUST_CREATED" -eq 1 ]; then created_note=".env was just created from .env.example by this script -- "; fi
        check ".env exists and defines GEMINI_API_KEY" 1 "${created_note}edit .env and set GEMINI_API_KEY=<your key> (get one at https://aistudio.google.com/apikey)"
    fi
fi

# --- 6. python -m pytest runs green -----------------------------------------
PYTEST_OK=1
if [ "$VENV_OK" -eq 0 ] && [ "$PIP_OK" -eq 0 ]; then
    # pytest.ini already sets addopts=-q; overriding addopts (rather than also
    # passing -q) avoids stacking quiet levels, which would suppress the final
    # "N passed" summary line this script greps for.
    pytest_out="$("$VENV_PY" -m pytest -o addopts="-q" 2>&1)"
    pytest_exit=$?
    passed_n="$(echo "$pytest_out" | grep -Eo '[0-9]+ passed' | grep -Eo '[0-9]+' | head -n1)"
    skipped_n="$(echo "$pytest_out" | grep -Eo '[0-9]+ skipped' | grep -Eo '[0-9]+' | head -n1)"
    failed_n="$(echo "$pytest_out" | grep -Eo '[0-9]+ failed' | grep -Eo '[0-9]+' | head -n1)"
    error_n="$(echo "$pytest_out" | grep -Eo '[0-9]+ error' | grep -Eo '[0-9]+' | head -n1)"

    note=""
    if [ -n "$passed_n" ] && [ -z "$failed_n" ] && [ -z "$error_n" ]; then
        if [ "$pytest_exit" -eq 0 ]; then PYTEST_OK=0; fi
        if [ -n "$skipped_n" ]; then
            note="$passed_n passed, $skipped_n skipped -- expected without data/ (1 test needs the local PFSD dataset; scripts/fetch_pfsd.py). Treated as PASS."
        else
            note="$passed_n passed -- full suite (data/ present)."
        fi
    else
        note="pytest did not report a clean summary; last line: $(echo "$pytest_out" | tail -n1)"
    fi

    if [ "$PYTEST_OK" -eq 0 ]; then
        check "python -m pytest" 0 "" "$note"
    else
        hint="run '.venv/bin/python -m pytest' yourself and read the failure(s)"
        if echo "$pytest_out" | grep -q "No module named pytest\|No module named 'pytest'"; then
            hint="pytest isn't installed -- re-run this script with --with-dev (or: .venv/bin/pip install -r requirements-dev.txt)"
        fi
        check "python -m pytest" 1 "$hint" "$note"
    fi
else
    check "python -m pytest" 1 "fix the venv/pip checks above first"
fi

# --- 7. optional live e2e ----------------------------------------------------
if [ "$WITH_E2E" -eq 1 ]; then
    if [ "$VENV_OK" -eq 0 ] && [ "$PIP_OK" -eq 0 ] && [ -f "$ENV_PATH" ] && [ "$HAS_KEY" -eq 0 ]; then
        e2e_out="$("$VENV_PY" -m scripts.e2e_live 2>&1)"
        e2e_exit=$?
        echo ""
        echo "--- live e2e output ---"
        echo "$e2e_out"
        echo "--- end live e2e output ---"
        got="$(echo "$e2e_out" | grep -Eo '[0-9]+/[0-9]+ live e2e cases passed' | grep -Eo '^[0-9]+')"
        total="$(echo "$e2e_out" | grep -Eo '[0-9]+/[0-9]+ live e2e cases passed' | grep -Eo '/[0-9]+' | tr -d '/')"
        e2e_ok=1
        note="see full output above for details"
        if [ -n "$got" ] && [ -n "$total" ]; then
            note="$got/$total live e2e cases passed"
            if [ "$got" = "$total" ] && [ "$e2e_exit" -eq 0 ]; then e2e_ok=0; fi
        fi
        check "python -m scripts.e2e_live (--with-e2e)" "$e2e_ok" "check GEMINI_API_KEY is valid and the venue network can reach the Gemini API" "$note"
    else
        check "python -m scripts.e2e_live (--with-e2e)" 1 "fix the venv/pip/.env checks above first"
    fi
else
    skip "python -m scripts.e2e_live" "pass --with-e2e to run it (needs a valid GEMINI_API_KEY; makes real network calls)"
fi

# --- summary -----------------------------------------------------------------
ELAPSED=$((SECONDS - START_TS))

echo ""
echo "=== Summary ==="
echo "PASS: $PASS_COUNT   FAIL: $FAIL_COUNT   SKIP: $SKIP_COUNT"
echo "Elapsed: ${ELAPSED}s"
echo ""

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "READY FOR DEMO"
    exit 0
else
    echo "NOT READY -- fix these, in order:"
    i=1
    for f in "${FIXES[@]}"; do
        echo "  $i. $f"
        i=$((i + 1))
    done
    exit 1
fi
