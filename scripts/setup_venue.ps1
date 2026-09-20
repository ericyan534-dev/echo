<#
.SYNOPSIS
    One-command venue bootstrap for Echo. Idempotent -- safe to re-run on a
    machine that's already partly set up (venv/pip steps are skip-if-present
    or naturally idempotent; nothing here deletes or overwrites your .env).

.DESCRIPTION
    Runs the checks a fresh laptop needs before a demo, in order, each
    printing an ASCII [PASS]/[FAIL] line with a one-line fix hint on failure:

        1. python >= 3.12 present
        2. .venv created and activatable
        3. pip install -r requirements.txt   (+ requirements-dev.txt with -WithDev)
        4. models/fillernet.pt present (committed plain file, not git-lfs)
        5. .env exists AND defines GEMINI_API_KEY (presence check only --
           this script never prints, echoes, or copies the key's value)
        6. python -m pytest runs green (122 passed with data/ present,
           121 passed + 1 skipped without -- both count as PASS)
        7. (optional, -WithE2E) python -m scripts.e2e_live -- 5/5 required

    Ends with a PASS/FAIL summary and either "READY FOR DEMO" or the ordered
    list of what to fix. Prints its own elapsed time.

.PARAMETER WithDev
    Also install requirements-dev.txt (pytest + eval/training extras). The
    mandatory pytest check in step 6 needs this -- omit only for a fast,
    runtime-only install where you don't care about the test-suite check.

.PARAMETER WithE2E
    After the offline suite passes, also run the live e2e suite
    (python -m scripts.e2e_live). Requires a real GEMINI_API_KEY in .env and
    makes real network calls to the Gemini API. Expects 5/5 to PASS.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_venue.ps1 -WithDev

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\setup_venue.ps1 -WithDev -WithE2E

.NOTES
    PowerShell 5.1 compatible: no &&/||, no ternary operator, no null-
    coalescing. See scripts/setup_venue.sh for a simpler bash mirror to use
    on a borrowed non-Windows laptop.
#>
param(
    [switch]$WithDev,
    [switch]$WithE2E
)

$ErrorActionPreference = "Continue"
$Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

# Repo root = parent of this script's directory (scripts\setup_venue.ps1 -> repo root).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$script:PassCount = 0
$script:FailCount = 0
$script:SkipCount = 0
$script:Fixes = New-Object System.Collections.ArrayList

function Write-Check {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][bool]$Ok,
        [string]$Hint = "",
        [string]$Note = ""
    )
    if ($Ok) {
        Write-Host "[PASS] $Name"
        $script:PassCount++
    } else {
        Write-Host "[FAIL] $Name"
        $script:FailCount++
        if ($Hint -ne "") {
            Write-Host "       fix: $Hint"
            [void]$script:Fixes.Add("$Name -- $Hint")
        } else {
            [void]$script:Fixes.Add("$Name")
        }
    }
    if ($Note -ne "") {
        Write-Host "       $Note"
    }
}

function Get-LastNonEmptyLine {
    # PS 5.1 quirk: `2>&1` on a native exe wraps each stderr line in an
    # ErrorRecord whose default .ToString() is just the exception type name
    # ("System.Management.Automation.RemoteException"), not the real message.
    # Piping through Out-String first renders the actual text for both
    # stdout and ErrorRecord lines, so line-splitting that is reliable.
    param([string]$Text)
    $lines = $Text -split "`r?`n" | Where-Object { $_.Trim() -ne "" }
    if ($lines.Count -eq 0) { return "" }
    return $lines[-1].Trim()
}

function Write-Skip {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [string]$Note = ""
    )
    Write-Host "[SKIP] $Name"
    if ($Note -ne "") { Write-Host "       $Note" }
    $script:SkipCount++
}

Write-Host "=== Echo venue bootstrap ==="
Write-Host "repo: $RepoRoot"
Write-Host ""

# --- 1. python >= 3.12 -----------------------------------------------------
$PythonOk = $false
$PythonCmd = $null
foreach ($cand in @("python", "py", "python3")) {
    $exists = Get-Command $cand -ErrorAction SilentlyContinue
    if ($null -eq $exists) { continue }
    try {
        $verOut = & $cand --version 2>&1
    } catch {
        continue
    }
    if ($verOut -match "Python (\d+)\.(\d+)") {
        $maj = [int]$Matches[1]
        $min = [int]$Matches[2]
        if (($maj -gt 3) -or (($maj -eq 3) -and ($min -ge 12))) {
            $PythonOk = $true
            $PythonCmd = $cand
            break
        }
    }
}
if ($PythonOk) {
    Write-Check -Name "python >= 3.12 present" -Ok $true -Note "using '$PythonCmd' ($verOut)"
} else {
    Write-Check -Name "python >= 3.12 present" -Ok $false `
        -Hint "install Python 3.12+ from https://www.python.org/downloads/ and ensure it's on PATH, then re-run this script"
}

# --- 2. venv created / activatable -----------------------------------------
$VenvDir = Join-Path $RepoRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$VenvActivate = Join-Path $VenvDir "Scripts\Activate.ps1"
$VenvOk = $false

if ($PythonOk) {
    if (-not (Test-Path $VenvPython)) {
        & $PythonCmd -m venv $VenvDir 2>&1 | Out-Null
    }
    $VenvOk = (Test-Path $VenvPython) -and (Test-Path $VenvActivate)
    if ($VenvOk) {
        Write-Check -Name ".venv created and activatable" -Ok $true `
            -Note "activate it yourself with: .venv\Scripts\Activate.ps1 (this script drives it directly via its python.exe)"
    } else {
        Write-Check -Name ".venv created and activatable" -Ok $false `
            -Hint "run manually: python -m venv .venv"
    }
} else {
    Write-Check -Name ".venv created and activatable" -Ok $false `
        -Hint "fix the python check above first"
}

# --- 3. pip install -----------------------------------------------------
$PipOk = $false
if ($VenvOk) {
    & $VenvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
    $reqOutput = & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements.txt") --quiet 2>&1 | Out-String
    $reqExit = $LASTEXITCODE
    $PipOk = ($reqExit -eq 0)
    $badOutput = $reqOutput
    $devNote = ""
    if ($PipOk -and $WithDev) {
        $devOutput = & $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-dev.txt") --quiet 2>&1 | Out-String
        $PipOk = ($LASTEXITCODE -eq 0)
        $badOutput = $devOutput
        $devNote = "requirements-dev.txt also installed (-WithDev)"
    } elseif (-not $WithDev) {
        $devNote = "requirements-dev.txt NOT installed (pass -WithDev to also install pytest/eval extras -- needed for the pytest check below)"
    }
    if ($PipOk) {
        Write-Check -Name "pip install -r requirements.txt" -Ok $true -Note $devNote
    } else {
        Write-Check -Name "pip install -r requirements.txt" -Ok $false `
            -Hint "run manually and read the error: .venv\Scripts\pip install -r requirements.txt" `
            -Note ("last pip line: " + (Get-LastNonEmptyLine $badOutput))
    }
} else {
    Write-Check -Name "pip install -r requirements.txt" -Ok $false `
        -Hint "fix the venv check above first"
}

# --- 4. models/fillernet.pt exists -----------------------------------------
$FillerNetPath = Join-Path $RepoRoot "models\fillernet.pt"
$FillerNetOk = Test-Path $FillerNetPath
Write-Check -Name "models/fillernet.pt present" -Ok $FillerNetOk `
    -Hint "this is a committed plain file (NOT git-lfs -- no 'git lfs pull' needed); re-clone or run 'git checkout -- models/fillernet.pt'"

# --- 5. .env exists AND defines GEMINI_API_KEY (presence only) -------------
$EnvPath = Join-Path $RepoRoot ".env"
$EnvExamplePath = Join-Path $RepoRoot ".env.example"
$EnvJustCreated = $false
if ((-not (Test-Path $EnvPath)) -and (Test-Path $EnvExamplePath)) {
    Copy-Item $EnvExamplePath $EnvPath
    $EnvJustCreated = $true
}
$EnvExists = Test-Path $EnvPath
$HasKey = $false   # boolean presence test only -- the key's value is never read into a printed variable
if ($EnvExists) {
    $envLines = Get-Content $EnvPath -ErrorAction SilentlyContinue
    foreach ($line in $envLines) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.+)$') {
            $rawVal = $Matches[2].Trim().Trim('"').Trim("'")
            if ($rawVal.Length -gt 0) {
                $HasKey = $true
            }
        }
    }
}
$EnvOk = $EnvExists -and $HasKey
if ($EnvOk) {
    Write-Check -Name ".env exists and defines GEMINI_API_KEY" -Ok $true
} else {
    if (-not $EnvExists) {
        Write-Check -Name ".env exists and defines GEMINI_API_KEY" -Ok $false `
            -Hint "copy the template and fill in your key: Copy-Item .env.example .env, then edit .env and set GEMINI_API_KEY=<your key>"
    } else {
        $createdNote = ""
        if ($EnvJustCreated) { $createdNote = ".env was just created from .env.example by this script -- " }
        Write-Check -Name ".env exists and defines GEMINI_API_KEY" -Ok $false `
            -Hint "${createdNote}edit .env and set GEMINI_API_KEY=<your key> (get one at https://aistudio.google.com/apikey)"
    }
}

# --- 6. python -m pytest runs green -----------------------------------------
$PytestOk = $false
$PytestNote = ""
if ($VenvOk -and $PipOk) {
    # pytest.ini already sets addopts=-q; overriding addopts here (rather than
    # also passing -q) avoids stacking quiet levels, which would suppress the
    # final "N passed" summary line this script parses.
    $pytestOutput = & $VenvPython -m pytest -o addopts="-q" 2>&1 | Out-String
    $pytestExit = $LASTEXITCODE
    $passedMatch = [regex]::Match($pytestOutput, "(\d+) passed")
    $skippedMatch = [regex]::Match($pytestOutput, "(\d+) skipped")
    $failedMatch = [regex]::Match($pytestOutput, "(\d+) failed")
    $errorMatch = [regex]::Match($pytestOutput, "(\d+) error")

    if ($passedMatch.Success -and (-not $failedMatch.Success) -and (-not $errorMatch.Success)) {
        $PytestOk = ($pytestExit -eq 0)
        $passedN = $passedMatch.Groups[1].Value
        if ($skippedMatch.Success) {
            $skippedN = $skippedMatch.Groups[1].Value
            $PytestNote = "$passedN passed, $skippedN skipped -- expected without data/ (1 test needs the local PFSD dataset; scripts/fetch_pfsd.py). Treated as PASS."
        } else {
            $PytestNote = "$passedN passed -- full suite (data/ present)."
        }
    } else {
        $PytestNote = "pytest did not report a clean summary; last line: " + `
            (($pytestOutput -split "`n" | Where-Object { $_.Trim() -ne "" } | Select-Object -Last 1))
    }

    if ($PytestOk) {
        Write-Check -Name "python -m pytest" -Ok $true -Note $PytestNote
    } else {
        $hint = "run '.venv\Scripts\python -m pytest' yourself and read the failure(s)"
        if ($pytestOutput -match "No module named (pytest|'pytest')") {
            $hint = "pytest isn't installed -- re-run this script with -WithDev (or: .venv\Scripts\pip install -r requirements-dev.txt)"
        }
        Write-Check -Name "python -m pytest" -Ok $false -Hint $hint -Note $PytestNote
    }
} else {
    Write-Check -Name "python -m pytest" -Ok $false `
        -Hint "fix the venv/pip checks above first"
}

# --- 7. optional live e2e ----------------------------------------------------
if ($WithE2E) {
    if ($VenvOk -and $PipOk -and $EnvOk) {
        $e2eOutput = & $VenvPython -m scripts.e2e_live 2>&1 | Out-String
        $e2eExit = $LASTEXITCODE
        $e2eMatch = [regex]::Match($e2eOutput, "(\d+)/(\d+) live e2e cases passed")
        $e2eOk = $false
        $e2eNote = "see full output above for details"
        if ($e2eMatch.Success) {
            $got = $e2eMatch.Groups[1].Value
            $total = $e2eMatch.Groups[2].Value
            $e2eOk = ($got -eq $total) -and ($e2eExit -eq 0)
            $e2eNote = "$got/$total live e2e cases passed"
        }
        Write-Host ""
        Write-Host "--- live e2e output ---"
        Write-Host $e2eOutput
        Write-Host "--- end live e2e output ---"
        Write-Check -Name "python -m scripts.e2e_live (-WithE2E)" -Ok $e2eOk `
            -Hint "check GEMINI_API_KEY is valid and the venue network can reach the Gemini API" `
            -Note $e2eNote
    } else {
        Write-Check -Name "python -m scripts.e2e_live (-WithE2E)" -Ok $false `
            -Hint "fix the venv/pip/.env checks above first"
    }
} else {
    Write-Skip -Name "python -m scripts.e2e_live" -Note "pass -WithE2E to run it (needs a valid GEMINI_API_KEY; makes real network calls)"
}

# --- summary -----------------------------------------------------------------
$Stopwatch.Stop()
$ElapsedS = [math]::Round($Stopwatch.Elapsed.TotalSeconds, 1)

Write-Host ""
Write-Host "=== Summary ==="
Write-Host "PASS: $script:PassCount   FAIL: $script:FailCount   SKIP: $script:SkipCount"
Write-Host "Elapsed: ${ElapsedS}s"
Write-Host ""

if ($script:FailCount -eq 0) {
    Write-Host "READY FOR DEMO"
    exit 0
} else {
    Write-Host "NOT READY -- fix these, in order:"
    $i = 1
    foreach ($f in $script:Fixes) {
        Write-Host "  $i. $f"
        $i++
    }
    exit 1
}
