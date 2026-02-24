<#
.SYNOPSIS
    Install/update/uninstall Gemini Memory MCP server and CLI tool (gmem).

.DESCRIPTION
    One-command installer for the Gemini Memory MCP system. Handles:
    - Prerequisites check (Python 3.10+, claude CLI)
    - API key resolution (parameter, existing MCP registration, env var, or prompt)
    - File deployment to install directory
    - Python venv creation and dependency installation
    - Platform-specific wrapper scripts (gmem, gmem-precompact, gmem-postcompact)
    - Claude Code MCP server registration
    - Claude Code hooks configuration (PreCompact + SessionStart)

    Re-run after `git pull` to update. Uses -Uninstall to remove everything.

.PARAMETER ApiKey
    Gemini API key. If omitted, tries existing MCP registration, then
    GEMINI_API_KEY env var, then prompts interactively.

.PARAMETER InstallDir
    Custom install location. Default: ~/tools/gemini-memory-mcp

.PARAMETER Uninstall
    Remove the MCP server, hooks, wrapper scripts, and install directory.

.PARAMETER SkipHooks
    Don't configure Claude Code hooks (PreCompact/SessionStart).

.PARAMETER SkipTest
    Don't run verification test after install.

.EXAMPLE
    ./install.ps1 -ApiKey "AIza..."
    ./install.ps1                      # detects existing key
    ./install.ps1 -Uninstall
#>

param(
    [string]$ApiKey,
    [string]$InstallDir,
    [switch]$Uninstall,
    [switch]$SkipHooks,
    [switch]$SkipTest
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$IsWindows_ = ($env:OS -eq "Windows_NT") -or ($PSVersionTable.PSVersion.Major -le 5)

if ($InstallDir) {
    $InstDir = $InstallDir
} elseif ($IsWindows_) {
    $InstDir = Join-Path $env:USERPROFILE "tools\gemini-memory-mcp"
} else {
    $InstDir = Join-Path $HOME "tools/gemini-memory-mcp"
}

$SettingsPath = if ($IsWindows_) {
    Join-Path $env:USERPROFILE ".claude\settings.json"
} else {
    Join-Path $HOME ".claude/settings.json"
}

$CoreFiles = @(
    "server.py",
    "gmem.py",
    "compact-hook.py",
    "post-compact-hook.py",
    "requirements.txt",
    "CLAUDE-MD-SNIPPET.md"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

function Write-Step($msg)  { Write-Host "  [+] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)    { Write-Host "  [OK] $msg" -ForegroundColor Green }
function Write-Warn($msg)  { Write-Host "  [!] $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  [X] $msg" -ForegroundColor Red }

function Test-Command($cmd) {
    $null = Get-Command $cmd -ErrorAction SilentlyContinue
    return $?
}

function Get-PythonCmd {
    # Try python3 first (Linux/macOS), then python (Windows)
    foreach ($cmd in @("python3", "python")) {
        if (Test-Command $cmd) {
            $ver = & $cmd --version 2>&1
            if ($ver -match "Python (\d+)\.(\d+)") {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -ge 3 -and $minor -ge 10) {
                    return $cmd
                }
            }
        }
    }
    return $null
}

function Get-VenvPython {
    if ($IsWindows_) {
        return Join-Path $InstDir ".venv\Scripts\python.exe"
    } else {
        return Join-Path $InstDir ".venv/bin/python"
    }
}

function Get-ExistingApiKey {
    # Try to extract API key from existing MCP registration
    try {
        $out = & claude mcp get gemini-memory -s user 2>&1
        $text = $out -join "`n"
        # Match GEMINI_API_KEY followed by separator and the key value
        if ($text -match 'GEMINI_API_KEY["''=:\s]+([A-Za-z0-9_-]{20,})') {
            return $Matches[1]
        }
    } catch {}

    # Try wrapper scripts (Windows: read from .bat files that set the key)
    if ($IsWindows_) {
        $gmemBatPath = Join-Path (Split-Path $InstDir) "gmem.bat"
        if (Test-Path $gmemBatPath) {
            $batContent = Get-Content $gmemBatPath -Raw
            if ($batContent -match "GEMINI_API_KEY=([A-Za-z0-9_-]{20,})") {
                return $Matches[1]
            }
        }
    }

    # Try environment variable
    if ($env:GEMINI_API_KEY) {
        return $env:GEMINI_API_KEY
    }

    return $null
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------

if ($Uninstall) {
    Write-Host "`n  Gemini Memory MCP - Uninstall" -ForegroundColor Magenta
    Write-Host "  $('=' * 40)" -ForegroundColor DarkGray

    # Remove MCP registration
    Write-Step "Removing MCP server registration..."
    try {
        & claude mcp remove gemini-memory 2>&1 | Out-Null
        Write-Ok "MCP registration removed"
    } catch {
        Write-Warn "MCP registration not found (already removed)"
    }

    # Remove hooks from settings.json
    if (Test-Path $SettingsPath) {
        Write-Step "Removing hooks from settings.json..."
        try {
            $settings = Get-Content $SettingsPath -Raw | ConvertFrom-Json
            $changed = $false

            if ($settings.hooks) {
                # Remove PreCompact entries that reference gmem
                if ($settings.hooks.PreCompact) {
                    $filtered = @($settings.hooks.PreCompact | Where-Object {
                        $hookJson = $_ | ConvertTo-Json -Depth 10
                        $hookJson -notmatch "gmem"
                    })
                    if ($filtered.Count -eq 0) {
                        $settings.hooks.PSObject.Properties.Remove("PreCompact")
                    } else {
                        $settings.hooks.PreCompact = $filtered
                    }
                    $changed = $true
                }

                # Remove SessionStart entries that reference gmem
                if ($settings.hooks.SessionStart) {
                    $filtered = @($settings.hooks.SessionStart | Where-Object {
                        $hookJson = $_ | ConvertTo-Json -Depth 10
                        $hookJson -notmatch "gmem"
                    })
                    if ($filtered.Count -eq 0) {
                        $settings.hooks.PSObject.Properties.Remove("SessionStart")
                    } else {
                        $settings.hooks.SessionStart = $filtered
                    }
                    $changed = $true
                }

                # Remove hooks object if empty
                $hookProps = @($settings.hooks.PSObject.Properties)
                if ($hookProps.Count -eq 0) {
                    $settings.PSObject.Properties.Remove("hooks")
                }
            }

            if ($changed) {
                $settings | ConvertTo-Json -Depth 10 | Set-Content $SettingsPath -Encoding UTF8
                Write-Ok "Hooks removed from settings.json"
            } else {
                Write-Warn "No gmem hooks found in settings.json"
            }
        } catch {
            Write-Warn "Could not update settings.json: $_"
        }
    }

    # Remove wrapper scripts
    Write-Step "Removing wrapper scripts..."
    if ($IsWindows_) {
        $wrapperDir = Split-Path $InstDir
        foreach ($name in @("gmem.bat", "gmem-precompact.bat", "gmem-postcompact.bat")) {
            $path = Join-Path $wrapperDir $name
            if (Test-Path $path) { Remove-Item $path -Force; Write-Ok "Removed $path" }
        }
    } else {
        foreach ($dir in @("$HOME/.local/bin", "/usr/local/bin")) {
            foreach ($name in @("gmem", "gmem-precompact", "gmem-postcompact")) {
                $path = Join-Path $dir $name
                if (Test-Path $path) { Remove-Item $path -Force; Write-Ok "Removed $path" }
            }
        }
    }

    # Remove install directory
    if (Test-Path $InstDir) {
        Write-Step "Removing install directory..."
        Remove-Item $InstDir -Recurse -Force
        Write-Ok "Removed $InstDir"
    }

    Write-Host "`n  Uninstall complete.`n" -ForegroundColor Green
    exit 0
}

# ---------------------------------------------------------------------------
# Install / Update
# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "  Gemini Memory MCP - Installer" -ForegroundColor Magenta
Write-Host "  $('=' * 40)" -ForegroundColor DarkGray
Write-Host ""

# --- Step 1: Prerequisites ---

Write-Step "Checking prerequisites..."

$pythonCmd = Get-PythonCmd
if (-not $pythonCmd) {
    Write-Err "Python 3.10+ is required but not found."
    Write-Host "    Install from https://www.python.org/downloads/" -ForegroundColor DarkGray
    exit 1
}
$pyVer = & $pythonCmd --version 2>&1
Write-Ok "Found $pyVer"

if (-not (Test-Command "claude")) {
    Write-Err "'claude' CLI not found. Install Claude Code first."
    Write-Host "    See https://docs.anthropic.com/en/docs/claude-code" -ForegroundColor DarkGray
    exit 1
}
Write-Ok "Found claude CLI"

# --- Step 2: Resolve API Key ---

Write-Step "Resolving API key..."

if (-not $ApiKey) {
    $ApiKey = Get-ExistingApiKey
}

if (-not $ApiKey) {
    Write-Host ""
    $ApiKey = Read-Host "    Enter your Gemini API key (from https://aistudio.google.com/apikey)"
    Write-Host ""
}

if (-not $ApiKey -or $ApiKey.Length -lt 10) {
    Write-Err "No valid API key provided. Use -ApiKey parameter or set GEMINI_API_KEY."
    exit 1
}
Write-Ok "API key resolved ($($ApiKey.Substring(0,8))...)"

# --- Step 3: Copy core files ---

Write-Step "Deploying files to $InstDir..."

if (-not (Test-Path $InstDir)) {
    New-Item -ItemType Directory -Path $InstDir -Force | Out-Null
}

foreach ($file in $CoreFiles) {
    $src = Join-Path $RepoDir $file
    if (-not (Test-Path $src)) {
        Write-Warn "Source file not found: $file (skipping)"
        continue
    }
    Copy-Item $src (Join-Path $InstDir $file) -Force
}
Write-Ok "Core files copied"

# --- Step 4: Create/update venv ---

Write-Step "Setting up Python virtual environment..."

$venvDir = Join-Path $InstDir ".venv"
$venvPython = Get-VenvPython

if (-not (Test-Path $venvPython)) {
    & $pythonCmd -m venv $venvDir
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment"
        exit 1
    }
    Write-Ok "Created venv"
} else {
    Write-Ok "Venv already exists"
}

Write-Step "Installing/updating dependencies..."
$reqFile = Join-Path $InstDir "requirements.txt"
& $venvPython -m pip install --upgrade pip --quiet 2>&1 | Out-Null
& $venvPython -m pip install -r $reqFile --quiet 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Err "Failed to install dependencies"
    exit 1
}
Write-Ok "Dependencies installed"

# --- Step 5: Create wrapper scripts ---

Write-Step "Creating wrapper scripts..."

$serverPy     = Join-Path $InstDir "server.py"
$gmemPy       = Join-Path $InstDir "gmem.py"
$compactPy    = Join-Path $InstDir "compact-hook.py"
$postCompactPy = Join-Path $InstDir "post-compact-hook.py"

if ($IsWindows_) {
    $wrapperDir = Split-Path $InstDir  # ~/tools/

    # gmem.bat
    $gmemBat = Join-Path $wrapperDir "gmem.bat"
    $batLines = @(
        "@echo off",
        "set GEMINI_API_KEY=$ApiKey",
        "set PYTHONIOENCODING=utf-8",
        "`"$venvPython`" `"$gmemPy`" %*"
    )
    $batLines -join "`r`n" | Set-Content $gmemBat -Encoding ASCII -NoNewline
    Write-Ok "Created $gmemBat"

    # gmem-precompact.bat
    $precompactBat = Join-Path $wrapperDir "gmem-precompact.bat"
    $batLines = @(
        "@echo off",
        "set GEMINI_API_KEY=$ApiKey",
        "set PYTHONIOENCODING=utf-8",
        "`"$venvPython`" `"$compactPy`""
    )
    $batLines -join "`r`n" | Set-Content $precompactBat -Encoding ASCII -NoNewline
    Write-Ok "Created $precompactBat"

    # gmem-postcompact.bat
    $postcompactBat = Join-Path $wrapperDir "gmem-postcompact.bat"
    $batLines = @(
        "@echo off",
        "set GEMINI_API_KEY=$ApiKey",
        "set PYTHONIOENCODING=utf-8",
        "`"$venvPython`" `"$postCompactPy`""
    )
    $batLines -join "`r`n" | Set-Content $postcompactBat -Encoding ASCII -NoNewline
    Write-Ok "Created $postcompactBat"

} else {
    # Determine bin dir
    $binDir = "$HOME/.local/bin"
    if (-not (Test-Path $binDir)) {
        New-Item -ItemType Directory -Path $binDir -Force | Out-Null
    }

    # gmem
    $gmemSh = Join-Path $binDir "gmem"
    $shLines = @(
        "#!/bin/bash",
        "export GEMINI_API_KEY=`"$ApiKey`"",
        "export PYTHONIOENCODING=utf-8",
        "exec `"$venvPython`" `"$gmemPy`" `"`$@`""
    )
    $shLines -join "`n" | Set-Content $gmemSh -Encoding UTF8 -NoNewline
    & chmod +x $gmemSh
    Write-Ok "Created $gmemSh"

    # gmem-precompact
    $precompactSh = Join-Path $binDir "gmem-precompact"
    $shLines = @(
        "#!/bin/bash",
        "export GEMINI_API_KEY=`"$ApiKey`"",
        "export PYTHONIOENCODING=utf-8",
        "exec `"$venvPython`" `"$compactPy`""
    )
    $shLines -join "`n" | Set-Content $precompactSh -Encoding UTF8 -NoNewline
    & chmod +x $precompactSh
    Write-Ok "Created $precompactSh"

    # gmem-postcompact
    $postcompactSh = Join-Path $binDir "gmem-postcompact"
    $shLines = @(
        "#!/bin/bash",
        "export GEMINI_API_KEY=`"$ApiKey`"",
        "export PYTHONIOENCODING=utf-8",
        "exec `"$venvPython`" `"$postCompactPy`""
    )
    $shLines -join "`n" | Set-Content $postcompactSh -Encoding UTF8 -NoNewline
    & chmod +x $postcompactSh
    Write-Ok "Created $postcompactSh"
}

# --- Step 6: Register MCP server ---

Write-Step "Registering MCP server with Claude Code..."

# Remove existing registration first (ignore errors)
try { & claude mcp remove gemini-memory -s user 2>&1 | Out-Null } catch {}

# Use forward slashes for the path (works cross-platform)
$serverPyPath = $serverPy.Replace("\", "/")
$venvPythonPath = $venvPython.Replace("\", "/")

& claude mcp add gemini-memory `
    -s user `
    -e "GEMINI_API_KEY=$ApiKey" `
    -- $venvPythonPath $serverPyPath `
    2>&1 | Out-Null

if ($LASTEXITCODE -ne 0) {
    Write-Warn "MCP registration may need manual setup (see README)"
} else {
    Write-Ok "MCP server registered as 'gemini-memory'"
}

# --- Step 7: Configure hooks ---

if (-not $SkipHooks) {
    Write-Step "Configuring Claude Code hooks..."

    # Ensure settings dir exists
    $settingsDir = Split-Path $SettingsPath
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Path $settingsDir -Force | Out-Null
    }

    # Load or create settings
    if (Test-Path $SettingsPath) {
        $settingsText = Get-Content $SettingsPath -Raw -Encoding UTF8
        $settings = $settingsText | ConvertFrom-Json
    } else {
        $settings = [PSCustomObject]@{}
    }

    # Ensure hooks object exists
    if (-not $settings.hooks) {
        $settings | Add-Member -NotePropertyName "hooks" -NotePropertyValue ([PSCustomObject]@{})
    }

    # Build hook command paths (forward slashes for cross-platform)
    if ($IsWindows_) {
        $precompactCmd = ($precompactBat.Replace("\", "/"))
        $postcompactCmd = ($postcompactBat.Replace("\", "/"))
    } else {
        $precompactCmd = $precompactSh
        $postcompactCmd = $postcompactSh
    }

    # PreCompact hook
    $preCompactHook = [PSCustomObject]@{
        hooks = @(
            [PSCustomObject]@{
                type = "command"
                command = $precompactCmd
                timeout = 120
                statusMessage = "Saving context to Gemini Memory..."
            }
        )
    }

    # SessionStart hook (matcher: compact - runs only after compaction)
    $sessionStartHook = [PSCustomObject]@{
        matcher = "compact"
        hooks = @(
            [PSCustomObject]@{
                type = "command"
                command = $postcompactCmd
                timeout = 60
                statusMessage = "Restoring context from Gemini Memory..."
            }
        )
    }

    # Replace gmem-related hooks, preserve others
    # PreCompact
    if ($settings.hooks.PreCompact) {
        $otherHooks = @($settings.hooks.PreCompact | Where-Object {
            $json = $_ | ConvertTo-Json -Depth 10
            $json -notmatch "gmem"
        })
        $settings.hooks.PreCompact = @($otherHooks) + @($preCompactHook)
    } else {
        $settings.hooks | Add-Member -NotePropertyName "PreCompact" -NotePropertyValue @($preCompactHook) -Force
    }

    # SessionStart
    if ($settings.hooks.SessionStart) {
        $otherHooks = @($settings.hooks.SessionStart | Where-Object {
            $json = $_ | ConvertTo-Json -Depth 10
            $json -notmatch "gmem"
        })
        $settings.hooks.SessionStart = @($otherHooks) + @($sessionStartHook)
    } else {
        $settings.hooks | Add-Member -NotePropertyName "SessionStart" -NotePropertyValue @($sessionStartHook) -Force
    }

    # Write back
    $settings | ConvertTo-Json -Depth 10 | Set-Content $SettingsPath -Encoding UTF8
    Write-Ok "Hooks configured in $SettingsPath"
}

# --- Step 8: Verification ---

if (-not $SkipTest) {
    Write-Step "Running quick verification..."
    try {
        $testResult = & $venvPython -c "from google import genai; from mcp.server.fastmcp import FastMCP; print('OK')" 2>&1
        if ($testResult -match "OK") {
            Write-Ok "Python imports verified"
        } else {
            Write-Warn "Import check returned: $testResult"
        }
    } catch {
        Write-Warn "Verification failed: $_"
    }
}

# --- Done ---

Write-Host ""
Write-Host "  Installation complete!" -ForegroundColor Green
Write-Host "  $('=' * 40)" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  MCP Server: gemini-memory (registered with Claude Code)" -ForegroundColor White
Write-Host "  Install dir: $InstDir" -ForegroundColor White
Write-Host "  CLI tool:    gmem --help" -ForegroundColor White
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor Yellow
Write-Host "    gmem stores               # list memory stores" -ForegroundColor DarkGray
Write-Host "    gmem init my-project      # create a project store" -ForegroundColor DarkGray
Write-Host "    gmem context my-project 'architecture'  # retrieve context" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  For CLAUDE.md integration, see CLAUDE-MD-SNIPPET.md" -ForegroundColor DarkGray
Write-Host ""
