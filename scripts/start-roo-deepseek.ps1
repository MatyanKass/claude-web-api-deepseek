[CmdletBinding()]
param(
    [string]$Workspace = "",
    [ValidateRange(1, 65535)]
    [int]$Port = 8766,
    [switch]$ServerOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$stateRoot = Join-Path $env:LOCALAPPDATA "claude-web-api\roo-xml"
$baseUrl = "http://127.0.0.1:$Port"
if ([string]::IsNullOrWhiteSpace($Workspace)) { $Workspace = $projectRoot }
if (-not (Test-Path -LiteralPath $Workspace -PathType Container)) {
    throw "Workspace does not exist: $Workspace"
}
New-Item -ItemType Directory -Path $stateRoot -Force | Out-Null

function Get-BridgeHealth {
    try { Invoke-RestMethod -Uri "$baseUrl/health" -TimeoutSec 3 }
    catch { return $null }
}

$health = Get-BridgeHealth
if ($null -eq $health -or $null -eq $health.browser) {
    # Refuse to take over a port belonging to another project.
    $listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    if ($listener) { throw "Port $Port is already used by another service. Choose another -Port." }
    $oldPort = $env:PORT
    try {
        $env:PORT = "$Port"
        $startArgs = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
            ('"{0}"' -f (Join-Path $PSScriptRoot "start.ps1")))
        $server = Start-Process powershell.exe -ArgumentList $startArgs -WindowStyle Hidden `
            -WorkingDirectory $projectRoot -PassThru `
            -RedirectStandardOutput (Join-Path $stateRoot "server-launch.stdout.log") `
            -RedirectStandardError (Join-Path $stateRoot "server-launch.stderr.log")
    }
    finally { $env:PORT = $oldPort }
    $deadline = (Get-Date).AddSeconds(120)
    do {
        Start-Sleep -Seconds 2
        $health = Get-BridgeHealth
        if ($server.HasExited) { throw "Server exited. See logs in $stateRoot." }
    } while (($null -eq $health -or $null -eq $health.browser) -and (Get-Date) -lt $deadline)
    if ($null -eq $health -or $null -eq $health.browser) {
        throw "Server did not start. See logs in $stateRoot."
    }
}

$ready = Invoke-RestMethod -Uri "$baseUrl/health/ready" -TimeoutSec 10
if ($ready.provider -ne "deepseek_web" -or -not $ready.ok) {
    throw "DeepSeek is not ready. Check $baseUrl/control/ and activate your DeepSeek profile."
}
Write-Host "API: $baseUrl/v1"
Write-Host "Model: deepseek-reasoner"
Write-Host "Provider: OpenAI Compatible; key placeholder: lm-studio"
if ($ServerOnly) { return }

# Keep the XML client isolated from the user's regular Roo installation.
$code = Join-Path $env:LOCALAPPDATA "Programs\Microsoft VS Code\bin\code.cmd"
if (-not (Test-Path -LiteralPath $code -PathType Leaf)) {
    $code = (Get-Command code.cmd -ErrorAction Stop).Source
}
$extensions = Join-Path $stateRoot "extensions"
$legacy = Join-Path $extensions "rooveterinaryinc.roo-cline-3.36.16"
if (-not (Test-Path -LiteralPath $legacy -PathType Container)) {
    throw "Roo 3.36.16 is missing in $extensions. See docs/roo-deepseek.md."
}
& (Join-Path $PSScriptRoot "repair-roo-ripgrep.ps1")
if ($Port -ne 8766) {
    Write-Host "Update the Local DeepSeek XML profile Base URL to $baseUrl/v1."
}
& $code --user-data-dir (Join-Path $stateRoot "user-data") --extensions-dir $extensions `
    --new-window --disable-workspace-trust $Workspace
if ($LASTEXITCODE -ne 0) { throw "VS Code could not open the workspace." }
