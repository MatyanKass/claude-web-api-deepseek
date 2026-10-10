# Roo 3.36.16 predates VS Code's ripgrep-universal directory layout.
$ErrorActionPreference = "Stop"
$rooState = Join-Path $env:LOCALAPPDATA "claude-web-api\roo-xml"
$extensionFile = Join-Path $rooState "extensions\rooveterinaryinc.roo-cline-3.36.16\dist\extension.js"
if (-not (Test-Path -LiteralPath $extensionFile -PathType Leaf)) {
    throw "Isolated Roo 3.36.16 is not installed."
}
$source = [System.IO.File]::ReadAllText($extensionFile)
$old = '||await e("node_modules.asar.unpacked/@vscode/ripgrep/bin/")}'
$new = '||await e("node_modules.asar.unpacked/@vscode/ripgrep/bin/")||await e("node_modules.asar.unpacked/@vscode/ripgrep-universal/bin/"+process.platform+"-"+process.arch+"/")||await e("node_modules/@vscode/ripgrep-universal/bin/"+process.platform+"-"+process.arch+"/")}'
if ($source.Contains($new)) { return }
if (($source.Split(@($old), [System.StringSplitOptions]::None).Count - 1) -ne 1) {
    throw "Unknown Roo ripgrep resolver. Refusing to modify this extension."
}
$backupDir = Join-Path $rooState "compat"
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
$backup = Join-Path $backupDir "roo-3.36.16-extension.js.original"
if (-not (Test-Path -LiteralPath $backup)) {
    Copy-Item -LiteralPath $extensionFile -Destination $backup
}
[System.IO.File]::WriteAllText($extensionFile, $source.Replace($old, $new), (New-Object System.Text.UTF8Encoding($false)))
Write-Host "Added current VS Code ripgrep path to isolated Roo. Original saved in $backupDir."
