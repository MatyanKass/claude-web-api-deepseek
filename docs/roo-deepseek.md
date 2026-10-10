# Roo Code with DeepSeek Web on localhost

This computer is configured with a separate VS Code user-data directory and
Roo Code **3.36.16**, using the **XML** tool protocol. Claude login is not required.
Run `start-roo-deepseek.cmd` from the project directory. It starts the bridge if
needed and opens the configured Roo window. An existing service using the port
is left running and reported as a conflict.

Connection settings for the `Local DeepSeek XML` profile:

| Setting | Value |
| --- | --- |
| API Provider | OpenAI Compatible (`openai`) |
| Base URL | `http://127.0.0.1:8766/v1` |
| Model | `deepseek-reasoner` |
| API key | `lm-studio` (a client placeholder, not an authentication secret) |
| Tool protocol | XML |
| Streaming | Enabled |
| R1 message format | Enabled |
| Include max tokens | Disabled |

The bridge binds to loopback. Port 8766 is used because another local project
currently uses 8765. The control panel is `http://127.0.0.1:8766/control/`.
The normal `start.cmd` still uses the default port 8765; use the Roo launcher
for this configuration.

## Why a separate Roo version

DeepSeek Web exposes a text stream, without a verified native function-calling
channel. The bridge rejects requests containing native tools. Roo 3.36.16 reads
XML tool instructions in its own agent loop, then executes approved tools in
VS Code. The bridge transports the model's real response without synthesizing
tool calls. Newer Roo versions require native tools, so the ordinary Roo profile
cannot run an agent against this backend.

The isolated client state is stored in
`%LOCALAPPDATA%\claude-web-api\roo-xml`. Extension auto-updates are disabled
there to preserve XML support. The user's regular VS Code and Roo installation
remain separate. File changes and commands require confirmation; the connection
test temporarily permits reads inside the workspace and restores confirmations
after completion.

The launcher also adds VS Code's newer `ripgrep-universal` directory to this
isolated extension's binary resolver. The original extension bundle is backed
up under `roo-xml\compat`. This small compatibility patch does not change tool
behavior or the regular VS Code installation.
The isolated window opens the explicitly selected workspace with VS Code's
workspace-trust gate disabled, so Roo can activate. Roo's own tool approval
prompts remain enabled.

## Reinstalling the isolated client

Use the installed VS Code CLI (`code.cmd`) and these PowerShell commands:

```powershell
$rooState = Join-Path $env:LOCALAPPDATA 'claude-web-api\roo-xml'
code.cmd --user-data-dir "$rooState\user-data" --extensions-dir "$rooState\extensions" --install-extension RooVeterinaryInc.roo-cline@3.36.16 --force
code.cmd --user-data-dir "$rooState\user-data" --extensions-dir "$rooState\extensions" --new-window 'D:\Projects\claude-web-api'
```

Disable extension auto-updates in this window and create the profile with the
settings above if its saved state was removed. If the DeepSeek session expires,
log into **DeepSeek**, then activate its profile in the control panel. A Claude
session is not used for this setup.

To open a different project:

```powershell
.\scripts\start-roo-deepseek.ps1 -Workspace 'D:\Projects\YourProject'
```
