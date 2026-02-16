# Installation

## Local development

```bash
pip install -e ".[dev]"
```

This installs face2face in editable mode with test dependencies. Requires Python 3.10+.

## Install on a remote machine

To deploy face2face on a second machine (e.g. the server laptop), you only need Python 3.10+ -- no git required.

**macOS / Linux:**

```bash
curl -sL "https://raw.githubusercontent.com/jtexp/face2face/main/deploy/bootstrap.py" \
  -o /tmp/f2f_bootstrap.py && python3 /tmp/f2f_bootstrap.py
```

**Windows PowerShell:**

```powershell
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/jtexp/face2face/main/deploy/bootstrap.py" -OutFile "$env:TEMP\f2f_bootstrap.py"; python "$env:TEMP\f2f_bootstrap.py"
```

This creates an isolated venv, installs face2face, writes a default config, and puts `face2face` on your PATH.

### Bootstrap flags

| Flag | Description |
|------|-------------|
| `--branch <name>` | Install from a specific branch |
| `--update` | Re-install into existing venv (skip venv creation) |
| `--uninstall` | Remove the installation |
| `--local <path>` | Install from a local directory or zip |

## Updating

After the initial install:

```bash
face2face update
```

Or from your dev machine via SSH: `ssh laptop "face2face update"`

## WSL2 and hardware setup

See [REAL_TEST_GUIDE.md](REAL_TEST_GUIDE.md) for detailed WSL2 webcam passthrough, GUI window setup, and single-machine testing instructions.
