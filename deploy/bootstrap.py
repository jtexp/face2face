#!/usr/bin/env python3
"""Bootstrap installer for face2face.

Self-contained script using only the Python standard library.
Downloads and installs face2face into an isolated venv with a
PATH-accessible command.

Usage:
    python bootstrap.py                    # install from default branch
    python bootstrap.py --branch main      # install from a specific branch
    python bootstrap.py --update           # update existing installation
    python bootstrap.py --uninstall        # remove installation
    python bootstrap.py --local ./face2face.zip  # install from local path
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

GITHUB_REPO = "jtexp/face2face"
DEFAULT_BRANCH = "claude/webcam-screen-http-proxy-jpxC1"
MIN_PYTHON = (3, 10)

DEFAULT_CONFIG = """\
# face2face configuration
# See README.md for all available options

[visual]
grid_cols = 16
grid_rows = 16
bits_per_cell = 2       # 4 colors - most robust for real hardware
cell_px = 28            # large cells for reliable decode

[renderer]
fullscreen = false
frame_hold_ms = 800     # hold each frame long enough for the webcam to capture
blank_hold_ms = 200     # sync gap between frames
display_padding = 80    # isolate frame from window title bar

[camera]
camera_index = 0        # change if using a USB webcam (try 0, 1, 2)
camera_width = 1280
camera_height = 720
camera_fps = 30

[error_correction]
ecc_nsym = 20           # Reed-Solomon redundancy (handles ~10 misread cells)

[flow_control]
ack_timeout = 5.0       # seconds to wait for ACK before retransmitting
max_retries = 10

[proxy]
proxy_host = "127.0.0.1"
proxy_port = 8080
proxy_timeout = 120.0   # seconds to wait for a full HTTP response
"""


def get_venv_dir() -> Path:
    """Platform-appropriate venv location."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "face2face" / ".venv"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "face2face" / ".venv"


def get_config_dir() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "face2face"
    else:
        return Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "face2face"


def get_pip(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "pip.exe"
    return venv / "bin" / "pip"


def get_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def get_entry_point(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "face2face.exe"
    return venv / "bin" / "face2face"


def check_python():
    if sys.version_info < MIN_PYTHON:
        print(f"Error: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
              f"found {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} OK")


def create_venv(venv_dir: Path):
    print(f"Creating venv at {venv_dir} ...")
    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([sys.executable, "-m", "venv", str(venv_dir)])
    # Upgrade pip — must use `python -m pip` instead of calling pip.exe
    # directly, because pip 25+ on Windows refuses to overwrite its own
    # running executable.
    subprocess.check_call(
        [str(get_python(venv_dir)), "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
    )
    print("  venv created.")


def install_package(venv_dir: Path, source: str):
    print(f"Installing face2face from {source} ...")
    subprocess.check_call([str(get_python(venv_dir)), "-m", "pip", "install", "--upgrade", source])
    print("  Installed.")


def deploy_config():
    config_dir = get_config_dir()
    config_file = config_dir / "config.toml"
    if config_file.exists():
        print(f"Config already exists at {config_file}, skipping.")
        return
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file.write_text(DEFAULT_CONFIG)
    print(f"Config written to {config_file}")


def install_shim_unix(venv_dir: Path):
    """Create a symlink in ~/.local/bin/ pointing to the venv entry point."""
    bin_dir = Path.home() / ".local" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    link = bin_dir / "face2face"
    target = get_entry_point(venv_dir)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)
    print(f"Symlink: {link} -> {target}")
    # Check if ~/.local/bin is on PATH
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    if str(bin_dir) not in path_dirs:
        print(f"\n  Note: {bin_dir} is not on your PATH.")
        shell = os.environ.get("SHELL", "")
        if "zsh" in shell:
            print(f'  Add it:  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.zshrc')
        else:
            print(f'  Add it:  echo \'export PATH="$HOME/.local/bin:$PATH"\' >> ~/.bashrc')


def install_shim_windows(venv_dir: Path):
    """Create a .cmd shim and add its directory to user PATH via the registry."""
    shim_dir = venv_dir.parent / "bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    shim = shim_dir / "face2face.cmd"
    target = get_entry_point(venv_dir)
    shim.write_text(f'@echo off\r\n"{target}" %*\r\n')
    print(f"Shim: {shim}")

    # Add to user PATH via registry
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Environment",
            0,
            winreg.KEY_READ | winreg.KEY_WRITE,
        )
        try:
            current_path, _ = winreg.QueryValueEx(key, "Path")
        except FileNotFoundError:
            current_path = ""
        shim_str = str(shim_dir)
        if shim_str.lower() not in current_path.lower():
            new_path = f"{current_path};{shim_str}" if current_path else shim_str
            winreg.SetValueEx(key, "Path", 0, winreg.REG_EXPAND_SZ, new_path)
            print(f"  Added {shim_dir} to user PATH (registry).")
            # Broadcast WM_SETTINGCHANGE so Explorer picks up the new PATH.
            # Running terminals still won't see it — they need to be reopened
            # or manually refresh $env:Path.
            try:
                import ctypes
                HWND_BROADCAST = 0xFFFF
                WM_SETTINGCHANGE = 0x001A
                SMTO_ABORTIFHUNG = 0x0002
                ctypes.windll.user32.SendMessageTimeoutW(
                    HWND_BROADCAST, WM_SETTINGCHANGE, 0,
                    "Environment", SMTO_ABORTIFHUNG, 5000, None,
                )
            except Exception:
                pass  # non-critical
        else:
            print(f"  {shim_dir} already on PATH.")
        winreg.CloseKey(key)
    except Exception as e:
        print(f"  Warning: Could not update PATH automatically: {e}")
        print(f"  Manually add {shim_dir} to your PATH.")


def install_shim(venv_dir: Path):
    if sys.platform == "win32":
        install_shim_windows(venv_dir)
    else:
        install_shim_unix(venv_dir)


def uninstall():
    venv_dir = get_venv_dir()
    print("Uninstalling face2face...")

    # Remove venv
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
        print(f"  Removed {venv_dir}")
    # Remove parent dir (contains bin/ shim on Windows)
    parent = venv_dir.parent
    if parent.exists() and parent.name == "face2face":
        shutil.rmtree(parent)
        print(f"  Removed {parent}")

    # Remove unix symlink
    unix_link = Path.home() / ".local" / "bin" / "face2face"
    if unix_link.is_symlink():
        unix_link.unlink()
        print(f"  Removed {unix_link}")

    config_dir = get_config_dir()
    if config_dir.exists():
        print(f"  Config preserved at {config_dir} (delete manually if desired)")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Bootstrap face2face installation")
    parser.add_argument("--branch", "-b", default=DEFAULT_BRANCH,
                        help=f"Git branch to install from (default: {DEFAULT_BRANCH})")
    parser.add_argument("--update", action="store_true",
                        help="Update existing installation (skip venv creation)")
    parser.add_argument("--uninstall", action="store_true",
                        help="Remove face2face installation")
    parser.add_argument("--local", metavar="PATH",
                        help="Install from a local directory or zip instead of GitHub")
    args = parser.parse_args()

    if args.uninstall:
        uninstall()
        return

    check_python()

    venv_dir = get_venv_dir()

    if args.update:
        if not venv_dir.exists():
            print(f"Error: No existing installation at {venv_dir}")
            print("Run without --update for first-time install.")
            sys.exit(1)
    else:
        if venv_dir.exists():
            print(f"Existing venv found at {venv_dir}, recreating...")
            shutil.rmtree(venv_dir)
        create_venv(venv_dir)

    # Determine install source
    if args.local:
        source = args.local
    else:
        source = f"https://github.com/{GITHUB_REPO}/archive/refs/heads/{args.branch}.zip"

    install_package(venv_dir, source)
    deploy_config()
    install_shim(venv_dir)

    venv_dir = get_venv_dir()
    shim_dir = venv_dir.parent / "bin"
    shim = shim_dir / "face2face.cmd"

    if sys.platform == "win32":
        print(textwrap.dedent(f"""
        ========================================
        face2face installed successfully!
        ========================================

        IMPORTANT: Open a NEW terminal window for PATH changes to take effect,
        then run:

            face2face --help

        Or use the full path right now (no new terminal needed):

            "{shim}" --help

        Start the server (on the machine with internet):
            face2face server

        Update later:
            python bootstrap.py --update

        Configuration:
            {get_config_dir() / 'config.toml'}
        """))
    else:
        print(textwrap.dedent(f"""
        ========================================
        face2face installed successfully!
        ========================================

        Test it:
            face2face --help

        Start the server (on the machine with internet):
            face2face server

        Update later:
            python bootstrap.py --update

        Configuration:
            {get_config_dir() / 'config.toml'}
        """))


if __name__ == "__main__":
    main()
