"""Configuration management — load/save TOML config files."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from ..protocol.channel import ChannelConfig
from ..protocol.flow import FlowConfig
from ..protocol.framing import FramingConfig
from ..protocol.link import LinkConfig
from ..visual.capture import CaptureConfig
from ..visual.codec import CodecConfig
from ..visual.decoder import DecoderConfig
from ..visual.ecc import ECCConfig
from ..visual.renderer import RendererConfig

DEFAULT_CONFIG_PATH = Path("~/.config/face2face/config.toml").expanduser()


@dataclass
class AppConfig:
    """Top-level application configuration."""

    # Proxy settings
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8080
    proxy_timeout: float = 120.0

    # Visual codec
    grid_cols: int = 32
    grid_rows: int = 32
    bits_per_cell: int = 2
    cell_px: int = 20
    border_px: int = 4

    # Screen renderer
    fullscreen: bool = False
    frame_hold_ms: int = 500
    blank_hold_ms: int = 100

    # Webcam capture
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    camera_fps: int = 30

    # Error correction
    ecc_nsym: int = 20

    # Flow control
    ack_timeout: float = 5.0
    max_retries: int = 10

    # Channel
    message_timeout: float = 60.0

    # Logging
    log_level: str = "INFO"

    # Debug
    debug_capture_dir: str | None = None

    # Monitor
    enable_monitor: bool = False

    def to_codec_config(self) -> CodecConfig:
        return CodecConfig(
            grid_cols=self.grid_cols,
            grid_rows=self.grid_rows,
            bits_per_cell=self.bits_per_cell,
            cell_px=self.cell_px,
            border_px=self.border_px,
        )

    def to_renderer_config(self) -> RendererConfig:
        return RendererConfig(
            fullscreen=self.fullscreen,
            frame_hold_ms=self.frame_hold_ms,
            blank_hold_ms=self.blank_hold_ms,
        )

    def to_capture_config(self) -> CaptureConfig:
        return CaptureConfig(
            camera_index=self.camera_index,
            width=self.camera_width,
            height=self.camera_height,
            fps=self.camera_fps,
        )

    def to_ecc_config(self) -> ECCConfig:
        return ECCConfig(nsym=self.ecc_nsym)

    def to_framing_config(self) -> FramingConfig:
        codec = self.to_codec_config()
        return FramingConfig(
            max_payload_per_frame=codec.payload_bytes,
            ecc_config=self.to_ecc_config(),
        )

    def to_flow_config(self) -> FlowConfig:
        return FlowConfig(
            ack_timeout=self.ack_timeout,
            max_retries=self.max_retries,
        )

    def to_link_config(self) -> LinkConfig:
        return LinkConfig(
            codec=self.to_codec_config(),
            renderer=self.to_renderer_config(),
            capture=self.to_capture_config(),
            decoder=DecoderConfig(debug_dir=self.debug_capture_dir),
            framing=self.to_framing_config(),
            flow=self.to_flow_config(),
            enable_monitor=self.enable_monitor,
        )

    def to_channel_config(self) -> ChannelConfig:
        return ChannelConfig(
            tx_link=self.to_link_config(),
            rx_link=self.to_link_config(),
            message_timeout=self.message_timeout,
        )


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load configuration from a TOML file.

    Falls back to defaults if the file doesn't exist.
    """
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)

    config = AppConfig()

    if not path.exists():
        return config

    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef]

    with open(path, "rb") as f:
        data = tomllib.load(f)

    # Flatten nested sections
    flat = _flatten_toml(data)

    for fld in fields(AppConfig):
        if fld.name in flat:
            setattr(config, fld.name, fld.type(flat[fld.name])
                    if fld.type in (int, float, str, bool) else flat[fld.name])

    return config


def save_config(config: AppConfig, path: Path | str | None = None) -> None:
    """Save configuration to a TOML file."""
    if path is None:
        path = DEFAULT_CONFIG_PATH
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# face2face configuration",
        "",
        "[proxy]",
        f'host = "{config.proxy_host}"',
        f"port = {config.proxy_port}",
        f"timeout = {config.proxy_timeout}",
        "",
        "[visual]",
        f"grid_cols = {config.grid_cols}",
        f"grid_rows = {config.grid_rows}",
        f"bits_per_cell = {config.bits_per_cell}",
        f"cell_px = {config.cell_px}",
        f"border_px = {config.border_px}",
        "",
        "[renderer]",
        f"fullscreen = {'true' if config.fullscreen else 'false'}",
        f"frame_hold_ms = {config.frame_hold_ms}",
        f"blank_hold_ms = {config.blank_hold_ms}",
        "",
        "[camera]",
        f"camera_index = {config.camera_index}",
        f"camera_width = {config.camera_width}",
        f"camera_height = {config.camera_height}",
        f"camera_fps = {config.camera_fps}",
        "",
        "[ecc]",
        f"ecc_nsym = {config.ecc_nsym}",
        "",
        "[flow]",
        f"ack_timeout = {config.ack_timeout}",
        f"max_retries = {config.max_retries}",
        "",
        "[channel]",
        f"message_timeout = {config.message_timeout}",
        "",
        "[logging]",
        f'log_level = "{config.log_level}"',
        "",
    ]

    with open(path, "w") as f:
        f.write("\n".join(lines))


def _flatten_toml(data: dict, prefix: str = "") -> dict:
    """Flatten nested TOML dict to a flat dict."""
    result = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result.update(_flatten_toml(value, f"{prefix}{key}_"))
        else:
            result[key] = value
    return result
