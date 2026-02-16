"""Screen renderer: display encoded frames in a window using OpenCV."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .codec import CodecConfig, FrameEncoder, FrameHeader


@dataclass
class RendererConfig:
    window_name: str = "face2face-tx"
    fullscreen: bool = False
    frame_hold_ms: int = 500     # how long to display each frame
    blank_hold_ms: int = 100     # blank gap between frames for sync
    show_info: bool = True       # overlay text info on the frame
    display_padding: int = 80    # black padding (px) around frame to isolate from window chrome


class ScreenRenderer:
    """Displays encoded frames on screen for webcam capture."""

    def __init__(self, codec_cfg: CodecConfig,
                 renderer_cfg: Optional[RendererConfig] = None):
        self.codec_cfg = codec_cfg
        self.cfg = renderer_cfg or RendererConfig()
        self.encoder = FrameEncoder(config=codec_cfg)
        self._window_created = False

    def _ensure_window(self) -> None:
        if self._window_created:
            return
        if self.cfg.fullscreen:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(self.cfg.window_name,
                                  cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_AUTOSIZE)
        self._window_created = True

    def _pad_image(self, image: np.ndarray) -> np.ndarray:
        """Add black padding around an image to isolate it from window chrome.

        Without padding, the window title bar can be included in the
        detected quadrilateral, shifting the grid and causing decode
        failures.
        """
        pad = self.cfg.display_padding
        if pad <= 0 or self.cfg.fullscreen:
            return image
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        if image.ndim == 3:
            padded = np.zeros((h + 2 * pad, w + 2 * pad, channels), dtype=image.dtype)
        else:
            padded = np.zeros((h + 2 * pad, w + 2 * pad), dtype=image.dtype)
        padded[pad:pad + h, pad:pad + w] = image
        return padded

    def show_frame(self, image: np.ndarray) -> None:
        """Display a pre-encoded frame image."""
        self._ensure_window()
        cv2.imshow(self.cfg.window_name, self._pad_image(image))
        cv2.waitKey(self.cfg.frame_hold_ms)

    def show_blank(self) -> None:
        """Show a blank (black) frame as a synchronization gap."""
        self._ensure_window()
        h = self.codec_cfg.image_height
        w = self.codec_cfg.image_width
        blank = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.imshow(self.cfg.window_name, self._pad_image(blank))
        cv2.waitKey(self.cfg.blank_hold_ms)

    def show_sync_pattern(self) -> None:
        """Show a distinctive sync pattern (all white) to signal frame boundary."""
        self._ensure_window()
        h = self.codec_cfg.image_height
        w = self.codec_cfg.image_width
        sync = np.full((h, w, 3), 255, dtype=np.uint8)
        cv2.imshow(self.cfg.window_name, self._pad_image(sync))
        cv2.waitKey(self.cfg.blank_hold_ms)

    def transmit_frame(self, payload: bytes, header: FrameHeader) -> None:
        """Encode and display a single data frame with sync gaps."""
        image = self.encoder.encode(payload, header)
        self.show_sync_pattern()
        self.show_frame(image)

    def transmit_frames(self, frames: list[tuple[bytes, FrameHeader]]) -> None:
        """Transmit a sequence of frames with sync gaps between them."""
        for payload, header in frames:
            self.transmit_frame(payload, header)

    def show_idle(self) -> None:
        """Show an idle pattern (e.g., alternating checkerboard)."""
        self._ensure_window()
        h = self.codec_cfg.image_height
        w = self.codec_cfg.image_width
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Checkerboard
        cell = 40
        for y in range(0, h, cell):
            for x in range(0, w, cell):
                if ((y // cell) + (x // cell)) % 2 == 0:
                    img[y:y + cell, x:x + cell] = (128, 128, 128)
        cv2.imshow(self.cfg.window_name, self._pad_image(img))
        cv2.waitKey(1)

    def destroy(self) -> None:
        """Close the display window."""
        if self._window_created:
            cv2.destroyWindow(self.cfg.window_name)
            self._window_created = False
