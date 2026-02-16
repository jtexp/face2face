"""Screen renderer: display encoded frames in a window using OpenCV."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

from .codec import CodecConfig, FrameEncoder, FrameHeader

log = logging.getLogger(__name__)


def _get_screen_size() -> Optional[Tuple[int, int]]:
    """Return (width, height) of the primary screen using tkinter.

    Returns None if tkinter is unavailable or detection fails.
    """
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = root.winfo_screenwidth()
        h = root.winfo_screenheight()
        root.destroy()
        return (w, h)
    except Exception:
        return None


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
            cv2.namedWindow(self.cfg.window_name, cv2.WINDOW_NORMAL)
            self._fit_window_to_screen()
        self._window_created = True

    def _fit_window_to_screen(self) -> None:
        """Resize the window to fit the screen if the image is too large."""
        pad = self.cfg.display_padding if not self.cfg.fullscreen else 0
        img_w = self.codec_cfg.image_width + 2 * pad
        img_h = self.codec_cfg.image_height + 2 * pad

        screen = _get_screen_size()
        if screen is None:
            log.debug("Could not detect screen size; skipping auto-fit")
            return

        screen_w, screen_h = screen
        max_w = int(screen_w * 0.9)
        max_h = int(screen_h * 0.9)

        if img_w <= max_w and img_h <= max_h:
            # Image fits fine — just set the window to the exact image size
            cv2.resizeWindow(self.cfg.window_name, img_w, img_h)
            return

        scale = min(max_w / img_w, max_h / img_h)
        win_w = int(img_w * scale)
        win_h = int(img_h * scale)
        log.debug("Auto-fitting window: %dx%d -> %dx%d (scale %.2f, screen %dx%d)",
                  img_w, img_h, win_w, win_h, scale, screen_w, screen_h)
        cv2.resizeWindow(self.cfg.window_name, win_w, win_h)

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

    def show_idle(self, payload: bytes, header: FrameHeader) -> None:
        """Show a pre-built frame so the grid is detectable at startup."""
        self._ensure_window()
        image = self.encoder.encode(payload, header)
        cv2.imshow(self.cfg.window_name, self._pad_image(image))
        cv2.waitKey(1)

    def destroy(self) -> None:
        """Close the display window."""
        if self._window_created:
            cv2.destroyWindow(self.cfg.window_name)
            self._window_created = False
