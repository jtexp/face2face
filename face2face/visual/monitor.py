"""Webcam self-monitor window for camera alignment.

Shows a live preview of the webcam feed with the detected frame quad
overlaid, so the user can aim the camera before and during operation.

Supports interactive ROI (Region of Interest) selection: click and drag
to draw a rectangle around the grid area, then the capture pipeline
crops to that region before decoding.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

WINDOW_NAME = "face2face-monitor"


class WebcamMonitor:
    """Small resizable OpenCV window showing the live webcam feed."""

    def __init__(self, width: int = 400, height: int = 300):
        self._width = width
        self._height = height
        self._window_created = False
        # ROI state
        self._roi: Optional[tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
        self._drawing: bool = False
        self._draw_start: tuple[int, int] = (0, 0)
        self._draw_current: tuple[int, int] = (0, 0)
        self._frame_size: tuple[int, int] = (1, 1)  # (w, h)

    def _on_mouse(self, event: int, x: int, y: int,
                  flags: int, param: object) -> None:
        """Mouse callback for ROI selection on the monitor window."""
        # Map display coords to original frame coords
        fw, fh = self._frame_size
        fx = int(x * fw / self._width)
        fy = int(y * fh / self._height)
        fx = max(0, min(fx, fw - 1))
        fy = max(0, min(fy, fh - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drawing = True
            self._draw_start = (fx, fy)
            self._draw_current = (fx, fy)
            self._roi = None
        elif event == cv2.EVENT_MOUSEMOVE and self._drawing:
            self._draw_current = (fx, fy)
        elif event == cv2.EVENT_LBUTTONUP and self._drawing:
            self._drawing = False
            x1 = min(self._draw_start[0], fx)
            y1 = min(self._draw_start[1], fy)
            x2 = max(self._draw_start[0], fx)
            y2 = max(self._draw_start[1], fy)
            if x2 - x1 >= 50 and y2 - y1 >= 50:
                self._roi = (x1, y1, x2, y2)
                logger.info("ROI set: (%d, %d) -> (%d, %d)  [%dx%d]",
                            x1, y1, x2, y2, x2 - x1, y2 - y1)
            else:
                logger.info("ROI too small (min 50x50), ignored")
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._roi = None
            self._drawing = False
            logger.info("ROI cleared")

    @property
    def roi(self) -> Optional[tuple[int, int, int, int]]:
        """Current ROI as (x1, y1, x2, y2) in frame coordinates, or None."""
        return self._roi

    def crop_frame(self, frame: np.ndarray) -> np.ndarray:
        """Crop frame to the current ROI, or return unchanged if no ROI."""
        if self._roi is None:
            return frame
        x1, y1, x2, y2 = self._roi
        return frame[y1:y2, x1:x2]

    def update(self, frame: np.ndarray,
               corners: Optional[np.ndarray] = None) -> None:
        """Show the webcam frame with optional quad overlay.

        Args:
            frame: BGR webcam image (full frame).
            corners: (4, 2) array of detected quad corners, or None.
        """
        if not self._window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self._width, self._height)
            cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
            self._window_created = True

        h, w = frame.shape[:2]
        self._frame_size = (w, h)

        preview = frame.copy()

        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2)

        # Draw ROI rectangle (cyan)
        roi_rect = None
        if self._drawing:
            roi_rect = (
                min(self._draw_start[0], self._draw_current[0]),
                min(self._draw_start[1], self._draw_current[1]),
                max(self._draw_start[0], self._draw_current[0]),
                max(self._draw_start[1], self._draw_current[1]),
            )
        elif self._roi is not None:
            roi_rect = self._roi

        if roi_rect is not None:
            cv2.rectangle(preview,
                          (roi_rect[0], roi_rect[1]),
                          (roi_rect[2], roi_rect[3]),
                          color=(255, 255, 0), thickness=2)

        # Status circle: green if corners detected, red otherwise
        radius = max(8, min(w, h) // 40)
        center = (w - radius - 10, radius + 10)
        color = (0, 255, 0) if corners is not None else (0, 0, 255)
        cv2.circle(preview, center, radius, color, -1)

        display = cv2.resize(preview, (self._width, self._height))
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('r'), ord('R')):
            self._roi = None
            self._drawing = False
            logger.info("ROI cleared (key)")

    def destroy(self) -> None:
        """Close the monitor window."""
        if self._window_created:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_created = False
