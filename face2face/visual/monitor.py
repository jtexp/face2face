"""Webcam self-monitor window for camera alignment.

Shows a live preview of the webcam feed with the detected frame quad
overlaid, so the user can aim the camera before and during operation.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

WINDOW_NAME = "face2face-monitor"


class WebcamMonitor:
    """Small resizable OpenCV window showing the live webcam feed."""

    def __init__(self, width: int = 400, height: int = 300):
        self._width = width
        self._height = height
        self._window_created = False

    def update(self, frame: np.ndarray,
               corners: Optional[np.ndarray] = None) -> None:
        """Show the webcam frame with optional quad overlay.

        Args:
            frame: BGR webcam image.
            corners: (4, 2) array of detected quad corners, or None.
        """
        if not self._window_created:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW_NAME, self._width, self._height)
            self._window_created = True

        preview = frame.copy()

        if corners is not None:
            pts = corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(preview, [pts], isClosed=True,
                          color=(0, 255, 0), thickness=2)

        # Status circle: green if corners detected, red otherwise
        h, w = preview.shape[:2]
        radius = max(8, min(w, h) // 40)
        center = (w - radius - 10, radius + 10)
        color = (0, 255, 0) if corners is not None else (0, 0, 255)
        cv2.circle(preview, center, radius, color, -1)

        display = cv2.resize(preview, (self._width, self._height))
        cv2.imshow(WINDOW_NAME, display)
        cv2.waitKey(1)

    def destroy(self) -> None:
        """Close the monitor window."""
        if self._window_created:
            cv2.destroyWindow(WINDOW_NAME)
            self._window_created = False
