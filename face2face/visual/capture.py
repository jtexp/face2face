"""Webcam capture: grab frames from a camera using OpenCV."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CaptureConfig:
    camera_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    auto_exposure: bool = True


class WebcamCapture:
    """Captures frames from a webcam."""

    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        """Open the webcam."""
        logger.info("Opening webcam (index=%d) ...", self.config.camera_index)
        # Use DSHOW on Windows, V4L2 on Linux to skip slow backend probing
        if hasattr(cv2, "CAP_DSHOW") and __import__("sys").platform == "win32":
            backend = cv2.CAP_DSHOW
        else:
            backend = cv2.CAP_V4L2
        self._cap = cv2.VideoCapture(self.config.camera_index, backend)
        if not self._cap.isOpened():
            # Fallback: let OpenCV auto-detect
            logger.debug("Backend %s failed, falling back to auto-detect", backend)
            self._cap = cv2.VideoCapture(self.config.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.config.camera_index}")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.config.fps)
        # Report actual resolution (driver may not honor the request)
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info("Webcam ready: %dx%d @ %.0f fps", actual_w, actual_h, actual_fps)

    def read(self) -> Optional[np.ndarray]:
        """Read a single frame. Returns BGR image or None on failure."""
        if self._cap is None:
            raise RuntimeError("Camera not opened. Call open() first.")
        ret, frame = self._cap.read()
        if not ret:
            return None
        return frame

    def read_preprocessed(self) -> Optional[np.ndarray]:
        """Read a frame with brightness/contrast normalization."""
        frame = self.read()
        if frame is None:
            return None
        # Convert to LAB, normalize L channel, convert back
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def close(self) -> None:
        """Release the webcam."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "WebcamCapture":
        self.open()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
