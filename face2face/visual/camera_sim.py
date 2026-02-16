"""Camera simulator: realistic degradation pipeline for testing.

Simulates what happens when a webcam photographs a screen:
  1. Screen bezel (dark border around the frame)
  2. Perspective warp (camera at an angle)
  3. Lens barrel distortion
  4. Defocus / Gaussian blur
  5. Color / white balance shift
  6. Vignetting (brightness falloff toward edges)
  7. Ambient light (additive brightness wash)
  8. Sensor noise (Gaussian)
  9. Resolution reduction (downscale to camera resolution)

Each effect can be enabled/disabled independently via CameraSimConfig.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class CameraSimConfig:
    """Configuration for camera simulation effects."""

    # Canvas: embed frame in a larger image with dark surround
    canvas_pad: int = 100          # pixels of dark surround
    canvas_color: tuple = (20, 20, 20)  # dark gray surround (BGR)

    # Perspective: camera angle in degrees (0 = head-on)
    perspective_enabled: bool = True
    perspective_x_angle: float = 8.0   # horizontal tilt degrees
    perspective_y_angle: float = 5.0   # vertical tilt degrees

    # Lens distortion
    barrel_enabled: bool = True
    barrel_k1: float = -0.05          # barrel distortion coefficient

    # Blur / defocus
    blur_enabled: bool = True
    blur_sigma: float = 0.8           # Gaussian blur sigma

    # Color / white balance shift
    color_shift_enabled: bool = True
    color_scale: tuple = (1.05, 0.97, 0.93)  # BGR multipliers (slight warm shift)
    color_offset: tuple = (3, -2, -5)          # BGR additive offset

    # Vignetting
    vignette_enabled: bool = True
    vignette_strength: float = 0.15   # 0 = none, 1 = full dark corners

    # Ambient light reflection
    ambient_enabled: bool = True
    ambient_strength: float = 0.03    # fraction of 255 added
    ambient_center: tuple = (0.3, 0.4)  # normalized (x, y) of hotspot

    # Sensor noise
    noise_enabled: bool = True
    noise_sigma: float = 4.0          # Gaussian noise std dev

    # Resolution reduction
    downscale_enabled: bool = False
    downscale_factor: float = 0.5     # scale factor

    # Random seed for reproducibility
    seed: int = 42


class CameraSimulator:
    """Simulates webcam capture of a screen display."""

    def __init__(self, config: CameraSimConfig | None = None):
        self.config = config or CameraSimConfig()
        self._rng = np.random.RandomState(self.config.seed)

    def simulate(self, frame: np.ndarray) -> np.ndarray:
        """Apply the full degradation pipeline to a rendered frame.

        Args:
            frame: BGR image as rendered by FrameEncoder (the screen content)

        Returns:
            Degraded BGR image as a webcam would capture it
        """
        cfg = self.config
        img = frame.copy()

        # 1. Embed in screen canvas (bezel / surround)
        img = self._add_canvas(img)

        # 2. Perspective warp
        if cfg.perspective_enabled:
            img = self._apply_perspective(img)

        # 3. Barrel distortion
        if cfg.barrel_enabled:
            img = self._apply_barrel_distortion(img)

        # 4. Blur / defocus
        if cfg.blur_enabled:
            img = self._apply_blur(img)

        # 5. Color shift
        if cfg.color_shift_enabled:
            img = self._apply_color_shift(img)

        # 6. Vignetting
        if cfg.vignette_enabled:
            img = self._apply_vignette(img)

        # 7. Ambient light
        if cfg.ambient_enabled:
            img = self._apply_ambient(img)

        # 8. Sensor noise
        if cfg.noise_enabled:
            img = self._apply_noise(img)

        # 9. Downscale
        if cfg.downscale_enabled:
            img = self._apply_downscale(img)

        return img

    def _add_canvas(self, frame: np.ndarray) -> np.ndarray:
        """Embed frame in a larger dark canvas (screen surround / bezel)."""
        pad = self.config.canvas_pad
        h, w = frame.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3),
                         self.config.canvas_color, dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = frame
        return canvas

    def _apply_perspective(self, img: np.ndarray) -> np.ndarray:
        """Apply perspective warp simulating a camera at an angle."""
        h, w = img.shape[:2]
        cfg = self.config

        # Convert angles to pixel offsets
        dx = int(w * np.tan(np.radians(cfg.perspective_x_angle)) * 0.1)
        dy = int(h * np.tan(np.radians(cfg.perspective_y_angle)) * 0.1)

        src = np.array([
            [0, 0],
            [w - 1, 0],
            [w - 1, h - 1],
            [0, h - 1],
        ], dtype=np.float32)

        # Shift corners to create perspective
        dst = np.array([
            [dx, dy],
            [w - 1 - dx // 2, dy // 2],
            [w - 1 - dx // 3, h - 1 - dy // 3],
            [dx // 2, h - 1 - dy],
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h),
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=self.config.canvas_color)

    def _apply_barrel_distortion(self, img: np.ndarray) -> np.ndarray:
        """Apply barrel/pincushion lens distortion."""
        h, w = img.shape[:2]
        k1 = self.config.barrel_k1

        # Camera matrix (assume center of image is optical center)
        fx = fy = max(w, h)
        cx, cy = w / 2.0, h / 2.0
        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.array([k1, 0, 0, 0, 0], dtype=np.float64)

        return cv2.undistort(img, camera_matrix, dist_coeffs)

    def _apply_blur(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur (defocus simulation)."""
        sigma = self.config.blur_sigma
        ksize = int(sigma * 4) | 1  # ensure odd
        ksize = max(ksize, 3)
        return cv2.GaussianBlur(img, (ksize, ksize), sigma)

    def _apply_color_shift(self, img: np.ndarray) -> np.ndarray:
        """Apply color/white balance shift."""
        result = img.astype(np.float64)
        scale = self.config.color_scale
        offset = self.config.color_offset
        for c in range(3):
            result[:, :, c] = result[:, :, c] * scale[c] + offset[c]
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_vignette(self, img: np.ndarray) -> np.ndarray:
        """Apply vignetting (brightness falloff from center)."""
        h, w = img.shape[:2]
        strength = self.config.vignette_strength

        # Create radial gradient
        Y, X = np.ogrid[:h, :w]
        cx, cy = w / 2.0, h / 2.0
        radius = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_radius = np.sqrt(cx ** 2 + cy ** 2)
        vignette = 1.0 - strength * (radius / max_radius) ** 2

        result = img.astype(np.float64)
        for c in range(3):
            result[:, :, c] *= vignette
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_ambient(self, img: np.ndarray) -> np.ndarray:
        """Apply ambient light reflection (bright hotspot)."""
        h, w = img.shape[:2]
        cfg = self.config
        strength = cfg.ambient_strength * 255

        # Create a soft bright spot
        cx = int(w * cfg.ambient_center[0])
        cy = int(h * cfg.ambient_center[1])
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        max_dist = np.sqrt(w ** 2 + h ** 2) * 0.5
        ambient = strength * np.exp(-(dist / max_dist) ** 2 * 3)

        result = img.astype(np.float64)
        for c in range(3):
            result[:, :, c] += ambient
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_noise(self, img: np.ndarray) -> np.ndarray:
        """Apply Gaussian sensor noise."""
        noise = self._rng.normal(0, self.config.noise_sigma, img.shape)
        result = img.astype(np.float64) + noise
        return np.clip(result, 0, 255).astype(np.uint8)

    def _apply_downscale(self, img: np.ndarray) -> np.ndarray:
        """Downscale to simulate lower camera resolution."""
        factor = self.config.downscale_factor
        h, w = img.shape[:2]
        new_w = int(w * factor)
        new_h = int(h * factor)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
