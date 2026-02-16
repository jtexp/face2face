"""Frame decoder: detect and decode a visual frame from a camera image.

Steps:
1. Find the 4 corner alignment markers in the camera image
2. Apply perspective transform to extract a clean rectangular grid
3. Calibrate colors using known marker cells (adaptive palette)
4. Sample each cell to determine its color/symbol
5. Pass the symbol grid to FrameDecoder.decode_grid()
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

from .codec import (
    CodecConfig,
    FrameDecoder as GridDecoder,
    FrameHeader,
    _marker_pattern,
)


@dataclass
class DecoderConfig:
    min_contour_area: int = 500
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    sample_margin: float = 0.25  # fraction of cell to skip at edges when sampling
    adaptive_palette: bool = True  # calibrate palette from marker cells
    debug_dir: Optional[str] = None  # save debug frames to this directory


class ImageFrameDecoder:
    """Detects and decodes visual frames from camera images."""

    def __init__(self, codec_cfg: CodecConfig,
                 decoder_cfg: Optional[DecoderConfig] = None):
        self.codec_cfg = codec_cfg
        self.cfg = decoder_cfg or DecoderConfig()
        self.grid_decoder = GridDecoder(config=codec_cfg)
        self._last_confidence: float = 0.0
        self._last_corners: Optional[np.ndarray] = None
        self._debug_seq: int = 0
        self._debug_last_save: float = 0.0

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    @property
    def last_corners(self) -> Optional[np.ndarray]:
        return self._last_corners

    def decode_image(self, image: np.ndarray) -> tuple[Optional[FrameHeader], Optional[bytes]]:
        """Attempt to decode a frame from a camera image.

        Returns (header, payload) or (None, None) if no valid frame found.
        """
        corners = self._find_corners(image)
        self._last_corners = corners
        if corners is None:
            self._last_confidence = 0.0
            return None, None

        warped = self._perspective_transform(image, corners)
        if warped is None:
            self._last_confidence = 0.0
            return None, None

        # Calibrate palette from marker cells if enabled
        if self.cfg.adaptive_palette:
            palette = self._calibrate_palette(warped)
        else:
            palette = self.codec_cfg.palette

        grid = self._sample_grid(warped, palette)
        self._last_confidence = 1.0

        # Debug capture: save raw, warped, and grid overlay frames
        if self.cfg.debug_dir is not None:
            now = time.monotonic()
            if now - self._debug_last_save >= 0.5:  # max 2 saves/sec
                self._debug_last_save = now
                self._save_debug_frames(image, warped, grid, palette)

        return self.grid_decoder.decode_grid(grid)

    def _save_debug_frames(self, raw: np.ndarray, warped: np.ndarray,
                           grid: np.ndarray, palette: np.ndarray) -> None:
        """Save debug frames to disk."""
        debug_dir = Path(self.cfg.debug_dir)
        seq = self._debug_seq
        self._debug_seq += 1

        try:
            prefix = str(debug_dir / f"{seq:04d}")
            cv2.imwrite(f"{prefix}_raw.png", raw)
            cv2.imwrite(f"{prefix}_warped.png", warped)
            grid_img = self._draw_grid_overlay(warped, grid, palette)
            cv2.imwrite(f"{prefix}_grid.png", grid_img)
            logger.debug("Saved debug frames %s_*.png", prefix)
        except Exception:
            logger.warning("Failed to save debug frame %04d", seq, exc_info=True)

    def _draw_grid_overlay(self, warped: np.ndarray, grid: np.ndarray,
                           palette: np.ndarray) -> np.ndarray:
        """Draw cell boundaries and decoded colors on a copy of the warped frame."""
        overlay = warped.copy()
        cfg = self.codec_cfg
        border_offset = cfg.border_px * cfg.cell_px

        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                y = border_offset + r * cfg.cell_px
                x = border_offset + c * cfg.cell_px
                sym = int(grid[r, c])
                color = tuple(int(v) for v in palette[sym])

                # Fill cell center with a semi-transparent decoded color
                cell_region = overlay[y:y + cfg.cell_px, x:x + cfg.cell_px]
                if cell_region.size > 0:
                    tinted = cv2.addWeighted(
                        cell_region, 0.5,
                        np.full_like(cell_region, color), 0.5, 0)
                    overlay[y:y + cfg.cell_px, x:x + cfg.cell_px] = tinted

                # Draw cell boundary
                cv2.rectangle(overlay, (x, y),
                              (x + cfg.cell_px - 1, y + cfg.cell_px - 1),
                              (0, 255, 0), 1)

        return overlay

    def _find_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corner alignment markers.

        Tries multiple threshold levels to handle varying brightness
        and color shifts. Returns array of shape (4, 2) with corner
        coordinates in order:
        [top-left, top-right, bottom-right, bottom-left]
        or None if not found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Try multiple thresholds to handle color shift / brightness variation
        for thresh_val in [200, 180, 160, 140]:
            _, thresh = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            corners = self._find_quad_in_thresh(thresh)
            if corners is not None:
                return corners

        # Try adaptive threshold as last resort
        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, self.cfg.adaptive_block_size, self.cfg.adaptive_c)
        return self._find_quad_in_thresh(adaptive)

    def _find_quad_in_thresh(self, thresh: np.ndarray) -> Optional[np.ndarray]:
        """Find the largest 4-sided contour in a thresholded image."""
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.cfg.min_contour_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4 and area > best_area:
                best = approx
                best_area = area

        if best is None:
            return None

        return self._order_corners(best.reshape(4, 2))

    def _order_corners(self, pts: np.ndarray) -> np.ndarray:
        """Order 4 points as: top-left, top-right, bottom-right, bottom-left."""
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left has smallest sum
        rect[2] = pts[np.argmax(s)]   # bottom-right has largest sum
        d = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(d)]   # top-right has smallest difference
        rect[3] = pts[np.argmax(d)]   # bottom-left has largest difference
        return rect

    def _perspective_transform(self, image: np.ndarray,
                               corners: np.ndarray) -> Optional[np.ndarray]:
        """Apply perspective transform to extract the grid area."""
        cfg = self.codec_cfg
        dst_w = cfg.image_width
        dst_h = cfg.image_height

        dst_corners = np.array([
            [0, 0],
            [dst_w - 1, 0],
            [dst_w - 1, dst_h - 1],
            [0, dst_h - 1],
        ], dtype=np.float32)

        src = corners.astype(np.float32)
        M = cv2.getPerspectiveTransform(src, dst_corners)
        warped = cv2.warpPerspective(image, M, (dst_w, dst_h))
        return warped

    def _calibrate_palette(self, warped: np.ndarray) -> np.ndarray:
        """Calibrate the color palette by sampling the alignment markers.

        The markers have a known pattern of black (symbol 0) and white
        (symbol 1) cells. By sampling what black and white actually look
        like in the captured image, we can estimate the color transform
        and adjust the full palette.
        """
        cfg = self.codec_cfg
        marker = _marker_pattern()
        mc = cfg.marker_cells
        palette = cfg.palette.copy().astype(np.float64)

        # Collect observed colors for symbol 0 (black) and symbol 1 (white)
        observed = {0: [], 1: []}

        # Sample all 4 marker corners
        corners = [
            (0, 0),
            (0, cfg.grid_cols - mc),
            (cfg.grid_rows - mc, 0),
            (cfg.grid_rows - mc, cfg.grid_cols - mc),
        ]

        border_offset = cfg.border_px * cfg.cell_px
        margin = int(cfg.cell_px * self.cfg.sample_margin)
        sample_h = cfg.cell_px - 2 * margin
        sample_w = cfg.cell_px - 2 * margin
        if sample_h <= 0 or sample_w <= 0:
            return palette.astype(np.uint8)

        for r0, c0 in corners:
            for dr in range(mc):
                for dc in range(mc):
                    sym = int(marker[dr, dc])
                    r = r0 + dr
                    c = c0 + dc
                    y = border_offset + r * cfg.cell_px + margin
                    x = border_offset + c * cfg.cell_px + margin

                    cell = warped[y:y + sample_h, x:x + sample_w]
                    if cell.size == 0:
                        continue
                    avg = cell.mean(axis=(0, 1))
                    observed[sym].append(avg)

        # Compute mean observed black and white
        if not observed[0] or not observed[1]:
            return palette.astype(np.uint8)

        obs_black = np.mean(observed[0], axis=0)  # what black looks like
        obs_white = np.mean(observed[1], axis=0)  # what white looks like

        # Expected palette values for black and white
        exp_black = palette[0].astype(np.float64)  # [0, 0, 0]
        exp_white = palette[1].astype(np.float64)  # [255, 255, 255]

        # Compute per-channel linear transform: observed = scale * expected + offset
        # From two points: scale = (obs_white - obs_black) / (exp_white - exp_black)
        #                  offset = obs_black - scale * exp_black
        denom = exp_white - exp_black
        # Avoid division by zero
        denom = np.where(np.abs(denom) < 1e-6, 1.0, denom)
        scale = (obs_white - obs_black) / denom
        offset = obs_black - scale * exp_black

        # Apply transform to all palette colors
        calibrated = np.zeros_like(palette)
        for i in range(len(palette)):
            calibrated[i] = scale * palette[i] + offset

        return np.clip(calibrated, 0, 255).astype(np.uint8)

    def _sample_grid(self, warped: np.ndarray,
                     palette: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample the color of each cell in the perspective-corrected image.

        Returns a grid of shape (grid_rows, grid_cols) with symbol indices.
        """
        cfg = self.codec_cfg
        if palette is None:
            palette = cfg.palette
        grid = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=np.uint8)

        border_offset = cfg.border_px * cfg.cell_px
        margin = int(cfg.cell_px * self.cfg.sample_margin)

        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                y = border_offset + r * cfg.cell_px + margin
                x = border_offset + c * cfg.cell_px + margin
                h = cfg.cell_px - 2 * margin
                w = cfg.cell_px - 2 * margin

                if h <= 0 or w <= 0:
                    h = max(1, cfg.cell_px)
                    w = max(1, cfg.cell_px)
                    y = border_offset + r * cfg.cell_px
                    x = border_offset + c * cfg.cell_px

                cell = warped[y:y + h, x:x + w]
                if cell.size == 0:
                    grid[r, c] = 0
                    continue

                # Average color of the cell
                avg_color = cell.mean(axis=(0, 1))

                # Find nearest palette color (Euclidean distance in BGR)
                distances = np.sqrt(
                    ((palette.astype(np.float64) - avg_color) ** 2).sum(axis=1)
                )
                grid[r, c] = int(np.argmin(distances))

        return grid

    def is_blank_frame(self, image: np.ndarray, threshold: int = 30) -> bool:
        """Check if the image is a blank/sync frame."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        return mean < threshold or mean > 225

    def is_sync_pattern(self, image: np.ndarray) -> bool:
        """Check if the image is the white sync pattern."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray.mean() > 225
