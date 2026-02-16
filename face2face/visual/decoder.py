"""Frame decoder: detect and decode a visual frame from a camera image.

Steps:
1. Find the 4 corner alignment markers in the camera image
2. Apply perspective transform to extract a clean rectangular grid
3. Sample each cell to determine its color/symbol
4. Pass the symbol grid to FrameDecoder.decode_grid()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .codec import CodecConfig, FrameDecoder as GridDecoder, FrameHeader


@dataclass
class DecoderConfig:
    min_contour_area: int = 500
    adaptive_block_size: int = 11
    adaptive_c: int = 2
    sample_margin: float = 0.25  # fraction of cell to skip at edges when sampling


class ImageFrameDecoder:
    """Detects and decodes visual frames from camera images."""

    def __init__(self, codec_cfg: CodecConfig,
                 decoder_cfg: Optional[DecoderConfig] = None):
        self.codec_cfg = codec_cfg
        self.cfg = decoder_cfg or DecoderConfig()
        self.grid_decoder = GridDecoder(config=codec_cfg)
        self._last_confidence: float = 0.0

    @property
    def last_confidence(self) -> float:
        return self._last_confidence

    def decode_image(self, image: np.ndarray) -> tuple[Optional[FrameHeader], Optional[bytes]]:
        """Attempt to decode a frame from a camera image.

        Returns (header, payload) or (None, None) if no valid frame found.
        """
        corners = self._find_corners(image)
        if corners is None:
            self._last_confidence = 0.0
            return None, None

        warped = self._perspective_transform(image, corners)
        if warped is None:
            self._last_confidence = 0.0
            return None, None

        grid = self._sample_grid(warped)
        self._last_confidence = 1.0  # TODO: compute from color distances

        return self.grid_decoder.decode_grid(grid)

    def _find_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Find the 4 corner alignment markers.

        Returns array of shape (4, 2) with corner coordinates in order:
        [top-left, top-right, bottom-left, bottom-right]
        or None if markers not found.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Look for the alternating border pattern
        # The outermost border is white, so we threshold and find the
        # largest white rectangle
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Find the largest rectangular contour (should be the border)
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

        # Order corners: top-left, top-right, bottom-right, bottom-left
        corners = self._order_corners(best.reshape(4, 2))
        return corners

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
        # The corners delineate the outer border. We want the inner grid area.
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

    def _sample_grid(self, warped: np.ndarray) -> np.ndarray:
        """Sample the color of each cell in the perspective-corrected image.

        Returns a grid of shape (grid_rows, grid_cols) with symbol indices.
        """
        cfg = self.codec_cfg
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
