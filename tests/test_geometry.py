"""Geometry tests: camera not pointed straight at the screen.

Tests the decoder's perspective correction under realistic geometric
distortions applied directly via cv2 transforms — not through the
CameraSimulator's simplified model.

Scenarios:
  - Camera rotated (tilted CW/CCW)
  - Camera off to one side (horizontal perspective)
  - Camera above/below (vertical perspective)
  - Combined rotation + perspective
  - Extreme keystoning
  - Small frame in a large field of view
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import ImageFrameDecoder


# Use 2-bit palette with large cells for robustness
BASE_CFG = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)


def _make_payload(cfg: CodecConfig, seed: int = 42) -> bytes:
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=cfg.payload_bytes, dtype=np.uint8))


def _encode(cfg: CodecConfig, payload: bytes, msg_id: int = 1) -> np.ndarray:
    encoder = FrameEncoder(config=cfg)
    header = FrameHeader(msg_id=msg_id, seq=0, total=1)
    return encoder.encode(payload, header)


def _embed_in_canvas(frame: np.ndarray, pad: int = 100,
                     bg: int = 20) -> np.ndarray:
    """Embed frame in a larger dark canvas."""
    h, w = frame.shape[:2]
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), bg, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = frame
    return canvas


def _warp_canvas(canvas: np.ndarray,
                 src_pts: np.ndarray,
                 dst_pts: np.ndarray) -> np.ndarray:
    """Apply a perspective transform to the canvas."""
    M = cv2.getPerspectiveTransform(src_pts.astype(np.float32),
                                    dst_pts.astype(np.float32))
    h, w = canvas.shape[:2]
    return cv2.warpPerspective(canvas, M, (w, h),
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(20, 20, 20))


def _rotate_canvas(canvas: np.ndarray, angle_deg: float,
                   scale: float = 1.0) -> np.ndarray:
    """Rotate the canvas around its center."""
    h, w = canvas.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, scale)

    # Compute new bounding size to avoid clipping
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    return cv2.warpAffine(canvas, M, (new_w, new_h),
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(20, 20, 20))


def _decode_and_check(cfg: CodecConfig, image: np.ndarray,
                      payload: bytes, label: str):
    """Decode and assert payload matches."""
    decoder = ImageFrameDecoder(cfg)
    dec_header, dec_payload = decoder.decode_image(image)
    assert dec_header is not None, f"{label}: header not found"
    assert dec_payload is not None, f"{label}: payload CRC failed"
    assert dec_payload == payload, f"{label}: payload mismatch"


# -------------------------------------------------------------------------
# Rotation tests
# -------------------------------------------------------------------------

class TestRotation:
    """Camera rotated (tilted) relative to the screen."""

    @pytest.mark.parametrize("angle", [2, 5, -3, -7])
    def test_small_rotation(self, angle):
        """Small rotations (< 10 degrees) should decode cleanly."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=angle + 100)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=120)
        rotated = _rotate_canvas(canvas, angle)

        _decode_and_check(cfg, rotated, payload, f"Rotation {angle}°")

    def test_10_degree_rotation(self):
        """10-degree rotation — still within tolerance."""
        cfg = BASE_CFG
        payload = _make_payload(cfg)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=150)
        rotated = _rotate_canvas(canvas, 10)

        _decode_and_check(cfg, rotated, payload, "Rotation 10°")

    def test_15_degree_rotation(self):
        """15-degree rotation — pushing limits."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=28)
        payload = _make_payload(cfg, seed=15)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=180)
        rotated = _rotate_canvas(canvas, 15)

        _decode_and_check(cfg, rotated, payload, "Rotation 15°")


# -------------------------------------------------------------------------
# Perspective tests (true projective transforms)
# -------------------------------------------------------------------------

class TestPerspective:
    """Camera off-axis — true projective distortion."""

    def test_camera_right_of_center(self):
        """Camera slightly to the right — left side of frame appears wider."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=50)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=100)
        h, w = canvas.shape[:2]

        # Squeeze the right side (camera is to the right)
        squeeze = 30
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [w - squeeze, squeeze],
            [w - squeeze, h - squeeze],
            [0, h],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        _decode_and_check(cfg, warped, payload, "Camera right")

    def test_camera_above(self):
        """Camera above the screen — top appears wider than bottom."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=60)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=100)
        h, w = canvas.shape[:2]

        squeeze = 25
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [w, 0],
            [w - squeeze, h - squeeze],
            [squeeze, h - squeeze],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        _decode_and_check(cfg, warped, payload, "Camera above")

    def test_camera_below_left(self):
        """Camera below and to the left."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=70)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=100)
        h, w = canvas.shape[:2]

        squeeze = 20
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [squeeze, squeeze],
            [w - squeeze * 2, 0],
            [w, h],
            [0, h - squeeze],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        _decode_and_check(cfg, warped, payload, "Camera below-left")

    def test_strong_keystone(self):
        """Strong keystone effect — top much narrower than bottom."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=28)
        payload = _make_payload(cfg, seed=80)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=120)
        h, w = canvas.shape[:2]

        squeeze = 50
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [squeeze, squeeze],
            [w - squeeze, squeeze],
            [w, h],
            [0, h],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        _decode_and_check(cfg, warped, payload, "Strong keystone")


# -------------------------------------------------------------------------
# Combined rotation + perspective
# -------------------------------------------------------------------------

class TestRotationPlusPerspective:
    """Camera tilted AND off-axis simultaneously."""

    def test_rotated_and_perspective(self):
        """5° rotation + mild perspective distortion."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=90)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=120)

        # Apply perspective first
        h, w = canvas.shape[:2]
        squeeze = 20
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [0, 0],
            [w - squeeze, squeeze],
            [w - squeeze // 2, h - squeeze // 2],
            [squeeze // 2, h],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        # Then rotate
        rotated = _rotate_canvas(warped, 5)

        _decode_and_check(cfg, rotated, payload, "Rotated + perspective")

    def test_rotated_and_keystone(self):
        """7° rotation + keystone distortion."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=28)
        payload = _make_payload(cfg, seed=95)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=150)

        # Keystone
        h, w = canvas.shape[:2]
        squeeze = 35
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [squeeze, squeeze],
            [w - squeeze, squeeze],
            [w, h],
            [0, h],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        # Rotate
        rotated = _rotate_canvas(warped, 7)

        _decode_and_check(cfg, rotated, payload, "Rotated + keystone")


# -------------------------------------------------------------------------
# Scale / distance tests
# -------------------------------------------------------------------------

class TestScaleAndDistance:
    """Frame appears small in the camera field of view."""

    def test_small_in_large_fov(self):
        """Frame is only ~25% of the camera image area."""
        cfg = BASE_CFG
        payload = _make_payload(cfg)
        frame = _encode(cfg, payload)

        # Embed in a very large canvas
        canvas = _embed_in_canvas(frame, pad=300)

        _decode_and_check(cfg, canvas, payload, "Small in large FOV")

    def test_small_rotated_in_large_fov(self):
        """Small frame, rotated, in a large field of view."""
        cfg = BASE_CFG
        payload = _make_payload(cfg, seed=111)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=250)
        rotated = _rotate_canvas(canvas, 4)

        _decode_and_check(cfg, rotated, payload, "Small + rotated")

    def test_downscaled_perspective(self):
        """Frame captured at lower resolution with perspective."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=28)
        payload = _make_payload(cfg, seed=120)
        frame = _encode(cfg, payload)
        canvas = _embed_in_canvas(frame, pad=100)

        # Mild perspective
        h, w = canvas.shape[:2]
        sq = 15
        src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst = np.array([
            [sq, 0], [w - sq, sq], [w, h - sq], [0, h],
        ], dtype=np.float32)
        warped = _warp_canvas(canvas, src, dst)

        # Downscale to 60%
        new_w = int(warped.shape[1] * 0.6)
        new_h = int(warped.shape[0] * 0.6)
        scaled = cv2.resize(warped, (new_w, new_h), interpolation=cv2.INTER_AREA)

        _decode_and_check(cfg, scaled, payload, "Downscaled + perspective")
