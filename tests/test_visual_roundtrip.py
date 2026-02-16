"""Visual round-trip tests: encode → write to file → read back → decode.

Tests the full image pipeline with real PNG/JPEG files, simulated camera
conditions (noise, embedding in larger background, JPEG artifacts), and
multiple codec configurations.
"""

from __future__ import annotations

import os
import tempfile

import cv2
import numpy as np
import pytest

from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import DecoderConfig, ImageFrameDecoder
from face2face.visual.ecc import ECCCodec, ECCConfig


def _make_test_payload(cfg: CodecConfig, seed: int = 42) -> bytes:
    """Generate a deterministic test payload that fills the frame."""
    rng = np.random.RandomState(seed)
    return bytes(rng.randint(0, 256, size=cfg.payload_bytes, dtype=np.uint8))


class TestPNGRoundTrip:
    """Encode → save as PNG (lossless) → load → decode."""

    @pytest.mark.parametrize("grid_size", [16, 32])
    @pytest.mark.parametrize("bits", [2, 4])
    def test_png_roundtrip(self, grid_size, bits, tmp_path):
        cfg = CodecConfig(grid_cols=grid_size, grid_rows=grid_size,
                          bits_per_cell=bits, cell_px=20)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=123, seq=0, total=1)
        image = encoder.encode(payload, header)

        # Write to PNG
        png_path = str(tmp_path / "frame.png")
        cv2.imwrite(png_path, image)

        # Read back
        loaded = cv2.imread(png_path)
        assert loaded is not None
        assert loaded.shape == image.shape

        # Decode
        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Failed to decode header from PNG"
        assert dec_payload is not None, "Failed to decode payload from PNG (CRC mismatch)"
        assert dec_header.msg_id == 123
        assert dec_header.seq == 0
        assert dec_header.total == 1
        assert dec_payload == payload

    def test_png_multiple_frames(self, tmp_path):
        """Encode and decode multiple frames sequentially."""
        cfg = CodecConfig(grid_cols=24, grid_rows=24, bits_per_cell=2, cell_px=16)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        for i in range(5):
            payload = _make_test_payload(cfg, seed=i)
            header = FrameHeader(msg_id=i + 1, seq=i, total=5)
            image = encoder.encode(payload, header)

            path = str(tmp_path / f"frame_{i}.png")
            cv2.imwrite(path, image)
            loaded = cv2.imread(path)

            dec_header, dec_payload = decoder.decode_image(loaded)
            assert dec_header is not None, f"Frame {i}: header decode failed"
            assert dec_payload is not None, f"Frame {i}: payload CRC failed"
            assert dec_header.msg_id == i + 1
            assert dec_payload == payload


class TestEmbeddedInBackground:
    """Frame rendered on a screen, surrounded by black — like a real camera view."""

    def test_frame_on_black_background(self, tmp_path):
        """Embed the frame in a larger black image and decode it."""
        cfg = CodecConfig(grid_cols=24, grid_rows=24, bits_per_cell=2, cell_px=16)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=77, seq=0, total=1)
        frame = encoder.encode(payload, header)

        # Embed in a larger black image (simulating screen with black border)
        pad = 80
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = frame

        path = str(tmp_path / "embedded.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Failed to find frame in larger image"
        assert dec_payload is not None, "CRC mismatch on embedded frame"
        assert dec_header.msg_id == 77
        assert dec_payload == payload

    def test_frame_on_gray_background(self, tmp_path):
        """Frame on a gray background (simulating ambient light)."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=20)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg, seed=99)
        header = FrameHeader(msg_id=88, seq=0, total=1)
        frame = encoder.encode(payload, header)

        pad = 60
        h, w = frame.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 50, dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = frame

        path = str(tmp_path / "gray_bg.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None
        assert dec_payload is not None
        assert dec_payload == payload


class TestJPEGRoundTrip:
    """JPEG compression introduces artifacts — tests robustness."""

    def test_jpeg_high_quality(self, tmp_path):
        """JPEG at quality 95 should decode cleanly with large cells."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=200, seq=0, total=1)
        frame = encoder.encode(payload, header)

        # Embed in background so contour detection works cleanly
        pad = 40
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = frame

        path = str(tmp_path / "frame.jpg")
        cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "JPEG q95: header decode failed"
        assert dec_payload is not None, "JPEG q95: payload CRC failed"
        assert dec_payload == payload

    def test_jpeg_medium_quality_with_ecc(self, tmp_path):
        """JPEG at quality 80 may corrupt some cells — ECC should recover."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)
        ecc = ECCCodec(ECCConfig(nsym=20))

        raw_payload = b"ECC test data!!"
        ecc_payload = ecc.encode(raw_payload)
        # Pad to fill the frame
        padded = ecc_payload + b"\x00" * (cfg.payload_bytes - len(ecc_payload))

        header = FrameHeader(msg_id=201, seq=0, total=1)
        frame = encoder.encode(padded, header)

        pad = 40
        h, w = frame.shape[:2]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = frame

        path = str(tmp_path / "frame_q80.jpg")
        cv2.imwrite(path, canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "JPEG q80: header not found"

        if dec_payload is not None:
            # If CRC passed, the raw decode was clean
            ecc_data = dec_payload[:len(ecc_payload)]
            recovered = ecc.decode(ecc_data)
            assert recovered == raw_payload
        else:
            # CRC failed — try ECC recovery on the raw symbols
            # Re-decode without CRC check by going through the grid
            from face2face.visual.codec import FrameDecoder, HEADER_BYTES, _symbols_to_bytes, _in_marker
            grid_decoder = FrameDecoder(config=cfg)

            # Get the warped image and sample grid
            corners = decoder._find_corners(loaded)
            assert corners is not None
            warped = decoder._perspective_transform(loaded, corners)
            grid = decoder._sample_grid(warped)

            # Extract raw bytes from grid
            symbols = []
            mc = cfg.marker_cells
            for r in range(cfg.grid_rows):
                for c in range(cfg.grid_cols):
                    if _in_marker(r, c, cfg.grid_rows, cfg.grid_cols, mc):
                        continue
                    symbols.append(int(grid[r, c]))

            total_bytes = HEADER_BYTES + cfg.payload_bytes
            raw = _symbols_to_bytes(symbols, cfg.bits_per_cell, total_bytes)
            raw_data = raw[HEADER_BYTES:HEADER_BYTES + len(ecc_payload)]

            recovered = ecc.decode(raw_data)
            assert recovered is not None, "ECC failed to recover from JPEG artifacts"
            assert recovered == raw_payload


class TestGaussianNoise:
    """Simulate camera sensor noise."""

    def test_light_noise(self, tmp_path):
        """Small amount of Gaussian noise — should decode fine with 2-bit palette."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=300, seq=0, total=1)
        frame = encoder.encode(payload, header)

        # Add Gaussian noise (sigma=5, very mild)
        noise = np.random.normal(0, 5, frame.shape).astype(np.float64)
        noisy = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        # Embed in background
        pad = 40
        h, w = noisy.shape[:2]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = noisy

        path = str(tmp_path / "noisy.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Light noise: header not found"
        assert dec_payload is not None, "Light noise: payload CRC failed"
        assert dec_payload == payload

    def test_moderate_noise_with_ecc(self, tmp_path):
        """Moderate noise may corrupt some cells — ECC to the rescue."""
        # Use large cells (32px) so the per-cell average is stable under noise,
        # and a fixed seed for determinism.
        # nsym=20 so ECC payload (18+20=38) fits within frame capacity (42 bytes).
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=32)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)
        ecc = ECCCodec(ECCConfig(nsym=20))

        raw_payload = b"Noisy channel test"
        ecc_payload = ecc.encode(raw_payload)
        assert len(ecc_payload) <= cfg.payload_bytes, \
            f"ECC payload ({len(ecc_payload)}B) exceeds frame capacity ({cfg.payload_bytes}B)"
        padded = ecc_payload + b"\x00" * (cfg.payload_bytes - len(ecc_payload))

        header = FrameHeader(msg_id=301, seq=0, total=1)
        frame = encoder.encode(padded, header)

        # Add moderate noise (sigma=10, fixed seed)
        rng = np.random.RandomState(123)
        noise = rng.normal(0, 10, frame.shape).astype(np.float64)
        noisy = np.clip(frame.astype(np.float64) + noise, 0, 255).astype(np.uint8)

        pad = 40
        h, w = noisy.shape[:2]
        canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + h, pad:pad + w] = noisy

        path = str(tmp_path / "moderate_noise.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Moderate noise: couldn't find frame"

        # Try direct decode first
        if dec_payload is not None:
            ecc_data = dec_payload[:len(ecc_payload)]
            recovered = ecc.decode(ecc_data)
            if recovered is not None:
                assert recovered == raw_payload
                return

        # Fall back to ECC recovery on raw grid
        from face2face.visual.codec import HEADER_BYTES, _in_marker, _symbols_to_bytes

        corners = decoder._find_corners(loaded)
        assert corners is not None
        warped = decoder._perspective_transform(loaded, corners)
        grid = decoder._sample_grid(warped)

        symbols = []
        mc = cfg.marker_cells
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                if _in_marker(r, c, cfg.grid_rows, cfg.grid_cols, mc):
                    continue
                symbols.append(int(grid[r, c]))

        total_bytes = HEADER_BYTES + cfg.payload_bytes
        raw = _symbols_to_bytes(symbols, cfg.bits_per_cell, total_bytes)
        ecc_data = raw[HEADER_BYTES:HEADER_BYTES + len(ecc_payload)]

        recovered = ecc.decode(ecc_data)
        assert recovered is not None, "ECC failed to recover from noise"
        assert recovered == raw_payload


class TestScaledImage:
    """Image resized — as a camera would capture at different resolutions."""

    def test_upscale_2x(self, tmp_path):
        """Frame scaled up 2x (camera closer than expected)."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=16)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=400, seq=0, total=1)
        frame = encoder.encode(payload, header)

        # Scale up 2x
        h, w = frame.shape[:2]
        scaled = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        # Embed in background
        pad = 60
        sh, sw = scaled.shape[:2]
        canvas = np.zeros((sh + 2 * pad, sw + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + sh, pad:pad + sw] = scaled

        path = str(tmp_path / "scaled_2x.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Upscale 2x: header not found"
        assert dec_payload is not None, "Upscale 2x: payload CRC failed"
        assert dec_payload == payload

    def test_downscale_half(self, tmp_path):
        """Frame scaled down 0.5x — cells get smaller, harder to read."""
        cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        payload = _make_test_payload(cfg)
        header = FrameHeader(msg_id=401, seq=0, total=1)
        frame = encoder.encode(payload, header)

        # Scale down
        h, w = frame.shape[:2]
        scaled = cv2.resize(frame, (w // 2, h // 2), interpolation=cv2.INTER_AREA)

        pad = 30
        sh, sw = scaled.shape[:2]
        canvas = np.zeros((sh + 2 * pad, sw + 2 * pad, 3), dtype=np.uint8)
        canvas[pad:pad + sh, pad:pad + sw] = scaled

        path = str(tmp_path / "scaled_half.png")
        cv2.imwrite(path, canvas)
        loaded = cv2.imread(path)

        dec_header, dec_payload = decoder.decode_image(loaded)
        assert dec_header is not None, "Downscale 0.5x: header not found"
        assert dec_payload is not None, "Downscale 0.5x: payload CRC failed"
        assert dec_payload == payload
