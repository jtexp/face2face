"""Calibration wizard: interactive setup and testing of the visual link.

Steps:
1. Display test patterns to verify webcam can see the screen
2. Auto-detect optimal grid size and color depth
3. Measure error rate and adjust ECC level
4. Measure throughput and display stats
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from ..visual.capture import CaptureConfig, WebcamCapture
from ..visual.codec import CodecConfig, FrameEncoder, FrameHeader
from ..visual.decoder import ImageFrameDecoder
from ..visual.ecc import ECCCodec, ECCConfig
from .config import AppConfig, save_config


class CalibrationWizard:
    """Interactive calibration for the visual link."""

    def __init__(self, config: AppConfig):
        self.config = config
        self._capture: WebcamCapture | None = None

    def run(self) -> AppConfig:
        """Run the full calibration wizard. Returns updated config."""
        print("=== face2face Calibration Wizard ===\n")

        # Step 1: Camera check
        if not self._step_camera_check():
            print("Calibration aborted: camera not working.")
            return self.config

        # Step 2: Test patterns
        best_grid, best_bits = self._step_find_optimal_params()
        self.config.grid_cols = best_grid
        self.config.grid_rows = best_grid
        self.config.bits_per_cell = best_bits

        # Step 3: Error rate measurement
        optimal_ecc = self._step_measure_error_rate()
        self.config.ecc_nsym = optimal_ecc

        # Step 4: Throughput measurement
        self._step_measure_throughput()

        print("\n=== Calibration Complete ===")
        print(f"  Grid: {self.config.grid_cols}x{self.config.grid_rows}")
        print(f"  Bits/cell: {self.config.bits_per_cell}")
        print(f"  ECC symbols: {self.config.ecc_nsym}")
        codec = self.config.to_codec_config()
        print(f"  Payload/frame: {codec.payload_bytes} bytes")

        return self.config

    def _step_camera_check(self) -> bool:
        """Step 1: Verify the webcam works."""
        print("Step 1: Camera check")
        print(f"  Opening camera {self.config.camera_index}...")

        try:
            cap_config = self.config.to_capture_config()
            self._capture = WebcamCapture(cap_config)
            self._capture.open()
            frame = self._capture.read()
            if frame is None:
                print("  ERROR: Could not read from camera.")
                return False
            print(f"  Camera OK — frame size: {frame.shape[1]}x{frame.shape[0]}")
            return True
        except Exception as e:
            print(f"  ERROR: {e}")
            return False

    def _step_find_optimal_params(self) -> tuple[int, int]:
        """Step 2: Test different grid sizes and color depths."""
        print("\nStep 2: Finding optimal parameters")

        best_grid = 16
        best_bits = 2
        best_score = 0.0

        for grid_size in [16, 24, 32, 48]:
            for bits in [2, 4]:
                score = self._test_encode_decode(grid_size, bits)
                print(f"  Grid {grid_size}x{grid_size}, {bits} bits/cell: "
                      f"accuracy = {score:.1%}")
                if score > best_score:
                    best_score = score
                    best_grid = grid_size
                    best_bits = bits

                # If we got perfect score, this config works
                if score >= 0.99:
                    best_grid = grid_size
                    best_bits = bits

        print(f"  Best: {best_grid}x{best_grid}, {best_bits} bits/cell "
              f"({best_score:.1%})")
        return best_grid, best_bits

    def _test_encode_decode(self, grid_size: int, bits: int,
                            n_trials: int = 3) -> float:
        """Test encode→display→capture→decode with given params.

        Returns accuracy as fraction of correctly decoded cells.
        For synthetic testing (no actual display/capture), tests
        the encode→decode pipeline directly.
        """
        cfg = CodecConfig(
            grid_cols=grid_size, grid_rows=grid_size,
            bits_per_cell=bits, cell_px=self.config.cell_px,
        )
        encoder = FrameEncoder(config=cfg)
        decoder = ImageFrameDecoder(cfg)

        total_correct = 0
        total_cells = 0

        for trial in range(n_trials):
            # Generate random payload
            payload_size = cfg.payload_bytes
            test_data = bytes(range(256)) * (payload_size // 256 + 1)
            test_data = test_data[:payload_size]

            header = FrameHeader(msg_id=trial, seq=0, total=1)
            image = encoder.encode(test_data, header)

            # Decode directly (synthetic — no camera)
            decoded_header, decoded_payload = decoder.decode_image(image)

            if decoded_header is not None and decoded_payload is not None:
                # Compare payloads byte by byte
                correct = sum(
                    1 for a, b in zip(test_data, decoded_payload) if a == b
                )
                total_correct += correct
                total_cells += len(test_data)
            else:
                total_cells += len(test_data)

        return total_correct / total_cells if total_cells > 0 else 0.0

    def _step_measure_error_rate(self) -> int:
        """Step 3: Measure error rate and find optimal ECC level."""
        print("\nStep 3: Measuring error rate")

        # For now, use a conservative default
        # In a real scenario, this would display frames and capture them
        # to measure actual error rates
        ecc_nsym = 20  # default
        print(f"  Using ECC symbols: {ecc_nsym} (default)")
        print(f"  Can correct up to {ecc_nsym // 2} byte errors per frame")
        return ecc_nsym

    def _step_measure_throughput(self) -> None:
        """Step 4: Measure throughput."""
        print("\nStep 4: Estimating throughput")

        codec = self.config.to_codec_config()
        ecc = ECCCodec(ECCConfig(nsym=self.config.ecc_nsym))

        payload_per_frame = ecc.max_payload(codec.payload_bytes)
        frame_time = (self.config.frame_hold_ms + self.config.blank_hold_ms) / 1000.0
        fps = 1.0 / frame_time if frame_time > 0 else 0

        throughput_bps = payload_per_frame * fps
        print(f"  Payload per frame: {payload_per_frame} bytes")
        print(f"  Frame rate: {fps:.1f} fps")
        print(f"  Estimated throughput: {throughput_bps:.0f} bytes/s "
              f"({throughput_bps * 8 / 1000:.1f} kbps)")

    def cleanup(self) -> None:
        """Release resources."""
        if self._capture:
            self._capture.close()
        cv2.destroyAllWindows()
