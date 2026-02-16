"""Live visual decode test: display a frame on screen, webcam tries to decode it.

Encodes a test message into a visual frame, displays it on screen, and
continuously captures from the webcam to attempt decoding. This validates
the full encode -> display -> capture -> decode pipeline with real hardware.

Usage:
    python tools/live_decode_test.py
    python tools/live_decode_test.py --grid 16 --cell-px 32 --bits 2
    python tools/live_decode_test.py --camera-index 1 --message "Custom message"

Controls:
    q  - quit
    s  - save current webcam frame as debug image
"""

import argparse
import time

import cv2
import numpy as np

from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import ImageFrameDecoder


def main():
    parser = argparse.ArgumentParser(description="Live webcam visual decode test")
    parser.add_argument("--grid", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--cell-px", type=int, default=28, help="Cell size in pixels (default: 28)")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 4, 6],
                        help="Bits per cell (default: 2, i.e. 4 colors)")
    parser.add_argument("--padding", type=int, default=80,
                        help="Black padding around frame to isolate from window chrome (default: 80)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--camera-width", type=int, default=1280, help="Camera width (default: 1280)")
    parser.add_argument("--camera-height", type=int, default=720, help="Camera height (default: 720)")
    parser.add_argument("--message", type=str, default="Hello from face2face!",
                        help="Test message to encode")
    args = parser.parse_args()

    cfg = CodecConfig(
        grid_cols=args.grid, grid_rows=args.grid,
        bits_per_cell=args.bits, cell_px=args.cell_px,
    )
    encoder = FrameEncoder(config=cfg)
    decoder = ImageFrameDecoder(cfg)

    # Encode test message
    msg = args.message.encode()
    if len(msg) > cfg.payload_bytes:
        print(f"WARNING: message truncated to {cfg.payload_bytes} bytes")
        msg = msg[:cfg.payload_bytes]
    payload = msg + b"\x00" * (cfg.payload_bytes - len(msg))
    header = FrameHeader(msg_id=1, seq=0, total=1)
    frame = encoder.encode(payload, header)

    # Pad with black so the window title bar doesn't interfere with detection
    pad = args.padding
    padded = np.zeros(
        (frame.shape[0] + 2 * pad, frame.shape[1] + 2 * pad, 3),
        dtype=np.uint8,
    )
    padded[pad:pad + frame.shape[0], pad:pad + frame.shape[1]] = frame

    print(f"Frame: {frame.shape[1]}x{frame.shape[0]}px (padded to {padded.shape[1]}x{padded.shape[0]})")
    print(f"Grid: {args.grid}x{args.grid}, cell: {args.cell_px}px, colors: {1 << args.bits}")
    print(f"Payload capacity: {cfg.payload_bytes} bytes")
    print(f"Message: {args.message!r}")
    print()
    print("Position your webcam to see the colored grid window.")
    print("Make sure the ENTIRE white border is visible in the webcam view.")
    print("Controls: q=quit, s=save debug image")
    print()

    # Display the padded frame
    win_name = "face2face TX"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win_name, padded)

    # Open webcam
    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)

    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {args.camera_index}")
        return

    decoded_count = 0
    attempt_count = 0
    last_status = ""
    last_decode_time = 0

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            fname = f"debug_webcam_{int(time.time())}.png"
            ret, save_frame = cap.read()
            if ret:
                cv2.imwrite(fname, save_frame)
                print(f"Saved: {fname}")

        ret, camera_frame = cap.read()
        if not ret:
            continue

        attempt_count += 1

        # Show webcam view
        cv2.imshow("Webcam View", camera_frame)

        # Try to decode
        try:
            dec_header, dec_payload = decoder.decode_image(camera_frame)
        except Exception as e:
            if attempt_count % 30 == 0:
                print(f"  decode error: {e}")
            continue

        if dec_header is not None and dec_payload is not None:
            decoded_msg = dec_payload.rstrip(b"\x00")
            decoded_count += 1
            now = time.time()
            if now - last_decode_time > 1.0:  # throttle output
                rate = decoded_count / max(attempt_count, 1) * 100
                print(f"DECODED #{decoded_count}: {decoded_msg}  ({rate:.0f}% success rate)")
                last_decode_time = now
        elif dec_header is not None:
            status = "Frame detected but CRC failed - adjust angle/distance"
            if status != last_status and attempt_count % 20 == 0:
                print(status)
                last_status = status
        else:
            if attempt_count % 60 == 0:
                print(f"  ({attempt_count} frames scanned, no frame detected yet...)")

    cap.release()
    cv2.destroyAllWindows()
    rate = decoded_count / max(attempt_count, 1) * 100
    print(f"\nDone. Decoded {decoded_count}/{attempt_count} frames ({rate:.1f}% success rate).")


if __name__ == "__main__":
    main()
