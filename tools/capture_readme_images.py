"""Capture and annotate images for the README.

Displays an encoded frame on screen, captures it with the webcam,
runs the decode pipeline, and saves annotated images showing each stage.

Usage:
    python tools/capture_readme_images.py

Produces:
    docs/images/01_encoded_frame.png    - The raw encoded color grid
    docs/images/02_webcam_capture.png   - What the webcam sees
    docs/images/03_warped.png           - After perspective correction
    docs/images/04_decoded_success.png  - Annotated webcam view with decode overlay
    docs/images/05_architecture.png     - System architecture diagram
"""

import time
import cv2
import numpy as np

from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import ImageFrameDecoder

OUT = "docs/images"
PAD = 80
GRID = 16
CELL_PX = 28
BITS = 2


def draw_text(img, text, pos, scale=0.7, color=(255, 255, 255),
              thickness=2, bg_color=(0, 0, 0), bg_pad=6):
    """Draw text with a background rectangle for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x - bg_pad, y - th - bg_pad),
                  (x + tw + bg_pad, y + baseline + bg_pad), bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_architecture(w=900, h=420):
    """Draw the system architecture diagram."""
    img = np.full((h, w, 3), 32, dtype=np.uint8)

    # Title
    draw_text(img, "face2face: HTTP Proxy via Screen + Webcam", (120, 35),
              scale=0.85, color=(0, 200, 255), bg_color=(32, 32, 32))

    box_color = (80, 80, 80)
    border_color = (180, 180, 180)

    # Machine A box
    cv2.rectangle(img, (30, 60), (420, 390), box_color, -1)
    cv2.rectangle(img, (30, 60), (420, 390), border_color, 2)
    draw_text(img, "Machine A (air-gapped)", (80, 90),
              scale=0.65, color=(100, 200, 255), bg_color=box_color)

    # Machine B box
    cv2.rectangle(img, (480, 60), (870, 390), box_color, -1)
    cv2.rectangle(img, (480, 60), (870, 390), border_color, 2)
    draw_text(img, "Machine B (has internet)", (530, 90),
              scale=0.65, color=(100, 255, 100), bg_color=box_color)

    # Machine A components
    components_a = [
        ("curl / git / browser", (70, 140), (0, 200, 255)),
        ("HTTP Proxy :8080", (70, 185), (0, 200, 255)),
        ("Visual Encoder", (70, 240), (0, 140, 255)),
        ("Screen (TX)", (70, 275), (0, 100, 255)),
        ("Webcam (RX)", (70, 330), (0, 100, 255)),
        ("Visual Decoder", (70, 365), (0, 140, 255)),
    ]

    for text, pos, color in components_a:
        draw_text(img, text, pos, scale=0.55, color=color, bg_color=(60, 60, 60))

    # Machine B components
    components_b = [
        ("Internet", (570, 140), (100, 255, 100)),
        ("HTTP Forwarder", (570, 185), (100, 255, 100)),
        ("Visual Decoder", (570, 240), (100, 200, 50)),
        ("Webcam (RX)", (570, 275), (100, 160, 50)),
        ("Screen (TX)", (570, 330), (100, 160, 50)),
        ("Visual Encoder", (570, 365), (100, 200, 50)),
    ]

    for text, pos, color in components_b:
        draw_text(img, text, pos, scale=0.55, color=color, bg_color=(60, 60, 60))

    # Arrows between machines
    arrow_color = (0, 180, 255)
    # Screen A -> Webcam B (top arrow)
    cv2.arrowedLine(img, (380, 260), (520, 260), arrow_color, 3, tipLength=0.08)
    draw_text(img, "light", (425, 250), scale=0.45, color=arrow_color, bg_color=(32, 32, 32))

    # Screen B -> Webcam A (bottom arrow)
    arrow_color2 = (100, 255, 100)
    cv2.arrowedLine(img, (520, 340), (380, 340), arrow_color2, 3, tipLength=0.08)
    draw_text(img, "light", (425, 330), scale=0.45, color=arrow_color2, bg_color=(32, 32, 32))

    # Vertical flow arrows (Machine A)
    for y1, y2 in [(148, 172), (193, 228), (283, 318)]:
        cv2.arrowedLine(img, (60, y1), (60, y2), (120, 120, 120), 2, tipLength=0.15)
    cv2.arrowedLine(img, (60, 370), (60, 193), (120, 120, 120), 2, tipLength=0.05)

    # Vertical flow arrows (Machine B)
    for y1, y2 in [(283, 228), (193, 148)]:
        cv2.arrowedLine(img, (560, y1), (560, y2), (120, 120, 120), 2, tipLength=0.15)
    cv2.arrowedLine(img, (560, 370), (560, 318), (120, 120, 120), 2, tipLength=0.1)

    return img


def annotate_encoded_frame(frame):
    """Add labels to the encoded frame image."""
    pad = 100
    h, w = frame.shape[:2]
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 32, dtype=np.uint8)
    canvas[pad:pad + h, pad:pad + w] = frame

    # Labels
    draw_text(canvas, "Alternating B/W border", (pad + w + 10, pad + 30),
              scale=0.5, color=(200, 200, 200))
    cv2.arrowedLine(canvas, (pad + w + 8, pad + 28), (pad + w - 5, pad + 10),
                    (200, 200, 200), 1, tipLength=0.15)

    draw_text(canvas, "Alignment marker (3x3)", (pad + w + 10, pad + 80),
              scale=0.5, color=(200, 200, 200))
    cv2.arrowedLine(canvas, (pad + w + 8, pad + 78), (pad + w - 40, pad + 40),
                    (200, 200, 200), 1, tipLength=0.1)

    draw_text(canvas, "Data cells (4 colors)", (pad + w + 10, pad + h // 2),
              scale=0.5, color=(0, 180, 255))
    cv2.arrowedLine(canvas, (pad + w + 8, pad + h // 2 - 2),
                    (pad + w - 60, pad + h // 2 - 2),
                    (0, 180, 255), 1, tipLength=0.1)

    # Color legend
    palette_labels = [("Black = 00", (0, 0, 0)),
                      ("White = 01", (255, 255, 255)),
                      ("Red   = 10", (0, 0, 255)),
                      ("Blue  = 11", (255, 0, 0))]
    y0 = pad + h // 2 + 50
    for i, (label, bgr) in enumerate(palette_labels):
        y = y0 + i * 30
        cv2.rectangle(canvas, (pad + w + 10, y - 12), (pad + w + 28, y + 6), bgr, -1)
        cv2.rectangle(canvas, (pad + w + 10, y - 12), (pad + w + 28, y + 6), (180, 180, 180), 1)
        draw_text(canvas, label, (pad + w + 35, y + 2),
                  scale=0.45, color=(200, 200, 200), bg_color=(32, 32, 32))

    draw_text(canvas, "Encoded Frame", (pad, pad - 15),
              scale=0.7, color=(0, 200, 255), bg_color=(32, 32, 32))

    return canvas


def annotate_webcam(cam, decoded_ok):
    """Annotate the webcam capture."""
    canvas = cam.copy()
    h, w = canvas.shape[:2]

    if decoded_ok:
        # Green border
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 255, 0), 4)
        draw_text(canvas, "DECODED: b'Hello from face2face!'",
                  (10, h - 20), scale=0.6, color=(0, 255, 0))
    else:
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
        draw_text(canvas, "Searching for frame...",
                  (10, h - 20), scale=0.6, color=(0, 0, 255))

    draw_text(canvas, "Webcam Capture", (10, 30),
              scale=0.7, color=(0, 200, 255))

    return canvas


def annotate_warped(warped):
    """Annotate the perspective-corrected image."""
    pad = 60
    h, w = warped.shape[:2]
    # Scale up for visibility
    scale = 2
    big = cv2.resize(warped, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    bh, bw = big.shape[:2]

    canvas = np.full((bh + 2 * pad, bw + 2 * pad, 3), 32, dtype=np.uint8)
    canvas[pad:pad + bh, pad:pad + bw] = big

    draw_text(canvas, "After Perspective Correction", (pad, pad - 15),
              scale=0.7, color=(0, 200, 255), bg_color=(32, 32, 32))
    draw_text(canvas, "Grid aligned, ready for cell sampling", (pad, pad + bh + 30),
              scale=0.5, color=(180, 180, 180), bg_color=(32, 32, 32))

    return canvas


def main():
    cfg = CodecConfig(grid_cols=GRID, grid_rows=GRID, bits_per_cell=BITS, cell_px=CELL_PX)
    encoder = FrameEncoder(config=cfg)
    decoder = ImageFrameDecoder(cfg)

    # Encode test message
    msg = b"Hello from face2face!"
    payload = msg + b"\x00" * (cfg.payload_bytes - len(msg))
    header = FrameHeader(msg_id=1, seq=0, total=1)
    frame = encoder.encode(payload, header)

    # --- Image 1: Annotated encoded frame ---
    annotated_frame = annotate_encoded_frame(frame)
    cv2.imwrite(f"{OUT}/01_encoded_frame.png", annotated_frame)
    print(f"Saved {OUT}/01_encoded_frame.png")

    # --- Image 5: Architecture diagram ---
    arch = draw_architecture()
    cv2.imwrite(f"{OUT}/05_architecture.png", arch)
    print(f"Saved {OUT}/05_architecture.png")

    # --- Images 2-4: Need webcam ---
    # Display padded frame on screen
    padded = np.zeros((frame.shape[0] + 2 * PAD, frame.shape[1] + 2 * PAD, 3), dtype=np.uint8)
    padded[PAD:PAD + frame.shape[0], PAD:PAD + frame.shape[1]] = frame

    cv2.namedWindow("face2face TX", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("face2face TX", padded)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("WARNING: No webcam. Skipping live capture images.")
        cap.release()
        cv2.destroyAllWindows()
        return

    print()
    print("Waiting for successful decode to capture images...")
    print("Point your webcam at the displayed frame.")
    print("Press q to skip webcam images.")
    print()

    # Wait for stable decode, then capture
    consecutive_ok = 0
    best_cam = None

    for attempt in range(600):  # ~30 seconds max
        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            print("Skipped webcam capture.")
            break

        ret, cam = cap.read()
        if not ret:
            continue

        cv2.imshow("Webcam", cam)

        try:
            corners = decoder._find_corners(cam)
            if corners is None:
                consecutive_ok = 0
                continue

            warped = decoder._perspective_transform(cam, corners)
            if warped is None:
                consecutive_ok = 0
                continue

            palette = decoder._calibrate_palette(warped)
            grid = decoder._sample_grid(warped, palette)
            dec_header, dec_payload = decoder.grid_decoder.decode_grid(grid)

            if dec_header and dec_payload:
                consecutive_ok += 1
                if consecutive_ok >= 5:
                    # Stable decode - capture everything
                    best_cam = cam.copy()

                    # Image 2: Webcam capture
                    annotated_cam = annotate_webcam(cam, True)
                    cv2.imwrite(f"{OUT}/02_webcam_capture.png", annotated_cam)
                    print(f"Saved {OUT}/02_webcam_capture.png")

                    # Image 3: Warped
                    annotated_warped = annotate_warped(warped)
                    cv2.imwrite(f"{OUT}/03_perspective_corrected.png", annotated_warped)
                    print(f"Saved {OUT}/03_perspective_corrected.png")

                    # Image 4: Decode success overlay
                    success_img = cam.copy()
                    h_c, w_c = success_img.shape[:2]

                    # Draw detected quad
                    pts = corners.astype(np.int32)
                    cv2.polylines(success_img, [pts], True, (0, 255, 0), 3)

                    # Corner labels
                    for i, (label, pt) in enumerate(zip(["TL", "TR", "BR", "BL"], pts)):
                        cv2.circle(success_img, tuple(pt), 8, (0, 255, 0), -1)
                        cv2.putText(success_img, label, (pt[0] + 12, pt[1] - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    decoded_text = dec_payload.rstrip(b"\x00").decode("utf-8", errors="replace")
                    draw_text(success_img, f'Decoded: "{decoded_text}"',
                              (10, h_c - 50), scale=0.7, color=(0, 255, 0))
                    draw_text(success_img, f"Grid: {GRID}x{GRID} | Colors: {cfg.n_colors} | "
                              f"Payload: {cfg.payload_bytes}B",
                              (10, h_c - 15), scale=0.5, color=(200, 200, 200))
                    draw_text(success_img, "Detected Frame Boundary", (10, 30),
                              scale=0.7, color=(0, 200, 255))

                    cv2.imwrite(f"{OUT}/04_decode_success.png", success_img)
                    print(f"Saved {OUT}/04_decode_success.png")
                    break
            else:
                consecutive_ok = 0
        except Exception:
            consecutive_ok = 0
            continue
    else:
        print("Could not get stable decode. Try adjusting webcam position.")

    cap.release()
    cv2.destroyAllWindows()
    print("\nDone! Images saved to docs/images/")


if __name__ == "__main__":
    main()
