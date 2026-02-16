"""Decode diagnostic: analyze webcam frames to identify decode issues.

Runs the decode pipeline step-by-step and shows exactly where failures
occur: corner detection, perspective transform, palette calibration,
or cell color matching.

Can operate in two modes:
  - Live: captures from webcam while displaying a test frame
  - Offline: analyzes saved webcam images (debug_webcam_*.png)

Usage:
    python tools/decode_diagnose.py                  # live mode
    python tools/decode_diagnose.py --offline         # analyze saved images
    python tools/decode_diagnose.py --image frame.png # analyze a specific image
"""

import argparse
import glob
import time

import cv2
import numpy as np

from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import ImageFrameDecoder


LABELS_4 = ["black", "white", "red", "blue"]


def diagnose_image(cam: np.ndarray, frame: np.ndarray, cfg: CodecConfig,
                   decoder: ImageFrameDecoder, save_prefix: str = ""):
    """Run decode pipeline step-by-step and print diagnostics."""
    labels = LABELS_4 if cfg.bits_per_cell == 2 else [str(i) for i in range(cfg.n_colors)]

    print(f"  Image size: {cam.shape[1]}x{cam.shape[0]}")

    # Step 1: Find corners
    corners = decoder._find_corners(cam)
    if corners is None:
        print("  FAIL: No corners found. Frame not visible to webcam.")
        return False
    print(f"  Corners: {corners.astype(int).tolist()}")

    # Step 2: Perspective transform
    warped = decoder._perspective_transform(cam, corners)
    if warped is None:
        print("  FAIL: Perspective transform failed.")
        return False

    if save_prefix:
        warped_path = f"{save_prefix}_warped.png"
        cv2.imwrite(warped_path, warped)
        print(f"  Warped saved: {warped_path}")

    # Step 3: Calibrate palette
    palette = decoder._calibrate_palette(warped)
    print(f"  Calibrated palette (BGR):")
    for i, color in enumerate(palette):
        expected = cfg.palette[i]
        label = labels[i] if i < len(labels) else f"color{i}"
        print(f"    {label}: expected={expected.tolist()} -> calibrated={color.tolist()}")

    # Step 4: Sample grid
    grid = decoder._sample_grid(warped, palette)

    # Get expected grid from the clean frame
    border_offset = cfg.border_px * cfg.cell_px
    margin = int(cfg.cell_px * 0.25)
    expected_grid = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=np.uint8)
    for r in range(cfg.grid_rows):
        for c in range(cfg.grid_cols):
            y = border_offset + r * cfg.cell_px + margin
            x = border_offset + c * cfg.cell_px + margin
            h = cfg.cell_px - 2 * margin
            w = cfg.cell_px - 2 * margin
            cell = frame[y:y + h, x:x + w]
            avg = cell.mean(axis=(0, 1))
            distances = np.sqrt(((cfg.palette.astype(np.float64) - avg) ** 2).sum(axis=1))
            expected_grid[r, c] = int(np.argmin(distances))

    # Compare
    total = cfg.grid_rows * cfg.grid_cols
    correct = int((grid == expected_grid).sum())
    wrong = total - correct
    print(f"\n  Grid: {correct}/{total} cells correct, {wrong} wrong ({100*correct/total:.1f}%)")

    if wrong > 0:
        print("  Mismatched cells:")
        count = 0
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                if grid[r, c] != expected_grid[r, c]:
                    count += 1
                    if count <= 20:
                        y = border_offset + r * cfg.cell_px + margin
                        x = border_offset + c * cfg.cell_px + margin
                        h = cfg.cell_px - 2 * margin
                        w = cfg.cell_px - 2 * margin
                        cell = warped[y:y + h, x:x + w]
                        avg = cell.mean(axis=(0, 1)).astype(int).tolist()
                        exp_l = labels[expected_grid[r, c]] if expected_grid[r, c] < len(labels) else "?"
                        got_l = labels[grid[r, c]] if grid[r, c] < len(labels) else "?"
                        print(f"    ({r:2d},{c:2d}): {exp_l:5s} -> {got_l:5s}  (observed BGR: {avg})")
        if count > 20:
            print(f"    ... and {count - 20} more")

    # Try decode
    dec_header, dec_payload = decoder.grid_decoder.decode_grid(grid)
    if dec_header and dec_payload:
        print(f"\n  DECODE SUCCESS: {dec_payload.rstrip(b'\\x00')}")
        return True
    elif dec_header:
        print(f"\n  Header OK but CRC failed ({wrong} misread cells)")
    else:
        print(f"\n  Header decode failed")
    return False


def run_offline(cfg, encoder, decoder, frame, images):
    """Analyze saved webcam images."""
    for img_path in images:
        print(f"\n{'='*60}")
        print(f"Analyzing: {img_path}")
        print(f"{'='*60}")

        cam = cv2.imread(img_path)
        if cam is None:
            print(f"  Cannot read {img_path}")
            continue

        prefix = img_path.rsplit(".", 1)[0]
        diagnose_image(cam, frame, cfg, decoder, save_prefix=prefix)


def run_live(cfg, encoder, decoder, frame, args):
    """Live mode: display frame and diagnose webcam captures."""
    pad = args.padding
    padded = np.zeros(
        (frame.shape[0] + 2 * pad, frame.shape[1] + 2 * pad, 3),
        dtype=np.uint8,
    )
    padded[pad:pad + frame.shape[0], pad:pad + frame.shape[1]] = frame

    cv2.namedWindow("face2face TX", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("face2face TX", padded)

    cap = cv2.VideoCapture(args.camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Press SPACE to capture and diagnose. Press q to quit.")
    snap_count = 0

    while True:
        ret, cam = cap.read()
        if ret:
            cv2.imshow("Webcam", cam)

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and ret:
            snap_count += 1
            prefix = f"diag_{snap_count:03d}"
            cv2.imwrite(f"{prefix}_raw.png", cam)
            print(f"\n{'='*60}")
            print(f"Snapshot #{snap_count}")
            print(f"{'='*60}")
            diagnose_image(cam, frame, cfg, decoder, save_prefix=prefix)

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Decode diagnostic tool")
    parser.add_argument("--grid", type=int, default=16, help="Grid size (default: 16)")
    parser.add_argument("--cell-px", type=int, default=28, help="Cell pixel size (default: 28)")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 4, 6], help="Bits per cell")
    parser.add_argument("--padding", type=int, default=80, help="Display padding (default: 80)")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index")
    parser.add_argument("--offline", action="store_true", help="Analyze saved debug_webcam_*.png files")
    parser.add_argument("--image", type=str, nargs="+", help="Specific image file(s) to analyze")
    args = parser.parse_args()

    cfg = CodecConfig(
        grid_cols=args.grid, grid_rows=args.grid,
        bits_per_cell=args.bits, cell_px=args.cell_px,
    )
    encoder = FrameEncoder(config=cfg)
    decoder = ImageFrameDecoder(cfg)

    msg = b"Hello from face2face!"
    payload = msg + b"\x00" * (cfg.payload_bytes - len(msg))
    header = FrameHeader(msg_id=1, seq=0, total=1)
    frame = encoder.encode(payload, header)

    print(f"Config: grid={args.grid}x{args.grid}, cell={args.cell_px}px, "
          f"bits={args.bits} ({cfg.n_colors} colors)")

    if args.offline or args.image:
        images = args.image or sorted(glob.glob("debug_webcam_*.png"))
        if not images:
            print("No images found. Use --image <path> or save debug images first.")
            return
        run_offline(cfg, encoder, decoder, frame, images)
    else:
        run_live(cfg, encoder, decoder, frame, args)


if __name__ == "__main__":
    main()
