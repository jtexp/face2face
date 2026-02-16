"""Quick webcam check: verify the webcam is accessible and working.

Usage:
    python tools/webcam_check.py [--index 0]
"""

import argparse
import cv2


def main():
    parser = argparse.ArgumentParser(description="Check webcam access")
    parser.add_argument("--index", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--save", type=str, default=None, help="Save a test frame to this path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.index)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera at index {args.index}")
        return 1

    ret, frame = cap.read()
    if ret:
        print(f"Camera OK: {frame.shape[1]}x{frame.shape[0]} at index {args.index}")
        if args.save:
            cv2.imwrite(args.save, frame)
            print(f"Saved test frame to {args.save}")
    else:
        print(f"ERROR: Camera opened but cannot read frame at index {args.index}")
        cap.release()
        return 1

    cap.release()
    return 0


if __name__ == "__main__":
    exit(main())
