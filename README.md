# face2face

An HTTP proxy that communicates through screens and webcams. It encodes HTTP traffic into visual frames displayed on a screen, which are captured by a webcam on another machine, decoded, and forwarded to the internet.

Two machines, two webcams, two screens -- no network cable required.

![Architecture](docs/images/05_architecture.png)

## Status

**Working end-to-end.** HTTP requests successfully proxy through the visual channel between two machines using webcams and screens. Tested with `curl` and `httpbin.org`.

## How it works

**Machine A** (air-gapped, no internet) runs an HTTP proxy on `localhost:8080`. When an app like `curl` or `git` makes a request, the proxy encodes it into a colored grid and displays it on screen.

**Machine B** (has internet) watches Machine A's screen with a webcam, decodes the request, makes the real HTTP call, then encodes the response and displays it on *its* screen for Machine A's webcam to pick up.

The visual link uses:
- A **color grid** where each cell encodes 1-6 bits (2 to 64 colors; defaults to black/white for reliability)
- **Alignment markers** in the corners for perspective correction
- **Adaptive palette calibration** to handle color shifts from different screens/cameras
- **Reed-Solomon error correction** to recover from misread cells
- **ARQ flow control** with ACK/NACK for reliable delivery

### Encoding

Binary data is packed into a grid of colored cells. By default each cell is black or white (1 bit), which maximizes camera decode reliability. Higher throughput modes use 4, 16, or 64 colors. An alternating black/white border and 3x3 checkerboard markers in each corner enable the decoder to find and correct the frame from any angle.

![Encoded Frame](docs/images/01_encoded_frame.png)

### Capture and decode

The webcam captures the screen, finds the frame boundary, and applies a perspective transform to straighten the grid. The decoder calibrates its color palette from the known marker cells, then reads each data cell.

| Webcam capture | Perspective corrected |
|:-:|:-:|
| ![Webcam Capture](docs/images/02_webcam_capture.png) | ![Perspective Corrected](docs/images/03_perspective_corrected.png) |

The detected quad (green outline) is used to warp the image back to a clean rectangle, aligning the grid for cell-by-cell sampling.

![Decode Success](docs/images/04_decode_success.png)

## Quick start

```bash
pip install -e ".[dev]"
```

### Full proxy (two machines)

```bash
# Machine B (has internet):
face2face server

# Machine A (air-gapped):
face2face client --port 8080

# Then on Machine A:
export http_proxy=http://localhost:8080
curl http://httpbin.org/get

# Override visual settings for faster throughput:
face2face client --colors 4 --grid 32x32 --cell-size 20
face2face server --colors 4 --grid 32x32 --cell-size 20
```

## Performance

| Setting | Default (B/W) | Fast (4-color) | Optimistic |
|---------|--------------|----------------|-----------|
| Grid | 24x24 | 32x32 | 48x48 |
| Colors | 2 (1-bit) | 4 (2-bit) | 16 (4-bit) |
| Bytes/frame | ~60 | ~201 | ~1,152 |
| Frame rate | 2 fps | 2 fps | 10 fps |
| **Throughput** | **~100 B/s** | **~336 B/s** | **~10 KB/s** |

The default black/white mode prioritizes decode reliability over speed. Use `--colors 4 --grid 32x32 --cell-size 20` on both sides to switch to 4-color mode for higher throughput once the link is stable.

At default settings, small HTTP responses return in seconds. Larger transfers (cloning a repo, downloading files) are slow but functional -- this is designed for air-gapped environments where *any* connectivity is valuable.

## Configuration

Settings are stored in `~/.config/face2face/config.toml`:

```toml
[visual]
grid_cols = 24        # grid size (default 24x24, up to 48x48 for more data)
grid_rows = 24
bits_per_cell = 1     # 1 = B/W (most reliable), 2 = 4 colors, 4 = 16 colors
cell_px = 28          # pixel size per cell

[camera]
camera_index = 0
camera_width = 1280
camera_height = 720

[renderer]
frame_hold_ms = 500   # display time per frame
blank_hold_ms = 100   # sync gap between frames

[error_correction]
ecc_nsym = 20         # Reed-Solomon redundancy symbols
```

## Debugging

Use `--monitor` / `-m` to open a live-preview window showing the webcam feed with frame detection overlay (green quad + status circle). Supports ROI selection -- click and drag to zoom the decoder into the grid area:

```bash
face2face client --monitor
face2face server --monitor
```

If the visual link fails to decode, use `--debug-capture <dir>` to save raw, warped, and grid-overlay images per capture for diagnosis:

```bash
face2face client --debug-capture ./debug-frames
face2face server --debug-capture ./debug-frames
```

## Further reading

- [INSTALL.md](INSTALL.md) -- remote machine bootstrap, updating
- [CONTRIBUTING.md](CONTRIBUTING.md) -- project structure, tests, tools
- [REAL_TEST_GUIDE.md](REAL_TEST_GUIDE.md) -- single-machine webcam testing, WSL2 setup
