# Real Test Guide: Single Computer, Webcam Pointed at Screen

This guide walks you through testing face2face on a single machine by
pointing your webcam at your own screen.

## Prerequisites

- A computer with a webcam (built-in or USB)
- Python 3.10+
- A display large enough for the webcam to see the pattern clearly

Install the project:

```bash
pip install -e ".[dev]"
```

Verify it installed:

```bash
face2face --help
```

## How It Works

In the real two-machine setup:

```
Machine A (no internet)          Machine B (has internet)
┌──────────────────────┐         ┌──────────────────────┐
│  App (git, curl...)  │         │                      │
│         │            │         │                      │
│   ProxyServer:8080   │         │   ProxyForwarder     │
│         │            │         │         │            │
│   Screen (TX) ───webcam────>  Webcam (RX)            │
│   Webcam (RX) <────screen──── Screen (TX)            │
└──────────────────────┘         └──────────────────────┘
```

On a single machine, both roles run in separate terminals, and your webcam
looks at the same screen that's displaying the frames.

## Step-by-Step

### 1. Position your webcam

Point your webcam at a section of your screen. Ideally:

- The webcam should be 30-60 cm (1-2 ft) from the screen
- Angle should be as straight-on as possible (the decoder handles up to ~15
  degrees, but straight is best)
- Minimize glare and reflections (dim overhead lights, close blinds)
- The display window should fill a good portion of the webcam's field of view

If using a laptop's built-in webcam, prop the laptop up so the webcam looks
down at an external monitor, or use a USB webcam on a small tripod pointed
at your laptop screen.

### 2. Check your webcam works

```bash
python -c "
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    print(f'Camera OK: {frame.shape[1]}x{frame.shape[0]}')
    cv2.imwrite('/tmp/webcam_test.png', frame)
    print('Saved test frame to /tmp/webcam_test.png')
else:
    print('ERROR: Cannot read from camera')
cap.release()
"
```

If your webcam is not at index 0, try index 1, 2, etc. Use the `--config`
option or a config file to change `camera_index`.

### 3. Run the calibration wizard

```bash
face2face calibrate --save
```

This tests the encode/decode pipeline and saves optimal settings to
`~/.config/face2face/config.toml`.

### 4. Quick visual decode test (no proxy, just the codec)

Before testing the full proxy stack, verify the webcam can see and decode
a frame. Run this script:

```bash
python -c "
import time
import cv2
import numpy as np
from face2face.visual.codec import CodecConfig, FrameEncoder, FrameHeader
from face2face.visual.decoder import ImageFrameDecoder

# Use conservative settings: large cells, few colors
cfg = CodecConfig(grid_cols=16, grid_rows=16, bits_per_cell=2, cell_px=24)
encoder = FrameEncoder(config=cfg)
decoder = ImageFrameDecoder(cfg)

# Encode a test payload
payload = b'Hello from face2face!' + b'\x00' * (cfg.payload_bytes - 21)
header = FrameHeader(msg_id=1, seq=0, total=1)
frame = encoder.encode(payload, header)

# Display the frame on screen
cv2.namedWindow('face2face-test', cv2.WINDOW_NORMAL)
cv2.resizeWindow('face2face-test', frame.shape[1], frame.shape[0])
cv2.imshow('face2face-test', frame)
print(f'Displaying {frame.shape[1]}x{frame.shape[0]} frame...')
print('Position your webcam to see this window.')
print('Press any key in the frame window to capture and decode.')
print('Press q to quit.')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    key = cv2.waitKey(100) & 0xFF
    if key == ord('q'):
        break

    ret, camera_frame = cap.read()
    if not ret:
        continue

    # Show what the webcam sees
    cv2.imshow('webcam-view', camera_frame)

    # Try to decode
    dec_header, dec_payload = decoder.decode_image(camera_frame)
    if dec_header is not None:
        if dec_payload is not None:
            msg = dec_payload.rstrip(b'\x00')
            print(f'DECODED: {msg}')
        else:
            print('Found frame but CRC failed (try adjusting angle/distance)')
    # (no output = frame not detected yet, keep adjusting)

cap.release()
cv2.destroyAllWindows()
"
```

What to look for:

- **"DECODED: b'Hello from face2face!'"** -- everything works
- **"Found frame but CRC failed"** -- webcam sees the frame but some cells
  are being misread. Try: moving the webcam closer, reducing glare, using
  larger cells (`cell_px=32`)
- **No output at all** -- the frame isn't being detected. Make sure the
  full frame (including the white border) is visible in the webcam view.

### 5. Adjust settings if needed

If step 4 doesn't decode, try these changes:

| Problem | Fix |
|---|---|
| Frame not detected | Move webcam closer so the frame fills more of the view |
| CRC failures | Increase `cell_px` (try 28, 32, or 40) |
| Color misreads | Use `bits_per_cell=2` (only 4 colors, most robust) |
| Glare/reflections | Tilt screen or webcam slightly, dim lights |
| Blurry image | Focus webcam (if manual focus), move back slightly |

You can also create a config file:

```toml
# ~/.config/face2face/config.toml

[visual]
grid_cols = 16
grid_rows = 16
bits_per_cell = 2
cell_px = 32       # bigger cells = more robust, less throughput

[camera]
camera_index = 0
camera_width = 1280
camera_height = 720

[renderer]
frame_hold_ms = 800   # hold each frame longer for the webcam to capture
blank_hold_ms = 200
```

### 6. Run the full proxy stack (single machine test)

Once basic decode works, test the full HTTP proxy. You need **two terminals**.

**Terminal 1 -- Server (has internet access):**

```bash
face2face server
```

This starts the visual receiver + HTTP forwarder. It watches the webcam for
incoming request frames, makes the real HTTP request, and displays the
response as frames on screen.

**Terminal 2 -- Client (proxy side):**

```bash
face2face client --port 8080
```

This starts the proxy listener + visual transmitter. It accepts HTTP
requests on `localhost:8080`, encodes them as visual frames, and watches
the webcam for response frames.

**Terminal 3 -- Make a request through the proxy:**

```bash
export http_proxy=http://localhost:8080
curl http://httpbin.org/get
```

Or with git:

```bash
export http_proxy=http://localhost:8080
git clone http://github.com/octocat/Hello-World
```

### 7. Single-machine window layout

The tricky part of single-machine testing is that both the client and
server display frames on the same screen, and the single webcam needs to
see both. Recommended layout:

```
┌─────────────────────────────┐
│  ┌──────────┐ ┌──────────┐  │
│  │ Server   │ │ Client   │  │  <- Screen
│  │ TX window│ │ TX window│  │
│  └──────────┘ └──────────┘  │
│                             │
│        ┌──────────┐         │
│        │ Webcam   │         │  <- Position webcam here,
│        │ (pointed │         │     aimed at the screen
│        │ at both  │         │
│        │ windows) │         │
│        └──────────┘         │
└─────────────────────────────┘
```

In practice this is difficult because a single webcam can't easily
distinguish which window belongs to which side. For proper single-machine
testing, the **loopback** or **PairedChannel** approach (used by the
automated tests) is more reliable.

### 8. Loopback benchmark (no webcam needed)

To test the proxy stack without any camera hardware:

```bash
face2face benchmark --loopback --duration 5
```

This uses an in-memory channel (no visual link) to measure the proxy's
raw throughput.

## Troubleshooting

### "Cannot open camera"

- Check another app isn't using the webcam
- Try a different `camera_index` (0, 1, 2)
- On Linux, check permissions: `ls -la /dev/video*`

### Frame detected but CRC always fails

The adaptive palette calibration adjusts for color shifts automatically,
but extreme conditions can overwhelm it:

1. Reduce screen brightness to ~70% to avoid webcam overexposure
2. Disable Night Shift / f.lux / blue light filters
3. Set your display to sRGB color profile if possible
4. Increase `cell_px` for bigger, easier-to-read cells

### Very slow decoding

- Ensure `frame_hold_ms` is long enough for the webcam to capture at least
  2-3 frames per displayed pattern (e.g., 800ms at 30fps = 24 captures)
- Use a lower resolution grid (`grid_cols=16, grid_rows=16`) for faster
  but less data per frame

### Works in test but not through proxy

- Verify both terminals can see each other's display windows
- Check that the proxy server is listening: `curl http://localhost:8080/`
  should return a connection error (not "connection refused")
- Try increasing `proxy_timeout` in the config

## Automated Tests (no hardware needed)

The project includes comprehensive automated tests that simulate the full
visual pipeline without needing a webcam or screen:

```bash
# Run all 112 tests
pytest tests/ -v

# Just the camera simulation tests
pytest tests/test_camera_sim.py -v

# Geometry (off-axis camera) tests
pytest tests/test_geometry.py -v

# Full proxy integration test (uses PairedChannel, no hardware)
pytest tests/test_integration.py -v

# Visual encode → save to file → load → decode
pytest tests/test_visual_roundtrip.py -v
```

These tests cover:

- **Camera simulation**: perspective, barrel distortion, blur, color shift,
  vignetting, ambient light, noise, downscaling
- **Geometry**: rotation up to 15 degrees, keystone distortion, combined
  rotation + perspective, variable distance/scale
- **Compression artifacts**: JPEG at various quality levels
- **Full proxy stack**: HTTP GET, POST, JSON, large responses, status codes
