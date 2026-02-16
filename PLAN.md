# face2face: HTTP Proxy Over Visual Channel (Webcam/Screen)

## Overview

An HTTP proxy that encodes network traffic into visual frames displayed on a screen, which are then captured by a webcam on another machine, decoded, and forwarded. This creates a unidirectional visual data link. Two such links (one in each direction) form a bidirectional channel capable of proxying HTTP traffic.

```
 Machine A (Client Side)                    Machine B (Server Side)
┌─────────────────────────┐                ┌─────────────────────────┐
│                         │                │                         │
│  curl/git/browser       │                │        Internet         │
│        │                │                │           ▲              │
│        ▼                │                │           │              │
│  HTTP Proxy Server      │                │  HTTP Forwarding Client │
│        │                │                │           ▲              │
│        ▼                │                │           │              │
│  Visual Encoder ──► SCREEN ─ ─ ─ ─►  WEBCAM ──► Visual Decoder   │
│                         │   light        │                         │
│  Visual Decoder ◄── WEBCAM ◄ ─ ─ ─ ─  SCREEN ◄── Visual Encoder  │
│                         │   light        │                         │
└─────────────────────────┘                └─────────────────────────┘
```

## Architecture

### System Components

#### 1. Visual Transport Layer (`visual/`)
The core innovation — encoding/decoding binary data as visual frames.

##### 1a. Frame Codec (`visual/codec.py`)
- **Encoding**: Binary data → grid of colored cells displayed on screen
- **Format**: Each frame is a grid of NxM cells, each cell is one of K colors
  - With 4 colors (2 bits/cell) and a 32x32 grid: 256 bytes/frame
  - With 16 colors (4 bits/cell) and a 32x32 grid: 512 bytes/frame
  - With 64 colors (6 bits/cell) and a 32x32 grid: 768 bytes/frame
- **Frame structure**:
  - Corner alignment markers (4 corners) for perspective correction
  - Header row: frame type, sequence number, total frames, chunk size, checksum
  - Data cells: the payload
  - Border: thick black/white alternating border for detection
- **Configurable**: grid size, color depth, cell size in pixels

##### 1b. Screen Renderer (`visual/renderer.py`)
- Opens a window (using pygame or OpenCV highgui) and displays encoded frames
- Renders the color grid with alignment markers
- Handles frame pacing — displays each frame for a configurable duration
- Shows synchronization patterns between frames (brief blank/marker frame)
- Full-screen mode support for maximum resolution

##### 1c. Webcam Capture (`visual/capture.py`)
- Captures frames from webcam using OpenCV
- Configurable camera index, resolution, FPS
- Frame preprocessing: brightness/contrast normalization

##### 1d. Frame Decoder (`visual/decoder.py`)
- Detects alignment markers to find the grid in the camera frame
- Applies perspective transform to correct for camera angle
- Samples each cell to determine its color value
- Extracts header, validates checksum
- Handles frame synchronization: detects new frames vs. repeated frames
- Reports decode errors/confidence

##### 1e. Error Correction (`visual/ecc.py`)
- Reed-Solomon error correction coding applied to each frame's payload
- Configurable redundancy level (e.g., 10-30% overhead)
- Allows recovery from misread cells due to lighting/angle issues

#### 2. Link Layer Protocol (`protocol/`)
Reliable data transfer over the unreliable visual channel.

##### 2a. Framing (`protocol/framing.py`)
- Splits large messages into frame-sized chunks
- Each chunk gets a sequence number and is independently encoded
- Reassembles chunks into complete messages on the receiving side
- Message format:
  ```
  [MSG_ID: 4 bytes][SEQ: 2 bytes][TOTAL: 2 bytes][FLAGS: 1 byte][PAYLOAD: N bytes][CRC32: 4 bytes]
  ```

##### 2b. Flow Control (`protocol/flow.py`)
- Stop-and-wait or sliding window ARQ
- ACK/NACK frames sent back over the reverse visual channel
- Timeout and retransmission for lost/corrupted frames
- Sequence numbers to detect duplicates and ordering

##### 2c. Link Manager (`protocol/link.py`)
- Manages one direction of the visual link (tx or rx)
- Coordinates encoder+renderer (tx) or capture+decoder (rx)
- Exposes async send/receive interface for upper layers
- Handles link establishment and keepalive

##### 2d. Channel (`protocol/channel.py`)
- Combines two unidirectional links into a bidirectional channel
- Multiplexes multiple logical streams (for concurrent HTTP requests)
- Stream IDs to match requests with responses
- Provides async `send(stream_id, data)` and `recv(stream_id) -> data` API

#### 3. HTTP Proxy Layer (`proxy/`)

##### 3a. Proxy Server (`proxy/server.py`) — Runs on Machine A
- Standard HTTP/HTTPS proxy server (listens on localhost:8080)
- Accepts HTTP CONNECT for HTTPS tunneling
- Accepts plain HTTP requests for HTTP proxying
- For each request:
  1. Serialize the request (method, URL, headers, body)
  2. Send over the visual channel
  3. Wait for response over the visual channel
  4. Return response to the client
- Compatible with `http_proxy`/`https_proxy` env vars
- Compatible with `git config http.proxy`

##### 3b. Proxy Forwarder (`proxy/forwarder.py`) — Runs on Machine B
- Receives serialized HTTP requests from the visual channel
- Makes the actual HTTP request to the target server
- Serializes the response (status, headers, body)
- Sends response back over the visual channel

##### 3c. Request Serialization (`proxy/serialization.py`)
- Compact binary serialization of HTTP requests and responses
- Includes: method, URL, headers, body
- Gzip compression of bodies
- Designed to minimize bytes over the visual channel

#### 4. Compression (`compression/`)

##### 4a. Payload Compression (`compression/compress.py`)
- All data sent over the visual channel is compressed
- zlib/gzip for general data
- Header-specific compression (HPACK-like dictionary for common HTTP headers)
- Adaptive: skip compression if data is already compressed (images, etc.)

#### 5. CLI & Configuration (`cli/`)

##### 5a. Main CLI (`cli/main.py`)
- `face2face client` — Start Machine A (proxy server + visual tx/rx)
- `face2face server` — Start Machine B (forwarder + visual tx/rx)
- `face2face calibrate` — Interactive calibration wizard
- `face2face benchmark` — Test throughput of the visual channel

##### 5b. Calibration (`cli/calibrate.py`)
- Interactive wizard to set up the visual link
- Step 1: Display test patterns, verify webcam can see screen
- Step 2: Auto-detect optimal grid size and color depth
- Step 3: Measure error rate and adjust ECC level
- Step 4: Measure throughput and display stats
- Saves calibration profile to config file

##### 5c. Configuration (`cli/config.py`)
- TOML configuration file support
- Settings: camera index, grid size, color depth, cell pixel size, ECC level, proxy port, timeouts, compression level

## Implementation Plan

### Phase 1: Visual Codec Foundation
**Files**: `visual/codec.py`, `visual/renderer.py`, `visual/capture.py`, `visual/decoder.py`

1. Implement frame encoding: binary data → color grid image (numpy array)
2. Implement screen renderer: display frames in a window using OpenCV
3. Implement webcam capture: grab frames from camera
4. Implement frame decoder: camera image → detect grid → read cells → binary data
5. Implement alignment marker detection and perspective correction
6. Add frame header with sequence number, checksum
7. Write tests: encode → decode round-trip with synthetic images

### Phase 2: Error Correction & Reliability
**Files**: `visual/ecc.py`, `protocol/framing.py`

1. Integrate Reed-Solomon ECC into frame codec
2. Implement message framing (split/reassemble large payloads)
3. Test error recovery with simulated noise

### Phase 3: Link Protocol
**Files**: `protocol/flow.py`, `protocol/link.py`, `protocol/channel.py`

1. Implement stop-and-wait ARQ with ACK/NACK
2. Build unidirectional link manager
3. Build bidirectional channel with stream multiplexing
4. Test with loopback (encode → decode on same machine)

### Phase 4: HTTP Proxy
**Files**: `proxy/server.py`, `proxy/forwarder.py`, `proxy/serialization.py`

1. Implement HTTP request/response serialization
2. Build proxy server (Machine A)
3. Build forwarder (Machine B)
4. Test with simple HTTP requests end-to-end

### Phase 5: Compression & Optimization
**Files**: `compression/compress.py`

1. Add zlib compression to all channel payloads
2. Add HTTP header dictionary compression
3. Tune parameters for throughput

### Phase 6: CLI & Calibration
**Files**: `cli/main.py`, `cli/calibrate.py`, `cli/config.py`

1. Build CLI with click/argparse
2. Implement calibration wizard
3. Add configuration file support
4. Add benchmark mode

### Phase 7: Integration & Polish

1. End-to-end testing with two machines
2. Git clone/push through the proxy
3. Performance tuning
4. Documentation

## Technology Stack

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python`)
- **Screen Display**: OpenCV `highgui` (simple, no extra deps)
- **Error Correction**: `reedsolo` (Reed-Solomon)
- **HTTP Proxy**: `aiohttp` or raw `asyncio` sockets
- **HTTP Client**: `aiohttp` or `httpx`
- **Compression**: `zlib` (stdlib)
- **Serialization**: `msgpack` (compact binary)
- **CLI**: `click`
- **Config**: `tomli` / `tomli-w`
- **Testing**: `pytest`

## Expected Performance

| Parameter | Conservative | Optimistic |
|-----------|-------------|------------|
| Grid size | 16x16 | 48x48 |
| Color depth | 4 colors (2 bit) | 16 colors (4 bit) |
| Bits per frame | 512 bits (64 B) | 9216 bits (1152 B) |
| Frame rate | 2 fps | 10 fps |
| ECC overhead | 30% | 10% |
| Raw throughput | ~90 B/s | ~10 KB/s |
| With compression | ~180 B/s | ~20 KB/s |

At ~200 B/s, a small git operation (a few KB) takes seconds. A larger clone (1 MB) would take ~90 minutes at conservative rates, or ~2 minutes at optimistic rates. This is functional for small operations.

## File Structure

```
face2face/
├── pyproject.toml
├── README.md (only if requested)
├── face2face/
│   ├── __init__.py
│   ├── visual/
│   │   ├── __init__.py
│   │   ├── codec.py          # Frame encoding/decoding
│   │   ├── renderer.py       # Screen display
│   │   ├── capture.py        # Webcam capture
│   │   ├── decoder.py        # Frame detection & reading
│   │   └── ecc.py            # Reed-Solomon error correction
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── framing.py        # Message chunking/reassembly
│   │   ├── flow.py           # ARQ flow control
│   │   ├── link.py           # Unidirectional link manager
│   │   └── channel.py        # Bidirectional multiplexed channel
│   ├── proxy/
│   │   ├── __init__.py
│   │   ├── server.py         # HTTP proxy server (Machine A)
│   │   ├── forwarder.py      # HTTP forwarder (Machine B)
│   │   └── serialization.py  # HTTP req/resp serialization
│   ├── compression/
│   │   ├── __init__.py
│   │   └── compress.py       # Payload compression
│   └── cli/
│       ├── __init__.py
│       ├── main.py           # CLI entry point
│       ├── calibrate.py      # Calibration wizard
│       └── config.py         # Configuration management
└── tests/
    ├── test_codec.py
    ├── test_ecc.py
    ├── test_framing.py
    ├── test_protocol.py
    ├── test_proxy.py
    └── test_e2e.py
```
