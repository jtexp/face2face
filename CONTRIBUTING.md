# Contributing

## Project structure

```
face2face/
  visual/          # Frame encoding/decoding, screen rendering, webcam capture
    codec.py       #   Color grid codec (encode/decode binary <-> color cells)
    decoder.py     #   Frame detection + perspective correction from camera images
    renderer.py    #   Screen display with OpenCV
    capture.py     #   Webcam capture
    camera_sim.py  #   Realistic camera degradation simulator (for tests)
    ecc.py         #   Reed-Solomon error correction
  protocol/        # Link-layer protocol
    framing.py     #   Message chunking and reassembly
    flow.py        #   ARQ flow control (stop-and-wait / sliding window)
    link.py        #   Unidirectional visual link manager
    channel.py     #   Bidirectional multiplexed channel
  proxy/           # HTTP proxy layer
    server.py      #   HTTP/HTTPS proxy server (Machine A)
    forwarder.py   #   HTTP request forwarder (Machine B)
    serialization.py # Compact binary serialization for HTTP messages
  compression/     # Payload compression (zlib + header dictionary)
  cli/             # CLI entry points (client, server, calibrate, benchmark)
tools/             # Real-hardware test and diagnostic scripts
tests/             # Automated test suite (118 tests, no hardware needed)
```

## Tests

All 118 tests run without any camera hardware:

```bash
pytest tests/ -v                        # full suite
pytest tests/test_camera_sim.py -v      # camera degradation simulation
pytest tests/test_geometry.py -v        # off-axis rotation/keystone
pytest tests/test_integration.py -v     # full proxy stack (loopback)
pytest tests/test_visual_roundtrip.py -v # encode -> save -> load -> decode
```

## Tools

| Script | Purpose |
|--------|---------|
| `tools/webcam_check.py` | Verify webcam is accessible |
| `tools/live_decode_test.py` | Live encode-display-capture-decode loop |
| `tools/decode_diagnose.py` | Step-by-step decode diagnostic |
| `tools/capture_readme_images.py` | Generate the images in the README |
