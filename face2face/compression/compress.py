"""Payload compression for the visual channel.

All data sent over the visual channel is compressed to minimize
the number of frames needed. Uses zlib for general compression with
a magic byte prefix to detect already-compressed data.
"""

from __future__ import annotations

import zlib

# Magic byte prefix to identify compressed data
COMPRESS_MAGIC = b"\xf2\xf2"
# Minimum size to bother compressing
MIN_COMPRESS_SIZE = 32


def compress(data: bytes, level: int = 6) -> bytes:
    """Compress data with zlib.

    Returns COMPRESS_MAGIC + compressed_data.
    If the data is too small or compression doesn't help, returns it as-is.
    """
    if len(data) < MIN_COMPRESS_SIZE:
        return data

    compressed = zlib.compress(data, level)

    # Only use compressed version if it's actually smaller
    if len(compressed) + len(COMPRESS_MAGIC) < len(data):
        return COMPRESS_MAGIC + compressed

    return data


def decompress(data: bytes) -> bytes:
    """Decompress data if it was compressed, otherwise return as-is."""
    if data[:len(COMPRESS_MAGIC)] == COMPRESS_MAGIC:
        return zlib.decompress(data[len(COMPRESS_MAGIC):])
    return data


# ---------------------------------------------------------------------------
# HTTP header dictionary compression (HPACK-like)
# ---------------------------------------------------------------------------

# Common HTTP header names — assigned short numeric codes
COMMON_HEADERS: dict[str, int] = {
    "accept": 1,
    "accept-encoding": 2,
    "accept-language": 3,
    "authorization": 4,
    "cache-control": 5,
    "connection": 6,
    "content-encoding": 7,
    "content-length": 8,
    "content-type": 9,
    "cookie": 10,
    "date": 11,
    "etag": 12,
    "host": 13,
    "if-modified-since": 14,
    "if-none-match": 15,
    "last-modified": 16,
    "location": 17,
    "pragma": 18,
    "referer": 19,
    "server": 20,
    "set-cookie": 21,
    "transfer-encoding": 22,
    "user-agent": 23,
    "vary": 24,
    "x-forwarded-for": 25,
    "x-requested-with": 26,
}

REVERSE_HEADERS: dict[int, str] = {v: k for k, v in COMMON_HEADERS.items()}


def compress_headers(headers: dict[str, str]) -> dict:
    """Replace common header names with short integer codes."""
    compressed = {}
    for key, value in headers.items():
        code = COMMON_HEADERS.get(key.lower())
        if code is not None:
            compressed[code] = value
        else:
            compressed[key] = value
    return compressed


def decompress_headers(headers: dict) -> dict[str, str]:
    """Restore header names from short integer codes."""
    restored = {}
    for key, value in headers.items():
        if isinstance(key, int):
            name = REVERSE_HEADERS.get(key, f"x-unknown-{key}")
        else:
            name = key
        restored[name] = value
    return restored
