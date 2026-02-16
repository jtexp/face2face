"""Tests for compression module."""

import pytest

from face2face.compression.compress import (
    compress,
    compress_headers,
    decompress,
    decompress_headers,
)


class TestCompression:
    def test_roundtrip(self):
        data = b"Hello, world! " * 100
        compressed = compress(data)
        decompressed = decompress(compressed)
        assert decompressed == data

    def test_small_data_not_compressed(self):
        data = b"Hi"
        result = compress(data)
        assert result == data  # too small to compress

    def test_already_compressed_data(self):
        data = bytes(range(256))  # random-ish, may not compress well
        result = compress(data)
        decompressed = decompress(result)
        assert decompressed == data

    def test_empty_data(self):
        assert decompress(compress(b"")) == b""

    def test_large_data(self):
        data = b"A" * 1_000_000
        compressed = compress(data)
        assert len(compressed) < len(data)
        assert decompress(compressed) == data

    def test_decompress_uncompressed(self):
        data = b"not compressed"
        assert decompress(data) == data


class TestHeaderCompression:
    def test_common_headers_compressed(self):
        headers = {
            "Content-Type": "text/html",
            "Host": "example.com",
            "X-Custom": "value",
        }
        compressed = compress_headers(headers)
        # content-type → 9, host → 13
        assert 9 in compressed
        assert 13 in compressed
        assert "X-Custom" in compressed

    def test_roundtrip(self):
        headers = {
            "Content-Type": "text/html",
            "Accept": "*/*",
            "X-Custom-Header": "test",
        }
        compressed = compress_headers(headers)
        restored = decompress_headers(compressed)
        assert restored["content-type"] == "text/html"
        assert restored["accept"] == "*/*"
        assert restored["X-Custom-Header"] == "test"

    def test_empty_headers(self):
        assert compress_headers({}) == {}
        assert decompress_headers({}) == {}
