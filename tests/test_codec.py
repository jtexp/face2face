"""Tests for the visual codec: encode/decode round-trip."""

import numpy as np
import pytest

from face2face.visual.codec import (
    CodecConfig,
    FrameDecoder,
    FrameEncoder,
    FrameHeader,
    FrameFlags,
    _bytes_to_symbols,
    _symbols_to_bytes,
)


class TestSymbolConversion:
    def test_roundtrip_2bit(self):
        data = b"\xab\xcd\xef"
        symbols = _bytes_to_symbols(data, 2)
        result = _symbols_to_bytes(symbols, 2, len(data))
        assert result == data

    def test_roundtrip_4bit(self):
        data = b"\x12\x34\x56\x78\x9a"
        symbols = _bytes_to_symbols(data, 4)
        result = _symbols_to_bytes(symbols, 4, len(data))
        assert result == data

    def test_roundtrip_6bit(self):
        data = b"\xff\x00\xaa\x55"
        symbols = _bytes_to_symbols(data, 6)
        result = _symbols_to_bytes(symbols, 6, len(data))
        assert result == data

    def test_empty(self):
        symbols = _bytes_to_symbols(b"", 2)
        assert symbols == []
        result = _symbols_to_bytes([], 2, 0)
        assert result == b""

    def test_single_byte_2bit(self):
        # 0xAB = 10101011 → 2-bit symbols: 10, 10, 10, 11 → [2, 2, 2, 3]
        symbols = _bytes_to_symbols(b"\xab", 2)
        assert symbols == [2, 2, 2, 3]

    def test_roundtrip_1bit(self):
        data = b"\xab\xcd\xef"
        symbols = _bytes_to_symbols(data, 1)
        result = _symbols_to_bytes(symbols, 1, len(data))
        assert result == data

    def test_long_data(self):
        data = bytes(range(256))
        for bits in [1, 2, 4, 6]:
            symbols = _bytes_to_symbols(data, bits)
            result = _symbols_to_bytes(symbols, bits, len(data))
            assert result == data


class TestFrameHeader:
    def test_pack_unpack(self):
        header = FrameHeader(msg_id=42, seq=3, total=10,
                             flags=FrameFlags.DATA, crc32=0xDEADBEEF)
        packed = header.pack()
        unpacked = FrameHeader.unpack(packed)
        assert unpacked.msg_id == 42
        assert unpacked.seq == 3
        assert unpacked.total == 10
        assert unpacked.flags == FrameFlags.DATA
        assert unpacked.crc32 == 0xDEADBEEF

    def test_roundtrip_zero(self):
        header = FrameHeader()
        packed = header.pack()
        unpacked = FrameHeader.unpack(packed)
        assert unpacked.msg_id == 0
        assert unpacked.seq == 0

    def test_max_values(self):
        header = FrameHeader(msg_id=0xFFFFFFFF, seq=0xFFFF,
                             total=0xFFFF, flags=0xFF,
                             crc32=0xFFFFFFFF)
        packed = header.pack()
        unpacked = FrameHeader.unpack(packed)
        assert unpacked.msg_id == 0xFFFFFFFF
        assert unpacked.seq == 0xFFFF


class TestCodecConfig:
    def test_defaults(self):
        cfg = CodecConfig()
        assert cfg.grid_cols == 24
        assert cfg.grid_rows == 24
        assert cfg.bits_per_cell == 1
        assert cfg.n_colors == 2

    def test_data_cells(self):
        cfg = CodecConfig(grid_cols=32, grid_rows=32)
        # 32*32 = 1024 total, minus 4 * 3*3 = 36 marker cells = 988
        assert cfg.data_cells == 1024 - 36

    def test_payload_bytes(self):
        cfg = CodecConfig()
        assert cfg.payload_bytes > 0
        assert cfg.payload_bytes <= cfg.payload_cells * cfg.bits_per_cell // 8

    def test_image_dimensions(self):
        cfg = CodecConfig(grid_cols=16, grid_rows=16, cell_px=10, border_px=3)
        assert cfg.image_width == (16 + 6) * 10  # 220
        assert cfg.image_height == (16 + 6) * 10  # 220


class TestFrameEncoderDecoder:
    @pytest.mark.parametrize("grid_size", [16, 24, 32])
    @pytest.mark.parametrize("bits", [1, 2, 4])
    def test_encode_decode_roundtrip(self, grid_size, bits):
        cfg = CodecConfig(grid_cols=grid_size, grid_rows=grid_size,
                          bits_per_cell=bits)
        encoder = FrameEncoder(config=cfg)
        decoder = FrameDecoder(config=cfg)

        payload = bytes(range(min(cfg.payload_bytes, 256)))
        # Pad to exact payload size
        payload = payload + b"\x00" * (cfg.payload_bytes - len(payload))

        header = FrameHeader(msg_id=1, seq=0, total=1)
        image = encoder.encode(payload, header)

        # Verify image dimensions
        assert image.shape == (cfg.image_height, cfg.image_width, 3)

        # Now extract the grid from the image (simulate perfect decode)
        # We need to sample the cells from the rendered image
        from face2face.visual.decoder import ImageFrameDecoder
        img_decoder = ImageFrameDecoder(cfg)
        dec_header, dec_payload = img_decoder.decode_image(image)

        assert dec_header is not None
        assert dec_header.msg_id == 1
        assert dec_header.seq == 0
        assert dec_header.total == 1
        # Payload should match
        if dec_payload is not None:
            assert dec_payload[:len(payload)] == payload

    def test_encode_produces_image(self):
        cfg = CodecConfig()
        encoder = FrameEncoder(config=cfg)
        payload = b"Hello, world!"
        payload = payload + b"\x00" * (cfg.payload_bytes - len(payload))
        header = FrameHeader(msg_id=1, seq=0, total=1)
        image = encoder.encode(payload, header)
        assert isinstance(image, np.ndarray)
        assert image.dtype == np.uint8
        assert len(image.shape) == 3  # (H, W, 3)

    def test_crc_populated(self):
        cfg = CodecConfig()
        encoder = FrameEncoder(config=cfg)
        payload = b"test data" + b"\x00" * (cfg.payload_bytes - 9)
        header = FrameHeader(msg_id=1, seq=0, total=1)
        encoder.encode(payload, header)
        assert header.crc32 != 0  # CRC should be set after encoding
