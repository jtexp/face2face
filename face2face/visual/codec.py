"""Frame codec: encode binary data into color grids and decode them back.

A frame is an image consisting of:
- 4 corner alignment markers (for perspective correction)
- A thick alternating black/white border (for detection)
- A header row embedded in the data area
- Data cells arranged in an NxM grid, each cell colored to represent bits

Color palettes:
- 4 colors  → 2 bits per cell
- 16 colors → 4 bits per cell
- 64 colors → 6 bits per cell
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKER_SIZE_CELLS = 3  # alignment marker occupies 3x3 cells in each corner
HEADER_BYTES = 15  # msg_id(4) + seq(2) + total(2) + flags(1) + hdr_crc16(2) + payload_crc32(4)


class FrameFlags(IntEnum):
    NONE = 0x00
    ACK = 0x01
    NACK = 0x02
    SYN = 0x04
    FIN = 0x08
    DATA = 0x10
    KEEPALIVE = 0x20


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

def _make_palette(bits_per_cell: int) -> np.ndarray:
    """Generate an evenly-spaced palette with 2**bits_per_cell colors in BGR."""
    n = 1 << bits_per_cell
    if n <= 4:
        # Hand-picked high-contrast palette (BGR)
        return np.array([
            [0, 0, 0],        # black  → 0b00
            [255, 255, 255],  # white  → 0b01
            [0, 0, 255],      # red    → 0b10
            [255, 0, 0],      # blue   → 0b11
        ], dtype=np.uint8)[:n]
    elif n <= 16:
        # 16-color palette — 4 brightness × 4 hues
        colors = []
        hues = [
            (0, 0, 0), (255, 255, 255),
            (0, 0, 255), (255, 0, 0),
            (0, 255, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0),
            (128, 0, 0), (0, 128, 0),
            (0, 0, 128), (128, 128, 0),
            (128, 0, 128), (0, 128, 128),
            (64, 64, 64), (192, 192, 192),
        ]
        for bgr in hues[:n]:
            colors.append(bgr)
        return np.array(colors, dtype=np.uint8)
    else:
        # Uniform cube in BGR, up to 64 colors (4x4x4)
        steps = int(round(n ** (1 / 3)))
        colors = []
        for b in np.linspace(0, 255, steps, dtype=np.uint8):
            for g in np.linspace(0, 255, steps, dtype=np.uint8):
                for r in np.linspace(0, 255, steps, dtype=np.uint8):
                    colors.append((int(b), int(g), int(r)))
        return np.array(colors[:n], dtype=np.uint8)


# Pre-built palettes
PALETTES = {
    2: _make_palette(2),
    4: _make_palette(4),
    6: _make_palette(6),
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class CodecConfig:
    """Configuration for the visual codec."""
    grid_cols: int = 32        # number of data cells horizontally
    grid_rows: int = 32        # number of data cells vertically
    bits_per_cell: int = 2     # 2, 4, or 6
    cell_px: int = 20          # pixel size of each cell
    border_px: int = 4         # border thickness in cells (alternating B/W)
    marker_cells: int = MARKER_SIZE_CELLS

    @property
    def n_colors(self) -> int:
        return 1 << self.bits_per_cell

    @property
    def palette(self) -> np.ndarray:
        return PALETTES[self.bits_per_cell]

    @property
    def data_cells(self) -> int:
        """Total data cells available (excluding alignment markers)."""
        total = self.grid_rows * self.grid_cols
        marker_area = 4 * (self.marker_cells ** 2)
        return total - marker_area

    @property
    def header_cells(self) -> int:
        """Number of cells needed for the frame header."""
        return _bytes_to_cells(HEADER_BYTES, self.bits_per_cell)

    @property
    def payload_cells(self) -> int:
        return self.data_cells - self.header_cells

    @property
    def payload_bytes(self) -> int:
        """Max payload bytes per frame (before ECC)."""
        return (self.payload_cells * self.bits_per_cell) // 8

    @property
    def image_width(self) -> int:
        return (self.grid_cols + 2 * self.border_px) * self.cell_px

    @property
    def image_height(self) -> int:
        return (self.grid_rows + 2 * self.border_px) * self.cell_px


def _bytes_to_cells(n_bytes: int, bits_per_cell: int) -> int:
    """How many cells are needed to encode *n_bytes*."""
    total_bits = n_bytes * 8
    return (total_bits + bits_per_cell - 1) // bits_per_cell


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def _bytes_to_symbols(data: bytes, bits_per_cell: int) -> list[int]:
    """Convert bytes to a list of cell symbols (each in 0..n_colors-1)."""
    symbols: list[int] = []
    mask = (1 << bits_per_cell) - 1
    bit_buf = 0
    bit_count = 0
    for byte in data:
        bit_buf = (bit_buf << 8) | byte
        bit_count += 8
        while bit_count >= bits_per_cell:
            bit_count -= bits_per_cell
            symbols.append((bit_buf >> bit_count) & mask)
    # Flush remaining bits (zero-padded)
    if bit_count > 0:
        symbols.append((bit_buf << (bits_per_cell - bit_count)) & mask)
    return symbols


def _symbols_to_bytes(symbols: list[int], bits_per_cell: int, n_bytes: int) -> bytes:
    """Convert cell symbols back to bytes."""
    bit_buf = 0
    bit_count = 0
    result = bytearray()
    for sym in symbols:
        bit_buf = (bit_buf << bits_per_cell) | sym
        bit_count += bits_per_cell
        while bit_count >= 8 and len(result) < n_bytes:
            bit_count -= 8
            result.append((bit_buf >> bit_count) & 0xFF)
    return bytes(result[:n_bytes])


# ---------------------------------------------------------------------------
# Marker patterns
# ---------------------------------------------------------------------------

def _marker_pattern() -> np.ndarray:
    """3x3 marker pattern — a bullseye-like pattern for corner detection.

    Returns array of shape (3, 3) with BGR color indices:
    B W B
    W B W
    B W B
    Where B=black(0), W=white(1).
    """
    return np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ], dtype=np.uint8)


# ---------------------------------------------------------------------------
# Frame header
# ---------------------------------------------------------------------------

def _header_crc16(packed_fields: bytes) -> int:
    """CRC16 over header fields (msg_id + seq + total + flags) for integrity."""
    return zlib.crc32(packed_fields) & 0xFFFF


@dataclass
class FrameHeader:
    msg_id: int = 0       # 4 bytes — message identifier
    seq: int = 0           # 2 bytes — sequence number within message
    total: int = 1         # 2 bytes — total frames in message
    flags: int = FrameFlags.DATA  # 1 byte
    crc32: int = 0         # 4 bytes — CRC32 of payload

    _FIELDS_FMT = ">IHHB"  # 9 bytes: msg_id + seq + total + flags
    _FULL_FMT = ">IHHBHI"  # 15 bytes: fields + hdr_crc16 + payload_crc32

    def pack(self) -> bytes:
        fields = struct.pack(self._FIELDS_FMT, self.msg_id, self.seq,
                             self.total, self.flags)
        hdr_crc = _header_crc16(fields)
        return fields + struct.pack(">HI", hdr_crc, self.crc32)

    @classmethod
    def unpack(cls, data: bytes) -> Optional["FrameHeader"]:
        """Unpack header from bytes. Returns None if header CRC fails."""
        if len(data) < HEADER_BYTES:
            return None
        msg_id, seq, total, flags, hdr_crc, crc = struct.unpack(
            cls._FULL_FMT, data[:HEADER_BYTES])
        # Verify header integrity
        fields = struct.pack(cls._FIELDS_FMT, msg_id, seq, total, flags)
        if _header_crc16(fields) != hdr_crc:
            return None
        return cls(msg_id=msg_id, seq=seq, total=total, flags=flags, crc32=crc)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

@dataclass
class FrameEncoder:
    config: CodecConfig = field(default_factory=CodecConfig)

    def encode(self, payload: bytes, header: FrameHeader) -> np.ndarray:
        """Encode payload + header into a BGR image (numpy array)."""
        cfg = self.config
        palette = cfg.palette

        # Compute CRC over the full payload area (including the zero-padding
        # the decoder will see) so that short payloads pass CRC on decode.
        padded = payload.ljust(cfg.payload_bytes, b"\x00")
        header.crc32 = zlib.crc32(padded) & 0xFFFFFFFF
        header_bytes = header.pack()

        # Convert header + payload to symbols
        all_data = header_bytes + payload
        symbols = _bytes_to_symbols(all_data, cfg.bits_per_cell)

        # Pad symbols to fill the grid
        total_data_cells = cfg.data_cells
        while len(symbols) < total_data_cells:
            symbols.append(0)

        # Build the cell grid (grid_rows x grid_cols) of symbol indices
        grid = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=np.uint8)
        marker = _marker_pattern()
        mc = cfg.marker_cells

        # Place data symbols, skipping marker corners
        sym_idx = 0
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                if _in_marker(r, c, cfg.grid_rows, cfg.grid_cols, mc):
                    continue
                if sym_idx < len(symbols):
                    grid[r, c] = symbols[sym_idx]
                    sym_idx += 1

        # Place alignment markers in 4 corners
        _place_marker(grid, 0, 0, marker, mc)
        _place_marker(grid, 0, cfg.grid_cols - mc, marker, mc)
        _place_marker(grid, cfg.grid_rows - mc, 0, marker, mc)
        _place_marker(grid, cfg.grid_rows - mc, cfg.grid_cols - mc, marker, mc)

        # Render to image
        return self._render_grid(grid, palette)

    def _render_grid(self, grid: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Render symbol grid to a BGR image with border."""
        cfg = self.config
        img_h = cfg.image_height
        img_w = cfg.image_width
        img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

        # Draw alternating border
        border_total = cfg.border_px * cfg.cell_px
        for i in range(cfg.border_px):
            color = (255, 255, 255) if i % 2 == 0 else (0, 0, 0)
            x0 = i * cfg.cell_px
            y0 = i * cfg.cell_px
            x1 = img_w - i * cfg.cell_px
            y1 = img_h - i * cfg.cell_px
            img[y0:y0 + cfg.cell_px, x0:x1] = color  # top
            img[y1 - cfg.cell_px:y1, x0:x1] = color  # bottom
            img[y0:y1, x0:x0 + cfg.cell_px] = color  # left
            img[y0:y1, x1 - cfg.cell_px:x1] = color  # right

        # Draw cells
        ox = border_total
        oy = border_total
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                color = palette[grid[r, c]]
                y = oy + r * cfg.cell_px
                x = ox + c * cfg.cell_px
                img[y:y + cfg.cell_px, x:x + cfg.cell_px] = color

        return img


def _in_marker(r: int, c: int, rows: int, cols: int, mc: int) -> bool:
    """Check if cell (r, c) falls inside any of the 4 corner markers."""
    in_top = r < mc
    in_bot = r >= rows - mc
    in_left = c < mc
    in_right = c >= cols - mc
    return (in_top and in_left) or (in_top and in_right) or \
           (in_bot and in_left) or (in_bot and in_right)


def _place_marker(grid: np.ndarray, r0: int, c0: int,
                  marker: np.ndarray, mc: int) -> None:
    """Write marker pattern into the grid."""
    for dr in range(mc):
        for dc in range(mc):
            grid[r0 + dr, c0 + dc] = marker[dr, dc]


# ---------------------------------------------------------------------------
# Decoder (from clean grid — perspective-corrected grid of symbol indices)
# ---------------------------------------------------------------------------

@dataclass
class FrameDecoder:
    config: CodecConfig = field(default_factory=CodecConfig)

    def decode_grid(self, grid: np.ndarray) -> tuple[Optional[FrameHeader], Optional[bytes]]:
        """Decode a symbol grid back to header + payload.

        *grid* has shape (grid_rows, grid_cols) with uint8 symbol indices.
        Returns (header, payload) or (None, None) on error.
        """
        cfg = self.config
        mc = cfg.marker_cells

        # Extract symbols in the same order the encoder placed them
        symbols: list[int] = []
        for r in range(cfg.grid_rows):
            for c in range(cfg.grid_cols):
                if _in_marker(r, c, cfg.grid_rows, cfg.grid_cols, mc):
                    continue
                symbols.append(int(grid[r, c]))

        # Convert symbols to bytes
        total_bytes = HEADER_BYTES + cfg.payload_bytes
        raw = _symbols_to_bytes(symbols, cfg.bits_per_cell, total_bytes)

        if len(raw) < HEADER_BYTES:
            return None, None

        header = FrameHeader.unpack(raw[:HEADER_BYTES])
        if header is None:
            return None, None

        payload = raw[HEADER_BYTES:HEADER_BYTES + cfg.payload_bytes]

        # Verify payload CRC
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != header.crc32:
            # CRC mismatch — may need ECC or the frame is corrupt
            return header, None

        return header, payload
