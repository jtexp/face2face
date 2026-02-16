"""Message framing: split large messages into frame-sized chunks and reassemble.

Each message gets a unique msg_id. The message is split into chunks, each
of which fits into a single visual frame payload. Each chunk is sent as a
frame with:
  - msg_id: identifies the message
  - seq: chunk sequence number (0-based)
  - total: total number of chunks
  - flags: DATA for data frames
  - crc32: per-chunk integrity check (handled by codec layer)

On the receiving side, chunks are collected and reassembled in order.
"""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from ..visual.codec import FrameFlags, FrameHeader
from ..visual.ecc import ECCCodec, ECCConfig


@dataclass
class FramingConfig:
    max_payload_per_frame: int = 200  # bytes per frame (after ECC overhead)
    ecc_config: ECCConfig = field(default_factory=ECCConfig)


class MessageFramer:
    """Splits a message into frame-sized chunks."""

    def __init__(self, config: Optional[FramingConfig] = None):
        self.config = config or FramingConfig()
        self.ecc = ECCCodec(self.config.ecc_config)
        self._msg_counter = 0
        self._lock = threading.Lock()

    def _next_msg_id(self) -> int:
        with self._lock:
            self._msg_counter = (self._msg_counter + 1) & 0xFFFFFFFF
            return self._msg_counter

    def frame_message(self, data: bytes, msg_id: int | None = None,
                      flags: int = FrameFlags.DATA
                      ) -> list[tuple[bytes, FrameHeader]]:
        """Split *data* into frame-ready (payload, header) pairs.

        Each payload has ECC applied. The caller should pass each pair
        to the visual encoder.
        """
        if msg_id is None:
            msg_id = self._next_msg_id()

        chunk_size = self.ecc.max_payload(self.config.max_payload_per_frame)
        if chunk_size <= 0:
            raise ValueError("ECC overhead exceeds frame payload capacity")

        chunks = []
        offset = 0
        while offset < len(data):
            chunks.append(data[offset:offset + chunk_size])
            offset += chunk_size

        if not chunks:
            chunks = [b""]

        total = len(chunks)
        frames = []
        for seq, chunk in enumerate(chunks):
            ecc_chunk = self.ecc.encode(chunk)
            header = FrameHeader(
                msg_id=msg_id,
                seq=seq,
                total=total,
                flags=flags,
            )
            frames.append((ecc_chunk, header))

        return frames

    def frame_control(self, flags: int, msg_id: int = 0,
                      seq: int = 0) -> tuple[bytes, FrameHeader]:
        """Create a control frame (ACK, NACK, SYN, FIN, KEEPALIVE)."""
        header = FrameHeader(
            msg_id=msg_id,
            seq=seq,
            total=1,
            flags=flags,
        )
        payload = self.ecc.encode(b"")
        return payload, header


class MessageAssembler:
    """Collects frame chunks and reassembles complete messages."""

    def __init__(self, config: Optional[FramingConfig] = None):
        self.config = config or FramingConfig()
        self.ecc = ECCCodec(self.config.ecc_config)
        # msg_id → {seq: payload}
        self._buffers: dict[int, dict[int, bytes]] = {}
        self._totals: dict[int, int] = {}
        self._timestamps: dict[int, float] = {}
        self._lock = threading.Lock()

    def add_frame(self, header: FrameHeader, raw_payload: bytes
                  ) -> Optional[bytes]:
        """Add a received frame. Returns the complete message if all chunks
        have been received, otherwise None.

        *raw_payload* should include ECC bytes (will be decoded here).
        """
        # Decode ECC
        payload = self.ecc.decode(raw_payload)
        if payload is None:
            return None  # uncorrectable error

        msg_id = header.msg_id
        seq = header.seq
        total = header.total

        with self._lock:
            if msg_id not in self._buffers:
                self._buffers[msg_id] = {}
                self._totals[msg_id] = total
                self._timestamps[msg_id] = time.monotonic()

            self._buffers[msg_id][seq] = payload

            if len(self._buffers[msg_id]) == self._totals[msg_id]:
                # All chunks received — reassemble
                message = b""
                for i in range(total):
                    message += self._buffers[msg_id][i]
                # Clean up
                del self._buffers[msg_id]
                del self._totals[msg_id]
                del self._timestamps[msg_id]
                return message

        return None

    def pending_messages(self) -> list[int]:
        """Return msg_ids of messages still being assembled."""
        with self._lock:
            return list(self._buffers.keys())

    def missing_seqs(self, msg_id: int) -> list[int]:
        """Return missing sequence numbers for a partially-received message."""
        with self._lock:
            if msg_id not in self._buffers:
                return []
            total = self._totals[msg_id]
            received = set(self._buffers[msg_id].keys())
            return [i for i in range(total) if i not in received]

    def cleanup_stale(self, max_age: float = 30.0) -> list[int]:
        """Remove messages older than *max_age* seconds. Returns removed msg_ids."""
        now = time.monotonic()
        stale = []
        with self._lock:
            for msg_id, ts in list(self._timestamps.items()):
                if now - ts > max_age:
                    stale.append(msg_id)
                    del self._buffers[msg_id]
                    del self._totals[msg_id]
                    del self._timestamps[msg_id]
        return stale
