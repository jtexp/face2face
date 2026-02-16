"""Bidirectional multiplexed channel over two visual links.

Combines a TransmitLink and ReceiveLink into a full-duplex channel,
with stream multiplexing so multiple HTTP requests can be in flight.

Stream protocol:
  Messages sent over the channel are prefixed with a stream header:
    [STREAM_ID: 4 bytes][LENGTH: 4 bytes][DATA: LENGTH bytes]

  This allows multiple logical streams (e.g., concurrent HTTP requests)
  to share the visual channel.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from dataclasses import dataclass, field
from typing import Optional

from ..visual.codec import CodecConfig, FrameFlags
from .link import LinkConfig, ReceiveLink, TransmitLink

logger = logging.getLogger(__name__)

STREAM_HEADER_FMT = ">II"  # stream_id (4 bytes) + length (4 bytes)
STREAM_HEADER_SIZE = struct.calcsize(STREAM_HEADER_FMT)


@dataclass
class ChannelConfig:
    tx_link: LinkConfig = field(default_factory=LinkConfig)
    rx_link: LinkConfig = field(default_factory=LinkConfig)
    message_timeout: float = 60.0


class VisualChannel:
    """Full-duplex multiplexed channel over webcam/screen links."""

    def __init__(self, config: Optional[ChannelConfig] = None):
        self.config = config or ChannelConfig()
        self.tx = TransmitLink(self.config.tx_link)
        self.rx = ReceiveLink(self.config.rx_link)

        # Stream demux: stream_id → asyncio.Queue of data
        self._streams: dict[int, asyncio.Queue[bytes]] = {}
        self._stream_lock = asyncio.Lock()
        self._running = False
        self._rx_task: Optional[asyncio.Task] = None
        self._dispatch_task: Optional[asyncio.Task] = None
        self._next_stream_id = 1
        self._stream_id_lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the channel — opens webcam, starts receive loop."""
        logger.info("Starting visual channel ...")
        await self.rx.start()
        self._running = True

        # Wire up ACK/NACK forwarding from rx → tx
        def on_control(kind: str, header):
            if kind == "ack":
                self.tx.arq.receive_ack(header)
            elif kind == "nack":
                self.tx.arq.receive_nack(header)
        self.rx.set_frame_callback(on_control)

        # Start receive and dispatch loops
        self._rx_task = asyncio.create_task(
            self.rx.run_receive_loop(self.tx))
        self._dispatch_task = asyncio.create_task(
            self._dispatch_loop())
        logger.info("Visual channel ready")

    async def stop(self) -> None:
        """Stop the channel."""
        self._running = False
        if self._rx_task:
            self._rx_task.cancel()
            try:
                await self._rx_task
            except asyncio.CancelledError:
                pass
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        await self.rx.stop()
        self.tx.destroy()

    async def allocate_stream(self) -> int:
        """Allocate a new stream ID."""
        async with self._stream_id_lock:
            sid = self._next_stream_id
            self._next_stream_id += 1
        async with self._stream_lock:
            self._streams[sid] = asyncio.Queue()
        return sid

    async def release_stream(self, stream_id: int) -> None:
        """Release a stream ID."""
        async with self._stream_lock:
            self._streams.pop(stream_id, None)

    async def send(self, stream_id: int, data: bytes) -> bool:
        """Send data on a stream. Returns True on success."""
        header = struct.pack(STREAM_HEADER_FMT, stream_id, len(data))
        message = header + data
        return await self.tx.send(message)

    async def recv(self, stream_id: int,
                   timeout: float | None = None) -> Optional[bytes]:
        """Receive data from a stream. Returns None on timeout."""
        if timeout is None:
            timeout = self.config.message_timeout

        async with self._stream_lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = asyncio.Queue()
            queue = self._streams[stream_id]

        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _dispatch_loop(self) -> None:
        """Dispatch received messages to stream queues."""
        while self._running:
            result = await self.rx.get_message(timeout=1.0)
            if result is None:
                continue

            msg_id, raw = result

            if len(raw) < STREAM_HEADER_SIZE:
                logger.warning("Received message too short (len=%d)", len(raw))
                continue

            stream_id, length = struct.unpack(
                STREAM_HEADER_FMT, raw[:STREAM_HEADER_SIZE])
            data = raw[STREAM_HEADER_SIZE:STREAM_HEADER_SIZE + length]

            if len(data) < length:
                logger.warning("Truncated stream message: expected %d, got %d",
                               length, len(data))

            async with self._stream_lock:
                if stream_id not in self._streams:
                    self._streams[stream_id] = asyncio.Queue()
                await self._streams[stream_id].put(data)

            logger.debug("Dispatched %d bytes to stream %d",
                         len(data), stream_id)


class LoopbackChannel:
    """A channel that loops back tx→rx for testing without hardware.

    Useful for unit tests and development without webcam/screen.
    """

    def __init__(self):
        self._streams: dict[int, asyncio.Queue[bytes]] = {}
        self._lock = asyncio.Lock()
        self._next_stream_id = 1

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def allocate_stream(self) -> int:
        async with self._lock:
            sid = self._next_stream_id
            self._next_stream_id += 1
            self._streams[sid] = asyncio.Queue()
        return sid

    async def release_stream(self, stream_id: int) -> None:
        async with self._lock:
            self._streams.pop(stream_id, None)

    async def send(self, stream_id: int, data: bytes) -> bool:
        async with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = asyncio.Queue()
            await self._streams[stream_id].put(data)
        return True

    async def recv(self, stream_id: int,
                   timeout: float | None = None) -> Optional[bytes]:
        timeout = timeout or 30.0
        async with self._lock:
            if stream_id not in self._streams:
                self._streams[stream_id] = asyncio.Queue()
            queue = self._streams[stream_id]
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
