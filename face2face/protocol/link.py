"""Link manager: manages one direction of the visual link (tx or rx).

Coordinates the visual encoder+renderer (for tx) or capture+decoder (for rx)
and exposes an async interface for upper layers.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..visual.capture import CaptureConfig, WebcamCapture
from ..visual.codec import CodecConfig, FrameEncoder, FrameFlags, FrameHeader
from ..visual.decoder import DecoderConfig, ImageFrameDecoder
from ..visual.renderer import RendererConfig, ScreenRenderer
from .framing import FramingConfig, MessageAssembler, MessageFramer
from .flow import ARQReceiver, ARQSender, FlowConfig

logger = logging.getLogger(__name__)


@dataclass
class LinkConfig:
    codec: CodecConfig = field(default_factory=CodecConfig)
    renderer: RendererConfig = field(default_factory=RendererConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    framing: FramingConfig = field(default_factory=FramingConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    rx_poll_interval: float = 0.05  # seconds between webcam reads
    enable_monitor: bool = False


class TransmitLink:
    """Manages outbound visual transmission (screen display)."""

    def __init__(self, config: Optional[LinkConfig] = None):
        self.config = config or LinkConfig()
        self.renderer = ScreenRenderer(self.config.codec, self.config.renderer)
        self.framer = MessageFramer(self.config.framing)
        self.arq = ARQSender(self.config.flow)
        self._tx_lock = asyncio.Lock()

    async def _tx_frame(self, payload: bytes, header: FrameHeader) -> None:
        """Physically transmit a frame via screen display."""
        self.renderer.transmit_frame(payload, header)

    async def send(self, data: bytes, msg_id: int | None = None) -> bool:
        """Send a complete message reliably.

        Splits into frames, displays each, and waits for ACKs.
        Returns True on success.
        """
        async with self._tx_lock:
            frames = self.framer.frame_message(data, msg_id=msg_id)
            logger.debug("TX: sending %d bytes as %d frame(s)", len(data), len(frames))
            result = await self.arq.send_reliable(frames, self._tx_frame)
            if result:
                logger.debug("TX: message sent successfully (%d bytes)", len(data))
            else:
                logger.warning("TX: message send failed after retries (%d bytes)", len(data))
            return result

    async def send_control(self, flags: int, msg_id: int = 0,
                           seq: int = 0) -> None:
        """Send a single control frame (ACK, NACK, etc.)."""
        payload, header = self.framer.frame_control(flags, msg_id, seq)
        await self._tx_frame(payload, header)

    def show_idle(self) -> None:
        """Display an idle frame (with ECC) so the grid is visible at startup."""
        payload, header = self.framer.frame_control(
            FrameFlags.KEEPALIVE, msg_id=0, seq=0)
        self.renderer.show_idle(payload, header)

    def destroy(self) -> None:
        self.renderer.destroy()


class ReceiveLink:
    """Manages inbound visual reception (webcam capture + decode)."""

    def __init__(self, config: Optional[LinkConfig] = None):
        self.config = config or LinkConfig()
        self.capture = WebcamCapture(self.config.capture)
        self.decoder = ImageFrameDecoder(self.config.codec, self.config.decoder)
        self.assembler = MessageAssembler(self.config.framing)
        self.arq = ARQReceiver(self.config.flow)
        self._running = False
        self._message_queue: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        self._frame_callback: Optional[Callable] = None
        self.monitor = None

    async def start(self) -> None:
        """Start the receive loop."""
        self.capture.open()
        self._running = True
        if self.config.enable_monitor:
            from ..visual.monitor import WebcamMonitor
            self.monitor = WebcamMonitor()
            logger.info("Webcam monitor window enabled")
        codec = self.config.codec
        logger.info("RX link started — grid=%dx%d, %d bits/cell, %d bytes/frame",
                     codec.grid_cols, codec.grid_rows, codec.bits_per_cell,
                     codec.payload_bytes)
        logger.info("Scanning for frames ...")

    async def stop(self) -> None:
        """Stop the receive loop."""
        self._running = False
        self.capture.close()
        if self.monitor is not None:
            self.monitor.destroy()
            self.monitor = None

    async def receive_once(self) -> Optional[tuple[FrameHeader, bytes, bool]]:
        """Capture and decode a single frame from the webcam.

        Returns (header, payload, valid) or None if no frame detected.
        Runs in a thread to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        full_frame = await loop.run_in_executor(None, self.capture.read_preprocessed)
        if full_frame is None:
            logger.debug("Webcam returned no frame")
            return None

        # Crop to ROI if the monitor has one set
        roi = self.monitor.roi if self.monitor is not None else None
        decode_frame = self.monitor.crop_frame(full_frame) if roi else full_frame

        # Skip blank/sync frames
        if self.decoder.is_blank_frame(decode_frame):
            logger.debug("Blank/sync frame detected, skipping")
            if self.monitor is not None:
                self.monitor.update(full_frame)
            return None

        header, payload = self.decoder.decode_image(decode_frame)
        # Map decoder corners back to full-frame coords for monitor overlay
        corners = self.decoder.last_corners
        if corners is not None and roi is not None:
            corners = corners + np.array([roi[0], roi[1]], dtype=corners.dtype)
        if self.monitor is not None:
            self.monitor.update(full_frame, corners)
        if header is None:
            logger.debug("No data frame detected in webcam image")
            return None

        valid = payload is not None
        if valid:
            logger.debug("Frame decoded OK: msg_id=%d seq=%d/%d (%d bytes)",
                         header.msg_id, header.seq, header.total,
                         len(payload) if payload else 0)
        else:
            logger.debug("Frame header found but payload CRC failed: msg_id=%d seq=%d/%d",
                         header.msg_id, header.seq, header.total)
        return header, payload if payload else b"", valid

    async def run_receive_loop(
        self,
        ack_sender: TransmitLink,
    ) -> None:
        """Continuously receive frames, send ACKs, and assemble messages.

        Complete messages are placed on the internal message queue.
        """
        while self._running:
            result = await self.receive_once()
            if result is None:
                await asyncio.sleep(self.config.rx_poll_interval)
                continue

            header, payload, valid = result

            # Ignore frames with failed CRC — header fields are unreliable
            if not valid:
                continue

            # Handle control frames
            if header.flags & FrameFlags.ACK:
                logger.debug("RX: ACK for msg_id=%d seq=%d", header.msg_id, header.seq)
                if self._frame_callback:
                    self._frame_callback("ack", header)
                continue
            if header.flags & FrameFlags.NACK:
                logger.debug("RX: NACK for msg_id=%d seq=%d", header.msg_id, header.seq)
                if self._frame_callback:
                    self._frame_callback("nack", header)
                continue
            if header.flags & FrameFlags.KEEPALIVE:
                logger.debug("RX: ignoring KEEPALIVE frame")
                continue

            # Data frame — send ACK back
            ack_payload, ack_header = self.arq.on_frame_received(header, valid)
            logger.debug("RX: sending ACK for msg_id=%d seq=%d/%d",
                         header.msg_id, header.seq, header.total)
            await ack_sender.send_control(
                FrameFlags.ACK, header.msg_id, header.seq)

            # Try to assemble
            complete = self.assembler.add_frame(header, payload)
            if complete is not None:
                logger.info("Message assembled: msg_id=%d, %d bytes",
                            header.msg_id, len(complete))
                self.arq.cleanup(header.msg_id)
                await self._message_queue.put((header.msg_id, complete))

    async def get_message(self, timeout: float = 30.0
                          ) -> Optional[tuple[int, bytes]]:
        """Wait for and return the next complete message.

        Returns (msg_id, data) or None on timeout.
        """
        try:
            return await asyncio.wait_for(
                self._message_queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def set_frame_callback(self, callback: Callable) -> None:
        """Set a callback for control frame events."""
        self._frame_callback = callback
