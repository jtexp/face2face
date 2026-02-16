"""Flow control: stop-and-wait ARQ with ACK/NACK over the visual channel.

Provides reliable delivery on top of the unreliable visual frame transport.
Uses a simple stop-and-wait protocol:
  1. Sender transmits a frame
  2. Sender waits for ACK from receiver (via reverse visual channel)
  3. If timeout → retransmit
  4. If NACK → retransmit
  5. If ACK → advance to next frame
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

from ..visual.codec import FrameFlags, FrameHeader

logger = logging.getLogger(__name__)


@dataclass
class FlowConfig:
    ack_timeout: float = 5.0      # seconds to wait for ACK
    max_retries: int = 10         # max retransmissions per frame
    ack_delay: float = 0.1        # small delay before sending ACK


class ARQSender:
    """Stop-and-wait ARQ sender.

    Works with async callbacks for transmitting frames and receiving ACKs.
    """

    def __init__(self, config: Optional[FlowConfig] = None):
        self.config = config or FlowConfig()
        self._ack_events: dict[tuple[int, int], asyncio.Event] = {}
        self._nack_received: set[tuple[int, int]] = set()

    async def send_reliable(
        self,
        frames: list[tuple[bytes, FrameHeader]],
        tx_func,  # async callable(payload, header) -> None
    ) -> bool:
        """Send all frames reliably using stop-and-wait.

        *tx_func* is an async callable that physically transmits a frame.
        Returns True if all frames were acknowledged, False on failure.
        """
        for payload, header in frames:
            key = (header.msg_id, header.seq)
            self._ack_events[key] = asyncio.Event()

            success = False
            for attempt in range(self.config.max_retries + 1):
                if attempt > 0:
                    logger.debug("Retransmit msg=%d seq=%d attempt=%d",
                                 header.msg_id, header.seq, attempt)

                await tx_func(payload, header)

                try:
                    await asyncio.wait_for(
                        self._ack_events[key].wait(),
                        timeout=self.config.ack_timeout,
                    )
                    success = True
                    break
                except asyncio.TimeoutError:
                    logger.debug("ACK timeout for msg=%d seq=%d",
                                 header.msg_id, header.seq)
                    self._ack_events[key].clear()

            del self._ack_events[key]
            self._nack_received.discard(key)

            if not success:
                logger.error("Failed to deliver msg=%d seq=%d after %d retries",
                             header.msg_id, header.seq, self.config.max_retries)
                return False

        return True

    def receive_ack(self, header: FrameHeader) -> None:
        """Called when an ACK frame is received from the remote side."""
        key = (header.msg_id, header.seq)
        if key in self._ack_events:
            self._ack_events[key].set()

    def receive_nack(self, header: FrameHeader) -> None:
        """Called when a NACK frame is received."""
        key = (header.msg_id, header.seq)
        self._nack_received.add(key)
        # Reset the event so sender retransmits
        if key in self._ack_events:
            self._ack_events[key].clear()


class ARQReceiver:
    """Stop-and-wait ARQ receiver.

    Tracks received sequence numbers and generates ACK/NACK frames.
    """

    def __init__(self, config: Optional[FlowConfig] = None):
        self.config = config or FlowConfig()
        self._received: set[tuple[int, int]] = set()

    def on_frame_received(self, header: FrameHeader, valid: bool
                          ) -> tuple[bytes, FrameHeader]:
        """Process a received data frame and generate an ACK or NACK.

        Returns (payload, header) for the ACK/NACK frame to send back.
        """
        key = (header.msg_id, header.seq)

        if valid and key not in self._received:
            self._received.add(key)
            ack_header = FrameHeader(
                msg_id=header.msg_id,
                seq=header.seq,
                total=header.total,
                flags=FrameFlags.ACK,
            )
            return b"", ack_header
        elif valid:
            # Duplicate — still ACK it
            ack_header = FrameHeader(
                msg_id=header.msg_id,
                seq=header.seq,
                total=header.total,
                flags=FrameFlags.ACK,
            )
            return b"", ack_header
        else:
            # Invalid frame — NACK
            nack_header = FrameHeader(
                msg_id=header.msg_id,
                seq=header.seq,
                total=header.total,
                flags=FrameFlags.NACK,
            )
            return b"", nack_header

    def is_duplicate(self, header: FrameHeader) -> bool:
        return (header.msg_id, header.seq) in self._received

    def cleanup(self, msg_id: int) -> None:
        """Remove tracking for a completed message."""
        self._received = {
            k for k in self._received if k[0] != msg_id
        }
