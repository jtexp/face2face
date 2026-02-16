"""HTTP forwarder — runs on Machine B (the server/internet side).

Receives serialized HTTP requests from the visual channel, makes the
actual HTTP requests, and sends responses back over the channel.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

import aiohttp

from ..compression.compress import compress, decompress
from .serialization import (
    deserialize_connect_request,
    deserialize_request,
    deserialize_tunnel_data,
    is_connect_request,
    serialize_response,
    serialize_tunnel_data,
)

logger = logging.getLogger(__name__)


class Channel(Protocol):
    """Protocol for the visual channel interface."""
    async def allocate_stream(self) -> int: ...
    async def release_stream(self, stream_id: int) -> None: ...
    async def send(self, stream_id: int, data: bytes) -> bool: ...
    async def recv(self, stream_id: int, timeout: float | None = None) -> bytes | None: ...


class ProxyForwarder:
    """Receives HTTP requests over the visual channel and forwards them."""

    def __init__(self, channel: Channel, timeout: float = 60.0):
        self.channel = channel
        self.timeout = timeout
        self._running = False
        self._session: aiohttp.ClientSession | None = None

    async def start(self) -> None:
        """Start the forwarder."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout)
        )
        self._running = True
        logger.info("Proxy forwarder started")

    async def stop(self) -> None:
        """Stop the forwarder."""
        self._running = False
        if self._session:
            await self._session.close()
            self._session = None
            # Allow underlying SSL transports to close cleanly
            await asyncio.sleep(0.25)

    async def run(self) -> None:
        """Main loop: receive requests from channel and forward them."""
        while self._running:
            # Listen for incoming messages on a well-known stream
            # In practice, the channel dispatches messages by stream_id
            # Here we process messages as they arrive on any stream
            await asyncio.sleep(0.1)

    async def handle_request(self, stream_id: int, raw_data: bytes) -> None:
        """Handle a single request received on *stream_id*."""
        try:
            data = decompress(raw_data)

            if is_connect_request(data):
                await self._handle_connect(stream_id, data)
                return

            method, url, headers, body = deserialize_request(data)
            logger.info("Forwarding %s %s", method, url)

            # Remove hop-by-hop headers
            for h in ("Connection", "Keep-Alive", "Proxy-Authenticate",
                       "Proxy-Authorization", "TE", "Trailers",
                       "Transfer-Encoding", "Upgrade"):
                headers.pop(h, None)

            async with self._session.request(
                method, url, headers=headers, data=body,
                allow_redirects=False, ssl=False,
            ) as resp:
                resp_body = await resp.read()
                resp_headers = dict(resp.headers)

                serialized = serialize_response(
                    resp.status, resp.reason or "", resp_headers, resp_body)
                compressed = compress(serialized)

                await self.channel.send(stream_id, compressed)
                logger.info("Responded %d for %s %s",
                            resp.status, method, url)

        except Exception as e:
            logger.exception("Error forwarding request")
            error_resp = serialize_response(
                502, "Bad Gateway",
                {"Content-Type": "text/plain"},
                str(e).encode(),
            )
            await self.channel.send(stream_id, compress(error_resp))

    async def _handle_connect(self, stream_id: int, data: bytes) -> None:
        """Handle a CONNECT tunnel request."""
        host, port = deserialize_connect_request(data)
        logger.info("Opening tunnel to %s:%d", host, port)

        try:
            reader, writer = await asyncio.open_connection(host, port)

            # Send connection established response
            resp = serialize_response(200, "Connection Established", {})
            await self.channel.send(stream_id, compress(resp))

            async def relay_channel_to_server():
                """Read from visual channel, write to remote server."""
                try:
                    while self._running:
                        msg = await self.channel.recv(stream_id, timeout=60.0)
                        if msg is None:
                            break
                        raw = decompress(msg)
                        tunnel_data, is_close = deserialize_tunnel_data(raw)
                        if is_close:
                            break
                        if tunnel_data:
                            writer.write(tunnel_data)
                            await writer.drain()
                except Exception:
                    pass
                finally:
                    writer.close()

            async def relay_server_to_channel():
                """Read from remote server, send over visual channel."""
                try:
                    while self._running:
                        data = await reader.read(4096)
                        if not data:
                            break
                        msg = compress(serialize_tunnel_data(data))
                        await self.channel.send(stream_id, msg)
                except Exception:
                    pass
                finally:
                    close_msg = compress(
                        serialize_tunnel_data(b"", is_close=True))
                    await self.channel.send(stream_id, close_msg)

            await asyncio.gather(
                relay_channel_to_server(),
                relay_server_to_channel(),
            )

        except Exception as e:
            logger.exception("Tunnel error to %s:%d", host, port)
            error_resp = serialize_response(
                502, "Bad Gateway",
                {"Content-Type": "text/plain"},
                f"Cannot connect to {host}:{port}: {e}".encode(),
            )
            await self.channel.send(stream_id, compress(error_resp))
