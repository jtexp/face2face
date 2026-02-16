"""HTTP proxy server — runs on Machine A (the client side).

Listens on localhost for HTTP/HTTPS proxy requests, serializes them,
sends them over the visual channel, and returns the response.

Usage:
    export http_proxy=http://localhost:8080
    export https_proxy=http://localhost:8080
    git clone http://github.com/user/repo
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from aiohttp import web

from ..compression.compress import compress, decompress
from .serialization import (
    deserialize_response,
    deserialize_tunnel_data,
    is_connect_request,
    serialize_connect_request,
    serialize_request,
    serialize_tunnel_data,
)

logger = logging.getLogger(__name__)


class Channel(Protocol):
    """Protocol for the visual channel interface."""
    async def allocate_stream(self) -> int: ...
    async def release_stream(self, stream_id: int) -> None: ...
    async def send(self, stream_id: int, data: bytes) -> bool: ...
    async def recv(self, stream_id: int, timeout: float | None = None) -> bytes | None: ...


class ProxyServer:
    """HTTP proxy server that forwards requests over a visual channel."""

    def __init__(self, channel: Channel, host: str = "127.0.0.1",
                 port: int = 8080, timeout: float = 120.0):
        self.channel = channel
        self.host = host
        self.port = port
        self.timeout = timeout
        self._app: web.Application | None = None
        self._runner: web.AppRunner | None = None

    async def start(self) -> None:
        """Start the proxy server."""
        self._app = web.Application()
        self._app.router.add_route("*", "/{path_info:.*}", self._handle_request)
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info("Proxy server listening on %s:%d", self.host, self.port)

    async def stop(self) -> None:
        """Stop the proxy server."""
        if self._runner:
            await self._runner.cleanup()

    async def _handle_request(self, request: web.Request) -> web.StreamResponse:
        """Handle an incoming proxy request."""
        if request.method == "CONNECT":
            return await self._handle_connect(request)
        return await self._handle_http(request)

    async def _handle_http(self, request: web.Request) -> web.Response:
        """Handle a plain HTTP proxy request."""
        stream_id = await self.channel.allocate_stream()
        try:
            # Read the full request body
            body = await request.read()

            # Build the absolute URL
            url = str(request.url)

            # Serialize and compress
            headers = dict(request.headers)
            # Remove proxy-specific headers
            headers.pop("Proxy-Connection", None)
            headers.pop("Proxy-Authorization", None)

            serialized = serialize_request(
                request.method, url, headers, body or None)
            compressed = compress(serialized)

            # Send over visual channel
            success = await self.channel.send(stream_id, compressed)
            if not success:
                return web.Response(status=502, text="Visual channel send failed")

            # Wait for response
            resp_data = await self.channel.recv(stream_id, timeout=self.timeout)
            if resp_data is None:
                return web.Response(status=504, text="Visual channel timeout")

            # Decompress and deserialize
            resp_raw = decompress(resp_data)
            status, reason, resp_headers, resp_body = deserialize_response(resp_raw)

            # Build response
            response = web.Response(
                status=status,
                reason=reason,
                body=resp_body,
            )
            for key, value in resp_headers.items():
                if key.lower() not in ("transfer-encoding", "content-encoding",
                                       "content-length"):
                    response.headers[key] = value

            return response

        except Exception as e:
            logger.exception("Error handling HTTP request")
            return web.Response(status=502, text=str(e))
        finally:
            await self.channel.release_stream(stream_id)

    async def _handle_connect(self, request: web.Request) -> web.StreamResponse:
        """Handle an HTTP CONNECT request (HTTPS tunneling)."""
        # Parse host:port from the request
        host_port = request.path_qs  # e.g., "example.com:443"
        if ":" in host_port:
            host, port_str = host_port.rsplit(":", 1)
            port = int(port_str)
        else:
            host = host_port
            port = 443

        stream_id = await self.channel.allocate_stream()
        try:
            # Send CONNECT request over visual channel
            connect_data = serialize_connect_request(host, port)
            compressed = compress(connect_data)
            success = await self.channel.send(stream_id, compressed)
            if not success:
                return web.Response(status=502, text="Visual channel send failed")

            # Wait for connection established response
            resp_data = await self.channel.recv(stream_id, timeout=self.timeout)
            if resp_data is None:
                return web.Response(status=504, text="Visual channel timeout")

            resp_raw = decompress(resp_data)
            status, reason, _, _ = deserialize_response(resp_raw)

            if status != 200:
                return web.Response(status=status, text=reason)

            # Upgrade to raw TCP tunnel
            response = web.StreamResponse(status=200, reason="Connection Established")
            response.force_close()
            await response.prepare(request)

            transport = request.transport
            if transport is None:
                return response

            # Bidirectional tunnel relay
            reader = request.content

            async def relay_client_to_channel():
                """Read from client, send over channel."""
                try:
                    while True:
                        data = await reader.read(4096)
                        if not data:
                            break
                        tunnel_msg = compress(serialize_tunnel_data(data))
                        await self.channel.send(stream_id, tunnel_msg)
                except Exception:
                    pass
                finally:
                    close_msg = compress(
                        serialize_tunnel_data(b"", is_close=True))
                    await self.channel.send(stream_id, close_msg)

            async def relay_channel_to_client():
                """Read from channel, send to client."""
                try:
                    while True:
                        msg = await self.channel.recv(stream_id, timeout=self.timeout)
                        if msg is None:
                            break
                        raw = decompress(msg)
                        tunnel_data, is_close = deserialize_tunnel_data(raw)
                        if is_close:
                            break
                        if tunnel_data:
                            await response.write(tunnel_data)
                except Exception:
                    pass

            await asyncio.gather(
                relay_client_to_channel(),
                relay_channel_to_client(),
            )

            return response

        except Exception as e:
            logger.exception("Error handling CONNECT request")
            return web.Response(status=502, text=str(e))
        finally:
            await self.channel.release_stream(stream_id)
