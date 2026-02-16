"""Integration test: full end-to-end HTTP proxy on a single machine.

Wires up:
  1. A target HTTP server (the "internet")
  2. A PairedChannel (simulates the visual link without hardware)
  3. A ProxyForwarder (server side) reading from the channel
  4. A ProxyServer (client side) accepting HTTP proxy requests
  5. An aiohttp client making requests through the proxy

Flow:
  client → ProxyServer(:proxy_port)
         → channel.send() → [paired queue] → forwarder receives
         → forwarder makes real HTTP request to target(:target_port)
         → forwarder sends response → [paired queue] → channel.recv()
         → ProxyServer returns response to client
"""

from __future__ import annotations

import asyncio
from typing import Optional

import aiohttp
import pytest
import pytest_asyncio
from aiohttp import web

# ---------------------------------------------------------------------------
# PairedChannel: cross-connected queues simulating the visual link
# ---------------------------------------------------------------------------


class PairedChannel:
    """Two-endpoint channel where one side's send is the other's recv.

    create() returns (client_channel, server_channel).
    """

    def __init__(self):
        # stream_id → queue for data flowing in this direction
        self._queues: dict[int, asyncio.Queue[bytes]] = {}
        self._peer: Optional[PairedChannel] = None
        self._lock = asyncio.Lock()
        self._next_stream_id = 1

    @classmethod
    def create(cls) -> tuple["PairedChannel", "PairedChannel"]:
        """Create a cross-connected pair of channels."""
        a = cls()
        b = cls()
        a._peer = b
        b._peer = a
        return a, b

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def allocate_stream(self) -> int:
        async with self._lock:
            sid = self._next_stream_id
            self._next_stream_id += 1
        # Pre-create queues on both sides
        await self._ensure_queue(sid)
        await self._peer._ensure_queue(sid)
        return sid

    async def release_stream(self, stream_id: int) -> None:
        async with self._lock:
            self._queues.pop(stream_id, None)
        if self._peer:
            async with self._peer._lock:
                self._peer._queues.pop(stream_id, None)

    async def send(self, stream_id: int, data: bytes) -> bool:
        """Send data — it arrives at the peer's recv queue."""
        peer = self._peer
        if peer is None:
            return False
        await peer._ensure_queue(stream_id)
        async with peer._lock:
            await peer._queues[stream_id].put(data)
        return True

    async def recv(self, stream_id: int,
                   timeout: float | None = None) -> Optional[bytes]:
        """Receive data sent by the peer."""
        timeout = timeout or 30.0
        await self._ensure_queue(stream_id)
        async with self._lock:
            queue = self._queues[stream_id]
        try:
            return await asyncio.wait_for(queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    async def _ensure_queue(self, stream_id: int) -> None:
        async with self._lock:
            if stream_id not in self._queues:
                self._queues[stream_id] = asyncio.Queue()


# ---------------------------------------------------------------------------
# Target HTTP server (the "internet" endpoint)
# ---------------------------------------------------------------------------


def make_target_app() -> web.Application:
    """Simple HTTP server to act as the target being proxied to."""
    app = web.Application()

    async def handle_hello(request: web.Request) -> web.Response:
        return web.Response(text="Hello from target!", status=200)

    async def handle_echo(request: web.Request) -> web.Response:
        body = await request.read()
        return web.Response(
            body=body,
            status=200,
            content_type=request.content_type,
        )

    async def handle_json(request: web.Request) -> web.Response:
        return web.Response(
            text='{"status": "ok", "value": 42}',
            content_type="application/json",
        )

    async def handle_large(request: web.Request) -> web.Response:
        return web.Response(
            body=b"X" * 50_000,
            content_type="application/octet-stream",
        )

    async def handle_headers(request: web.Request) -> web.Response:
        """Echo back request headers as JSON."""
        import json
        headers = dict(request.headers)
        return web.Response(
            text=json.dumps(headers),
            content_type="application/json",
        )

    async def handle_status(request: web.Request) -> web.Response:
        code = int(request.match_info["code"])
        return web.Response(status=code, text=f"Status {code}")

    app.router.add_get("/hello", handle_hello)
    app.router.add_post("/echo", handle_echo)
    app.router.add_get("/json", handle_json)
    app.router.add_get("/large", handle_large)
    app.router.add_get("/headers", handle_headers)
    app.router.add_get("/status/{code}", handle_status)

    return app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def target_server():
    """Start the target HTTP server on a random port."""
    app = make_target_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)  # port 0 = random
    await site.start()

    # Extract the actual port
    sockets = site._server.sockets
    port = sockets[0].getsockname()[1]
    url = f"http://127.0.0.1:{port}"

    yield url

    await runner.cleanup()


@pytest_asyncio.fixture
async def proxy_stack(target_server):
    """Set up the full proxy stack: paired channel + server + forwarder.

    Yields the proxy URL (http://127.0.0.1:<proxy_port>) and target URL.
    """
    from face2face.proxy.forwarder import ProxyForwarder
    from face2face.proxy.server import ProxyServer

    client_channel, server_channel = PairedChannel.create()

    # Start proxy server on a random port
    proxy = ProxyServer(client_channel, host="127.0.0.1", port=0, timeout=10.0)
    # We need to manually set up on port 0 and grab the port
    proxy._app = web.Application()
    proxy._app.router.add_route("*", "/{path_info:.*}", proxy._handle_request)
    proxy._runner = web.AppRunner(proxy._app)
    await proxy._runner.setup()
    proxy_site = web.TCPSite(proxy._runner, "127.0.0.1", 0)
    await proxy_site.start()
    proxy_port = proxy_site._server.sockets[0].getsockname()[1]
    proxy_url = f"http://127.0.0.1:{proxy_port}"

    # Start forwarder
    forwarder = ProxyForwarder(server_channel, timeout=10.0)
    await forwarder.start()

    # Dispatcher task: reads from server_channel and dispatches to forwarder
    async def dispatcher():
        while forwarder._running:
            # Poll all known streams for incoming messages
            # In the real system this is driven by the rx loop;
            # here we watch for new data on any stream
            try:
                async with server_channel._lock:
                    stream_ids = list(server_channel._queues.keys())
                for sid in stream_ids:
                    try:
                        data = await asyncio.wait_for(
                            server_channel._queues[sid].get(), timeout=0.05)
                        asyncio.create_task(
                            forwarder.handle_request(sid, data))
                    except (asyncio.TimeoutError, KeyError):
                        continue
                await asyncio.sleep(0.01)
            except Exception:
                await asyncio.sleep(0.05)

    dispatch_task = asyncio.create_task(dispatcher())

    yield proxy_url, target_server

    # Cleanup
    forwarder._running = False
    dispatch_task.cancel()
    try:
        await dispatch_task
    except asyncio.CancelledError:
        pass
    await forwarder.stop()
    await proxy._runner.cleanup()


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegrationE2E:
    """End-to-end tests: client → proxy → channel → forwarder → target."""

    @pytest.mark.asyncio
    async def test_simple_get(self, proxy_stack):
        """GET request through the full proxy stack."""
        proxy_url, target_url = proxy_stack

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{target_url}/hello",
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                assert resp.status == 200
                text = await resp.text()
                assert text == "Hello from target!"

    @pytest.mark.asyncio
    async def test_post_echo(self, proxy_stack):
        """POST request with body echoed back."""
        proxy_url, target_url = proxy_stack

        payload = b"This is test data sent through the visual proxy!"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{target_url}/echo",
                proxy=proxy_url,
                data=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                assert resp.status == 200
                body = await resp.read()
                assert body == payload

    @pytest.mark.asyncio
    async def test_json_response(self, proxy_stack):
        """GET request returning JSON."""
        proxy_url, target_url = proxy_stack

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{target_url}/json",
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                assert resp.status == 200
                data = await resp.json()
                assert data["status"] == "ok"
                assert data["value"] == 42

    @pytest.mark.asyncio
    async def test_large_response(self, proxy_stack):
        """GET request with a 50KB response body."""
        proxy_url, target_url = proxy_stack

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{target_url}/large",
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                assert resp.status == 200
                body = await resp.read()
                assert body == b"X" * 50_000

    @pytest.mark.asyncio
    async def test_status_codes(self, proxy_stack):
        """Various HTTP status codes are preserved through the proxy."""
        proxy_url, target_url = proxy_stack

        async with aiohttp.ClientSession() as session:
            for code in [200, 201, 204, 400, 404, 500]:
                async with session.get(
                    f"{target_url}/status/{code}",
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    assert resp.status == code, f"Expected {code}, got {resp.status}"

    @pytest.mark.asyncio
    async def test_multiple_sequential_requests(self, proxy_stack):
        """Multiple requests in sequence through the same proxy."""
        proxy_url, target_url = proxy_stack

        async with aiohttp.ClientSession() as session:
            for i in range(5):
                async with session.get(
                    f"{target_url}/hello",
                    proxy=proxy_url,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    assert resp.status == 200
                    text = await resp.text()
                    assert text == "Hello from target!"
