"""End-to-end tests using the loopback channel."""

import asyncio

import pytest

from face2face.compression.compress import compress, decompress
from face2face.protocol.channel import LoopbackChannel
from face2face.proxy.serialization import (
    deserialize_request,
    deserialize_response,
    serialize_request,
    serialize_response,
)


class TestEndToEnd:
    @pytest.mark.asyncio
    async def test_http_request_response_via_loopback(self):
        """Simulate a full HTTP request/response cycle through loopback."""
        channel = LoopbackChannel()
        await channel.start()

        stream_id = await channel.allocate_stream()

        # Client side: serialize and send request
        req_data = serialize_request(
            "GET", "http://example.com/test",
            {"Host": "example.com", "Accept": "*/*"},
        )
        compressed_req = compress(req_data)
        await channel.send(stream_id, compressed_req)

        # Server side: receive and deserialize request
        received = await channel.recv(stream_id, timeout=1.0)
        assert received is not None
        raw_req = decompress(received)
        method, url, headers, body = deserialize_request(raw_req)
        assert method == "GET"
        assert url == "http://example.com/test"

        # Server side: serialize and send response
        resp_data = serialize_response(
            200, "OK",
            {"Content-Type": "text/html"},
            b"<html>Hello</html>",
        )
        compressed_resp = compress(resp_data)
        await channel.send(stream_id, compressed_resp)

        # Client side: receive and deserialize response
        received_resp = await channel.recv(stream_id, timeout=1.0)
        assert received_resp is not None
        raw_resp = decompress(received_resp)
        status, reason, resp_headers, resp_body = deserialize_response(raw_resp)
        assert status == 200
        assert resp_body == b"<html>Hello</html>"

        await channel.release_stream(stream_id)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_concurrent_streams(self):
        """Multiple concurrent request/response cycles."""
        channel = LoopbackChannel()
        await channel.start()

        async def do_request(stream_id: int, path: str):
            req = compress(serialize_request(
                "GET", f"http://example.com{path}", {}))
            await channel.send(stream_id, req)

            received = await channel.recv(stream_id, timeout=1.0)
            assert received is not None
            return decompress(received)

        s1 = await channel.allocate_stream()
        s2 = await channel.allocate_stream()

        # Send requests
        await channel.send(s1, compress(serialize_request(
            "GET", "http://example.com/a", {})))
        await channel.send(s2, compress(serialize_request(
            "GET", "http://example.com/b", {})))

        # Receive on each stream
        r1 = await channel.recv(s1, timeout=1.0)
        r2 = await channel.recv(s2, timeout=1.0)
        assert r1 is not None
        assert r2 is not None

        # Verify they went to the right streams
        m1, u1, _, _ = deserialize_request(decompress(r1))
        m2, u2, _, _ = deserialize_request(decompress(r2))
        assert "/a" in u1
        assert "/b" in u2

        await channel.release_stream(s1)
        await channel.release_stream(s2)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_large_response(self):
        """Test with a larger response body."""
        channel = LoopbackChannel()
        await channel.start()

        stream_id = await channel.allocate_stream()

        # Send a large response (simulating a git pack or large file)
        large_body = b"X" * 50_000
        resp = serialize_response(200, "OK",
                                  {"Content-Type": "application/octet-stream"},
                                  large_body)
        compressed = compress(resp)
        await channel.send(stream_id, compressed)

        received = await channel.recv(stream_id, timeout=1.0)
        assert received is not None
        status, _, _, body = deserialize_response(decompress(received))
        assert status == 200
        assert body == large_body

        await channel.release_stream(stream_id)
        await channel.stop()
