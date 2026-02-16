"""Tests for the protocol layer: channel, flow control."""

import asyncio

import pytest

from face2face.protocol.channel import LoopbackChannel


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestLoopbackChannel:
    @pytest.mark.asyncio
    async def test_send_recv(self):
        channel = LoopbackChannel()
        await channel.start()

        stream_id = await channel.allocate_stream()
        data = b"Hello from loopback!"

        await channel.send(stream_id, data)
        received = await channel.recv(stream_id, timeout=1.0)

        assert received == data
        await channel.release_stream(stream_id)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_multiple_streams(self):
        channel = LoopbackChannel()
        await channel.start()

        s1 = await channel.allocate_stream()
        s2 = await channel.allocate_stream()

        await channel.send(s1, b"stream1")
        await channel.send(s2, b"stream2")

        r1 = await channel.recv(s1, timeout=1.0)
        r2 = await channel.recv(s2, timeout=1.0)

        assert r1 == b"stream1"
        assert r2 == b"stream2"

        await channel.release_stream(s1)
        await channel.release_stream(s2)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_recv_timeout(self):
        channel = LoopbackChannel()
        await channel.start()

        stream_id = await channel.allocate_stream()
        result = await channel.recv(stream_id, timeout=0.1)
        assert result is None

        await channel.release_stream(stream_id)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_multiple_messages_same_stream(self):
        channel = LoopbackChannel()
        await channel.start()

        stream_id = await channel.allocate_stream()

        for i in range(5):
            await channel.send(stream_id, f"msg-{i}".encode())

        for i in range(5):
            received = await channel.recv(stream_id, timeout=1.0)
            assert received == f"msg-{i}".encode()

        await channel.release_stream(stream_id)
        await channel.stop()

    @pytest.mark.asyncio
    async def test_stream_ids_increment(self):
        channel = LoopbackChannel()
        s1 = await channel.allocate_stream()
        s2 = await channel.allocate_stream()
        assert s2 == s1 + 1
