"""Tests for message framing and assembly."""

import pytest

from face2face.protocol.framing import (
    FramingConfig,
    MessageAssembler,
    MessageFramer,
)
from face2face.visual.codec import FrameFlags, FrameHeader
from face2face.visual.ecc import ECCConfig


class TestMessageFramer:
    def test_small_message(self):
        """A small message should fit in a single frame."""
        framer = MessageFramer()
        data = b"Hello!"
        frames = framer.frame_message(data)
        assert len(frames) == 1
        payload, header = frames[0]
        assert header.seq == 0
        assert header.total == 1
        assert header.flags == FrameFlags.DATA

    def test_large_message_splits(self):
        """A large message should be split into multiple frames."""
        config = FramingConfig(
            max_payload_per_frame=50,
            ecc_config=ECCConfig(nsym=10),
        )
        framer = MessageFramer(config)
        # Max payload per frame = 50 - 10 = 40 bytes
        data = b"X" * 100
        frames = framer.frame_message(data)
        assert len(frames) == 3  # 40 + 40 + 20
        for i, (payload, header) in enumerate(frames):
            assert header.seq == i
            assert header.total == 3
            assert header.msg_id == frames[0][1].msg_id

    def test_empty_message(self):
        framer = MessageFramer()
        frames = framer.frame_message(b"")
        assert len(frames) == 1

    def test_msg_ids_increment(self):
        framer = MessageFramer()
        frames1 = framer.frame_message(b"a")
        frames2 = framer.frame_message(b"b")
        assert frames1[0][1].msg_id != frames2[0][1].msg_id

    def test_control_frame(self):
        framer = MessageFramer()
        payload, header = framer.frame_control(FrameFlags.ACK, msg_id=42, seq=3)
        assert header.flags == FrameFlags.ACK
        assert header.msg_id == 42
        assert header.seq == 3


class TestMessageAssembler:
    def test_single_frame_assembly(self):
        """A single-frame message should be immediately complete."""
        config = FramingConfig(
            max_payload_per_frame=200,
            ecc_config=ECCConfig(nsym=10),
        )
        framer = MessageFramer(config)
        assembler = MessageAssembler(config)

        data = b"Hello!"
        frames = framer.frame_message(data)
        assert len(frames) == 1

        payload, header = frames[0]
        result = assembler.add_frame(header, payload)
        assert result is not None
        assert result == data

    def test_multi_frame_assembly(self):
        """Multiple frames should be assembled into the original message."""
        config = FramingConfig(
            max_payload_per_frame=50,
            ecc_config=ECCConfig(nsym=10),
        )
        framer = MessageFramer(config)
        assembler = MessageAssembler(config)

        data = b"A" * 100
        frames = framer.frame_message(data)
        assert len(frames) > 1

        for i, (payload, header) in enumerate(frames):
            result = assembler.add_frame(header, payload)
            if i < len(frames) - 1:
                assert result is None
            else:
                assert result is not None
                assert result == data

    def test_out_of_order_assembly(self):
        """Frames received out of order should still assemble correctly."""
        config = FramingConfig(
            max_payload_per_frame=50,
            ecc_config=ECCConfig(nsym=10),
        )
        framer = MessageFramer(config)
        assembler = MessageAssembler(config)

        data = b"B" * 100
        frames = framer.frame_message(data)

        # Send in reverse order
        for payload, header in reversed(frames):
            result = assembler.add_frame(header, payload)

        assert result is not None
        assert result == data

    def test_missing_seqs(self):
        config = FramingConfig(
            max_payload_per_frame=50,
            ecc_config=ECCConfig(nsym=10),
        )
        framer = MessageFramer(config)
        assembler = MessageAssembler(config)

        data = b"C" * 100
        frames = framer.frame_message(data)
        msg_id = frames[0][1].msg_id

        # Only send first frame
        assembler.add_frame(frames[0][1], frames[0][0])

        missing = assembler.missing_seqs(msg_id)
        assert len(missing) == len(frames) - 1

    def test_cleanup_stale(self):
        assembler = MessageAssembler()
        header = FrameHeader(msg_id=99, seq=0, total=5, flags=FrameFlags.DATA)
        assembler.add_frame(header, b"data")

        assert 99 in assembler.pending_messages()
        stale = assembler.cleanup_stale(max_age=0)
        assert 99 in stale
        assert 99 not in assembler.pending_messages()
