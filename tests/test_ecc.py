"""Tests for Reed-Solomon error correction."""

import pytest

from face2face.visual.ecc import ECCCodec, ECCConfig


class TestECCCodec:
    def test_roundtrip(self):
        ecc = ECCCodec()
        data = b"Hello, world!"
        encoded = ecc.encode(data)
        decoded = ecc.decode(encoded)
        assert decoded == data

    def test_overhead(self):
        config = ECCConfig(nsym=20)
        ecc = ECCCodec(config)
        assert ecc.overhead == 20

    def test_error_correction(self):
        ecc = ECCCodec(ECCConfig(nsym=20))
        data = b"Test data for ECC"
        encoded = bytearray(ecc.encode(data))

        # Introduce some errors (up to nsym//2 = 10 byte errors)
        for i in range(5):
            encoded[i] ^= 0xFF  # flip all bits

        decoded = ecc.decode(bytes(encoded))
        assert decoded == data

    def test_too_many_errors(self):
        ecc = ECCCodec(ECCConfig(nsym=10))
        data = b"Test data"
        encoded = bytearray(ecc.encode(data))

        # Introduce more errors than can be corrected
        for i in range(len(encoded)):
            encoded[i] ^= 0xFF

        decoded = ecc.decode(bytes(encoded))
        assert decoded is None

    def test_empty_data(self):
        ecc = ECCCodec()
        encoded = ecc.encode(b"")
        decoded = ecc.decode(encoded)
        assert decoded == b""

    def test_large_data(self):
        ecc = ECCCodec()
        data = bytes(range(256)) * 4  # 1 KB
        encoded = ecc.encode(data)
        decoded = ecc.decode(encoded)
        assert decoded == data

    def test_max_payload(self):
        ecc = ECCCodec(ECCConfig(nsym=20))
        assert ecc.max_payload(100) == 80
        assert ecc.max_payload(20) == 0
        assert ecc.max_payload(50) == 30

    def test_different_nsym(self):
        for nsym in [10, 20, 30, 50]:
            ecc = ECCCodec(ECCConfig(nsym=nsym))
            data = b"test" * 10
            encoded = ecc.encode(data)
            assert len(encoded) == len(data) + nsym
            decoded = ecc.decode(encoded)
            assert decoded == data
