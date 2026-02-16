"""Reed-Solomon error correction for visual frames.

Wraps the `reedsolo` library to add forward error correction to frame
payloads, allowing recovery from misread cells caused by lighting,
angle, or camera noise.
"""

from __future__ import annotations

from dataclasses import dataclass

from reedsolo import RSCodec


@dataclass
class ECCConfig:
    """Error correction configuration.

    nsym: number of error-correction symbols. Higher = more redundancy.
          Can correct up to nsym//2 symbol errors.
          Overhead is nsym bytes per block.
    """
    nsym: int = 20  # ~20 bytes overhead → can correct 10 byte errors


class ECCCodec:
    """Reed-Solomon error correction encoder/decoder."""

    def __init__(self, config: ECCConfig | None = None):
        self.config = config or ECCConfig()
        self._rs = RSCodec(self.config.nsym)

    @property
    def overhead(self) -> int:
        """Number of bytes of ECC overhead added per encode."""
        return self.config.nsym

    def encode(self, data: bytes) -> bytes:
        """Add error correction codes to data.

        Returns data + ECC parity bytes.
        """
        return bytes(self._rs.encode(data))

    def decode(self, data: bytes) -> bytes | None:
        """Decode data with error correction.

        Returns the corrected original data, or None if uncorrectable.
        """
        try:
            decoded = self._rs.decode(data)
            # reedsolo returns (data, remainders, errata_pos)
            return bytes(decoded[0])
        except Exception:
            return None

    def max_payload(self, block_size: int) -> int:
        """Maximum payload bytes that fit in a block of *block_size* bytes."""
        return block_size - self.config.nsym
