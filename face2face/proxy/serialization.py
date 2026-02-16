"""Compact binary serialization of HTTP requests and responses.

Uses msgpack for efficient encoding. Requests and responses are
serialized into compact dictionaries, then msgpack-encoded and
optionally compressed.
"""

from __future__ import annotations

from typing import Any

import msgpack


def serialize_request(method: str, url: str,
                      headers: dict[str, str],
                      body: bytes | None = None) -> bytes:
    """Serialize an HTTP request to compact bytes."""
    obj: dict[str, Any] = {
        "m": method,
        "u": url,
        "h": headers,
    }
    if body:
        obj["b"] = body
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_request(data: bytes) -> tuple[str, str, dict[str, str], bytes | None]:
    """Deserialize an HTTP request.

    Returns (method, url, headers, body).
    """
    obj = msgpack.unpackb(data, raw=False)
    return (
        obj["m"],
        obj["u"],
        obj.get("h", {}),
        obj.get("b"),
    )


def serialize_response(status: int, reason: str,
                       headers: dict[str, str],
                       body: bytes | None = None) -> bytes:
    """Serialize an HTTP response to compact bytes."""
    obj: dict[str, Any] = {
        "s": status,
        "r": reason,
        "h": headers,
    }
    if body:
        obj["b"] = body
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_response(data: bytes) -> tuple[int, str, dict[str, str], bytes | None]:
    """Deserialize an HTTP response.

    Returns (status, reason, headers, body).
    """
    obj = msgpack.unpackb(data, raw=False)
    return (
        obj["s"],
        obj.get("r", ""),
        obj.get("h", {}),
        obj.get("b"),
    )


def serialize_connect_request(host: str, port: int) -> bytes:
    """Serialize an HTTP CONNECT tunnel request."""
    obj = {
        "m": "CONNECT",
        "host": host,
        "port": port,
    }
    return msgpack.packb(obj, use_bin_type=True)


def serialize_tunnel_data(data: bytes, is_close: bool = False) -> bytes:
    """Serialize raw tunnel data (for CONNECT tunnels)."""
    obj: dict[str, Any] = {"td": data}
    if is_close:
        obj["close"] = True
    return msgpack.packb(obj, use_bin_type=True)


def deserialize_tunnel_data(data: bytes) -> tuple[bytes, bool]:
    """Deserialize tunnel data. Returns (data, is_close)."""
    obj = msgpack.unpackb(data, raw=False)
    return obj.get("td", b""), obj.get("close", False)


def is_connect_request(data: bytes) -> bool:
    """Check if serialized data is a CONNECT request."""
    try:
        obj = msgpack.unpackb(data, raw=False)
        return obj.get("m") == "CONNECT"
    except Exception:
        return False


def deserialize_connect_request(data: bytes) -> tuple[str, int]:
    """Deserialize a CONNECT request. Returns (host, port)."""
    obj = msgpack.unpackb(data, raw=False)
    return obj["host"], obj["port"]
