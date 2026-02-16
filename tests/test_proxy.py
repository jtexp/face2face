"""Tests for HTTP proxy serialization and components."""

import pytest

from face2face.proxy.serialization import (
    deserialize_connect_request,
    deserialize_request,
    deserialize_response,
    deserialize_tunnel_data,
    is_connect_request,
    serialize_connect_request,
    serialize_request,
    serialize_response,
    serialize_tunnel_data,
)


class TestRequestSerialization:
    def test_roundtrip_get(self):
        data = serialize_request("GET", "http://example.com/page",
                                 {"Host": "example.com", "Accept": "*/*"})
        method, url, headers, body = deserialize_request(data)
        assert method == "GET"
        assert url == "http://example.com/page"
        assert headers["Host"] == "example.com"
        assert body is None

    def test_roundtrip_post_with_body(self):
        body = b'{"key": "value"}'
        data = serialize_request("POST", "http://api.example.com/data",
                                 {"Content-Type": "application/json"}, body)
        method, url, headers, body_out = deserialize_request(data)
        assert method == "POST"
        assert body_out == body

    def test_empty_headers(self):
        data = serialize_request("GET", "http://example.com", {})
        method, url, headers, body = deserialize_request(data)
        assert headers == {}

    def test_no_body(self):
        data = serialize_request("DELETE", "http://example.com/1", {})
        method, url, headers, body = deserialize_request(data)
        assert body is None


class TestResponseSerialization:
    def test_roundtrip_200(self):
        body = b"<html>OK</html>"
        data = serialize_response(200, "OK",
                                  {"Content-Type": "text/html"}, body)
        status, reason, headers, body_out = deserialize_response(data)
        assert status == 200
        assert reason == "OK"
        assert headers["Content-Type"] == "text/html"
        assert body_out == body

    def test_roundtrip_404(self):
        data = serialize_response(404, "Not Found", {})
        status, reason, headers, body = deserialize_response(data)
        assert status == 404
        assert reason == "Not Found"
        assert body is None

    def test_large_body(self):
        body = b"X" * 100_000
        data = serialize_response(200, "OK", {}, body)
        status, reason, headers, body_out = deserialize_response(data)
        assert body_out == body


class TestConnectSerialization:
    def test_roundtrip(self):
        data = serialize_connect_request("github.com", 443)
        assert is_connect_request(data)
        host, port = deserialize_connect_request(data)
        assert host == "github.com"
        assert port == 443

    def test_not_connect(self):
        data = serialize_request("GET", "http://example.com", {})
        assert not is_connect_request(data)


class TestTunnelData:
    def test_roundtrip(self):
        data = serialize_tunnel_data(b"\x16\x03\x01\x00")
        payload, is_close = deserialize_tunnel_data(data)
        assert payload == b"\x16\x03\x01\x00"
        assert not is_close

    def test_close(self):
        data = serialize_tunnel_data(b"", is_close=True)
        payload, is_close = deserialize_tunnel_data(data)
        assert payload == b""
        assert is_close
