from __future__ import annotations

import pytest

from ragtime.remote_client.tunnel_protocol import MAX_DATA_PAYLOAD, TunnelMessageType, TunnelProtocolError, decode_frame, encode_frame


def test_tunnel_frame_round_trip_and_payload_limit() -> None:
    frame = encode_frame(TunnelMessageType.DATA, 7, b"ssh")
    assert decode_frame(frame) == (TunnelMessageType.DATA, 7, b"ssh")
    with pytest.raises(TunnelProtocolError, match="64 KiB"):
        encode_frame(TunnelMessageType.DATA, 7, b"x" * (MAX_DATA_PAYLOAD + 1))


@pytest.mark.parametrize("frame", [b"", b"\x02\x00\x00\x00\x00", b"\x09\x00\x00\x00\x01"])
def test_tunnel_protocol_rejects_malformed_stream_identifiers(frame: bytes) -> None:
    with pytest.raises(TunnelProtocolError):
        decode_frame(frame)
