"""Framing primitives shared by the Ragtime relay and tunnel clients."""

from __future__ import annotations

from enum import IntEnum

HEADER_SIZE = 5
MAX_DATA_PAYLOAD = 64 * 1024
MAX_CONTROL_PAYLOAD = 1024


class TunnelMessageType(IntEnum):
    OPEN = 1
    DATA = 2
    CLOSE = 3
    PING = 4
    PONG = 5


class TunnelProtocolError(ValueError):
    """A peer sent an invalid tunnel frame."""


def encode_frame(message_type: TunnelMessageType, stream_id: int = 0, payload: bytes = b"") -> bytes:
    """Encode one protocol message, rejecting out-of-contract values."""
    if not 0 <= stream_id <= 0xFFFFFFFF:
        raise TunnelProtocolError("stream id is outside uint32 range")
    if message_type is TunnelMessageType.DATA and len(payload) > MAX_DATA_PAYLOAD:
        raise TunnelProtocolError("data payload exceeds 64 KiB")
    if message_type is not TunnelMessageType.DATA and len(payload) > MAX_CONTROL_PAYLOAD:
        raise TunnelProtocolError("control payload exceeds 1 KiB")
    return bytes((message_type,)) + stream_id.to_bytes(4, "big") + payload


def decode_frame(frame: bytes) -> tuple[TunnelMessageType, int, bytes]:
    """Validate and decode one complete binary WebSocket message."""
    if len(frame) < HEADER_SIZE:
        raise TunnelProtocolError("frame is shorter than its header")
    try:
        message_type = TunnelMessageType(frame[0])
    except ValueError as error:
        raise TunnelProtocolError("unknown tunnel message type") from error
    stream_id = int.from_bytes(frame[1:HEADER_SIZE], "big")
    payload = frame[HEADER_SIZE:]
    if message_type is TunnelMessageType.DATA and len(payload) > MAX_DATA_PAYLOAD:
        raise TunnelProtocolError("data payload exceeds 64 KiB")
    if message_type is not TunnelMessageType.DATA and len(payload) > MAX_CONTROL_PAYLOAD:
        raise TunnelProtocolError("control payload exceeds 1 KiB")
    if message_type in {TunnelMessageType.PING, TunnelMessageType.PONG} and stream_id:
        raise TunnelProtocolError("heartbeat stream id must be zero")
    if message_type in {TunnelMessageType.OPEN, TunnelMessageType.DATA, TunnelMessageType.CLOSE} and not stream_id:
        raise TunnelProtocolError("stream message id must be non-zero")
    return message_type, stream_id, payload
