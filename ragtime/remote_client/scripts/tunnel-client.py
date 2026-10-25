#!/usr/bin/env python3
"""Outbound Ragtime SSH tunnel client; invoke with only a JSON config path."""

import argparse
import base64
import hashlib
import json
import os
import queue
import random
import socket
import ssl
import struct
import sys
import threading
import time
from urllib.parse import urlsplit

OPEN, DATA, CLOSE, PING, PONG = range(1, 6)
HEADER_SIZE = 5
MAX_DATA = 64 * 1024
MAX_MESSAGE = 1024 * 1024
HEARTBEAT_TIMEOUT = 75
MAX_STREAM_BUFFER = 1024 * 1024
SUBPROTOCOL = "ragtime-tunnel-v1"


class ProtocolError(Exception):
    pass


def frame(message_type, stream_id=0, payload=b""):
    if not 0 <= stream_id <= 0xFFFFFFFF:
        raise ProtocolError("invalid stream id")
    if message_type == DATA and len(payload) > MAX_DATA:
        raise ProtocolError("data payload exceeds 64 KiB")
    if message_type != DATA and len(payload) > 1024:
        raise ProtocolError("control payload is too large")
    return bytes(bytearray([message_type])) + struct.pack("!I", stream_id) + payload


def parse(message):
    if len(message) < HEADER_SIZE:
        raise ProtocolError("short tunnel message")
    kind = message[0]
    stream_id = struct.unpack("!I", message[1:HEADER_SIZE])[0]
    payload = message[HEADER_SIZE:]
    if kind not in (OPEN, DATA, CLOSE, PING, PONG):
        raise ProtocolError("unknown tunnel message")
    if kind == DATA and len(payload) > MAX_DATA:
        raise ProtocolError("oversize tunnel data")
    if kind != DATA and len(payload) > 1024:
        raise ProtocolError("oversize tunnel control")
    if kind in (PING, PONG) and stream_id:
        raise ProtocolError("heartbeat stream id is not zero")
    if kind in (OPEN, DATA, CLOSE) and not stream_id:
        raise ProtocolError("stream id is zero")
    return kind, stream_id, payload


def read_exact(connection, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = connection.recv(remaining)
        if not chunk:
            raise EOFError("websocket closed")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class WebSocket(object):
    def __init__(self, connection, initial_data=b""):
        self.connection = connection
        self.receive_buffer = bytearray(initial_data)
        self.send_lock = threading.Lock()
        self.closed = False

    def _read_exact(self, size):
        buffered = bytes(self.receive_buffer[:size])
        del self.receive_buffer[:size]
        return buffered + read_exact(self.connection, size - len(buffered))

    def send_binary(self, payload):
        self._send_frame(2, payload)

    def _send_frame(self, opcode, payload=b""):
        if self.closed:
            raise EOFError("websocket closed")
        size = len(payload)
        if size > MAX_MESSAGE:
            raise ProtocolError("websocket message too large")
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        if size < 126:
            header.append(0x80 | size)
        elif size <= 0xFFFF:
            header.append(0x80 | 126)
            header.extend(struct.pack("!H", size))
        else:
            header.append(0x80 | 127)
            header.extend(struct.pack("!Q", size))
        masked = bytes(bytearray(value ^ mask[index % 4] for index, value in enumerate(bytearray(payload))))
        with self.send_lock:
            self.connection.sendall(bytes(header) + mask + masked)

    def receive_binary(self):
        parts = []
        opcode = None
        total = 0
        while True:
            first, second = bytearray(self._read_exact(2))
            if first & 0x70:
                raise ProtocolError("websocket extensions were not negotiated")
            fin, incoming_opcode = bool(first & 0x80), first & 0x0F
            masked, size = bool(second & 0x80), second & 0x7F
            if masked:
                raise ProtocolError("server websocket frames must not be masked")
            if size == 126:
                size = struct.unpack("!H", self._read_exact(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self._read_exact(8))[0]
            if size > MAX_MESSAGE:
                raise ProtocolError("websocket frame too large")
            payload = self._read_exact(size)
            if incoming_opcode >= 8:
                if incoming_opcode not in (8, 9, 10) or not fin or len(payload) > 125:
                    raise ProtocolError("invalid control frame")
                if incoming_opcode == 8:
                    if len(payload) == 1:
                        raise ProtocolError("invalid websocket close payload")
                    self._send_frame(8, payload)
                    self.closed = True
                    raise EOFError("websocket closed")
                if incoming_opcode == 9:
                    self._send_frame(10, payload)
                continue
            if incoming_opcode in (1, 2):
                if opcode is not None:
                    raise ProtocolError("new websocket message before continuation")
                opcode = incoming_opcode
            elif incoming_opcode != 0 or opcode is None:
                raise ProtocolError("invalid websocket continuation")
            total += len(payload)
            if total > MAX_MESSAGE:
                raise ProtocolError("websocket message too large")
            parts.append(payload)
            if fin:
                if opcode != 2:
                    raise ProtocolError("tunnel requires binary websocket messages")
                return b"".join(parts)

    def close(self):
        if not self.closed:
            try:
                self._send_frame(8, struct.pack("!H", 1000))
            except (EOFError, OSError):
                pass
        self.closed = True
        self.connection.close()


def connect(config):
    target = urlsplit(config["server_url"])
    if target.scheme != "wss" or not target.hostname:
        raise ValueError("server_url must be an absolute wss:// URL")
    if target.username or target.password or target.query or target.fragment:
        raise ValueError("server_url must not contain credentials, query, or fragment")
    raw = socket.create_connection((target.hostname, target.port or (443 if target.scheme == "wss" else 80)), timeout=20)
    if target.scheme == "wss":
        context = ssl.create_default_context(cafile=config.get("ca_file") or None)
        raw = context.wrap_socket(raw, server_hostname=target.hostname)
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    path = target.path or "/"
    request = "\r\n".join(
        (
            "GET {0} HTTP/1.1".format(path),
            "Host: {0}".format(target.netloc),
            "Upgrade: websocket",
            "Connection: Upgrade",
            "Sec-WebSocket-Key: {0}".format(key),
            "Sec-WebSocket-Version: 13",
            "Sec-WebSocket-Protocol: {0}".format(SUBPROTOCOL),
            "Authorization: Bearer {0}".format(config["credential"]),
            "X-Ragtime-Pairing-ID: {0}".format(config["pairing_id"]),
            "\r\n",
        )
    ).encode("ascii")
    raw.sendall(request)
    response = b""
    while b"\r\n\r\n" not in response:
        chunk = raw.recv(4096)
        if not chunk:
            raise EOFError("websocket handshake ended before headers")
        response += chunk
        if len(response) > 16 * 1024:
            raise ProtocolError("oversize websocket handshake")
    header, extra = response.split(b"\r\n\r\n", 1)
    lines = header.decode("iso-8859-1").split("\r\n")
    if not lines or not lines[0].startswith("HTTP/1.1 101"):
        raise ProtocolError("websocket upgrade was rejected")
    headers = {}
    for line in lines[1:]:
        name, separator, value = line.partition(":")
        if not separator:
            raise ProtocolError("malformed websocket handshake")
        headers[name.lower()] = value.strip()
    expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")).digest()).decode("ascii")
    if headers.get("sec-websocket-accept") != expected or headers.get("sec-websocket-protocol") != SUBPROTOCOL:
        raise ProtocolError("websocket server did not validate the requested protocol")
    if headers.get("sec-websocket-extensions"):
        raise ProtocolError("websocket extensions are not supported")
    if headers.get("upgrade", "").lower() != "websocket" or "upgrade" not in headers.get("connection", "").lower():
        raise ProtocolError("websocket server did not complete an upgrade")
    raw.settimeout(HEARTBEAT_TIMEOUT)
    return WebSocket(raw, extra)


class _ClientStream(object):
    def __init__(self):
        self.connection = None
        self.connected = threading.Event()
        self.closed = False
        self.local_eof = False
        self.remote_eof = False
        self.pending = queue.Queue()
        self.pending_bytes = 0
        self.pending_lock = threading.Lock()
        self.write_in_flight = False
        self.remote_write_closed = False


class Tunnel(object):
    def __init__(self, websocket, sshd_port):
        self.websocket = websocket
        self.sshd_port = sshd_port
        self.streams = {}
        self.lock = threading.Lock()

    def run(self):
        try:
            while True:
                kind, stream_id, payload = parse(self.websocket.receive_binary())
                if kind == PING:
                    self.websocket.send_binary(frame(PONG))
                elif kind == PONG:
                    pass
                elif kind == OPEN:
                    self._open(stream_id)
                elif kind == DATA:
                    self._write(stream_id, payload)
                elif kind == CLOSE:
                    self._close_remote(stream_id)
        finally:
            self.close()

    def _open(self, stream_id):
        with self.lock:
            if stream_id in self.streams or len(self.streams) >= 32:
                self.websocket.send_binary(frame(CLOSE, stream_id))
                return
            stream = _ClientStream()
            self.streams[stream_id] = stream
        thread = threading.Thread(target=self._connect_local, args=(stream_id, stream))
        thread.daemon = True
        thread.start()

    def _connect_local(self, stream_id, stream):
        try:
            connection = socket.create_connection(("127.0.0.1", self.sshd_port), timeout=15)
            connection.settimeout(None)
            with self.lock:
                if self.streams.get(stream_id) is not stream or stream.closed:
                    connection.close()
                    return
                stream.connection = connection
                stream.connected.set()
            self._shutdown_local_write_if_drained(stream_id, stream)
            self._start_worker(stream_id, stream, self._read_local)
            self._start_worker(stream_id, stream, self._write_local)
        except OSError:
            self._fail_stream(stream_id, stream)

    def _start_worker(self, stream_id, stream, worker):
        thread = threading.Thread(target=worker, args=(stream_id, stream))
        thread.daemon = True
        thread.start()

    def _write(self, stream_id, payload):
        with self.lock:
            stream = self.streams.get(stream_id)
        if stream is None or stream.remote_eof:
            return
        with stream.pending_lock:
            if stream.pending_bytes + len(payload) > MAX_STREAM_BUFFER:
                self._fail_stream(stream_id, stream)
                return
            stream.pending.put(payload)
            stream.pending_bytes += len(payload)

    def _write_local(self, stream_id, stream):
        try:
            while True:
                data = stream.pending.get()
                if data is None:
                    return
                with stream.pending_lock:
                    stream.pending_bytes -= len(data)
                    stream.write_in_flight = True
                stream.connected.wait()
                if stream.closed:
                    return
                try:
                    stream.connection.sendall(data)
                finally:
                    with stream.pending_lock:
                        stream.write_in_flight = False
                    self._shutdown_local_write_if_drained(stream_id, stream)
        except OSError:
            self._fail_stream(stream_id, stream)

    def _read_local(self, stream_id, stream):
        try:
            while True:
                data = stream.connection.recv(MAX_DATA)
                if not data:
                    break
                self.websocket.send_binary(frame(DATA, stream_id, data))
        except (EOFError, OSError, ProtocolError):
            pass
        finally:
            self._local_eof(stream_id, stream)

    def _close_remote(self, stream_id):
        with self.lock:
            stream = self.streams.get(stream_id)
        if stream is not None:
            stream.remote_eof = True
            self._shutdown_local_write_if_drained(stream_id, stream)
            if stream.local_eof:
                self._finalize(stream_id, stream)

    def _shutdown_local_write_if_drained(self, stream_id, stream):
        with stream.pending_lock:
            ready = (
                stream.connection is not None
                and stream.remote_eof
                and not stream.write_in_flight
                and not stream.pending_bytes
                and not stream.remote_write_closed
            )
            if ready:
                stream.remote_write_closed = True
        if not ready or stream.connection is None:
            return
        try:
            stream.connection.shutdown(socket.SHUT_WR)
        except OSError:
            self._fail_stream(stream_id, stream)

    def _local_eof(self, stream_id, stream):
        stream.local_eof = True
        try:
            self.websocket.send_binary(frame(CLOSE, stream_id))
        except (EOFError, OSError, ProtocolError):
            pass
        if stream.remote_eof:
            self._finalize(stream_id, stream)

    def _fail_stream(self, stream_id, stream):
        try:
            self.websocket.send_binary(frame(CLOSE, stream_id))
        except (EOFError, OSError, ProtocolError):
            pass
        self._finalize(stream_id, stream)

    def _finalize(self, stream_id, stream):
        with self.lock:
            if self.streams.get(stream_id) is not stream:
                return
            self.streams.pop(stream_id, None)
            stream.closed = True
            stream.pending.put(None)
        if stream.connection is not None:
            stream.connection.close()

    def close(self):
        with self.lock:
            streams, self.streams = list(self.streams.values()), {}
        for stream in streams:
            stream.closed = True
            stream.pending.put(None)
            if stream.connection is not None:
                stream.connection.close()
        self.websocket.close()


def load_config(path):
    with open(path, "r") as handle:
        config = json.load(handle)
    allowed = {"server_url", "pairing_id", "credential", "sshd_port", "ca_file"}
    if not isinstance(config, dict) or set(config) - allowed or set(config) < {"server_url", "pairing_id", "credential", "sshd_port"}:
        raise ValueError("invalid tunnel client configuration")
    if not all(isinstance(config[key], str) and config[key] for key in ("server_url", "pairing_id", "credential")):
        raise ValueError("configuration text fields must be non-empty")
    if not isinstance(config["sshd_port"], int) or not 1 <= config["sshd_port"] <= 65535:
        raise ValueError("sshd_port must be between 1 and 65535")
    if config.get("ca_file") and not os.path.isfile(config["ca_file"]):
        raise ValueError("ca_file does not exist")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to tunnel client JSON configuration")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    delay = 1.0
    while True:
        connected_at = None
        try:
            tunnel = Tunnel(connect(config), config["sshd_port"])
            connected_at = time.monotonic()
            tunnel.run()
        except (EOFError, OSError, ProtocolError, ValueError) as error:
            print("Ragtime tunnel disconnected: {0}".format(error), file=sys.stderr)
        if connected_at is not None and time.monotonic() - connected_at >= 60:
            delay = 1.0
        time.sleep(min(60.0, delay) * random.uniform(0.8, 1.2))
        delay = min(60.0, delay * 2)


if __name__ == "__main__":
    main()
