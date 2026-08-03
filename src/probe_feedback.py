"""Versioned binary messages for bounded Track B capacity probes."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct


PROBE_VERSION = 1
MSG_CAPACITY_PROBE = 7
MSG_CAPACITY_PROBE_ACK = 8
PROBE_FLAG_START = 1
PROBE_FLAG_END = 2

PROBE_HEADER_FORMAT = "<BBQQBdI"
PROBE_HEADER_SIZE = struct.calcsize(PROBE_HEADER_FORMAT)
PROBE_ACK_FORMAT = "<BBQQBdI"
PROBE_ACK_SIZE = struct.calcsize(PROBE_ACK_FORMAT)


@dataclass(frozen=True)
class ProbeData:
    probe_id: int
    sequence: int
    flags: int
    sender_ts: float
    payload_bytes: int
    payload: bytes


@dataclass(frozen=True)
class ProbeAck:
    probe_id: int
    sequence: int
    flags: int
    receiver_ts: float
    confirmed_payload_bytes: int


def encode_probe_data(
    probe_id: int,
    sequence: int,
    flags: int,
    sender_ts: float,
    payload: bytes,
) -> bytes:
    payload = bytes(payload)
    if (
        int(probe_id) <= 0
        or int(sequence) <= 0
        or int(flags) & ~(PROBE_FLAG_START | PROBE_FLAG_END)
        or not math.isfinite(float(sender_ts))
        or float(sender_ts) < 0.0
    ):
        raise ValueError("invalid capacity probe")
    return struct.pack(
        PROBE_HEADER_FORMAT,
        MSG_CAPACITY_PROBE,
        PROBE_VERSION,
        int(probe_id),
        int(sequence),
        int(flags),
        float(sender_ts),
        len(payload),
    ) + payload


def decode_probe_data(message: bytes) -> ProbeData | None:
    if not isinstance(message, bytes) or len(message) < PROBE_HEADER_SIZE:
        return None
    values = struct.unpack(PROBE_HEADER_FORMAT, message[:PROBE_HEADER_SIZE])
    msg_type, version, probe_id, sequence, flags, sender_ts, payload_bytes = values
    if (
        msg_type != MSG_CAPACITY_PROBE
        or version != PROBE_VERSION
        or probe_id <= 0
        or sequence <= 0
        or flags & ~(PROBE_FLAG_START | PROBE_FLAG_END)
        or not math.isfinite(sender_ts)
        or sender_ts < 0.0
        or payload_bytes != len(message) - PROBE_HEADER_SIZE
    ):
        return None
    return ProbeData(
        probe_id=int(probe_id),
        sequence=int(sequence),
        flags=int(flags),
        sender_ts=float(sender_ts),
        payload_bytes=int(payload_bytes),
        payload=message[PROBE_HEADER_SIZE:],
    )


def encode_probe_ack(ack: ProbeAck) -> bytes:
    if (
        int(ack.probe_id) <= 0
        or int(ack.sequence) <= 0
        or int(ack.flags) & ~(PROBE_FLAG_START | PROBE_FLAG_END)
        or not math.isfinite(float(ack.receiver_ts))
        or float(ack.receiver_ts) < 0.0
        or int(ack.confirmed_payload_bytes) < 0
    ):
        raise ValueError("invalid capacity probe acknowledgement")
    return struct.pack(
        PROBE_ACK_FORMAT,
        MSG_CAPACITY_PROBE_ACK,
        PROBE_VERSION,
        int(ack.probe_id),
        int(ack.sequence),
        int(ack.flags),
        float(ack.receiver_ts),
        int(ack.confirmed_payload_bytes),
    )


def decode_probe_ack(message: bytes) -> ProbeAck | None:
    if not isinstance(message, bytes) or len(message) != PROBE_ACK_SIZE:
        return None
    values = struct.unpack(PROBE_ACK_FORMAT, message)
    msg_type, version, probe_id, sequence, flags, receiver_ts, confirmed = values
    if (
        msg_type != MSG_CAPACITY_PROBE_ACK
        or version != PROBE_VERSION
        or probe_id <= 0
        or sequence <= 0
        or flags & ~(PROBE_FLAG_START | PROBE_FLAG_END)
        or not math.isfinite(receiver_ts)
        or receiver_ts < 0.0
    ):
        return None
    return ProbeAck(
        probe_id=int(probe_id),
        sequence=int(sequence),
        flags=int(flags),
        receiver_ts=float(receiver_ts),
        confirmed_payload_bytes=int(confirmed),
    )
