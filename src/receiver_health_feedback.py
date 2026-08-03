"""Versioned binary receiver-health feedback shared by ReVo peers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Optional


MSG_RECEIVER_HEALTH = 7
RECEIVER_HEALTH_VERSION = 2

# type:u8, version:u8, feedback_seq:u64, receiver_ts:f64,
# v1: window_start/end/total/full/partial/decode/frozen:u32
# v2: window_start/end/total/full/partial/decode/reference/frozen:u32
# impairment_rate:f64
V1_FORMAT = "<BBQdIIIIIIId"
V2_FORMAT = "<BBQdIIIIIIIId"
# Current-format alias retained for callers that construct boundary tests.
FORMAT = V2_FORMAT
V1_SIZE = struct.calcsize(V1_FORMAT)
SIZE = struct.calcsize(V2_FORMAT)
RECEIVER_HEALTH_SIZES = frozenset((V1_SIZE, SIZE))


@dataclass(frozen=True)
class ReceiverHealthFeedback:
    feedback_seq: int
    receiver_ts: float
    window_start_frame: int
    window_end_frame: int
    total_frames: int
    full_misses: int
    partial_frames: int
    decode_failures: int
    reference_unavailable: int
    frozen_frames: int
    impairment_rate: float
    version: int = RECEIVER_HEALTH_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


def _u32(value: int, name: str) -> int:
    value = int(value)
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError(f"{name} must fit uint32")
    return value


def encode_receiver_health(feedback: ReceiverHealthFeedback | dict) -> bytes:
    if isinstance(feedback, dict):
        feedback = ReceiverHealthFeedback(**feedback)
    if feedback.version not in (1, RECEIVER_HEALTH_VERSION):
        raise ValueError("unsupported receiver-health version")
    if feedback.version == 1 and feedback.reference_unavailable:
        raise ValueError("v1 feedback cannot encode reference-unavailable frames")
    seq = int(feedback.feedback_seq)
    if not 0 <= seq <= 0xFFFFFFFFFFFFFFFF:
        raise ValueError("feedback_seq must fit uint64")
    receiver_ts = float(feedback.receiver_ts)
    rate = float(feedback.impairment_rate)
    if not math.isfinite(receiver_ts) or not math.isfinite(rate):
        raise ValueError("timestamps and rates must be finite")
    if not 0.0 <= rate <= 1.0:
        raise ValueError("impairment_rate must be in [0, 1]")
    fields = (
        _u32(feedback.window_start_frame, "window_start_frame"),
        _u32(feedback.window_end_frame, "window_end_frame"),
        _u32(feedback.total_frames, "total_frames"),
        _u32(feedback.full_misses, "full_misses"),
        _u32(feedback.partial_frames, "partial_frames"),
        _u32(feedback.decode_failures, "decode_failures"),
        _u32(feedback.reference_unavailable, "reference_unavailable"),
        _u32(feedback.frozen_frames, "frozen_frames"),
    )
    if any(value > fields[2] for value in fields[3:]):
        raise ValueError("component counts cannot exceed total_frames")
    wire_fields = fields if feedback.version == 2 else fields[:6] + fields[7:]
    return struct.pack(
        V2_FORMAT if feedback.version == 2 else V1_FORMAT,
        MSG_RECEIVER_HEALTH,
        feedback.version,
        seq,
        receiver_ts,
        *wire_fields,
        rate,
    )


def decode_receiver_health(data: bytes) -> Optional[ReceiverHealthFeedback]:
    if (
        not isinstance(data, (bytes, bytearray, memoryview))
        or len(data) not in RECEIVER_HEALTH_SIZES
    ):
        return None
    payload = bytes(data)
    version = payload[1] if len(payload) >= 2 else 0
    if version == 1 and len(payload) == V1_SIZE:
        unpacked = struct.unpack(V1_FORMAT, payload)
        reference_unavailable = 0
        frozen_index = 10
        rate_index = 11
    elif version == RECEIVER_HEALTH_VERSION and len(payload) == SIZE:
        unpacked = struct.unpack(V2_FORMAT, payload)
        reference_unavailable = unpacked[10]
        frozen_index = 11
        rate_index = 12
    else:
        return None
    if unpacked[0] != MSG_RECEIVER_HEALTH:
        return None
    feedback = ReceiverHealthFeedback(
        version=unpacked[1],
        feedback_seq=unpacked[2],
        receiver_ts=unpacked[3],
        window_start_frame=unpacked[4],
        window_end_frame=unpacked[5],
        total_frames=unpacked[6],
        full_misses=unpacked[7],
        partial_frames=unpacked[8],
        decode_failures=unpacked[9],
        reference_unavailable=reference_unavailable,
        frozen_frames=unpacked[frozen_index],
        impairment_rate=unpacked[rate_index],
    )
    try:
        encode_receiver_health(feedback)
    except ValueError:
        return None
    if feedback.total_frames == 0:
        return None
    return feedback


class ReceiverHealthInbox:
    """Sender-side freshness and replay protection using local receive time."""

    def __init__(
        self, freshness_s: float, *, shared_monotonic_clock: bool = False
    ):
        if freshness_s <= 0:
            raise ValueError("freshness_s must be positive")
        self.freshness_s = float(freshness_s)
        self._feedback: Optional[ReceiverHealthFeedback] = None
        self._received_ts: Optional[float] = None
        self.shared_monotonic_clock = bool(shared_monotonic_clock)
        self._feedback_delay_s: Optional[float] = None

    def accept(self, payload: bytes, received_ts: float) -> bool:
        feedback = decode_receiver_health(payload)
        if feedback is None:
            return False
        if self._feedback is not None and feedback.feedback_seq <= self._feedback.feedback_seq:
            return False
        self._feedback = feedback
        self._received_ts = float(received_ts)
        self._feedback_delay_s = (
            max(0.0, self._received_ts - feedback.receiver_ts)
            if self.shared_monotonic_clock
            else None
        )
        return True

    def snapshot(self, now: float) -> dict:
        if self._feedback is None or self._received_ts is None:
            return {
                "impairment_rate": None,
                "health_age_s": None,
                "health_fresh": False,
                "feedback_seq": 0,
                "receiver_timestamp": None,
                "feedback_delay_s": None,
                "window_start_frame": None,
                "window_end_frame": None,
                "total_frames": None,
                "full_misses": None,
                "partial_frames": None,
                "decode_failures": None,
                "reference_unavailable": None,
                "frozen_frames": None,
            }
        age = max(0.0, float(now) - self._received_ts)
        return {
            "impairment_rate": self._feedback.impairment_rate,
            "health_age_s": age,
            "health_fresh": age <= self.freshness_s,
            "feedback_seq": self._feedback.feedback_seq,
            "receiver_timestamp": self._feedback.receiver_ts,
            "feedback_delay_s": self._feedback_delay_s,
            "window_start_frame": self._feedback.window_start_frame,
            "window_end_frame": self._feedback.window_end_frame,
            "total_frames": self._feedback.total_frames,
            "full_misses": self._feedback.full_misses,
            "partial_frames": self._feedback.partial_frames,
            "decode_failures": self._feedback.decode_failures,
            "reference_unavailable": self._feedback.reference_unavailable,
            "frozen_frames": self._feedback.frozen_frames,
        }
