"""Causal bounded active probing for Track B upgrade authorization."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import threading
from typing import Optional

from probe_feedback import (
    PROBE_FLAG_END,
    PROBE_FLAG_START,
    PROBE_HEADER_SIZE,
    ProbeAck,
)


def frames_until_keyframe(frame_id: int, gop_size: int) -> int:
    """Return zero on a keyframe, otherwise frames to the next GOP boundary."""
    frame_id = int(frame_id)
    gop_size = int(gop_size)
    if frame_id < 0 or gop_size <= 0:
        raise ValueError("frame_id must be nonnegative and gop_size positive")
    remainder = frame_id % gop_size
    return 0 if remainder == 0 else gop_size - remainder


class ProbeBufferGrowthBaseline:
    """Separates causal media-queue growth from measured probe enqueue bytes."""

    def __init__(self, rgb_buffer: int, depth_buffer: int):
        self._previous_rgb = max(0, int(rgb_buffer))
        self._probe_rgb_credit = 0
        media = max(self._previous_rgb, max(0, int(depth_buffer)))
        self._media_low_water = media
        self._media_buffer = media

    def observe(self, rgb_buffer: int, depth_buffer: int) -> dict:
        rgb = max(0, int(rgb_buffer))
        depth = max(0, int(depth_buffer))
        rgb_drain = max(0, self._previous_rgb - rgb)
        self._probe_rgb_credit = max(
            0, self._probe_rgb_credit - rgb_drain
        )
        media_rgb = max(0, rgb - self._probe_rgb_credit)
        self._media_buffer = max(media_rgb, depth)
        self._media_low_water = min(
            self._media_low_water, self._media_buffer
        )
        self._previous_rgb = rgb
        return self.snapshot()

    def record_probe_enqueue(self, before_rgb: int, after_rgb: int) -> dict:
        before = max(0, int(before_rgb))
        after = max(0, int(after_rgb))
        self._probe_rgb_credit += max(0, after - before)
        self._previous_rgb = after
        return self.snapshot()

    def snapshot(self) -> dict:
        return {
            "probe_buffer_credit_bytes": self._probe_rgb_credit,
            "media_buffer_baseline_bytes": self._media_low_water,
            "media_buffer_bytes": self._media_buffer,
            "media_buffer_growth_bytes": max(
                0, self._media_buffer - self._media_low_water
            ),
        }


class ProbeState(str, Enum):
    IDLE = "IDLE"
    WAITING_FOR_ELIGIBILITY = "WAITING_FOR_ELIGIBILITY"
    PROBING = "PROBING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class ProbeConfig:
    payload_bytes: int = 1024
    min_duration_s: float = 0.40
    max_duration_s: float = 0.60
    max_total_bytes: int = 512 * 1024
    max_additional_rate_mbps: float = 20.0
    max_rate_multiplier: float = 8.0
    min_confirmed_fraction: float = 0.90
    ack_timeout_s: float = 0.25
    cooldown_s: float = 2.0
    authorization_ttl_s: float = 1.5
    buffer_abort_bytes: int = 128 * 1024
    buffer_growth_abort_bytes: int = 8 * 1024
    impairment_abort_threshold: float = 0.05
    capacity_safety_margin: float = 0.85
    boundary_guard_s: float = 0.08

    def __post_init__(self):
        if self.payload_bytes <= 0 or self.max_total_bytes < self.payload_bytes:
            raise ValueError("invalid probe byte bounds")
        if not 0 < self.min_duration_s <= self.max_duration_s:
            raise ValueError("invalid probe durations")
        if not 0 < self.min_confirmed_fraction <= 1:
            raise ValueError("invalid confirmation fraction")
        if not 0 < self.capacity_safety_margin <= 1:
            raise ValueError("invalid safety margin")


@dataclass(frozen=True)
class ProbeInputs:
    now: float
    current_quality: str
    rgb_buffer: int
    depth_buffer: int
    buffer_trend: int
    receiver_impairment: Optional[float]
    receiver_health_fresh: bool
    dwell_complete: bool
    transport_connected: bool
    seconds_to_keyframe: float
    current_offered_video_mbps: Optional[float]
    predicted_low_demand_mbps: Optional[float]
    predicted_high_demand_mbps: Optional[float]
    severe_downgrade: bool = False
    passive_deterioration_unaligned: bool = False
    frame_id: int = -1
    gop_id: int = -1
    keyframe: bool = False
    probe_buffer_credit_bytes: int = 0
    media_buffer_baseline_bytes: Optional[int] = None
    media_buffer_bytes: Optional[int] = None


@dataclass(frozen=True)
class ProbePacket:
    probe_id: int
    sequence: int
    flags: int
    payload_bytes: int


class CapacityProbeController:
    """One-probe-at-a-time state machine; all decisions use present/past data."""

    def __init__(self, config: ProbeConfig | None = None):
        self.config = config or ProbeConfig()
        self._lock = threading.Lock()
        self._next_probe_id = 1
        self.reset()

    def reset(self):
        with getattr(self, "_lock", threading.Lock()):
            self.state = ProbeState.IDLE
            self.reason = "idle"
            self.probe_id = 0
            self.sequence = 0
            self.start_ts = None
            self.end_ts = None
            self.target_additional_mbps = None
            self.target_high_mbps = None
            self.current_offered_mbps = None
            self.offered_bytes = 0
            self.confirmed_bytes = 0
            self.offered_application_bytes = 0
            self.confirmed_application_bytes = 0
            self.last_offer_ts = None
            self.next_offer_ts = None
            self.last_ack_ts = None
            self.first_data_ack_ts = None
            self.last_data_ack_ts = None
            self.last_accepted_ack_sequence = 0
            self._sent_packets = {}
            self._accepted_ack_sequences = set()
            self.end_marker_sent = False
            self.end_marker_confirmed = False
            self.cooldown_until = 0.0
            self.authorization_expiry = None
            self.measured_probe_mbps = None
            self._passive_deterioration_unaligned = False

    def stop_for_stream_shutdown(self, now: float):
        """End an in-flight probe neutrally when the requested stream ends."""
        now = float(now)
        with self._lock:
            if self.state == ProbeState.PROBING:
                self.state = ProbeState.IDLE
                self.reason = "normal_stream_shutdown"
                self.end_ts = now
                self.authorization_expiry = None
            return self._snapshot_unlocked(now)

    def _authorization_valid(self, now):
        return (
            self.authorization_expiry is not None
            and now <= self.authorization_expiry
        )

    def _healthy(self, inputs):
        return (
            max(inputs.rgb_buffer, inputs.depth_buffer)
            < self.config.buffer_abort_bytes
            and inputs.buffer_trend < self.config.buffer_growth_abort_bytes
            and inputs.receiver_health_fresh
            and inputs.receiver_impairment is not None
            and inputs.receiver_impairment
            < self.config.impairment_abort_threshold
            and inputs.transport_connected
            and not inputs.severe_downgrade
        )

    def _target(self, inputs):
        if (
            inputs.current_offered_video_mbps is None
            or inputs.predicted_high_demand_mbps is None
            or inputs.predicted_low_demand_mbps is None
        ):
            return None
        required_total = (
            inputs.predicted_high_demand_mbps
            / self.config.capacity_safety_margin
        )
        additional = max(
            0.0, required_total - inputs.current_offered_video_mbps
        )
        causal_base = max(
            inputs.current_offered_video_mbps,
            inputs.predicted_low_demand_mbps,
            1e-9,
        )
        bounded = min(
            additional,
            self.config.max_additional_rate_mbps,
            self.config.max_rate_multiplier * causal_base,
        )
        return inputs.predicted_high_demand_mbps, bounded

    def evaluate(self, inputs: ProbeInputs):
        now = float(inputs.now)
        with self._lock:
            # Passive conditional service and offered-video EWMA do not share a
            # proven causal window. Preserve the observation for audit logging,
            # but never convert it into an active-probe abort.
            self._passive_deterioration_unaligned = bool(
                inputs.passive_deterioration_unaligned
            )
            if inputs.severe_downgrade:
                self.authorization_expiry = None
                if self.state == ProbeState.PROBING:
                    self._terminal(ProbeState.ABORTED, "severe_downgrade", now)
                    return self._snapshot_unlocked(now)
            if self.state == ProbeState.PROBING:
                if not self._healthy(inputs):
                    if not inputs.transport_connected:
                        reason = "transport_closed"
                    elif not inputs.receiver_health_fresh:
                        reason = "receiver_health_stale"
                    elif (
                        inputs.receiver_impairment is not None
                        and inputs.receiver_impairment
                        >= self.config.impairment_abort_threshold
                    ):
                        reason = "receiver_health_bad"
                    elif max(inputs.rgb_buffer, inputs.depth_buffer) >= (
                        self.config.buffer_abort_bytes
                    ):
                        reason = "buffer_abort"
                    else:
                        reason = "buffer_growth_abort"
                    self._terminal(ProbeState.ABORTED, reason, now)
                elif (
                    self.end_marker_confirmed
                    and len(self._accepted_ack_sequences)
                    == len(self._sent_packets)
                ):
                    self._finish_from_evidence(now)
                elif (
                    self.end_marker_confirmed
                    and self.last_ack_ts is not None
                    and now - self.last_ack_ts > self.config.ack_timeout_s
                ):
                    # The unordered channel may deliver the end ACK before
                    # earlier data ACKs. Allow one existing ACK timeout for
                    # causal reordering, then decide from all evidence received.
                    self._finish_from_evidence(now)
                elif (
                    self.offered_bytes > 0
                    and self.last_ack_ts is not None
                    and now - self.last_ack_ts > self.config.ack_timeout_s
                ):
                    self._terminal(ProbeState.FAILED, "ack_stalled", now)
                elif (
                    self.offered_bytes > 0
                    and self.last_ack_ts is None
                    and self.start_ts is not None
                    and now - self.start_ts > self.config.ack_timeout_s
                ):
                    self._terminal(ProbeState.FAILED, "ack_timeout", now)
                elif (
                    self.start_ts is not None
                    and now - self.start_ts
                    > self.config.max_duration_s + self.config.ack_timeout_s
                ):
                    self._terminal(ProbeState.FAILED, "probe_timeout", now)
            elif self.state in (
                ProbeState.SUCCEEDED,
                ProbeState.FAILED,
                ProbeState.ABORTED,
            ):
                self.state = ProbeState.COOLDOWN
                self.reason = "cooldown"
            elif self.state == ProbeState.COOLDOWN and now >= self.cooldown_until:
                self.state = ProbeState.IDLE
                self.reason = "idle"

            if self.state in (ProbeState.IDLE, ProbeState.WAITING_FOR_ELIGIBILITY):
                eligible = (
                    inputs.current_quality == "low"
                    and not inputs.keyframe
                    and self._healthy(inputs)
                    and inputs.dwell_complete
                    and now >= self.cooldown_until
                    and not self._authorization_valid(now)
                    and inputs.seconds_to_keyframe
                    >= self.config.max_duration_s + self.config.boundary_guard_s
                )
                target = self._target(inputs)
                if eligible and target is not None and target[1] > 0.0:
                    self._start(now, target, inputs.current_offered_video_mbps)
                else:
                    self.state = ProbeState.WAITING_FOR_ELIGIBILITY
                    self.reason = "waiting_for_eligibility"
            return self._snapshot_unlocked(now)

    def _start(self, now, target, current_offered):
        self.state = ProbeState.PROBING
        self.reason = "probing"
        self.probe_id = self._next_probe_id
        self._next_probe_id += 1
        self.sequence = 0
        self.start_ts = now
        self.end_ts = None
        self.target_high_mbps, self.target_additional_mbps = target
        self.current_offered_mbps = current_offered
        self.offered_bytes = self.confirmed_bytes = 0
        self.offered_application_bytes = self.confirmed_application_bytes = 0
        self.last_offer_ts = self.last_ack_ts = None
        self.next_offer_ts = now
        self.first_data_ack_ts = self.last_data_ack_ts = None
        self.last_accepted_ack_sequence = 0
        self._sent_packets = {}
        self._accepted_ack_sequences = set()
        self.end_marker_sent = self.end_marker_confirmed = False
        self.measured_probe_mbps = None

    def next_packet(self, now: float) -> ProbePacket | None:
        now = float(now)
        with self._lock:
            if self.state != ProbeState.PROBING:
                return None
            elapsed = now - self.start_ts
            target_bytes = int(
                math.ceil(
                    self.target_additional_mbps
                    * 1_000_000.0
                    / 8.0
                    * self.config.max_duration_s
                )
            ) + self.config.payload_bytes
            if elapsed >= self.config.max_duration_s:
                remaining_target = (
                    min(target_bytes, self.config.max_total_bytes)
                    - self.offered_bytes
                )
                if remaining_target > 0:
                    self.sequence += 1
                    payload = min(self.config.payload_bytes, remaining_target)
                    self.offered_bytes += payload
                    self.offered_application_bytes += (
                        payload + PROBE_HEADER_SIZE
                    )
                    self.last_offer_ts = now
                    packet = ProbePacket(
                        self.probe_id, self.sequence, 0, payload
                    )
                    self._sent_packets[packet.sequence] = (
                        packet.flags, packet.payload_bytes
                    )
                    return packet
            if (
                elapsed >= self.config.max_duration_s
                or self.offered_bytes >= self.config.max_total_bytes
            ):
                if self.end_marker_sent:
                    return None
                self.sequence += 1
                self.end_marker_sent = True
                self.last_offer_ts = now
                packet = ProbePacket(
                    self.probe_id, self.sequence, PROBE_FLAG_END, 0
                )
                self._sent_packets[packet.sequence] = (
                    packet.flags, packet.payload_bytes
                )
                return packet
            interval = (
                self.config.payload_bytes
                * 8.0
                / (self.target_additional_mbps * 1_000_000.0)
            )
            if self.next_offer_ts is not None and now < self.next_offer_ts:
                return None
            remaining = self.config.max_total_bytes - self.offered_bytes
            payload = min(self.config.payload_bytes, remaining)
            if payload <= 0:
                return None
            self.sequence += 1
            flags = PROBE_FLAG_START if self.sequence == 1 else 0
            self.offered_bytes += payload
            self.offered_application_bytes += payload + PROBE_HEADER_SIZE
            self.last_offer_ts = now
            self.next_offer_ts = (
                now if self.next_offer_ts is None else self.next_offer_ts
            ) + interval
            packet = ProbePacket(
                self.probe_id, self.sequence, flags, payload
            )
            self._sent_packets[packet.sequence] = (
                packet.flags, packet.payload_bytes
            )
            return packet

    def accept_ack(self, ack: ProbeAck, now: float) -> str:
        now = float(now)
        with self._lock:
            if self.state != ProbeState.PROBING or ack.probe_id != self.probe_id:
                return "inactive_or_wrong_probe"
            expected = self._sent_packets.get(ack.sequence)
            if expected is None:
                return (
                    "invalid_future_sequence"
                    if ack.sequence > self.sequence
                    else "unknown_sequence"
                )
            if ack.sequence in self._accepted_ack_sequences:
                return "duplicate"
            if (ack.flags, ack.confirmed_payload_bytes) != expected:
                return "payload_or_flags_mismatch"
            reordered = ack.sequence < self.last_accepted_ack_sequence
            self._accepted_ack_sequences.add(ack.sequence)
            self.last_accepted_ack_sequence = max(
                self.last_accepted_ack_sequence, ack.sequence
            )
            self.confirmed_bytes += ack.confirmed_payload_bytes
            if ack.confirmed_payload_bytes > 0:
                self.confirmed_application_bytes += (
                    ack.confirmed_payload_bytes + PROBE_HEADER_SIZE
                )
            self.last_ack_ts = now
            if ack.confirmed_payload_bytes > 0:
                if self.first_data_ack_ts is None:
                    self.first_data_ack_ts = now
                self.last_data_ack_ts = now
            if ack.flags & PROBE_FLAG_END:
                self.end_marker_confirmed = True
            return "accepted_reordered" if reordered else "accepted"

    def _finish_from_evidence(self, now):
        elapsed = max(1e-9, now - self.start_ts)
        delivery_elapsed = (
            elapsed
            if self.first_data_ack_ts is None
            or self.last_data_ack_ts is None
            else max(
                1e-9, self.last_data_ack_ts - self.first_data_ack_ts
            )
        )
        self.measured_probe_mbps = (
            self.confirmed_application_bytes
            * 8.0 / delivery_elapsed / 1_000_000.0
        )
        expected = (
            self.target_additional_mbps
            * 1_000_000.0
            / 8.0
            * min(elapsed, self.config.max_duration_s)
        )
        enough_duration = elapsed >= self.config.min_duration_s
        enough_offered = self.offered_bytes >= (
            expected * self.config.min_confirmed_fraction
        )
        enough_confirmed = (
            self.confirmed_bytes
            >= self.offered_bytes * self.config.min_confirmed_fraction
        )
        total_delivery = (
            (self.current_offered_mbps or 0.0) + self.measured_probe_mbps
        )
        rate_ok = (
            total_delivery * self.config.capacity_safety_margin
            >= self.target_high_mbps
        )
        if enough_duration and enough_offered and enough_confirmed and rate_ok:
            self.authorization_expiry = now + self.config.authorization_ttl_s
            self._terminal(ProbeState.SUCCEEDED, "target_supported", now)
        else:
            reasons = []
            if not enough_duration:
                reasons.append("insufficient_duration")
            if not enough_offered:
                reasons.append("insufficient_offered_bytes")
            if not enough_confirmed:
                reasons.append("insufficient_confirmed_bytes")
            if not rate_ok:
                reasons.append("target_rate_not_supported")
            self._terminal(ProbeState.FAILED, "|".join(reasons), now)

    def _terminal(self, state, reason, now):
        self.state = state
        self.reason = reason
        self.end_ts = now
        self.cooldown_until = now + self.config.cooldown_s
        if state != ProbeState.SUCCEEDED:
            self.authorization_expiry = None

    def cancel_authorization(self, reason, now):
        with self._lock:
            self.authorization_expiry = None
            if self.state == ProbeState.PROBING:
                self._terminal(ProbeState.ABORTED, reason, float(now))

    def snapshot(self, now):
        with self._lock:
            return self._snapshot_unlocked(float(now))

    def _snapshot_unlocked(self, now):
        age = None if self.end_ts is None else max(0.0, now - self.end_ts)
        ack_age = (
            None if self.last_ack_ts is None else max(0.0, now - self.last_ack_ts)
        )
        elapsed = (
            0.0 if self.start_ts is None else max(0.0, now - self.start_ts)
        )
        return {
            "state": self.state.value,
            "reason": self.reason,
            "probe_id": self.probe_id,
            "probe_sequence": self.sequence,
            "target_additional_mbps": self.target_additional_mbps,
            "target_high_mbps": self.target_high_mbps,
            "offered_bytes": self.offered_bytes,
            "confirmed_bytes": self.confirmed_bytes,
            "offered_application_bytes": self.offered_application_bytes,
            "confirmed_application_bytes": self.confirmed_application_bytes,
            "elapsed_s": elapsed,
            "measured_probe_mbps": self.measured_probe_mbps,
            "ack_age_s": ack_age,
            "authorization_valid": self._authorization_valid(now),
            "authorization_expiry": self.authorization_expiry,
            "result_age_s": age,
            "passive_deterioration_ignored": (
                self.state == ProbeState.PROBING
                and bool(
                    getattr(
                        self,
                        "_passive_deterioration_unaligned",
                        False,
                    )
                )
            ),
        }


class CausalOfferedRate:
    """EWMA of actually selected application bytes per completed frame."""

    def __init__(self, alpha=0.2):
        if not 0 < float(alpha) <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.mbps = None

    def observe(self, application_bytes, frame_interval_s):
        rate = (
            int(application_bytes)
            * 8.0
            / float(frame_interval_s)
            / 1_000_000.0
        )
        self.mbps = rate if self.mbps is None else (
            self.alpha * rate + (1.0 - self.alpha) * self.mbps
        )
        return self.mbps
