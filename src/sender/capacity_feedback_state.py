"""Sender-local storage and causal aging for receiver capacity feedback."""

from __future__ import annotations

import math
import threading
from typing import Optional


def _finite_optional(value, name):
    if value is None:
        return None
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class CapacityFeedbackState:
    """Order feedback by receiver generation and age it on one clock.

    Application sequence IDs remain ACK/accounting identifiers. Receiver
    feedback sequence IDs alone determine capacity-state replacement.
    """

    def __init__(self, *, shared_monotonic_clock: bool = False):
        self.shared_monotonic_clock = bool(shared_monotonic_clock)
        self._lock = threading.Lock()
        self.reset()

    @staticmethod
    def _empty_state():
        return {
            "raw_estimated_capacity_mbps": None,
            "filtered_estimated_capacity_mbps": None,
            "published_estimated_capacity_mbps": None,
            "last_published_capacity_mbps": None,
            "raw_update_age_s": None,
            "published_estimate_age_s": None,
            "estimate_age_s": None,
            "estimate_fresh": False,
            "estimate_stale": False,
            "estimate_recovering": False,
            "estimate_unavailable": True,
            "estimate_state_reason": "startup_warmup",
            "freshness_threshold_s": None,
            "last_valid_update_ts": None,
            "last_published_estimate_ts": None,
            "feedback_received_ts": None,
            "feedback_sequence_id": 0,
            "application_sequence_id": 0,
            "packet_rtt_s": None,
            "feedback_delivery_delay_s": None,
            "feedback_clock_status": "unavailable",
        }

    def reset(self):
        with self._lock:
            self._state = self._empty_state()

    def accept(
        self,
        *,
        feedback_sequence_id: int,
        application_sequence_id: int,
        receiver_timestamp: float,
        receiver_raw_update_age_s: Optional[float],
        receiver_published_estimate_age_s: Optional[float],
        receiver_estimate_fresh: bool,
        receiver_estimate_stale: bool,
        receiver_estimate_recovering: bool,
        receiver_estimate_unavailable: bool,
        estimate_state_reason: str,
        freshness_threshold_s: float,
        raw_capacity_mbps: Optional[float],
        filtered_capacity_mbps: Optional[float],
        last_valid_update_ts: Optional[float],
        last_published_estimate_ts: Optional[float],
        feedback_received_ts: float,
        packet_rtt_s: Optional[float],
    ) -> str:
        """Accept newer feedback, returning an explicit state-update result."""
        feedback_sequence_id = int(feedback_sequence_id)
        application_sequence_id = int(application_sequence_id)
        if feedback_sequence_id <= 0 or application_sequence_id <= 0:
            return "invalid"
        try:
            receiver_timestamp = _finite_optional(
                receiver_timestamp, "receiver_timestamp"
            )
            raw_age = _finite_optional(
                receiver_raw_update_age_s, "receiver_raw_update_age_s"
            )
            publication_age = _finite_optional(
                receiver_published_estimate_age_s,
                "receiver_published_estimate_age_s",
            )
            threshold = _finite_optional(
                freshness_threshold_s, "freshness_threshold_s"
            )
            raw_capacity = _finite_optional(
                raw_capacity_mbps, "raw_capacity_mbps"
            )
            filtered_capacity = _finite_optional(
                filtered_capacity_mbps, "filtered_capacity_mbps"
            )
            last_valid = _finite_optional(
                last_valid_update_ts, "last_valid_update_ts"
            )
            last_published = _finite_optional(
                last_published_estimate_ts, "last_published_estimate_ts"
            )
            received_ts = _finite_optional(
                feedback_received_ts, "feedback_received_ts"
            )
            packet_rtt = _finite_optional(packet_rtt_s, "packet_rtt_s")
        except (TypeError, ValueError):
            return "invalid"

        state_flags = (
            bool(receiver_estimate_fresh),
            bool(receiver_estimate_stale),
            bool(receiver_estimate_recovering),
            bool(receiver_estimate_unavailable),
        )
        if (
            receiver_timestamp is None
            or received_ts is None
            or threshold is None
            or threshold <= 0.0
            or (raw_age is not None and raw_age < 0.0)
            or (publication_age is not None and publication_age < 0.0)
            or (raw_capacity is not None and raw_capacity <= 0.0)
            or (filtered_capacity is not None and filtered_capacity <= 0.0)
            or (packet_rtt is not None and packet_rtt < 0.0)
            or sum(state_flags) != 1
        ):
            return "invalid"

        with self._lock:
            newest_sequence = self._state["feedback_sequence_id"]
            if feedback_sequence_id < newest_sequence:
                return "reordered"
            if feedback_sequence_id == newest_sequence:
                return "duplicate"

            if self.shared_monotonic_clock:
                delivery_delay = received_ts - receiver_timestamp
                if delivery_delay < 0.0:
                    delivery_delay = None
                    clock_status = "receiver_timestamp_after_receipt"
                else:
                    clock_status = "shared_monotonic"
            else:
                delivery_delay = None
                clock_status = "not_comparable_age_uncertainty"

            # Only the reverse path after receiver publication can contribute
            # to age. Full packet RTT includes time before the estimate exists
            # and remains diagnostic only.
            if delivery_delay is not None:
                if raw_age is not None:
                    raw_age += delivery_delay
                if publication_age is not None:
                    publication_age += delivery_delay

            fresh = (
                bool(receiver_estimate_fresh)
                and publication_age is not None
                and publication_age <= threshold
                and filtered_capacity is not None
            )
            if fresh:
                stale = recovering = unavailable = False
                state_reason = "published"
                last_published_capacity = filtered_capacity
            else:
                recovering = bool(receiver_estimate_recovering)
                unavailable = bool(receiver_estimate_unavailable)
                stale = bool(receiver_estimate_stale) or (
                    bool(receiver_estimate_fresh)
                    and publication_age is not None
                    and publication_age > threshold
                )
                if stale:
                    recovering = unavailable = False
                    state_reason = "feedback_delivery_expired"
                else:
                    state_reason = str(estimate_state_reason)
                last_published_capacity = self._state[
                    "last_published_capacity_mbps"
                ]

            self._state = {
                "raw_estimated_capacity_mbps": raw_capacity,
                "filtered_estimated_capacity_mbps": filtered_capacity,
                "published_estimated_capacity_mbps": (
                    filtered_capacity if fresh else None
                ),
                "last_published_capacity_mbps": last_published_capacity,
                "raw_update_age_s": raw_age,
                "published_estimate_age_s": publication_age,
                "estimate_age_s": publication_age,
                "estimate_fresh": fresh,
                "estimate_stale": stale,
                "estimate_recovering": recovering,
                "estimate_unavailable": unavailable,
                "estimate_state_reason": state_reason,
                "freshness_threshold_s": threshold,
                "last_valid_update_ts": last_valid,
                "last_published_estimate_ts": last_published,
                "feedback_received_ts": received_ts,
                "feedback_sequence_id": feedback_sequence_id,
                "application_sequence_id": application_sequence_id,
                "packet_rtt_s": packet_rtt,
                "feedback_delivery_delay_s": delivery_delay,
                "feedback_clock_status": clock_status,
            }
            return "accepted_fresh" if fresh else f"accepted_{state_reason}"

    def snapshot(self, sender_now: float):
        """Return an immutable view aged from local receipt time exactly once."""
        sender_now = float(sender_now)
        if not math.isfinite(sender_now):
            raise ValueError("sender_now must be finite")
        with self._lock:
            state = dict(self._state)
        received_ts = state["feedback_received_ts"]
        if received_ts is not None:
            if sender_now < received_ts:
                raise ValueError("sender monotonic clock moved backwards")
            elapsed = sender_now - received_ts
            for field in ("raw_update_age_s", "published_estimate_age_s"):
                if state[field] is not None:
                    state[field] += elapsed
        state["estimate_age_s"] = state["published_estimate_age_s"]
        age = state["published_estimate_age_s"]
        threshold = state["freshness_threshold_s"]
        fresh = (
            state["estimate_fresh"]
            and age is not None
            and threshold is not None
            and age <= threshold
        )
        if state["estimate_fresh"] and not fresh:
            state["estimate_stale"] = True
            state["estimate_recovering"] = False
            state["estimate_unavailable"] = False
            state["estimate_state_reason"] = "sender_local_expired"
        state["estimate_fresh"] = fresh
        if not fresh:
            state["published_estimated_capacity_mbps"] = None
        return state
