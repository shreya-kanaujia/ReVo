"""Shared, strict logging contracts for every Track B adaptation mode."""

from __future__ import annotations


CONTROLLER_FIELDNAMES = (
    "timestamp_monotonic", "decision_sequence", "frame_id", "gop_id",
    "keyframe", "adaptation_mode", "controller_state",
    "current_quality", "requested_quality", "applied_quality",
    "switch_requested", "switch_applied", "primary_decision_reason",
    "active_reason_flags", "raw_capacity_mbps",
    "filtered_capacity_mbps", "published_capacity_mbps",
    "capacity_raw_update_age_s", "capacity_published_estimate_age_s",
    "capacity_age_s", "capacity_fresh", "capacity_stale",
    "capacity_recovering", "capacity_unavailable",
    "capacity_state_reason", "capacity_feedback_sequence",
    "safe_capacity_mbps", "predicted_high_demand_mbps",
    "predicted_low_demand_mbps", "passive_service_estimate_mbps",
    "current_offered_video_mbps", "passive_deterioration_ignored",
    "current_quality_demand_mbps", "target_high_quality_demand_mbps",
    "probe_state", "probe_authorization_valid", "probe_result_age_s",
    "probe_measured_rate_mbps", "probe_failure_reason",
    "rgb_bufferedAmount", "depth_bufferedAmount", "max_bufferedAmount",
    "buffer_trend_bytes", "receiver_impairment_rate",
    "receiver_health_age_s", "receiver_health_fresh", "dwell_gops",
    "cooldown_active", "health_feedback_sequence",
    "health_window_start_frame", "health_window_end_frame",
    "health_total_frames", "health_full_misses", "health_partial_frames",
    "health_decode_failures", "health_reference_unavailable",
    "health_frozen_frames", "high_rgb_bytes",
    "high_depth_bytes", "low_rgb_bytes", "low_depth_bytes",
    "high_rgb_encode_s", "high_depth_encode_s", "low_rgb_encode_s",
    "low_depth_encode_s", "outstanding_messages", "feedback_delay_s",
    "capacity_packet_rtt_s", "capacity_feedback_delivery_delay_s",
    "capacity_feedback_clock_status", "fallback_reason",
)

CAPACITY_PROBE_FIELDNAMES = (
    "monotonic_timestamp", "event", "probe_id", "probe_sequence",
    "probe_state", "probe_reason", "target_additional_rate_mbps",
    "target_total_high_quality_rate_mbps", "offered_probe_bytes",
    "confirmed_probe_bytes", "offered_probe_application_bytes",
    "confirmed_probe_application_bytes", "elapsed_s",
    "measured_probe_delivery_rate_mbps", "rgb_buffer", "depth_buffer",
    "maximum_buffer", "buffer_trend", "probe_buffer_credit_bytes",
    "media_buffer_baseline_bytes", "media_buffer_bytes",
    "receiver_impairment",
    "receiver_health_fresh", "acknowledgement_age_s",
    "success_safety_margin", "abort_buffer_threshold",
    "authorization_created", "authorization_expiry", "quality_before",
    "requested_quality", "applied_quality", "frame_id", "gop_id",
    "keyframe", "current_offered_video_mbps",
    "passive_deterioration_ignored",
)

PROBE_SNAPSHOT_FIELDS = (
    "state", "reason", "probe_id", "probe_sequence",
    "target_additional_mbps", "target_high_mbps", "offered_bytes",
    "confirmed_bytes", "offered_application_bytes",
    "confirmed_application_bytes", "elapsed_s", "measured_probe_mbps",
    "ack_age_s",
    "authorization_valid", "authorization_expiry", "result_age_s",
    "passive_deterioration_ignored",
)


def neutral_probe_snapshot(reason="probe_disabled_for_mode"):
    """Return the complete neutral probe state used outside combined mode."""
    return {
        "state": "IDLE",
        "reason": str(reason),
        "probe_id": 0,
        "probe_sequence": 0,
        "target_additional_mbps": None,
        "target_high_mbps": None,
        "offered_bytes": 0,
        "confirmed_bytes": 0,
        "offered_application_bytes": 0,
        "confirmed_application_bytes": 0,
        "elapsed_s": 0.0,
        "measured_probe_mbps": None,
        "ack_age_s": None,
        "authorization_valid": False,
        "authorization_expiry": None,
        "result_age_s": None,
        "passive_deterioration_ignored": False,
    }


def require_exact_keys(mapping, expected_fields, label):
    """Fail at the producer with an actionable schema error."""
    actual = set(mapping)
    expected = set(expected_fields)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(
            f"{label} schema mismatch: missing={missing}, extra={extra}"
        )


def write_strict_row(writer, row, fieldnames, label):
    require_exact_keys(row, fieldnames, label)
    writer.writerow(row)
