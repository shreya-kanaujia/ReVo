"""Binary receiver-capacity feedback shared by the ReVo peers."""

import struct


MSG_CAPACITY_FEEDBACK = 6

# type, sequence, message bytes, nine f64 fields, freshness, skip, filter.
FORMAT = "<BQIdddddddddBBB"
SIZE = struct.calcsize(FORMAT)

SKIP_REASONS = {
    "": 0,
    "initial_sample": 1,
    "sequence_gap": 2,
    "sender_idle_boundary": 3,
    "stale_boundary": 4,
    "recovering": 5,
}
FILTER_REASONS = {
    "": 0,
    "no_raw_estimate": 1,
    "no_valid_spacing_sample": 2,
    "stale_boundary_excluded": 3,
    "recovery_buffering": 4,
    "recovery_seeded": 5,
    "sample_accepted": 6,
    "rolling_time_median": 7,
}
SKIP_CODES = {value: key for key, value in SKIP_REASONS.items()}
FILTER_CODES = {value: key for key, value in FILTER_REASONS.items()}


def _encode_optional(value):
    return -1.0 if value is None else float(value)


def _decode_optional(value):
    return None if value < 0.0 else value


def encode_capacity_feedback(feedback):
    skip_reason = feedback.get("skip_reason", "")
    filter_reason = feedback.get("filter_reason", "")
    if skip_reason not in SKIP_REASONS:
        raise ValueError(f"unknown capacity-feedback skip reason: {skip_reason}")
    if filter_reason not in FILTER_REASONS:
        raise ValueError(
            f"unknown capacity-feedback filter reason: {filter_reason}"
        )
    return struct.pack(
        FORMAT,
        MSG_CAPACITY_FEEDBACK,
        int(feedback["seq_id"]),
        int(feedback["packet_size"]),
        float(feedback["receiver_ts"]),
        _encode_optional(feedback.get("raw_interarrival")),
        float(feedback.get("sender_grace_period", 0.0)),
        _encode_optional(feedback.get("corrected_interarrival")),
        _encode_optional(feedback.get("ewma_interarrival")),
        _encode_optional(feedback.get("filtered_ewma_interarrival")),
        _encode_optional(feedback.get("estimate_age_s")),
        float(feedback["freshness_threshold_s"]),
        _encode_optional(feedback.get("last_valid_update_ts")),
        1 if feedback.get("estimate_fresh") else 0,
        SKIP_REASONS[skip_reason],
        FILTER_REASONS[filter_reason],
    )


def decode_capacity_feedback(message):
    if not isinstance(message, bytes) or len(message) != SIZE:
        return None
    values = struct.unpack(FORMAT, message)
    if values[0] != MSG_CAPACITY_FEEDBACK:
        return None
    (
        _message_type,
        seq_id,
        packet_size,
        receiver_ts,
        raw_interarrival,
        sender_grace_period,
        corrected_interarrival,
        ewma_interarrival,
        filtered_ewma_interarrival,
        estimate_age_s,
        freshness_threshold_s,
        last_valid_update_ts,
        estimate_fresh,
        skip_code,
        filter_code,
    ) = values
    if skip_code not in SKIP_CODES or filter_code not in FILTER_CODES:
        raise ValueError("unknown capacity-feedback reason code")
    return {
        "type": "capacity_feedback",
        "seq_id": int(seq_id),
        "packet_size": int(packet_size),
        "receiver_ts": receiver_ts,
        "raw_interarrival": _decode_optional(raw_interarrival),
        "sender_grace_period": sender_grace_period,
        "corrected_interarrival": _decode_optional(corrected_interarrival),
        "ewma_interarrival": _decode_optional(ewma_interarrival),
        "filtered_ewma_interarrival": _decode_optional(
            filtered_ewma_interarrival
        ),
        "estimate_age_s": _decode_optional(estimate_age_s),
        "freshness_threshold_s": freshness_threshold_s,
        "last_valid_update_ts": _decode_optional(last_valid_update_ts),
        "estimate_fresh": bool(estimate_fresh),
        "skip_reason": SKIP_CODES[skip_code],
        "filter_reason": FILTER_CODES[filter_code],
    }
