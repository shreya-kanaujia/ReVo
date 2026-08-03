"""Binary receiver-capacity feedback shared by the ReVo peers."""

import struct
import math


MSG_CAPACITY_FEEDBACK = 6
CAPACITY_FEEDBACK_VERSION = 2

# type, version, receiver feedback sequence, application sequence, message bytes,
# eleven f64 fields, four mutually exclusive state flags, skip/filter/state codes.
FORMAT = "<BBQQI" + ("d" * 11) + ("B" * 7)
SIZE = struct.calcsize(FORMAT)

SKIP_REASONS = {
    "": 0,
    "initial_sample": 1,
    "sequence_gap": 2,
    "sender_idle_boundary": 3,
    "stale_boundary": 4,
    "recovering": 5,
    "reordered_message": 6,
    "reorder_boundary": 7,
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
STATE_REASONS = {
    "startup_warmup": 0,
    "sequence_gap": 1,
    "confirmation_gap": 2,
    "recovery": 3,
    "published": 4,
    "publication_expired": 5,
}
SKIP_CODES = {value: key for key, value in SKIP_REASONS.items()}
FILTER_CODES = {value: key for key, value in FILTER_REASONS.items()}
STATE_CODES = {value: key for key, value in STATE_REASONS.items()}


def _encode_optional(value):
    return -1.0 if value is None else float(value)


def _decode_optional(value):
    return None if value < 0.0 else value


def validate_capacity_feedback(feedback):
    """Return whether decoded feedback is finite and internally consistent."""
    try:
        if (
            int(feedback["feedback_sequence_id"]) <= 0
            or int(feedback["seq_id"]) <= 0
            or int(feedback["packet_size"]) <= 0
        ):
            return False
        receiver_ts = float(feedback["receiver_ts"])
        sender_grace = float(feedback.get("sender_grace_period", 0.0))
        threshold = float(feedback["freshness_threshold_s"])
        if (
            not math.isfinite(receiver_ts)
            or not math.isfinite(sender_grace)
            or not math.isfinite(threshold)
            or receiver_ts < 0.0
            or sender_grace < 0.0
            or threshold <= 0.0
        ):
            return False
        for name in (
            "raw_interarrival",
            "corrected_interarrival",
            "ewma_interarrival",
            "filtered_ewma_interarrival",
            "raw_update_age_s",
            "published_estimate_age_s",
            "last_valid_update_ts",
            "last_published_estimate_ts",
        ):
            value = feedback.get(name)
            if value is not None and (
                not math.isfinite(float(value)) or float(value) < 0.0
            ):
                return False
        if feedback.get("estimate_fresh"):
            age = feedback.get("published_estimate_age_s")
            filtered = feedback.get("filtered_ewma_interarrival")
            if (
                age is None
                or filtered is None
                or float(filtered) <= 0.0
                or float(age) > threshold
            ):
                return False
        state_flags = [
            bool(feedback.get("estimate_fresh")),
            bool(feedback.get("estimate_stale")),
            bool(feedback.get("estimate_recovering")),
            bool(feedback.get("estimate_unavailable")),
        ]
        if sum(state_flags) != 1:
            return False
        return (
            feedback.get("skip_reason", "") in SKIP_REASONS
            and feedback.get("filter_reason", "") in FILTER_REASONS
            and feedback.get("estimate_state_reason") in STATE_REASONS
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def encode_capacity_feedback(feedback):
    skip_reason = feedback.get("skip_reason", "")
    filter_reason = feedback.get("filter_reason", "")
    if skip_reason not in SKIP_REASONS:
        raise ValueError(f"unknown capacity-feedback skip reason: {skip_reason}")
    if filter_reason not in FILTER_REASONS:
        raise ValueError(
            f"unknown capacity-feedback filter reason: {filter_reason}"
        )
    if not validate_capacity_feedback(feedback):
        raise ValueError("invalid capacity feedback")
    return struct.pack(
        FORMAT,
        MSG_CAPACITY_FEEDBACK,
        CAPACITY_FEEDBACK_VERSION,
        int(feedback["feedback_sequence_id"]),
        int(feedback["seq_id"]),
        int(feedback["packet_size"]),
        float(feedback["receiver_ts"]),
        _encode_optional(feedback.get("raw_interarrival")),
        float(feedback.get("sender_grace_period", 0.0)),
        _encode_optional(feedback.get("corrected_interarrival")),
        _encode_optional(feedback.get("ewma_interarrival")),
        _encode_optional(feedback.get("filtered_ewma_interarrival")),
        _encode_optional(feedback.get("raw_update_age_s")),
        _encode_optional(feedback.get("published_estimate_age_s")),
        float(feedback["freshness_threshold_s"]),
        _encode_optional(feedback.get("last_valid_update_ts")),
        _encode_optional(feedback.get("last_published_estimate_ts")),
        1 if feedback.get("estimate_fresh") else 0,
        1 if feedback.get("estimate_stale") else 0,
        1 if feedback.get("estimate_recovering") else 0,
        1 if feedback.get("estimate_unavailable") else 0,
        SKIP_REASONS[skip_reason],
        FILTER_REASONS[filter_reason],
        STATE_REASONS[feedback["estimate_state_reason"]],
    )


def decode_capacity_feedback(message):
    if not isinstance(message, bytes) or len(message) != SIZE:
        return None
    values = struct.unpack(FORMAT, message)
    if (
        values[0] != MSG_CAPACITY_FEEDBACK
        or values[1] != CAPACITY_FEEDBACK_VERSION
    ):
        return None
    (
        _message_type,
        _version,
        feedback_sequence_id,
        seq_id,
        packet_size,
        receiver_ts,
        raw_interarrival,
        sender_grace_period,
        corrected_interarrival,
        ewma_interarrival,
        filtered_ewma_interarrival,
        raw_update_age_s,
        published_estimate_age_s,
        freshness_threshold_s,
        last_valid_update_ts,
        last_published_estimate_ts,
        estimate_fresh,
        estimate_stale,
        estimate_recovering,
        estimate_unavailable,
        skip_code,
        filter_code,
        state_code,
    ) = values
    if (
        skip_code not in SKIP_CODES
        or filter_code not in FILTER_CODES
        or state_code not in STATE_CODES
        or any(
            flag not in (0, 1)
            for flag in (
                estimate_fresh,
                estimate_stale,
                estimate_recovering,
                estimate_unavailable,
            )
        )
    ):
        return None
    feedback = {
        "type": "capacity_feedback",
        "version": CAPACITY_FEEDBACK_VERSION,
        "feedback_sequence_id": int(feedback_sequence_id),
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
        "raw_update_age_s": _decode_optional(raw_update_age_s),
        "published_estimate_age_s": _decode_optional(
            published_estimate_age_s
        ),
        "estimate_age_s": _decode_optional(published_estimate_age_s),
        "freshness_threshold_s": freshness_threshold_s,
        "last_valid_update_ts": _decode_optional(last_valid_update_ts),
        "last_published_estimate_ts": _decode_optional(
            last_published_estimate_ts
        ),
        "estimate_fresh": bool(estimate_fresh),
        "estimate_stale": bool(estimate_stale),
        "estimate_recovering": bool(estimate_recovering),
        "estimate_unavailable": bool(estimate_unavailable),
        "skip_reason": SKIP_CODES[skip_code],
        "filter_reason": FILTER_CODES[filter_code],
        "estimate_state_reason": STATE_CODES[state_code],
    }
    return feedback if validate_capacity_feedback(feedback) else None
