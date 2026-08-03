#!/usr/bin/env python3
"""Round-trip checks for coordinated receiver-capacity feedback."""

import math
import struct
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capacity_feedback import (  # noqa: E402
    FORMAT,
    SIZE,
    decode_capacity_feedback,
    encode_capacity_feedback,
    validate_capacity_feedback,
)


def assert_equal(actual, expected, field):
    if isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{field}: expected {expected}, got {actual}")
    elif actual != expected:
        raise AssertionError(f"{field}: expected {expected!r}, got {actual!r}")


def round_trip(sample):
    encoded = encode_capacity_feedback(sample)
    if len(encoded) != SIZE:
        raise AssertionError(f"encoded size: expected {SIZE}, got {len(encoded)}")
    decoded = decode_capacity_feedback(encoded)
    for field, expected in sample.items():
        assert_equal(decoded[field], expected, field)


def main():
    round_trip({
        "feedback_sequence_id": 2**64 - 2,
        "seq_id": 2**64 - 1,
        "packet_size": 2**32 - 1,
        "receiver_ts": 123456.125,
        "raw_interarrival": 0.00125,
        "sender_grace_period": 0.0,
        "corrected_interarrival": 0.00125,
        "ewma_interarrival": 0.0015,
        "filtered_ewma_interarrival": 0.0014,
        "raw_update_age_s": 0.002,
        "published_estimate_age_s": 0.012,
        "freshness_threshold_s": 0.05,
        "last_valid_update_ts": 123456.113,
        "last_published_estimate_ts": 123456.100,
        "estimate_fresh": True,
        "estimate_stale": False,
        "estimate_recovering": False,
        "estimate_unavailable": False,
        "estimate_state_reason": "published",
        "skip_reason": "",
        "filter_reason": "rolling_time_median",
    })
    round_trip({
        "feedback_sequence_id": 1,
        "seq_id": 1,
        "packet_size": 1,
        "receiver_ts": 0.0,
        "raw_interarrival": None,
        "sender_grace_period": 1.0,
        "corrected_interarrival": None,
        "ewma_interarrival": None,
        "filtered_ewma_interarrival": None,
        "raw_update_age_s": None,
        "published_estimate_age_s": None,
        "freshness_threshold_s": 1.0,
        "last_valid_update_ts": None,
        "last_published_estimate_ts": None,
        "estimate_fresh": False,
        "estimate_stale": False,
        "estimate_recovering": False,
        "estimate_unavailable": True,
        "estimate_state_reason": "startup_warmup",
        "skip_reason": "stale_boundary",
        "filter_reason": "stale_boundary_excluded",
    })
    invalid = {
        "feedback_sequence_id": 1,
        "seq_id": 0,
        "packet_size": 1,
        "receiver_ts": 0.0,
        "raw_interarrival": None,
        "sender_grace_period": 0.0,
        "corrected_interarrival": None,
        "ewma_interarrival": None,
        "filtered_ewma_interarrival": None,
        "raw_update_age_s": None,
        "published_estimate_age_s": None,
        "freshness_threshold_s": 1.0,
        "last_valid_update_ts": None,
        "last_published_estimate_ts": None,
        "estimate_fresh": False,
        "estimate_stale": False,
        "estimate_recovering": False,
        "estimate_unavailable": True,
        "estimate_state_reason": "startup_warmup",
        "skip_reason": "initial_sample",
        "filter_reason": "no_raw_estimate",
    }
    if validate_capacity_feedback(invalid):
        raise AssertionError("sequence zero must be rejected")
    try:
        encode_capacity_feedback(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feedback must not encode")
    valid = encode_capacity_feedback({
        **invalid,
        "seq_id": 1,
    })
    fields = list(struct.unpack(FORMAT, valid))
    fields[-3] = 255
    if decode_capacity_feedback(struct.pack(FORMAT, *fields)) is not None:
        raise AssertionError("unknown skip code must be rejected")
    fields = list(struct.unpack(FORMAT, valid))
    fields[-7] = 2
    if decode_capacity_feedback(struct.pack(FORMAT, *fields)) is not None:
        raise AssertionError("invalid freshness flag must be rejected")
    print(f"capacity feedback round-trip test passed ({SIZE} bytes)")


if __name__ == "__main__":
    main()
