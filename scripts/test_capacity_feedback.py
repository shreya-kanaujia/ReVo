#!/usr/bin/env python3
"""Round-trip checks for coordinated receiver-capacity feedback."""

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from capacity_feedback import (  # noqa: E402
    SIZE,
    decode_capacity_feedback,
    encode_capacity_feedback,
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
        "seq_id": 2**64 - 1,
        "packet_size": 2**32 - 1,
        "receiver_ts": 123456.125,
        "raw_interarrival": 0.00125,
        "sender_grace_period": 0.0,
        "corrected_interarrival": 0.00125,
        "ewma_interarrival": 0.0015,
        "filtered_ewma_interarrival": 0.0014,
        "estimate_age_s": 0.012,
        "freshness_threshold_s": 0.05,
        "last_valid_update_ts": 123456.113,
        "estimate_fresh": True,
        "skip_reason": "",
        "filter_reason": "rolling_time_median",
    })
    round_trip({
        "seq_id": 0,
        "packet_size": 0,
        "receiver_ts": 0.0,
        "raw_interarrival": None,
        "sender_grace_period": 1.0,
        "corrected_interarrival": None,
        "ewma_interarrival": None,
        "filtered_ewma_interarrival": None,
        "estimate_age_s": None,
        "freshness_threshold_s": 1.0,
        "last_valid_update_ts": None,
        "estimate_fresh": False,
        "skip_reason": "stale_boundary",
        "filter_reason": "stale_boundary_excluded",
    })
    print(f"capacity feedback round-trip test passed ({SIZE} bytes)")


if __name__ == "__main__":
    main()
