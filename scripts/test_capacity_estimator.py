#!/usr/bin/env python3
"""Deterministic checks for the passive Week 1B capacity estimator."""

import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "receiver"))

from capacity_estimator import ArrivalCapacityEstimator  # noqa: E402


def assert_close(actual, expected, label, tol=1e-9):
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main():
    est = ArrivalCapacityEstimator(alpha=0.1)

    first = est.observe(receiver_ts=10.000, sender_grace_period=0.0)
    if first["raw_interarrival"] is not None:
        raise AssertionError("first sample should not have an inter-arrival time")

    second = est.observe(receiver_ts=10.020, sender_grace_period=0.0)
    assert_close(second["raw_interarrival"], 0.020, "raw sample 2")
    assert_close(second["corrected_interarrival"], 0.020, "corrected sample 2")
    assert_close(second["ewma_interarrival"], 0.020, "ewma sample 2")

    third = est.observe(receiver_ts=10.140, sender_grace_period=0.100)
    assert_close(third["raw_interarrival"], 0.120, "raw sample 3")
    assert_close(third["sender_grace_period"], 0.100, "grace sample 3")
    if third["corrected_interarrival"] is not None:
        raise AssertionError("sender-idle boundary must not update service time")
    assert_close(third["ewma_interarrival"], 0.020, "idle preserves EWMA")

    fourth = est.observe(receiver_ts=10.180, sender_grace_period=0.0)
    assert_close(fourth["raw_interarrival"], 0.040, "raw sample 4")
    assert_close(fourth["corrected_interarrival"], 0.040, "corrected sample 4")
    assert_close(fourth["ewma_interarrival"], 0.022, "ewma sample 4")

    clamped = est.observe(receiver_ts=10.180, sender_grace_period=0.0)
    assert_close(clamped["corrected_interarrival"], 1e-6, "clamped corrected sample")

    sequenced = ArrivalCapacityEstimator(alpha=0.1)
    sequenced.observe(20.000, sequence_id=10)
    normal = sequenced.observe(20.010, sequence_id=11)
    assert_close(normal["ewma_interarrival"], 0.010, "contiguous sequence sample")
    gap = sequenced.observe(20.040, sequence_id=13)
    if gap["corrected_interarrival"] is not None:
        raise AssertionError("non-contiguous sequence must not update service time")
    assert_close(gap["ewma_interarrival"], 0.010, "sequence gap preserves EWMA")
    resumed = sequenced.observe(20.052, sequence_id=14)
    assert_close(resumed["corrected_interarrival"], 0.012, "post-gap adjacent sample")

    variable = ArrivalCapacityEstimator(alpha=0.1)
    variable.observe(30.000, sequence_id=1, packet_size_bytes=1000)
    small = variable.observe(30.010, sequence_id=2, packet_size_bytes=1000)
    assert_close(1000 / small["ewma_interarrival"], 100_000.0,
                 "small-message byte rate", tol=1e-7)
    large = variable.observe(30.030, sequence_id=3, packet_size_bytes=2000)
    assert_close(2000 / large["ewma_interarrival"], 100_000.0,
                 "variable-size byte rate", tol=1e-7)
    idle_large = variable.observe(
        30.130, sender_grace_period=0.080, sequence_id=4,
        packet_size_bytes=2000,
    )
    if idle_large["corrected_interarrival"] is not None:
        raise AssertionError("variable-size idle boundary must be excluded")
    assert_close(2000 / idle_large["ewma_interarrival"], 100_000.0,
                 "idle preserves variable-size byte rate", tol=1e-7)
    print("capacity estimator unit test passed")


if __name__ == "__main__":
    main()
