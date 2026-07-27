#!/usr/bin/env python3
"""Deterministic estimator/filter/analyzer audit checks."""

import inspect
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "receiver"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capacity_estimator import ArrivalCapacityEstimator  # noqa: E402
from analyze_estimator_audit import (  # noqa: E402
    binned_metrics,
    capacity_mbps,
    extract_estimator_columns,
    replay_rows,
)


def close(actual, expected, tolerance=1e-9):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"expected {expected}, got {actual}")


def test_named_columns_and_single_conversion():
    row = {
        "published_estimated_capacity_mbps": "3.25",
        "unrelated": "999",
        "filtered_estimated_capacity_mbps": "4.5",
        "estimated_capacity_mbps": "5.75",
    }
    values = extract_estimator_columns(row)
    assert values == {
        "raw_ewma_mbps": 5.75,
        "filtered_mbps": 4.5,
        "published_mbps": 3.25,
    }
    close(capacity_mbps(1000, 0.001), 8.0)


def test_median_bounds_and_causality():
    estimator = ArrivalCapacityEstimator(alpha=1.0, robust_window_s=1.0)
    estimator.observe(0.0, sequence_id=1, packet_size_bytes=1000)
    timestamp = 0.0
    intervals = [0.001, 0.004, 0.002, 0.003, 0.005]
    samples = []
    for sequence, interval in enumerate(intervals, 2):
        timestamp += interval
        sample = estimator.observe(
            timestamp, sequence_id=sequence, packet_size_bytes=1000
        )
        samples.append(sample)
        filtered = sample["filtered_ewma_interarrival"]
        active = intervals[:sequence - 1]
        if not min(active) <= filtered <= max(active):
            raise AssertionError("rolling median left its active sample range")
    before_future = samples[-1]["filtered_ewma_interarrival"]
    estimator.observe(timestamp + 0.1, sequence_id=7, packet_size_bytes=1000)
    close(samples[-1]["filtered_ewma_interarrival"], before_future)


def test_stale_gap_and_recovery_state():
    estimator = ArrivalCapacityEstimator(
        alpha=1.0,
        robust_window_s=0.02,
        freshness_multiplier=2.0,
        freshness_min_s=0.01,
        freshness_max_s=0.02,
        recovery_samples=5,
        recovery_window_s=0.02,
    )
    estimator.observe(1.000, sequence_id=1, packet_size_bytes=1000)
    estimator.observe(1.005, sequence_id=2, packet_size_bytes=1000)
    stale = estimator.snapshot(1.100, packet_size_bytes=1000)
    assert not stale["estimate_fresh"]
    boundary = estimator.observe(1.100, sequence_id=3, packet_size_bytes=1000)
    assert boundary["corrected_interarrival"] is None
    assert boundary["skip_reason"] == "stale_boundary"
    assert boundary["filtered_ewma_interarrival"] is None

    # Five compressed adjacent samples alone are insufficient because they do
    # not span the configured causal recovery window.
    last = None
    for sequence in range(4, 9):
        last = estimator.observe(
            1.100 + (sequence - 3) * 0.0001,
            sequence_id=sequence,
            packet_size_bytes=1000,
        )
    assert not last["estimate_fresh"]
    assert last["filtered_ewma_interarrival"] is None

    # A gap resets both count and buffered recovery samples.
    gap = estimator.observe(1.105, sequence_id=10, packet_size_bytes=1000)
    assert gap["corrected_interarrival"] is None
    assert gap["skip_reason"] == "sequence_gap"
    assert gap["filtered_ewma_interarrival"] is None

    timestamps = [1.110, 1.115, 1.120, 1.125, 1.130]
    for index, timestamp in enumerate(timestamps, 11):
        last = estimator.observe(
            timestamp, sequence_id=index, packet_size_bytes=1000
        )
    assert last["estimate_fresh"]
    assert last["filter_reason"] == "recovery_seeded"
    close(last["filtered_ewma_interarrival"], 0.005)


def test_idle_and_ordinary_arrivals():
    estimator = ArrivalCapacityEstimator(
        alpha=1.0,
        recovery_samples=2,
        recovery_window_s=0.01,
    )
    estimator.observe(2.000, sequence_id=1, packet_size_bytes=1000)
    ordinary = estimator.observe(
        2.010, sequence_id=2, packet_size_bytes=1000
    )
    assert ordinary["corrected_interarrival"] is not None
    idle = estimator.observe(
        2.100,
        sender_grace_period=0.09,
        sequence_id=3,
        packet_size_bytes=1000,
    )
    assert idle["corrected_interarrival"] is None
    assert idle["skip_reason"] == "sender_idle_boundary"
    assert not idle["estimate_fresh"]


def test_constant_rate_and_clean_steps():
    estimator = ArrivalCapacityEstimator(alpha=0.1, robust_window_s=0.05)
    timestamp = 10.0
    sequence = 1
    estimator.observe(timestamp, sequence_id=sequence, packet_size_bytes=1000)
    interval_values = []
    for rate in (8.0, 3.0, 6.0):
        segment = []
        interval = 1000 * 8.0 / rate / 1_000_000.0
        for _ in range(300):
            sequence += 1
            timestamp += interval
            sample = estimator.observe(
                timestamp, sequence_id=sequence, packet_size_bytes=1000
            )
            if sample["published_ewma_interarrival"] is not None:
                segment.append(capacity_mbps(
                    1000, sample["published_ewma_interarrival"]
                ))
        interval_values.append(segment)
    for values, expected in zip(interval_values, (8.0, 3.0, 6.0)):
        tail = values[-50:]
        close(sum(tail) / len(tail), expected, tolerance=1e-6)


def test_replay_logging_and_matching_windows():
    source = [
        {
            "receiver_timestamp": "20.000",
            "sequence_id": "1",
            "packet_size_bytes": "1000",
            "sender_grace_period": "0",
        },
        {
            "receiver_timestamp": "20.001",
            "sequence_id": "2",
            "packet_size_bytes": "1000",
            "sender_grace_period": "0",
        },
        {
            "receiver_timestamp": "20.002",
            "sequence_id": "3",
            "packet_size_bytes": "1000",
            "sender_grace_period": "0",
        },
    ]
    replayed = replay_rows(source, ArrivalCapacityEstimator(alpha=1.0))
    assert replayed[1]["instantaneous_service_rate_mbps"] == "8.000000000"
    assert replayed[1]["estimated_capacity_mbps"] == "8.000000000"
    metrics = binned_metrics(replayed, 20.0, 21.0)
    assert len(metrics["goodput"]) == len(metrics["published"]) == 1
    close(metrics["active_service"][0], 8.0)


def test_no_ground_truth_inputs():
    forbidden = {"bandwidth", "loss", "trace", "ground_truth", "expected"}
    parameters = inspect.signature(ArrivalCapacityEstimator).parameters
    if any(any(word in name for word in forbidden) for name in parameters):
        raise AssertionError("ground truth entered estimator constructor")


def main():
    tests = [
        test_named_columns_and_single_conversion,
        test_median_bounds_and_causality,
        test_stale_gap_and_recovery_state,
        test_idle_and_ordinary_arrivals,
        test_constant_rate_and_clean_steps,
        test_replay_logging_and_matching_windows,
        test_no_ground_truth_inputs,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} estimator audit checks passed.")


if __name__ == "__main__":
    main()
