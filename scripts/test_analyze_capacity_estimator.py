#!/usr/bin/env python3
"""Deterministic checks for finite Week 1 validation alignment."""

import importlib.util
import math
import tempfile
import csv
from pathlib import Path


SCRIPT = Path(__file__).with_name("analyze_capacity_estimator.py")
SPEC = importlib.util.spec_from_file_location("capacity_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def assert_close(actual, expected, label, tolerance=1e-9):
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def sample(timestamp, estimate):
    return {
        "timestamp": float(timestamp),
        "estimated_mbps": float(estimate),
        "raw_estimated_mbps": float(estimate),
        "max_frame_size_bytes": 1000.0,
    }


def main():
    gt = [
        {"timestamp": 0.0, "bandwidth_mbps": 8.0},
        {"timestamp": 10.0, "bandwidth_mbps": 3.0},
        {"timestamp": 20.0, "bandwidth_mbps": 3.0},
    ]
    samples = [sample(5, 8), sample(10, 3), sample(15, 3), sample(20, 3), sample(21, 100)]
    probe = {"last_sent_timestamp": 22.0}
    start, end, schedule_end = analysis.comparison_bounds(gt, samples, probe, 20.0)
    assert_close(start, 5.0, "partial comparison start")
    assert_close(end, 20.0, "finite comparison end")
    assert_close(schedule_end, 20.0, "configured schedule end")

    cropped = analysis.crop_timestamped(samples, start, end)
    if [row["timestamp"] for row in cropped] != [5.0, 10.0, 15.0, 20.0]:
        raise AssertionError("post-trace estimator tail was not cropped")
    excluded = analysis.comparison_exclusion_counts(samples, start, end)
    if excluded != {"before_comparison_start": 0, "after_comparison_end": 1}:
        raise AssertionError(f"unexpected comparison exclusions: {excluded}")
    metrics = analysis.error_metrics([
        (row["timestamp"], row["estimated_mbps"], analysis.gt_at(gt, row["timestamp"]))
        for row in cropped
    ])
    assert_close(metrics["mae_mbps"], 0.0, "post-trace tail excluded from errors")

    plot_x, _ = analysis.ground_truth_plot_series(gt, start, end)
    assert_close(plot_x[0], start, "overlay starts at comparison start")
    assert_close(plot_x[-1], end, "overlay stops at schedule end")

    status, coverage = analysis.coverage_status(0.0, 10.0, start, end, schedule_end)
    if status != "partial_start":
        raise AssertionError(f"expected partial_start, got {status}")
    assert_close(coverage, 0.5, "partial-start coverage")

    lag_samples = [sample(19.1, 6), sample(19.9, 6), sample(20.1, 6), sample(21.2, 6)]
    valid_lag_samples = analysis.crop_timestamped(lag_samples, 0.0, schedule_end)
    if analysis.transition_lag(valid_lag_samples, 19.0, 6.0, 0.25, 2.0) is not None:
        raise AssertionError("transition incorrectly used samples beyond schedule end")

    corr_gt = [
        {"timestamp": 0.0, "bandwidth_mbps": 8.0},
        {"timestamp": 10.0, "bandwidth_mbps": 3.0},
        {"timestamp": 20.0, "bandwidth_mbps": 3.0},
    ]
    corr_samples = [sample(t, 8 if t < 10 else 3) for t in range(1, 20)]
    expected_corr = analysis.cross_correlation_lag(corr_samples, corr_gt, 2.0, 1.0)
    tailed = corr_samples + [sample(21, 100), sample(22, 100)]
    cropped_tailed = analysis.crop_timestamped(tailed, 1.0, 20.0)
    actual_corr = analysis.cross_correlation_lag(cropped_tailed, corr_gt, 2.0, 1.0)
    if actual_corr != expected_corr:
        raise AssertionError("post-trace tail contaminated cross-correlation")

    with tempfile.TemporaryDirectory() as temp_dir:
        estimator_csv = Path(temp_dir) / "estimator.csv"
        with estimator_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "acknowledged_sequence_number",
                    "corrected_interarrival_time",
                    "smoothed_tau",
                    "packet_size_bytes",
                    "estimated_capacity_mbps",
                ],
            )
            writer.writeheader()
            for timestamp, seq in [(1.0, 1), (2.0, 3), (3.0, 2), (4.0, 2)]:
                writer.writerow({
                    "timestamp": timestamp,
                    "acknowledged_sequence_number": seq,
                    "corrected_interarrival_time": 0.001,
                    "smoothed_tau": 0.001,
                    "packet_size_bytes": 1000,
                    "estimated_capacity_mbps": 8.0,
                })
        _, sequence_report = analysis.read_estimator(estimator_csv, 0.0)
        if sequence_report["missing_sequence_ids"] != 0:
            raise AssertionError("sequence range accounting is incorrect")
        if sequence_report["duplicate_ack_sequence_ids"] != 1:
            raise AssertionError("duplicate ACK was not counted")
        if sequence_report["reordered_ack_sequence_ids"] != 1:
            raise AssertionError("reordered ACK was not counted")

    print("capacity analyzer unit test passed")


if __name__ == "__main__":
    main()
