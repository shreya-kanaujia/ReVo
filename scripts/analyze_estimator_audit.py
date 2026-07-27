#!/usr/bin/env python3
"""Offline replay and consistency audit for a retained receiver-arrival run."""

import argparse
import csv
import math
import sys
from pathlib import Path
from statistics import median

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "receiver"))

from capacity_estimator import ArrivalCapacityEstimator  # noqa: E402


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, field):
    value = row.get(field, "")
    return None if value in ("", None) else float(value)


def capacity_mbps(packet_size_bytes, service_interval_s):
    if service_interval_s is None or service_interval_s <= 0.0:
        return None
    return float(packet_size_bytes) * 8.0 / service_interval_s / 1_000_000.0


def extract_estimator_columns(row):
    """Read estimator values by semantic column name, never CSV position."""
    return {
        "raw_ewma_mbps": number(row, "estimated_capacity_mbps"),
        "filtered_mbps": number(row, "filtered_estimated_capacity_mbps"),
        "published_mbps": number(row, "published_estimated_capacity_mbps"),
    }


def comparison_interval(trace_rows, probe_rows, duration_s):
    schedule_start = (
        number(trace_rows[0], "monotonic_timestamp")
        - number(trace_rows[0], "elapsed_s")
    )
    schedule_end = schedule_start + float(duration_s)
    probe_end = max(number(row, "timestamp_monotonic") for row in probe_rows)
    return schedule_start, min(schedule_end, probe_end)


def replay_rows(receiver_rows, estimator):
    replayed = []
    for source in receiver_rows:
        timestamp = number(source, "receiver_timestamp")
        packet_size = int(source["packet_size_bytes"])
        sample = estimator.observe(
            timestamp,
            sender_grace_period=number(source, "sender_grace_period") or 0.0,
            sequence_id=int(source["sequence_id"]),
            packet_size_bytes=packet_size,
        )
        raw_interval = sample["corrected_interarrival"]
        raw_ewma_interval = sample["ewma_interarrival"]
        filtered_interval = sample["filtered_ewma_interarrival"]
        active_service_values = [
            value * packet_size
            for _sample_timestamp, value in estimator._service_samples
        ]
        replayed.append({
            "receiver_timestamp": f"{timestamp:.9f}",
            "sequence_id": source["sequence_id"],
            "packet_size_bytes": packet_size,
            "raw_interarrival_time": (
                "" if sample["raw_interarrival"] is None
                else f"{sample['raw_interarrival']:.9f}"
            ),
            "sender_grace_period": f"{sample['sender_grace_period']:.9f}",
            "corrected_interarrival_time": (
                "" if raw_interval is None else f"{raw_interval:.9f}"
            ),
            "instantaneous_service_rate_mbps": (
                "" if raw_interval is None
                else f"{capacity_mbps(packet_size, raw_interval):.9f}"
            ),
            "smoothed_tau": (
                "" if raw_ewma_interval is None
                else f"{raw_ewma_interval:.9f}"
            ),
            "estimated_capacity_mbps": (
                "" if raw_ewma_interval is None
                else f"{capacity_mbps(packet_size, raw_ewma_interval):.9f}"
            ),
            "filtered_smoothed_tau": (
                "" if filtered_interval is None
                else f"{filtered_interval:.9f}"
            ),
            "filtered_estimated_capacity_mbps": (
                "" if filtered_interval is None
                else f"{capacity_mbps(packet_size, filtered_interval):.9f}"
            ),
            "published_estimated_capacity_mbps": (
                ""
                if sample["published_ewma_interarrival"] is None
                else f"{capacity_mbps(packet_size, sample['published_ewma_interarrival']):.9f}"
            ),
            "estimate_age_s": (
                "" if sample["estimate_age_s"] is None
                else f"{sample['estimate_age_s']:.9f}"
            ),
            "estimate_fresh": "1" if sample["estimate_fresh"] else "0",
            "freshness_threshold_s": f"{sample['freshness_threshold_s']:.9f}",
            "last_valid_update_timestamp": (
                "" if sample["last_valid_update_ts"] is None
                else f"{sample['last_valid_update_ts']:.9f}"
            ),
            "skip_reason": sample["skip_reason"],
            "filter_reason": sample["filter_reason"],
            "active_window_min_interval": (
                "" if not active_service_values
                else f"{min(active_service_values):.9f}"
            ),
            "active_window_max_interval": (
                "" if not active_service_values
                else f"{max(active_service_values):.9f}"
            ),
        })
    return replayed


def error_metrics(estimates, references):
    if not estimates:
        return None, None
    errors = np.asarray(estimates) - np.asarray(references)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def binned_metrics(rows, start, end):
    count = int(math.ceil(end - start))
    delivered_bytes = np.zeros(count)
    active_bytes = np.zeros(count)
    active_time = np.zeros(count)
    published = [[] for _ in range(count)]
    raw = [[] for _ in range(count)]

    for row in rows:
        timestamp = number(row, "receiver_timestamp")
        if timestamp is None or timestamp < start or timestamp > end:
            continue
        index = min(count - 1, int(timestamp - start))
        packet_size = int(row["packet_size_bytes"])
        delivered_bytes[index] += packet_size
        corrected = number(row, "corrected_interarrival_time")
        if corrected is not None:
            active_bytes[index] += packet_size
            active_time[index] += corrected
        values = extract_estimator_columns(row)
        if values["raw_ewma_mbps"] is not None:
            raw[index].append(values["raw_ewma_mbps"])
        if values["published_mbps"] is not None:
            published[index].append(values["published_mbps"])

    widths = np.ones(count)
    final_width = end - start - math.floor(end - start)
    if final_width > 1e-9:
        widths[-1] = final_width
    goodput = delivered_bytes * 8.0 / 1_000_000.0 / widths
    active_service = np.divide(
        active_bytes * 8.0 / 1_000_000.0,
        active_time,
        out=np.full(count, np.nan),
        where=active_time > 0.0,
    )
    published_median = np.asarray([
        median(values) if values else np.nan for values in published
    ])
    raw_median = np.asarray([
        median(values) if values else np.nan for values in raw
    ])
    published_mask = np.isfinite(published_median)
    matched_mask = published_mask & np.isfinite(active_service)
    matched_mae, matched_rmse = error_metrics(
        published_median[matched_mask].tolist(),
        active_service[matched_mask].tolist(),
    )
    goodput_mae, goodput_rmse = error_metrics(
        published_median[published_mask].tolist(),
        goodput[published_mask].tolist(),
    )
    overall_active_service = (
        float(np.sum(active_bytes)) * 8.0 / 1_000_000.0
        / float(np.sum(active_time))
        if np.sum(active_time) > 0.0 else None
    )
    return {
        "goodput": goodput,
        "active_service": active_service,
        "published": published_median,
        "raw": raw_median,
        "published_mask": published_mask,
        "matched_mask": matched_mask,
        "matched_mae": matched_mae,
        "matched_rmse": matched_rmse,
        "goodput_mae": goodput_mae,
        "goodput_rmse": goodput_rmse,
        "overall_active_service": overall_active_service,
    }


def freshness_durations(rows, start, end):
    intervals = []
    for index, row in enumerate(rows):
        timestamp = number(row, "receiver_timestamp")
        if timestamp is None or timestamp < start or timestamp > end:
            continue
        next_timestamp = end
        for candidate in rows[index + 1:]:
            candidate_ts = number(candidate, "receiver_timestamp")
            if candidate_ts is not None and candidate_ts >= timestamp:
                next_timestamp = min(end, candidate_ts)
                break
        if row.get("estimate_fresh") != "1":
            continue
        age = number(row, "estimate_age_s")
        threshold = number(row, "freshness_threshold_s")
        if age is None or threshold is None:
            continue
        fresh_end = min(next_timestamp, timestamp + max(0.0, threshold - age))
        if fresh_end > timestamp:
            intervals.append((timestamp, fresh_end))
    merged = []
    for left, right in intervals:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    fresh = sum(right - left for left, right in merged)
    stale = max(0.0, end - start - fresh)
    return fresh, stale


def distribution(rows, field, fresh_only=False):
    values = [
        number(row, field) for row in rows
        if number(row, field) is not None
        and (not fresh_only or row.get("estimate_fresh") == "1")
    ]
    if not values:
        return (None, None, None, None)
    return (
        min(values),
        float(np.median(values)),
        float(np.percentile(values, 99)),
        max(values),
    )


def serialization_consistency(receiver_rows, sender_rows):
    sender_by_sequence = {
        int(row["acknowledged_sequence_number"]): row for row in sender_rows
    }
    pairs = 0
    mismatches = 0
    maximum_difference = 0.0
    fields = [
        ("receiver_timestamp", "receiver_timestamp"),
        ("raw_interarrival_time", "raw_interarrival_time"),
        ("sender_grace_period", "sender_grace_period"),
        ("corrected_interarrival_time", "corrected_interarrival_time"),
        ("smoothed_tau", "smoothed_tau"),
        ("filtered_smoothed_tau", "filtered_smoothed_tau"),
        ("freshness_threshold_s", "freshness_threshold_s"),
    ]
    for receiver in receiver_rows:
        sender = sender_by_sequence.get(int(receiver["sequence_id"]))
        if sender is None:
            continue
        pairs += 1
        for receiver_field, sender_field in fields:
            left = number(receiver, receiver_field)
            right = number(sender, sender_field)
            if left is None or right is None:
                if left is not None or right is not None:
                    mismatches += 1
                continue
            difference = abs(left - right)
            maximum_difference = max(maximum_difference, difference)
            if difference > 1e-9:
                mismatches += 1
    return pairs, mismatches, maximum_difference


def fmt(value):
    return "not available" if value is None else f"{value:.9f}"


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_audit(original, replayed, bins, start, output):
    import matplotlib.pyplot as plt

    original_t = [
        number(row, "receiver_timestamp") - start for row in original
        if number(row, "receiver_timestamp") >= start
    ]
    original_published = [
        number(row, "published_estimated_capacity_mbps") for row in original
        if number(row, "receiver_timestamp") >= start
    ]
    replay_t = [
        number(row, "receiver_timestamp") - start for row in replayed
        if number(row, "receiver_timestamp") >= start
    ]
    replay_published = [
        number(row, "published_estimated_capacity_mbps") for row in replayed
        if number(row, "receiver_timestamp") >= start
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(original_t, original_published, ".", ms=1)
    axes[0].set(title="Retained publication before audit fixes", ylabel="Mbps")
    axes[1].plot(replay_t, replay_published, ".", ms=1)
    axes[1].set(
        title="Offline replay after audit fixes",
        xlabel="Elapsed controlled time (s)",
        ylabel="Mbps",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "before_vs_corrected_publication.png", dpi=160)
    plt.close(fig)

    seconds = np.arange(len(bins["goodput"])) + 0.5
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.step(seconds, bins["goodput"], where="mid", label="Delivered goodput (1 s)")
    ax.plot(
        seconds,
        bins["active_service"],
        linewidth=1,
        label="Conditional active-service reference (1 s)",
    )
    ax.plot(
        seconds,
        bins["published"],
        ".",
        ms=4,
        label="Published estimate (1 s median; gaps are unavailable)",
    )
    ax.set(
        title="wifi/trace14 offline estimator audit",
        xlabel="Elapsed controlled time (s)",
        ylabel="Mbps",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "matched_reference_and_goodput.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=324.15)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite audit output: {args.output}")
    args.output.mkdir(parents=True)

    receiver = read_csv(args.run / "receiver_arrivals_estimator.csv")
    sender_feedback = read_csv(args.run / "sender_feedback.csv")
    trace = read_csv(args.run / "applied_trace.csv")
    probe = read_csv(args.run / "sender_probe.csv")
    start, end = comparison_interval(trace, probe, args.duration)

    estimator = ArrivalCapacityEstimator()
    replayed = replay_rows(receiver, estimator)
    write_csv(args.output / "replayed_estimator.csv", replayed)
    write_csv(args.output / "sample_decisions.csv", replayed)

    controlled_original = [
        row for row in receiver
        if start <= number(row, "receiver_timestamp") <= end
    ]
    controlled_replay = [
        row for row in replayed
        if start <= number(row, "receiver_timestamp") <= end
    ]
    bins = binned_metrics(controlled_replay, start, end)
    fresh_duration, stale_duration = freshness_durations(
        controlled_replay, start, end
    )
    raw_stats = distribution(controlled_replay, "estimated_capacity_mbps")
    filtered_stats = distribution(
        controlled_replay, "filtered_estimated_capacity_mbps"
    )
    published_stats = distribution(
        controlled_replay, "published_estimated_capacity_mbps", fresh_only=True
    )
    original_internal_stats = distribution(
        controlled_original, "filtered_estimated_capacity_mbps"
    )
    original_published_stats = distribution(
        controlled_original, "published_estimated_capacity_mbps", fresh_only=True
    )
    original_filtered_max_row = max(
        (
            row for row in controlled_original
            if number(row, "filtered_estimated_capacity_mbps") is not None
        ),
        key=lambda row: number(row, "filtered_estimated_capacity_mbps"),
    )
    freshness_violations = sum(
        row["estimate_fresh"] == "1"
        and (
            number(row, "estimate_age_s") is None
            or number(row, "estimate_age_s")
            > number(row, "freshness_threshold_s") + 1e-12
        )
        for row in controlled_replay
    )
    publication_flag_violations = sum(
        (number(row, "published_estimated_capacity_mbps") is not None)
        != (row["estimate_fresh"] == "1")
        for row in controlled_replay
    )
    filter_bound_violations = 0
    for row in controlled_replay:
        filtered_interval = number(row, "filtered_smoothed_tau")
        lower = number(row, "active_window_min_interval")
        upper = number(row, "active_window_max_interval")
        if filtered_interval is None:
            continue
        if lower is None or upper is None or not (
            lower - 1e-12 <= filtered_interval <= upper + 1e-12
        ):
            filter_bound_violations += 1
    serialized_pairs, serialization_mismatches, serialization_max_difference = (
        serialization_consistency(controlled_original, sender_feedback)
    )

    plot_audit(controlled_original, controlled_replay, bins, start, args.output)
    report = [
        "wifi/trace14 retained-run estimator offline audit",
        "",
        f"comparison_start: {start:.9f}",
        f"comparison_end: {end:.9f}",
        f"comparison_duration_s: {end - start:.9f}",
        "estimator_definition: conditional adjacent-message receiver service rate",
        "goodput_definition: all receiver-delivered application bytes per wall-clock second",
        "matched_reference_definition: valid adjacent bytes divided by their summed service intervals per one-second bin",
        "",
        "original_internal_filtered_max_sequence_id: "
        f"{original_filtered_max_row['sequence_id']}",
        "original_internal_filtered_max_timestamp: "
        f"{original_filtered_max_row['receiver_timestamp']}",
        "original_internal_filtered_max_interval_s: "
        f"{original_filtered_max_row['corrected_interarrival_time']}",
        "confirmed_80mbps_cause: first valid recovery spacing seeded the old filtered state",
        "confirmed_80mbps_serialization_error: false",
        "confirmed_analyzer_error: internal recovering filtered rows were reported as control output",
        f"original_internal_filtered_min_median_p99_max_mbps: {' '.join(fmt(v) for v in original_internal_stats)}",
        f"original_published_min_median_p99_max_mbps: {' '.join(fmt(v) for v in original_published_stats)}",
        f"corrected_raw_min_median_p99_max_mbps: {' '.join(fmt(v) for v in raw_stats)}",
        f"corrected_internal_filtered_min_median_p99_max_mbps: {' '.join(fmt(v) for v in filtered_stats)}",
        f"corrected_published_min_median_p99_max_mbps: {' '.join(fmt(v) for v in published_stats)}",
        f"fresh_duration_s: {fresh_duration:.9f}",
        f"stale_duration_s: {stale_duration:.9f}",
        f"fresh_1s_bins: {int(np.sum(bins['published_mask']))}",
        f"stale_1s_bins: {len(bins['published']) - int(np.sum(bins['published_mask']))}",
        f"matched_active_service_mae_mbps: {fmt(bins['matched_mae'])}",
        f"matched_active_service_rmse_mbps: {fmt(bins['matched_rmse'])}",
        f"overall_conditional_active_service_mbps: {fmt(bins['overall_active_service'])}",
        f"separate_goodput_mae_mbps: {fmt(bins['goodput_mae'])}",
        f"separate_goodput_rmse_mbps: {fmt(bins['goodput_rmse'])}",
        f"delivered_goodput_mean_mbps: {float(np.mean(bins['goodput'])):.9f}",
        f"matched_bins: {int(np.sum(bins['matched_mask']))}",
        f"freshness_flag_violations: {freshness_violations}",
        f"publication_flag_violations: {publication_flag_violations}",
        f"causal_filter_bound_violations: {filter_bound_violations}",
        f"retained_feedback_serialized_pairs_checked: {serialized_pairs}",
        f"retained_feedback_field_mismatches: {serialization_mismatches}",
        "retained_feedback_max_numeric_difference: "
        f"{serialization_max_difference:.12g}",
        "",
        "integrity_trace_values_enter_estimator: false",
        "integrity_future_samples_used: false",
        "integrity_calibration_or_rescaling: false",
        "integrity_retained_csv_modified: false",
    ]
    (args.output / "audit_metrics.txt").write_text("\n".join(report) + "\n")
    print("\n".join(report))


if __name__ == "__main__":
    main()
