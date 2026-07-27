#!/usr/bin/env python3
"""Compare the retained before/post-fix wifi/trace14 runs without calibration."""

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


TRACE_DURATION_S = 324.15


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, field):
    value = row.get(field, "")
    return None if value in ("", None) else float(value)


def percentile(values, q):
    return None if not values else float(np.percentile(values, q))


def fmt(value, digits=9):
    return "not available" if value is None else f"{value:.{digits}f}"


def error_metrics(estimate, reference):
    if not estimate:
        return None, None
    errors = np.asarray(estimate) - np.asarray(reference)
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def sequence_stats(sequence):
    counts = Counter(sequence)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    reordered = sum(current < previous for previous, current in zip(sequence, sequence[1:]))
    gaps = []
    for previous, current in zip(sequence, sequence[1:]):
        if current > previous + 1:
            gaps.append((previous, current, current - previous - 1))
    return duplicates, reordered, gaps


def fresh_intervals(rows, start, end):
    events = []
    for row in rows:
        timestamp = number(row, "timestamp")
        if timestamp is None or timestamp < start or timestamp > end:
            continue
        fresh = row.get("estimate_fresh") == "1"
        age = number(row, "estimate_age_s")
        threshold = number(row, "freshness_threshold_s")
        events.append((timestamp, fresh, age, threshold))

    intervals = []
    for index, event in enumerate(events):
        timestamp, fresh, age, threshold = event
        next_timestamp = events[index + 1][0] if index + 1 < len(events) else end
        if fresh and age is not None and threshold is not None:
            fresh_end = min(next_timestamp, timestamp + max(0.0, threshold - age))
            if fresh_end > timestamp:
                intervals.append((timestamp, fresh_end))
    return intervals


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def stale_summary(fresh, start, end):
    merged = merge_intervals(fresh)
    fresh_duration = sum(right - left for left, right in merged)
    stale = []
    cursor = start
    for left, right in merged:
        if left > cursor:
            stale.append((cursor, left))
        cursor = max(cursor, right)
    if cursor < end:
        stale.append((cursor, end))
    longest = max((right - left for left, right in stale), default=0.0)
    return max(0.0, end - start - fresh_duration), longest


def parse_probe_summary(log_path):
    text = log_path.read_text(errors="replace")
    matches = re.findall(
        r"\[PacedProbe\] completed elapsed=([0-9.]+)s packets=(\d+) "
        r"bytes=(\d+) offered=([0-9.]+) Mbps pauses=(\d+) "
        r"missed_deadlines=(\d+) max_buffered=(\d+) channel=(\w+)",
        text,
    )
    if not matches:
        return {}
    elapsed, packets, byte_count, offered, pauses, missed, buffered, channel = matches[-1]
    return {
        "elapsed_s": float(elapsed),
        "packets": int(packets),
        "bytes": int(byte_count),
        "offered_mbps": float(offered),
        "pauses": int(pauses),
        "missed_deadlines": int(missed),
        "max_buffered": int(buffered),
        "channel": channel,
    }


def load_run(path, has_publication):
    trace = read_csv(path / "applied_trace.csv")
    probe = read_csv(path / "sender_probe.csv")
    receiver = read_csv(path / "receiver_arrivals_estimator.csv")
    feedback = read_csv(path / "sender_feedback.csv")
    sender_diag = read_csv(path / "sender_diagnostics.csv")
    receiver_diag = read_csv(path / "receiver_diagnostics.csv")

    schedule_start = number(trace[0], "monotonic_timestamp") - number(trace[0], "elapsed_s")
    schedule_end = schedule_start + TRACE_DURATION_S
    comparison_start = schedule_start
    comparison_end = min(
        schedule_end,
        max(number(row, "timestamp_monotonic") for row in probe),
    )

    all_sent_rows = [row for row in probe if row.get("sent") == "1"]
    sent_rows = [
        row for row in probe
        if row.get("sent") == "1"
        and comparison_start <= number(row, "timestamp_monotonic") <= comparison_end
    ]
    all_receiver_rows = receiver
    receiver_rows = [
        row for row in receiver
        if comparison_start <= number(row, "receiver_timestamp") <= comparison_end
    ]
    all_feedback_rows = feedback
    feedback_rows = [
        row for row in feedback
        if comparison_start <= number(row, "timestamp") <= comparison_end
    ]
    sent_seq = [int(row["probe_sequence_id"]) for row in all_sent_rows]
    recv_seq = [int(row["sequence_id"]) for row in all_receiver_rows]
    ack_seq = [int(row["acknowledged_sequence_number"]) for row in all_feedback_rows]
    recv_duplicates, recv_reordered, recv_gaps = sequence_stats(recv_seq)
    ack_duplicates, ack_reordered, ack_gaps = sequence_stats(ack_seq)

    all_valid_rows = [
        row for row in all_receiver_rows
        if number(row, "corrected_interarrival_time") is not None
    ]
    valid_rows = [
        row for row in receiver_rows
        if number(row, "corrected_interarrival_time") is not None
    ]
    raw = [
        number(row, "estimated_capacity_mbps") for row in valid_rows
        if number(row, "estimated_capacity_mbps") is not None
    ]
    internal_filtered = [
        number(row, "filtered_estimated_capacity_mbps") for row in valid_rows
        if number(row, "filtered_estimated_capacity_mbps") is not None
    ]
    filtered = [
        number(row, "published_estimated_capacity_mbps") for row in receiver_rows
        if number(row, "published_estimated_capacity_mbps") is not None
        and row.get("estimate_fresh") == "1"
    ]
    valid_timestamps = [
        number(row, "receiver_timestamp") for row in valid_rows
    ]
    update_boundaries = [comparison_start, *valid_timestamps, comparison_end]
    longest_no_update = max(
        right - left for left, right in zip(update_boundaries, update_boundaries[1:])
    )

    bin_count = int(math.ceil(comparison_end - comparison_start))
    delivered_bytes = np.zeros(bin_count)
    raw_bins = [[] for _ in range(bin_count)]
    filtered_bins = [[] for _ in range(bin_count)]
    published_bins = [[] for _ in range(bin_count)]
    for row in receiver_rows:
        index = int(number(row, "receiver_timestamp") - comparison_start)
        if 0 <= index < bin_count:
            delivered_bytes[index] += int(row["packet_size_bytes"])
            if number(row, "corrected_interarrival_time") is not None:
                raw_value = number(row, "estimated_capacity_mbps")
                filtered_value = number(row, "filtered_estimated_capacity_mbps")
                if raw_value is not None:
                    raw_bins[index].append(raw_value)
                if filtered_value is not None:
                    filtered_bins[index].append(filtered_value)
    if has_publication:
        for row in feedback_rows:
            value = number(row, "published_estimated_capacity_mbps")
            index = int(number(row, "timestamp") - comparison_start)
            if value is not None and 0 <= index < bin_count:
                published_bins[index].append(value)

    bin_widths = np.ones(bin_count)
    final_width = comparison_end - comparison_start - math.floor(
        comparison_end - comparison_start
    )
    if final_width > 1e-9:
        bin_widths[-1] = final_width
    delivered_mbps = delivered_bytes * 8.0 / 1_000_000.0 / bin_widths
    raw_median = np.array([
        np.median(values) if values else np.nan for values in raw_bins
    ])
    filtered_median = np.array([
        np.median(values) if values else np.nan for values in filtered_bins
    ])
    published_median = np.array([
        np.median(values) if values else np.nan for values in published_bins
    ])
    fresh_mask = np.isfinite(published_median)
    fresh_mae, fresh_rmse = error_metrics(
        published_median[fresh_mask].tolist(),
        delivered_mbps[fresh_mask].tolist(),
    )

    raw_held = np.full(bin_count, np.nan)
    valid_pairs = list(zip(valid_timestamps, raw))
    cursor = 0
    held = None
    for index in range(bin_count):
        bin_center = comparison_start + index + 0.5
        while cursor < len(valid_pairs) and valid_pairs[cursor][0] <= bin_center:
            held = valid_pairs[cursor][1]
            cursor += 1
        if held is not None:
            raw_held[index] = held
    held_mask = np.isfinite(raw_held)
    held_mae, held_rmse = error_metrics(
        raw_held[held_mask].tolist(), delivered_mbps[held_mask].tolist()
    )

    if has_publication:
        fresh = fresh_intervals(feedback_rows, comparison_start, comparison_end)
        stale_duration, longest_stale = stale_summary(
            fresh, comparison_start, comparison_end
        )
        incorrectly_fresh_rows = [
            row for row in feedback_rows
            if row.get("estimate_fresh") == "1"
            and number(row, "estimate_age_s") is not None
            and number(row, "freshness_threshold_s") is not None
            and number(row, "estimate_age_s")
            > number(row, "freshness_threshold_s")
        ]
        incorrectly_fresh_duration = 0.0 if not incorrectly_fresh_rows else None
    else:
        stale_duration = None
        longest_stale = longest_no_update
        incorrectly_fresh_duration = longest_no_update

    sender_set = set(sent_seq)
    receiver_set = set(recv_seq)
    ack_set = set(ack_seq)
    pause_rows = [row for row in probe if row.get("paused_due_to_backpressure") == "1"]
    grace_rows = [
        row for row in sent_rows
        if number(row, "sender_grace_period") not in (None, 0.0)
    ]
    gap_recovery_s = []
    unrecovered_gaps = 0
    for index, row in enumerate(receiver_rows):
        if index == 0:
            continue
        if int(row["sequence_id"]) <= int(receiver_rows[index - 1]["sequence_id"]) + 1:
            continue
        recovery = next(
            (
                candidate
                for candidate in receiver_rows[index + 1:]
                if number(candidate, "corrected_interarrival_time") is not None
            ),
            None,
        )
        if recovery is None:
            unrecovered_gaps += 1
        else:
            gap_recovery_s.append(
                number(recovery, "receiver_timestamp")
                - number(row, "receiver_timestamp")
            )
    max_pause_duration = max(
        (
            number(row, "backpressure_pause_duration_seconds") or 0.0
            for row in pause_rows
        ),
        default=0.0,
    )
    max_outstanding = max(
        (int(row["outstanding_packets"]) for row in sender_diag),
        default=0,
    )

    return {
        "path": path,
        "trace": trace,
        "probe": probe,
        "receiver": receiver_rows,
        "feedback": feedback_rows,
        "sender_diag": sender_diag,
        "receiver_diag": receiver_diag,
        "start": comparison_start,
        "end": comparison_end,
        "duration": comparison_end - comparison_start,
        "sent": len(all_sent_rows),
        "controlled_sent": len(sent_rows),
        "received": len(all_receiver_rows),
        "controlled_received": len(receiver_rows),
        "feedback_count": len(all_feedback_rows),
        "controlled_feedback_count": len(feedback_rows),
        "data_missing": len(sender_set - receiver_set),
        "feedback_missing": len(receiver_set - ack_set),
        "receiver_duplicates": recv_duplicates,
        "receiver_reordered": recv_reordered,
        "feedback_duplicates": ack_duplicates,
        "feedback_reordered": ack_reordered,
        "sequence_gaps": len(recv_gaps),
        "gap_missing": sum(gap[2] for gap in recv_gaps),
        "feedback_gaps": len(ack_gaps),
        "valid": len(all_valid_rows),
        "controlled_valid": len(valid_rows),
        "valid_percent_sent": 100.0 * len(all_valid_rows) / max(1, len(all_sent_rows)),
        "valid_percent_received": (
            100.0 * len(all_valid_rows) / max(1, len(all_receiver_rows))
        ),
        "longest_no_update": longest_no_update,
        "incorrectly_fresh_duration": incorrectly_fresh_duration,
        "stale_duration": stale_duration,
        "longest_stale": longest_stale,
        "raw": raw,
        "filtered": filtered,
        "internal_filtered": internal_filtered,
        "delivered": delivered_mbps,
        "delivered_overall_mbps": (
            sum(int(row["packet_size_bytes"]) for row in receiver_rows)
            * 8.0 / 1_000_000.0 / (comparison_end - comparison_start)
        ),
        "raw_bins": raw_median,
        "filtered_bins": filtered_median,
        "published_bins": published_median,
        "fresh_bin_count": int(np.sum(fresh_mask)),
        "stale_bin_count": int(bin_count - np.sum(fresh_mask)),
        "fresh_mae": fresh_mae,
        "fresh_rmse": fresh_rmse,
        "held_mae": held_mae,
        "held_rmse": held_rmse,
        "pause_rows": len(pause_rows),
        "max_pause_duration": max_pause_duration,
        "grace_rows": len(grace_rows),
        "schedule_reset_rows": sum(
            row.get("schedule_reset_due_to_lateness") == "1"
            for row in sent_rows
        ),
        "probe_summary": parse_probe_summary(path / "logs" / "sender.log"),
        "max_outstanding": max_outstanding,
        "source_fresh_rows": sum(
            row.get("estimate_fresh") == "1" for row in receiver_rows
        ),
        "published_feedback_rows": sum(
            number(row, "published_estimated_capacity_mbps") is not None
            for row in feedback_rows
        ),
        "gap_recovered_count": len(gap_recovery_s),
        "gap_unrecovered_count": unrecovered_gaps,
        "gap_recovery_median_s": percentile(gap_recovery_s, 50),
        "gap_recovery_max_s": max(gap_recovery_s) if gap_recovery_s else None,
        "receiver_sequence": recv_seq,
    }


def plot_results(before, after, output):
    seconds = np.arange(len(after["delivered"])) + 0.5

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(seconds, after["delivered"], where="mid", label="Delivered application throughput (1 s)")
    raw = after["raw_bins"]
    ax.plot(seconds[np.isfinite(raw)], raw[np.isfinite(raw)], ".", ms=4, label="Raw EWMA (bin median)")
    ax.set(xlabel="Elapsed trace time (s)", ylabel="Mbps",
           title="wifi/trace14 post-fix raw estimate versus delivered throughput")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "raw_estimate_vs_delivered_throughput.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(seconds, after["delivered"], where="mid", label="Delivered application throughput (1 s)")
    published = after["published_bins"]
    ax.plot(
        seconds,
        published,
        marker=".",
        ms=4,
        linewidth=1,
        label="Published fresh/filtered estimate",
    )
    ax.set(xlabel="Elapsed trace time (s)", ylabel="Mbps",
           title="wifi/trace14 post-fix control-safe estimate (gaps are stale)")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "published_estimate_vs_delivered_throughput.png", dpi=160)
    plt.close(fig)

    diag_t, ages, thresholds, fresh = [], [], [], []
    for row in after["sender_diag"]:
        timestamp = number(row, "timestamp_monotonic")
        if timestamp is None or not after["start"] <= timestamp <= after["end"]:
            continue
        diag_t.append(timestamp - after["start"])
        ages.append(number(row, "estimate_age_s") or 0.0)
        thresholds.append(number(row, "freshness_threshold_s") or 0.0)
        fresh.append(str(row.get("estimate_fresh")).lower() in ("1", "true"))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(diag_t, ages, label="Estimate age")
    ax.plot(diag_t, thresholds, label="Freshness threshold")
    ax.fill_between(diag_t, 0, ages, where=np.logical_not(fresh), alpha=0.2, color="red", label="Stale")
    ax.set(xlabel="Elapsed trace time (s)", ylabel="Seconds",
           title="wifi/trace14 post-fix estimate age and stale state")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "estimate_age_and_stale_intervals.png", dpi=160)
    plt.close(fig)

    trace_t = [number(row, "monotonic_timestamp") - after["start"] for row in after["trace"]]
    losses = [100.0 * number(row, "loss_ratio") for row in after["trace"]]
    gap_t = []
    previous = None
    for row in after["receiver"]:
        sequence = int(row["sequence_id"])
        if previous is not None and sequence > previous + 1:
            gap_t.append(number(row, "receiver_timestamp") - after["start"])
        previous = sequence
    pause_t = [
        number(row, "timestamp_monotonic") - after["start"]
        for row in after["probe"]
        if row.get("paused_due_to_backpressure") == "1"
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(trace_t, losses, linewidth=0.7, alpha=0.7, label="Configured packet loss (%)")
    ax.vlines(gap_t, 0, 100, color="tab:red", alpha=0.25, linewidth=0.5, label="Receiver sequence gap")
    if pause_t:
        ax.scatter(pause_t, np.full(len(pause_t), -4), s=2, color="black", label="Backpressure pause")
    ax.set(xlabel="Elapsed trace time (s)", ylabel="Loss / event marker",
           title="wifi/trace14 loss, sequence gaps, and sender backpressure")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "loss_gaps_and_backpressure.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for ax, run, title in (
        (axes[0], before, "Before fix (raw estimate)"),
        (axes[1], after, "Post fix (published fresh/filtered estimate)"),
    ):
        run_seconds = np.arange(len(run["delivered"])) + 0.5
        ax.step(run_seconds, run["delivered"], where="mid", label="Delivered throughput")
        values = run["raw_bins"] if run is before else run["published_bins"]
        ax.plot(run_seconds, values, ".", ms=3, label="Estimator")
        ax.set(ylabel="Mbps", title=title)
        ax.grid(alpha=0.25)
        ax.legend()
    axes[1].set_xlabel("Elapsed trace time (s)")
    fig.tight_layout()
    fig.savefig(output / "before_vs_after.png", dpi=160)
    plt.close(fig)


def describe(prefix, run):
    raw = run["raw"]
    filtered = run["filtered"]
    internal_filtered = run["internal_filtered"]
    summary = run["probe_summary"]
    lines = [
        f"{prefix}_comparison_start: {run['start']:.9f}",
        f"{prefix}_comparison_end: {run['end']:.9f}",
        f"{prefix}_controlled_duration_s: {run['duration']:.9f}",
        f"{prefix}_probe_elapsed_s: {fmt(summary.get('elapsed_s'))}",
        f"{prefix}_probe_channel_final_state: {summary.get('channel', 'not available')}",
        f"{prefix}_sent_messages: {run['sent']}",
        f"{prefix}_controlled_sent_messages: {run['controlled_sent']}",
        f"{prefix}_received_messages: {run['received']}",
        f"{prefix}_controlled_received_messages: {run['controlled_received']}",
        f"{prefix}_feedback_messages: {run['feedback_count']}",
        f"{prefix}_controlled_feedback_messages: {run['controlled_feedback_count']}",
        f"{prefix}_data_missing: {run['data_missing']}",
        f"{prefix}_feedback_missing: {run['feedback_missing']}",
        f"{prefix}_receiver_duplicates: {run['receiver_duplicates']}",
        f"{prefix}_receiver_reordered: {run['receiver_reordered']}",
        f"{prefix}_feedback_duplicates: {run['feedback_duplicates']}",
        f"{prefix}_feedback_reordered: {run['feedback_reordered']}",
        f"{prefix}_sequence_gap_events: {run['sequence_gaps']}",
        f"{prefix}_missing_ids_crossed_by_gaps: {run['gap_missing']}",
        f"{prefix}_valid_adjacent_samples: {run['valid']}",
        f"{prefix}_controlled_valid_adjacent_samples: {run['controlled_valid']}",
        f"{prefix}_valid_adjacent_percent_of_sent: {run['valid_percent_sent']:.9f}",
        f"{prefix}_valid_adjacent_percent_of_received: {run['valid_percent_received']:.9f}",
        f"{prefix}_gap_recovered_count: {run['gap_recovered_count']}",
        f"{prefix}_gap_unrecovered_count: {run['gap_unrecovered_count']}",
        f"{prefix}_gap_recovery_median_s: {fmt(run['gap_recovery_median_s'])}",
        f"{prefix}_gap_recovery_max_s: {fmt(run['gap_recovery_max_s'])}",
        f"{prefix}_longest_raw_no_update_s: {run['longest_no_update']:.9f}",
        f"{prefix}_longest_incorrectly_fresh_s: {fmt(run['incorrectly_fresh_duration'])}",
        f"{prefix}_stale_state_duration_s: {fmt(run['stale_duration'])}",
        f"{prefix}_published_fresh_duration_percent: "
        + (
            "not available"
            if run["stale_duration"] is None
            else f"{100.0 * (run['duration'] - run['stale_duration']) / run['duration']:.9f}"
        ),
        f"{prefix}_longest_stale_interval_s: {fmt(run['longest_stale'])}",
        f"{prefix}_raw_min_mbps: {fmt(min(raw) if raw else None)}",
        f"{prefix}_raw_median_mbps: {fmt(percentile(raw, 50))}",
        f"{prefix}_raw_p99_mbps: {fmt(percentile(raw, 99))}",
        f"{prefix}_raw_max_mbps: {fmt(max(raw) if raw else None)}",
        f"{prefix}_filtered_min_mbps: {fmt(min(filtered) if filtered else None)}",
        f"{prefix}_filtered_median_mbps: {fmt(percentile(filtered, 50))}",
        f"{prefix}_filtered_p99_mbps: {fmt(percentile(filtered, 99))}",
        f"{prefix}_filtered_max_mbps: {fmt(max(filtered) if filtered else None)}",
        f"{prefix}_internal_recovery_filtered_max_mbps: "
        f"{fmt(max(internal_filtered) if internal_filtered else None)}",
        f"{prefix}_delivered_overall_mbps: {run['delivered_overall_mbps']:.9f}",
        f"{prefix}_delivered_unweighted_1s_bin_mean_mbps: {np.mean(run['delivered']):.9f}",
        f"{prefix}_delivered_median_1s_mbps: {np.median(run['delivered']):.9f}",
        f"{prefix}_fresh_1s_bins: {run['fresh_bin_count']}",
        f"{prefix}_stale_1s_bins: {run['stale_bin_count']}",
        f"{prefix}_fresh_bin_mae_mbps: {fmt(run['fresh_mae'])}",
        f"{prefix}_fresh_bin_rmse_mbps: {fmt(run['fresh_rmse'])}",
        f"{prefix}_raw_held_all_available_bin_mae_mbps: {fmt(run['held_mae'])}",
        f"{prefix}_raw_held_all_available_bin_rmse_mbps: {fmt(run['held_rmse'])}",
        f"{prefix}_backpressure_pause_rows: {run['pause_rows']}",
        f"{prefix}_max_backpressure_pause_duration_s: {run['max_pause_duration']:.9f}",
        f"{prefix}_sender_grace_rows: {run['grace_rows']}",
        f"{prefix}_schedule_reset_rows: {run['schedule_reset_rows']}",
        f"{prefix}_schedule_missed_deadlines: {summary.get('missed_deadlines', 'not available')}",
        f"{prefix}_max_buffered_amount: {summary.get('max_buffered', 'not available')}",
        f"{prefix}_max_outstanding_packets: {run['max_outstanding']}",
        f"{prefix}_receiver_source_fresh_rows: {run['source_fresh_rows']}",
        f"{prefix}_sender_published_feedback_rows: {run['published_feedback_rows']}",
    ]
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()

    before = load_run(args.before, has_publication=False)
    after = load_run(args.after, has_publication=True)
    plot_results(before, after, args.after)

    report = [
        "wifi/trace14 heavy-loss estimator before-versus-after report",
        "",
        "Method: identical absolute monotonic trace window; one-second delivered",
        "throughput is confirmed receiver application bytes * 8 / 1e6. Raw,",
        "filtered, and published estimates are never rescaled or calibrated.",
        "Fresh-bin errors use only explicitly published bins; every stale bin is",
        "reported separately, and raw causal-hold all-bin errors are retained.",
        "",
        *describe("before", before),
        "",
        *describe("after", after),
        "",
        "integrity_no_trace_values_enter_estimator: true",
        "integrity_no_ground_truth_calibration_or_rescaling: true",
        "integrity_filter_is_causal: true",
        "integrity_old_run_modified: false",
        "integrity_controlled_interval_shared_by_plots_and_metrics: true",
    ]
    (args.after / "before_after_metrics.txt").write_text(
        "\n".join(report) + "\n"
    )
    print("\n".join(report))


if __name__ == "__main__":
    main()
