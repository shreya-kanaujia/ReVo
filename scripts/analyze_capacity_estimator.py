#!/usr/bin/env python3
"""Analyze Week 1 capacity-estimator validation on one monotonic clock."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COMPLETENESS_TOLERANCE_S = 0.5


def median(values):
    values = sorted(v for v in values if v is not None and math.isfinite(v))
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2.0


def read_ground_truth(path):
    with open(path, newline="") as f:
        rows = [
            {
                "timestamp": float(row["monotonic_timestamp"]),
                "bandwidth_mbps": float(row["bandwidth_mbps"]),
            }
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"empty ground-truth CSV: {path}")
    rows.sort(key=lambda row: row["timestamp"])
    return rows


def read_estimator(path, discard_startup_s):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty estimator CSV: {path}")

    timestamps = [float(row["timestamp"]) for row in rows]
    first_ts = timestamps[0]
    valid = []
    discarded = 0
    nonpositive_corrected = 0
    duplicate_acks = 0
    reordered_acks = 0
    seen_seq = set()
    sequence_ids = []
    previous_seq = None

    for row, timestamp in zip(rows, timestamps):
        seq_key = (
            "acknowledged_sequence_number"
            if "acknowledged_sequence_number" in row
            else "latest_sequence_id"
        )
        seq = int(row[seq_key])
        sequence_ids.append(seq)
        if seq in seen_seq:
            duplicate_acks += 1
        if previous_seq is not None and seq < previous_seq:
            reordered_acks += 1
        seen_seq.add(seq)
        previous_seq = seq

        corrected_raw = row.get("corrected_interarrival_time", "")
        if corrected_raw and float(corrected_raw) <= 0:
            nonpositive_corrected += 1

        est_raw = row.get("estimated_capacity_mbps", "")
        ewma_raw = row.get("smoothed_tau", row.get("ewma_interarrival_time", ""))
        if timestamp - first_ts < discard_startup_s or not est_raw or not ewma_raw:
            discarded += 1
            continue
        estimate = float(est_raw)
        if not math.isfinite(estimate) or estimate <= 0:
            discarded += 1
            continue
        corrected = float(corrected_raw) if corrected_raw else None
        packet_size = float(row["packet_size_bytes"]) if row.get("packet_size_bytes") else None
        valid.append({
            "timestamp": timestamp,
            "estimated_mbps": estimate,
            "raw_estimated_mbps": (
                packet_size / corrected * 8.0 / 1_000_000.0
                if packet_size is not None and corrected is not None and corrected > 0
                else None
            ),
            "max_frame_size_bytes": (
                float(row["max_frame_size_bytes"]) if row.get("max_frame_size_bytes") else None
            ),
            "packets_outstanding": (
                float(row["packets_outstanding"]) if row.get("packets_outstanding") else None
            ),
            "rgb_bufferedAmount": (
                float(row["rgb_bufferedAmount"]) if row.get("rgb_bufferedAmount") else None
            ),
            "depth_bufferedAmount": (
                float(row["depth_bufferedAmount"]) if row.get("depth_bufferedAmount") else None
            ),
            "seq": seq,
        })

    return valid, {
        "rows": len(rows),
        "valid_samples": len(valid),
        "discarded_samples": discarded,
        "nonpositive_corrected_samples": nonpositive_corrected,
        "duplicate_ack_sequence_ids": duplicate_acks,
        "reordered_ack_sequence_ids": reordered_acks,
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "duration_s": max(timestamps) - min(timestamps),
        "first_sequence_id": min(sequence_ids),
        "last_sequence_id": max(sequence_ids),
        "unique_sequence_ids": len(set(sequence_ids)),
        "missing_sequence_ids": (
            max(sequence_ids) - min(sequence_ids) + 1 - len(set(sequence_ids))
        ),
    }


def read_probe_sender(path):
    if not path:
        return [], None
    with open(path, newline="") as f:
        raw_rows = list(csv.DictReader(f))
    if not raw_rows:
        raise ValueError(f"empty probe-sender CSV: {path}")

    rows = []
    sequence_ids = []
    reordered_sequence_ids = 0
    previous_seq = None
    for row in raw_rows:
        timestamp = float(row["timestamp_monotonic"])
        seq_raw = row.get("probe_sequence_id", "")
        seq = int(seq_raw) if seq_raw else None
        if seq is not None:
            sequence_ids.append(seq)
            if previous_seq is not None and seq < previous_seq:
                reordered_sequence_ids += 1
            previous_seq = seq
        rows.append({
            "timestamp": timestamp,
            "seq": seq,
            "payload_bytes": int(row["payload_bytes"]),
            "target_offered_bitrate_mbps": float(row["target_offered_bitrate_mbps"]),
            "actual_send_interval_seconds": (
                float(row["actual_send_interval_seconds"])
                if row.get("actual_send_interval_seconds") else None
            ),
            "datachannel_bufferedAmount": float(row["datachannel_bufferedAmount"]),
            "sent": row.get("sent") == "1",
            "paused_due_to_backpressure": row.get("paused_due_to_backpressure") == "1",
            "cumulative_probe_bytes_sent": int(row["cumulative_probe_bytes_sent"]),
        })
    timestamps = [row["timestamp"] for row in rows]
    sent_rows = [row for row in rows if row["sent"]]
    return rows, {
        "rows": len(rows),
        "sent_packets": len(sent_rows),
        "pause_rows": sum(row["paused_due_to_backpressure"] for row in rows),
        "first_timestamp": min(timestamps),
        "last_timestamp": max(timestamps),
        "duration_s": max(timestamps) - min(timestamps),
        "first_sent_timestamp": sent_rows[0]["timestamp"] if sent_rows else None,
        "last_sent_timestamp": sent_rows[-1]["timestamp"] if sent_rows else None,
        "sent_duration_s": (
            sent_rows[-1]["timestamp"] - sent_rows[0]["timestamp"] if sent_rows else 0.0
        ),
        "first_sequence_id": min(sequence_ids) if sequence_ids else None,
        "last_sequence_id": max(sequence_ids) if sequence_ids else None,
        "unique_sequence_ids": len(set(sequence_ids)),
        "reordered_sequence_ids": reordered_sequence_ids,
    }


def gt_at(gt_rows, timestamp):
    """Zero-order hold on absolute monotonic timestamps."""
    current = gt_rows[0]["bandwidth_mbps"]
    for row in gt_rows:
        if row["timestamp"] <= timestamp:
            current = row["bandwidth_mbps"]
        else:
            break
    return current


def comparison_bounds(gt_rows, samples, probe, expected_duration_s=None):
    """Return the real overlap with the configured, finite tc schedule."""
    schedule_start = gt_rows[0]["timestamp"]
    schedule_end = gt_rows[-1]["timestamp"]
    if expected_duration_s is not None:
        schedule_end = min(schedule_end, schedule_start + expected_duration_s)
    comparison_start = max(schedule_start, samples[0]["timestamp"])
    end_candidates = [schedule_end, samples[-1]["timestamp"]]
    if probe is not None:
        end_candidates.append(probe["last_sent_timestamp"])
    comparison_end = min(end_candidates)
    if comparison_end <= comparison_start:
        raise ValueError("no valid estimator / ground-truth / probe overlap")
    return comparison_start, comparison_end, schedule_end


def crop_timestamped(rows, start, end):
    return [row for row in rows if start <= row["timestamp"] <= end]


def comparison_exclusion_counts(rows, start, end):
    """Count valid estimator samples excluded on either side of the overlap."""
    return {
        "before_comparison_start": sum(row["timestamp"] < start for row in rows),
        "after_comparison_end": sum(row["timestamp"] > end for row in rows),
    }


def ground_truth_plot_series(gt_rows, start, end):
    """Finite zero-order-hold series bounded to the comparison window."""
    timestamps = [start]
    values = [gt_at(gt_rows, start)]
    for row in gt_rows:
        if start < row["timestamp"] < end:
            timestamps.append(row["timestamp"])
            values.append(row["bandwidth_mbps"])
    timestamps.append(end)
    values.append(gt_at(gt_rows, end))
    return timestamps, values


def ground_truth_transitions(gt_rows):
    transitions = []
    previous = gt_rows[0]["bandwidth_mbps"]
    for row in gt_rows[1:]:
        if row["bandwidth_mbps"] != previous:
            transitions.append({
                "timestamp": row["timestamp"],
                "old_capacity": previous,
                "new_capacity": row["bandwidth_mbps"],
            })
            previous = row["bandwidth_mbps"]
    return transitions


def schedule_intervals(gt_rows, transitions, schedule_end):
    boundaries = [gt_rows[0]["timestamp"]]
    boundaries.extend(item["timestamp"] for item in transitions)
    boundaries.append(schedule_end)
    return [
        {
            "index": index + 1,
            "start": start,
            "end": end,
            "ground_truth_mbps": gt_at(gt_rows, start),
        }
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        if end > start
    ]


def coverage_status(start, end, observation_start, observation_end, gt_evidence_end):
    if observation_start is None or observation_end is None or observation_end < start or observation_start >= end:
        return "not_observed", 0.0
    covered_start = max(start, observation_start)
    covered_end = min(end, observation_end)
    coverage = max(0.0, covered_end - covered_start) / (end - start)
    missing_start = observation_start > start + COMPLETENESS_TOLERANCE_S
    missing_end = observation_end < end - COMPLETENESS_TOLERANCE_S or gt_evidence_end < end - COMPLETENESS_TOLERANCE_S
    if missing_start and missing_end:
        return "partial_start_and_end", coverage
    if missing_start:
        return "partial_start", coverage
    if missing_end:
        return "partial_end", coverage
    return "complete", coverage


def error_metrics(pairs):
    errors = [estimate - truth for _, estimate, truth in pairs]
    if not errors:
        return None
    absolute = [abs(value) for value in errors]
    return {
        "mae_mbps": sum(absolute) / len(absolute),
        "median_abs_error_mbps": median(absolute),
        "rmse_mbps": math.sqrt(sum(value * value for value in errors) / len(errors)),
    }


def transition_lag(samples, change_timestamp, new_capacity, tolerance_fraction, stable_s):
    lower = new_capacity * (1.0 - tolerance_fraction)
    upper = new_capacity * (1.0 + tolerance_fraction)
    for sample in samples:
        start = sample["timestamp"]
        if start < change_timestamp:
            continue
        end = start + stable_s
        window = [item for item in samples if start <= item["timestamp"] <= end]
        if window and window[-1]["timestamp"] >= end - 1e-3 and all(
            lower <= item["estimated_mbps"] <= upper for item in window
        ):
            return start - change_timestamp
    return None


def cross_correlation_lag(samples, gt_rows, max_lag_s, step_s):
    if len(samples) < 3:
        return None, None
    first = samples[0]["timestamp"]
    last = samples[-1]["timestamp"]
    best_lag = None
    best_score = None
    lag = -max_lag_s
    while lag <= max_lag_s + 1e-9:
        paired = []
        for sample in samples:
            shifted = sample["timestamp"] - lag
            if first <= shifted <= last:
                paired.append((sample["estimated_mbps"], gt_at(gt_rows, shifted)))
        if len(paired) >= 3:
            est_mean = sum(a for a, _ in paired) / len(paired)
            gt_mean = sum(b for _, b in paired) / len(paired)
            numerator = sum((a - est_mean) * (b - gt_mean) for a, b in paired)
            est_denominator = math.sqrt(sum((a - est_mean) ** 2 for a, _ in paired))
            gt_denominator = math.sqrt(sum((b - gt_mean) ** 2 for _, b in paired))
            if est_denominator > 0 and gt_denominator > 0:
                score = numerator / (est_denominator * gt_denominator)
                if best_score is None or score > best_score:
                    best_score = score
                    best_lag = lag
        lag += step_s
    return best_lag, best_score


def log_failure_lines(paths):
    needles = ("datachannel", "webrtc", "connection", "consent", "ice state", "traceback", "error")
    matches = []
    for path in paths:
        if not path:
            continue
        log_path = Path(path)
        if not log_path.exists():
            matches.append(f"missing log: {log_path}")
            continue
        for line in log_path.read_text(errors="replace").splitlines():
            lowered = line.lower()
            if any(needle in lowered for needle in needles) and (
                "closed" in lowered or "failed" in lowered or "expired" in lowered
                or "error" in lowered or "traceback" in lowered
            ):
                matches.append(f"{log_path.name}: {line.strip()}")
    return matches[-10:]


def interval_report_lines(samples, probe_rows, intervals, gt_evidence_end):
    lines = ["", "Per-interval report (absolute aligned boundaries):"]
    estimator_start = samples[0]["timestamp"] if samples else None
    estimator_end = samples[-1]["timestamp"] if samples else None
    sent_probe = [row for row in probe_rows if row["sent"]]
    probe_start = sent_probe[0]["timestamp"] if sent_probe else None
    probe_end = sent_probe[-1]["timestamp"] if sent_probe else None
    for interval in intervals:
        start = interval["start"]
        end = interval["end"]
        label = f"interval_{interval['index']}"
        interval_samples = [sample for sample in samples if start <= sample["timestamp"] < end]
        interval_probe = [row for row in probe_rows if start <= row["timestamp"] < end]
        sent = [row for row in interval_probe if row["sent"]]
        status, coverage = coverage_status(start, end, estimator_start, estimator_end, gt_evidence_end)
        probe_status, probe_coverage = coverage_status(start, end, probe_start, probe_end, gt_evidence_end)
        actual_offered = None
        if len(sent) >= 2:
            span = sent[-1]["timestamp"] - sent[0]["timestamp"]
            byte_delta = sent[-1]["cumulative_probe_bytes_sent"] - sent[0]["cumulative_probe_bytes_sent"]
            actual_offered = byte_delta * 8.0 / span / 1_000_000.0 if span > 0 else None
        smoothed_median = median([sample["estimated_mbps"] for sample in interval_samples])
        raw_median = median([sample["raw_estimated_mbps"] for sample in interval_samples])
        max_frame_median = median([sample["max_frame_size_bytes"] for sample in interval_samples])
        lines.extend([
            f"{label}_start_monotonic: {start:.9f}",
            f"{label}_end_monotonic: {end:.9f}",
            f"{label}_ground_truth_mbps: {interval['ground_truth_mbps']:.6f}",
            f"{label}_estimator_coverage_status: {status}",
            f"{label}_estimator_coverage_fraction: {coverage:.6f}",
            f"{label}_probe_coverage_status: {probe_status}",
            f"{label}_probe_coverage_fraction: {probe_coverage:.6f}",
            f"{label}_valid_estimator_samples: {len(interval_samples)}",
            f"{label}_median_raw_estimated_service_rate_mbps: {'' if raw_median is None else f'{raw_median:.6f}'}",
            f"{label}_median_smoothed_estimated_service_rate_mbps: {'' if smoothed_median is None else f'{smoothed_median:.6f}'}",
            f"{label}_median_max_frame_size_bytes: {'' if max_frame_median is None else f'{max_frame_median:.6f}'}",
            f"{label}_actual_offered_mbps: {'' if actual_offered is None else f'{actual_offered:.6f}'}",
            f"{label}_probe_rows: {len(interval_probe)}",
            f"{label}_probe_sent_packets: {len(sent)}",
            f"{label}_probe_backpressure_pauses: {sum(row['paused_due_to_backpressure'] for row in interval_probe)}",
        ])
    return lines


def main():
    parser = argparse.ArgumentParser(description="Analyze ReVo Week 1 capacity estimator validation")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--estimator", required=True)
    parser.add_argument("--plot", required=True)
    parser.add_argument("--max-frame-plot", default=None)
    parser.add_argument("--report", required=True)
    parser.add_argument("--probe-sender", default=None)
    parser.add_argument("--sender-log", default=None)
    parser.add_argument("--receiver-log", default=None)
    parser.add_argument("--expect-duration-s", type=float, default=None)
    parser.add_argument(
        "--discard-startup-s",
        type=float,
        default=2.0,
        help="Fixed initial EWMA warm-up excluded consistently from plots and metrics",
    )
    parser.add_argument("--lag-tolerance-fraction", type=float, default=0.25)
    parser.add_argument("--lag-stable-s", type=float, default=2.0)
    args = parser.parse_args()

    gt_rows = read_ground_truth(args.ground_truth)
    samples, estimator = read_estimator(args.estimator, args.discard_startup_s)
    probe_rows, probe = read_probe_sender(args.probe_sender)
    if not samples:
        raise SystemExit("no valid estimator samples after filtering")

    transitions = ground_truth_transitions(gt_rows)
    gt_start = gt_rows[0]["timestamp"]
    gt_end = gt_rows[-1]["timestamp"]
    expected_end = gt_start + args.expect_duration_s if args.expect_duration_s is not None else gt_end
    comparison_start, comparison_end, schedule_end = comparison_bounds(
        gt_rows, samples, probe, args.expect_duration_s
    )
    comparison_samples = crop_timestamped(samples, comparison_start, comparison_end)
    comparison_probe_rows = crop_timestamped(probe_rows, comparison_start, comparison_end)
    excluded = comparison_exclusion_counts(samples, comparison_start, comparison_end)
    intervals = schedule_intervals(gt_rows, transitions, schedule_end)
    pairs = [
        (sample["timestamp"], sample["estimated_mbps"], gt_at(gt_rows, sample["timestamp"]))
        for sample in comparison_samples
    ]
    metrics = error_metrics(pairs)
    if metrics is None:
        raise SystemExit("no estimator samples overlap the ground-truth schedule")

    transition_results = [
        (
            item,
            transition_lag(
                comparison_samples, item["timestamp"], item["new_capacity"],
                args.lag_tolerance_fraction, args.lag_stable_s,
            ),
        )
        for item in transitions
    ]
    xcorr_lag, xcorr_score = cross_correlation_lag(
        comparison_samples, gt_rows, 10.0, 0.25
    )

    gt_complete = (
        args.expect_duration_s is None
        or gt_end >= expected_end - COMPLETENESS_TOLERANCE_S
    )
    estimator_complete = (
        args.expect_duration_s is None
        or estimator["duration_s"] >= args.expect_duration_s - COMPLETENESS_TOLERANCE_S
    )
    probe_complete = (
        probe is None
        or args.expect_duration_s is None
        or probe["sent_duration_s"] >= args.expect_duration_s - COMPLETENESS_TOLERANCE_S
    )
    timestamp_match = True
    sequence_match = True
    if probe is not None:
        timestamp_match = (
            abs(estimator["first_timestamp"] - probe["first_sent_timestamp"]) <= 1.0
            and estimator["last_timestamp"] <= probe["last_timestamp"] + 5.0
            and gt_start <= min(estimator["first_timestamp"], probe["first_sent_timestamp"])
        )
        sequence_match = (
            estimator["first_sequence_id"] == probe["first_sequence_id"] == 1
            and estimator["last_sequence_id"] <= probe["last_sequence_id"]
            and estimator["rows"] == estimator["unique_sequence_ids"]
            and estimator["reordered_ack_sequence_ids"] == 0
            and probe["unique_sequence_ids"]
            == probe["last_sequence_id"] - probe["first_sequence_id"] + 1
            and probe["sent_packets"] == probe["unique_sequence_ids"]
            and probe["reordered_sequence_ids"] == 0
        )
    run_complete = gt_complete and estimator_complete and probe_complete and timestamp_match and sequence_match
    failure_reasons = []
    if not gt_complete:
        failure_reasons.append("ground truth did not reach the full expected schedule duration")
    if not estimator_complete:
        failure_reasons.append("estimator ACK stream ended before the expected duration")
    if not probe_complete:
        failure_reasons.append("probe sender ended before the expected duration")
    if not timestamp_match:
        failure_reasons.append("CSV timestamp ranges do not identify one overlapping run")
    if not sequence_match:
        failure_reasons.append("probe and estimator sequence-ID ranges are inconsistent")

    plot_path = Path(args.plot)
    report_path = Path(args.report)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    gt_x, gt_y = ground_truth_plot_series(gt_rows, comparison_start, comparison_end)
    elapsed_gt_x = [timestamp - comparison_start for timestamp in gt_x]
    elapsed_sample_x = [sample["timestamp"] - comparison_start for sample in comparison_samples]

    plt.figure(figsize=(11, 5))
    plt.step(elapsed_gt_x, gt_y, where="post", label="Injected tc bandwidth (Mbps)", linewidth=2)
    plt.plot(elapsed_sample_x, [s["estimated_mbps"] for s in comparison_samples],
             label="Delivered DataChannel application-throughput estimate (Mbps)",
             alpha=0.75)
    plt.xlabel("Elapsed validation time (s)")
    plt.ylabel("Mbps")
    plt.title("ReVo Week 1 Real tc Application-Throughput Validation")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=160)
    plt.close()

    if args.max_frame_plot:
        frame_samples = [
            sample for sample in comparison_samples
            if sample["max_frame_size_bytes"] is not None
        ]
        if frame_samples:
            frame_path = Path(args.max_frame_plot)
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(11, 4))
            plt.plot([s["timestamp"] - comparison_start for s in frame_samples],
                     [s["max_frame_size_bytes"] for s in frame_samples],
                     label="Salsify max_frame_size_bytes")
            plt.xlabel("Elapsed validation time (s)")
            plt.ylabel("Bytes")
            plt.title("Passive Salsify Max Frame Size")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(frame_path, dpi=160)
            plt.close()

    lines = [
        "validation_status: " + ("COMPLETE" if run_complete else "INCOMPLETE"),
        "validation_claim: " + ("eligible_for_rate_accuracy_review" if run_complete else "FAIL_partial_run_no_success_claim"),
        "Ground truth interpolation: zero-order hold on absolute monotonic timestamps.",
        "Intervals and transitions: derived from changes in the ground-truth CSV; no fixed 0-20/20-40/40-60 alignment.",
        "Plot, error metrics, cross-correlation, transition lags, and interval summaries use the same cropped estimator samples.",
        "Lag definition: time from the absolute ground-truth change timestamp until the estimate enters",
        f"and remains inside +/-{args.lag_tolerance_fraction * 100:.1f}% of the new capacity for {args.lag_stable_s:.1f}s.",
        "",
        f"expected_duration_s: {'' if args.expect_duration_s is None else f'{args.expect_duration_s:.6f}'}",
        f"ground_truth_rows: {len(gt_rows)}",
        f"ground_truth_first_timestamp: {gt_start:.9f}",
        f"ground_truth_last_timestamp: {gt_end:.9f}",
        f"ground_truth_duration_s: {gt_end - gt_start:.6f}",
        f"configured_schedule_end_timestamp: {schedule_end:.9f}",
        f"comparison_start_timestamp: {comparison_start:.9f}",
        f"comparison_end_timestamp: {comparison_end:.9f}",
        f"comparison_duration_s: {comparison_end - comparison_start:.6f}",
        f"ground_truth_completed_expected_duration: {gt_complete}",
        f"estimator_rows: {estimator['rows']}",
        f"estimator_valid_samples: {estimator['valid_samples']}",
        f"comparison_estimator_samples: {len(comparison_samples)}",
        f"valid_estimator_samples_before_comparison_start: {excluded['before_comparison_start']}",
        f"valid_estimator_samples_after_comparison_end: {excluded['after_comparison_end']}",
        f"estimator_discarded_samples: {estimator['discarded_samples']}",
        f"estimator_startup_discard_s: {args.discard_startup_s:.6f}",
        f"estimator_first_timestamp: {estimator['first_timestamp']:.9f}",
        f"estimator_last_timestamp: {estimator['last_timestamp']:.9f}",
        f"estimator_duration_s: {estimator['duration_s']:.6f}",
        f"estimator_first_sequence_id: {estimator['first_sequence_id']}",
        f"estimator_last_sequence_id: {estimator['last_sequence_id']}",
        f"estimator_unique_sequence_ids: {estimator['unique_sequence_ids']}",
        f"estimator_missing_sequence_ids: {estimator['missing_sequence_ids']}",
        f"estimator_duplicate_ack_sequence_ids: {estimator['duplicate_ack_sequence_ids']}",
        f"estimator_reordered_ack_sequence_ids: {estimator['reordered_ack_sequence_ids']}",
        f"estimator_nonpositive_corrected_samples: {estimator['nonpositive_corrected_samples']}",
        f"estimator_completed_expected_duration: {estimator_complete}",
        f"csv_timestamp_ranges_match_one_run: {timestamp_match}",
        f"csv_sequence_ranges_match_one_run: {sequence_match}",
        f"mae_mbps: {metrics['mae_mbps']:.6f}",
        f"median_abs_error_mbps: {metrics['median_abs_error_mbps']:.6f}",
        f"rmse_mbps: {metrics['rmse_mbps']:.6f}",
        f"cross_correlation_lag_s: {'' if xcorr_lag is None else f'{xcorr_lag:.6f}'}",
        f"cross_correlation_score: {'' if xcorr_score is None else f'{xcorr_score:.6f}'}",
    ]
    if probe is not None:
        lines.extend([
            f"probe_sender_rows: {probe['rows']}",
            f"probe_sender_sent_packets: {probe['sent_packets']}",
            f"probe_sender_pause_rows: {probe['pause_rows']}",
            f"probe_sender_first_timestamp: {probe['first_timestamp']:.9f}",
            f"probe_sender_last_timestamp: {probe['last_timestamp']:.9f}",
            f"probe_sender_duration_s: {probe['duration_s']:.6f}",
            f"probe_sender_sent_duration_s: {probe['sent_duration_s']:.6f}",
            f"probe_sender_first_sequence_id: {probe['first_sequence_id']}",
            f"probe_sender_last_sequence_id: {probe['last_sequence_id']}",
            f"probe_sender_unique_sequence_ids: {probe['unique_sequence_ids']}",
            f"probe_sender_reordered_sequence_ids: {probe['reordered_sequence_ids']}",
            f"probe_completed_expected_duration: {probe_complete}",
            f"probe_to_estimator_start_delta_s: {estimator['first_timestamp'] - probe['first_sent_timestamp']:.6f}",
        ])
    lines.append("")
    lines.append("Ground-truth transitions:")
    for index, (transition, lag) in enumerate(transition_results, 1):
        lines.extend([
            f"transition_{index}_timestamp_monotonic: {transition['timestamp']:.9f}",
            f"transition_{index}_old_capacity_mbps: {transition['old_capacity']:.6f}",
            f"transition_{index}_new_capacity_mbps: {transition['new_capacity']:.6f}",
            f"transition_{index}_lag_s: {'not_reached' if lag is None else f'{lag:.6f}'}",
        ])
    lines.extend(interval_report_lines(
        comparison_samples, comparison_probe_rows, intervals, schedule_end
    ))
    if failure_reasons:
        lines.extend(["", "Incomplete-run failures:"])
        lines.extend(f"- {reason}" for reason in failure_reasons)
        errors = log_failure_lines([args.sender_log, args.receiver_log])
        lines.append("DataChannel/WebRTC errors:")
        lines.extend(f"- {error}" for error in errors)
        if not errors:
            lines.append("- no matching error line supplied/found in sender or receiver logs")

    report_path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not run_complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
