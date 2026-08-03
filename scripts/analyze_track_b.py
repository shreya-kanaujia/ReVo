#!/usr/bin/env python3
"""Reproducible per-run and cross-mode Track B analysis."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from validate_track_b_run import inspect_run


MODES = ("fixed_high", "fixed_low", "buffer_health", "estimator_only", "combined")


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row, name, default=None):
    value = row.get(name, "")
    if value in ("", None):
        return default
    return float(value)


def boolean(row, name):
    return str(row.get(name, "")).lower() in ("1", "true")


def capacity_availability_counts(receiver_rows, sender_rows, controller_rows):
    """Keep publication, accepted feedback, and controller sampling distinct."""
    def is_new_receiver_publication(row):
        if not boolean(row, "estimate_fresh"):
            return False
        publication_ts = number(
            row, "last_published_estimate_timestamp", None
        )
        receiver_ts = number(row, "receiver_timestamp", None)
        if publication_ts is not None and receiver_ts is not None:
            return math.isclose(
                publication_ts, receiver_ts, rel_tol=0.0, abs_tol=1e-8
            )
        # Compatibility for older CSVs that predate explicit publication time.
        return row.get("filter_reason") in (
            "sample_accepted",
            "rolling_time_median",
            "recovery_seeded",
        )

    controller_fresh = sum(
        boolean(row, "capacity_fresh") for row in controller_rows
    )
    return {
        "receiver_fresh_publications": sum(
            is_new_receiver_publication(row) for row in receiver_rows
        ),
        "receiver_fresh_feedback_rows": sum(
            boolean(row, "estimate_fresh") for row in receiver_rows
        ),
        "receiver_capacity_feedback_sent": sum(
            boolean(row, "feedback_sent") for row in receiver_rows
        ),
        "sender_capacity_feedback_received": len(sender_rows),
        "sender_accepted_fresh_feedback": sum(
            row.get("capacity_state_update_status") == "accepted_fresh"
            for row in sender_rows
        ),
        "controller_fresh_capacity_decisions": controller_fresh,
        "controller_capacity_decisions": len(controller_rows),
        "controller_fresh_capacity_fraction": (
            0.0
            if not controller_rows
            else float(controller_fresh / len(controller_rows))
        ),
    }


def receiver_controlled_window(sender_feedback_rows, sender_start, sender_end):
    """Map a sender-controlled interval into the receiver monotonic domain.

    Capacity feedback retains the receiver arrival timestamp and the sender's
    local receipt timestamp in the same row.  Selecting those pairs by sender
    receipt time yields a trace-independent receiver interval without assuming
    that separate hosts share CLOCK_MONOTONIC.
    """
    receiver_timestamps = []
    for row in sender_feedback_rows:
        sender_timestamp = number(row, "timestamp", None)
        receiver_timestamp = number(row, "receiver_timestamp", None)
        if (
            sender_timestamp is not None
            and receiver_timestamp is not None
            and sender_start <= sender_timestamp <= sender_end
        ):
            receiver_timestamps.append(receiver_timestamp)
    if not receiver_timestamps:
        return float(sender_start), float(sender_end), "shared_clock_fallback"
    receiver_start = min(receiver_timestamps)
    receiver_end = max(receiver_timestamps)
    if receiver_end <= receiver_start:
        return float(sender_start), float(sender_end), "shared_clock_fallback"
    return receiver_start, receiver_end, "paired_capacity_feedback"


def percentile(values, q):
    return None if not values else float(np.percentile(values, q))


def frame_ssim(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    result = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / (
        (mu_a * mu_a + mu_b * mu_b + c1)
        * (sigma_a2 + sigma_b2 + c2)
    )
    return float(np.mean(result))


def compare_video(gt_path, reconstructed_path, frame_limit):
    gt = cv2.VideoCapture(str(gt_path))
    reconstructed = cv2.VideoCapture(str(reconstructed_path))
    if not gt.isOpened() or not reconstructed.isOpened():
        raise RuntimeError(f"cannot open SSIM inputs: {gt_path}, {reconstructed_path}")
    values = []
    last_reconstructed = None
    missing_reconstructed = 0
    try:
        while len(values) < frame_limit:
            ok_gt, gt_frame = gt.read()
            ok_reconstructed, reconstructed_frame = reconstructed.read()
            if not ok_gt:
                break
            if not ok_reconstructed:
                missing_reconstructed += 1
                if last_reconstructed is None:
                    reconstructed_frame = np.zeros_like(gt_frame)
                else:
                    reconstructed_frame = last_reconstructed
            else:
                last_reconstructed = reconstructed_frame
            if gt_frame.shape != reconstructed_frame.shape:
                reconstructed_frame = cv2.resize(
                    reconstructed_frame, (gt_frame.shape[1], gt_frame.shape[0])
                )
            values.append(
                frame_ssim(
                    cv2.cvtColor(gt_frame, cv2.COLOR_BGR2GRAY),
                    cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2GRAY),
                )
            )
    finally:
        gt.release()
        reconstructed.release()
    return values, missing_reconstructed


def longest_true_duration(rows, predicate):
    longest = current = 0.0
    for before, after in zip(rows, rows[1:]):
        dt = max(
            0.0,
            number(after, "timestamp_monotonic", 0)
            - number(before, "timestamp_monotonic", 0),
        )
        if predicate(before):
            current += dt
            longest = max(longest, current)
        else:
            current = 0.0
    return longest


def one_second_rate(rows, timestamp_name, bytes_name, start, end):
    bins = max(1, int(math.ceil(end - start)))
    byte_counts = np.zeros(bins)
    for row in rows:
        timestamp = number(row, timestamp_name)
        byte_count = number(row, bytes_name)
        if timestamp is None or byte_count is None or not start <= timestamp <= end:
            continue
        index = min(bins - 1, int(timestamp - start))
        byte_counts[index] += byte_count
    return np.arange(bins) + 0.5, byte_counts * 8 / 1_000_000


def estimator_reference_metrics(rows, start, end):
    """Compare one-second published medians with matching active service."""
    count = max(1, int(math.ceil(end - start)))
    active_bytes = np.zeros(count)
    active_time = np.zeros(count)
    delivered_bytes = np.zeros(count)
    published = [[] for _ in range(count)]
    for row in rows:
        timestamp = number(row, "receiver_timestamp")
        if timestamp is None or not start <= timestamp <= end:
            continue
        index = min(count - 1, int(timestamp - start))
        packet_size = number(row, "packet_size_bytes", 0)
        delivered_bytes[index] += packet_size
        corrected = number(row, "corrected_interarrival_time")
        if corrected is not None:
            active_bytes[index] += packet_size
            active_time[index] += corrected
        value = number(row, "published_estimated_capacity_mbps")
        if value is not None:
            published[index].append(value)
    widths = np.ones(count)
    final_width = end - start - math.floor(end - start)
    if final_width > 1e-9:
        widths[-1] = final_width
    goodput = delivered_bytes * 8 / 1_000_000 / widths
    active_service = np.divide(
        active_bytes * 8 / 1_000_000,
        active_time,
        out=np.full(count, np.nan),
        where=active_time > 0,
    )
    published_median = np.asarray(
        [median(values) if values else np.nan for values in published]
    )
    fresh = np.isfinite(published_median)
    matched = fresh & np.isfinite(active_service)

    def errors(reference, mask):
        if not np.any(mask):
            return None, None
        error = published_median[mask] - reference[mask]
        return float(np.mean(np.abs(error))), float(np.sqrt(np.mean(error ** 2)))

    matched_mae, matched_rmse = errors(active_service, matched)
    goodput_mae, goodput_rmse = errors(goodput, fresh)
    return {
        "estimator_fresh_one_second_bins": int(np.sum(fresh)),
        "estimator_matched_active_service_bins": int(np.sum(matched)),
        "estimator_active_service_mae_mbps": matched_mae,
        "estimator_active_service_rmse_mbps": matched_rmse,
        "estimator_goodput_mae_mbps": goodput_mae,
        "estimator_goodput_rmse_mbps": goodput_rmse,
    }


def parse_receiver_summary(path):
    if not path.exists():
        return {}
    match = re.search(
        r"Receiver Summary\] Total=(\d+) Decoded=(\d+) "
        r"LostFull=(\d+) LostPartial=(\d+).*?Loss=([0-9.]+)%",
        path.read_text(errors="replace"),
    )
    if not match:
        return {}
    return {
        "receiver_total_frames_legacy_summary": int(match.group(1)),
        "receiver_decoded_frames": int(match.group(2)),
        "receiver_full_misses_legacy_summary": int(match.group(3)),
        "receiver_partial_frames_legacy_summary": int(match.group(4)),
        "receiver_loss_percent_legacy_summary": float(match.group(5)),
    }


def process_resources(rows):
    points = [
        (
            number(row, "timestamp_monotonic"),
            number(row, "process_cpu_time_s"),
            number(row, "process_max_rss_bytes"),
        )
        for row in rows
    ]
    points = [point for point in points if point[0] is not None]
    cpu = []
    for before, after in zip(points, points[1:]):
        dt = after[0] - before[0]
        if dt > 0 and before[1] is not None and after[1] is not None:
            cpu.append(max(0.0, (after[1] - before[1]) / dt * 100))
    rss = [point[2] for point in points if point[2] is not None]
    return {
        "cpu_mean_percent": None if not cpu else float(np.mean(cpu)),
        "cpu_p95_percent": percentile(cpu, 95),
        "rss_max_bytes": None if not rss else int(max(rss)),
    }


def media_deadline_audit(controller, measurements, timing_rows):
    """Join sender and receiver causal timing evidence by frame ID."""
    high_keyframes = {
        int(number(row, "frame_id", -1))
        for row in controller
        if boolean(row, "keyframe") and row.get("applied_quality") == "high"
    }
    sender_by_frame = {}
    for row in measurements:
        if row.get("stream") != "rgb" or row.get("frame_type") != "I":
            continue
        sender_by_frame[int(number(row, "frame_id", -1))] = row
    receiver_by_frame = {}
    display_by_frame = {}
    for row in timing_rows:
        frame_id = int(number(row, "frame_id", -1))
        if row.get("event") == "assembly_decode":
            receiver_by_frame[frame_id] = row
        elif row.get("event") == "display":
            display_by_frame[frame_id] = row
    result = []
    for frame_id in sorted(high_keyframes):
        sender = sender_by_frame.get(frame_id, {})
        receiver = receiver_by_frame.get(frame_id, {})
        display = display_by_frame.get(frame_id, {})
        send_start = number(sender, "send_start_monotonic")
        send_end = number(sender, "send_end_monotonic")
        result.append({
            "frame_id": frame_id,
            "production_timestamp": number(
                sender, "production_timestamp_monotonic"
            ),
            "enqueue_start": send_start,
            "enqueue_end": send_end,
            "enqueue_interval_s": (
                None
                if send_start is None or send_end is None
                else send_end - send_start
            ),
            "causal_encode_lead_s": number(
                sender, "causal_encode_lead_s"
            ),
            "first_chunk_arrival": number(
                receiver, "first_chunk_arrival"
            ),
            "last_chunk_arrival": number(
                receiver, "last_chunk_arrival"
            ),
            "fec_ready_timestamp": number(
                receiver, "fec_ready_timestamp"
            ),
            "assembly_deadline": number(
                receiver, "assembly_deadline"
            ),
            "decode_start": number(receiver, "decode_start"),
            "decode_end": number(receiver, "decode_end"),
            "display_deadline": number(receiver, "display_deadline"),
            "assembly_ready": boolean(receiver, "assembly_ready"),
            "assembly_reason": receiver.get("assembly_reason", "missing_timing"),
            "frozen_display": boolean(display, "frozen_display"),
        })
    return result


def full_impairment_windows(controller):
    """Return unique receiver-health windows that reported 100% impairment."""
    windows = {}
    for row in controller:
        sequence = row.get("health_feedback_sequence", "")
        rate = number(row, "receiver_impairment_rate")
        if not sequence or rate is None or rate < 1.0:
            continue
        windows[sequence] = {
            "feedback_sequence": int(number(row, "health_feedback_sequence", 0)),
            "window_start_frame": int(
                number(row, "health_window_start_frame", -1)
            ),
            "window_end_frame": int(
                number(row, "health_window_end_frame", -1)
            ),
            "full_misses": int(number(row, "health_full_misses", 0)),
            "partial_frames": int(number(row, "health_partial_frames", 0)),
            "codec_decode_failures": int(
                number(row, "health_decode_failures", 0)
            ),
            "reference_unavailable": int(
                number(row, "health_reference_unavailable", 0)
            ),
            "frozen_frames": int(number(row, "health_frozen_frames", 0)),
        }
    return list(windows.values())


def analyze_run(
    run_dir, output_dir, gt_rgb=None, gt_depth=None, metrics_only=False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    controller = read_csv(run_dir / "controller_decisions.csv")
    mode = (
        controller[0].get("adaptation_mode", "combined")
        if controller
        else "combined"
    )
    validation = inspect_run(run_dir, mode)
    if not validation["valid"]:
        failure_path = output_dir / "analysis_failure.json"
        failure_path.write_text(json.dumps(validation, indent=2) + "\n")
        raise RuntimeError(
            f"incomplete Track B run; details written to {failure_path}"
        )
    receiver_capacity = read_csv(run_dir / "receiver_capacity.csv")
    sender_capacity = read_csv(run_dir / "sender_capacity_feedback.csv")
    probe_rows = read_csv(run_dir / "capacity_probe.csv")
    measurements = read_csv(run_dir / "sender_measurements.csv")
    sender_diagnostics = read_csv(run_dir / "sender_diagnostics.csv")
    receiver_diagnostics = read_csv(run_dir / "receiver_diagnostics.csv")
    receiver_media_timing = read_csv(run_dir / "receiver_media_timing.csv")
    applied_trace = read_csv(run_dir / "applied_trace.csv")

    timestamps = np.asarray(
        [float(row["timestamp_monotonic"]) for row in controller]
    )
    start, end = float(timestamps[0]), float(timestamps[-1])
    receiver_start, receiver_end, receiver_window_method = (
        receiver_controlled_window(sender_capacity, start, end)
    )
    elapsed = timestamps - start
    qualities = np.asarray([row["applied_quality"] == "high" for row in controller])
    requested = np.asarray(
        [row["requested_quality"] == "high" for row in controller]
    )
    capacity = np.asarray(
        [number(row, "published_capacity_mbps", np.nan) for row in controller]
    )
    capacity[[
        not boolean(row, "capacity_fresh") for row in controller
    ]] = np.nan
    demand_high = np.asarray(
        [number(row, "predicted_high_demand_mbps", np.nan) for row in controller]
    )
    demand_low = np.asarray(
        [number(row, "predicted_low_demand_mbps", np.nan) for row in controller]
    )
    buffers = np.asarray(
        [int(number(row, "max_bufferedAmount", 0)) for row in controller]
    )
    health = np.asarray(
        [number(row, "receiver_impairment_rate", np.nan) for row in controller]
    )
    health_fresh = np.asarray(
        [boolean(row, "receiver_health_fresh") for row in controller]
    )
    health[~health_fresh] = np.nan

    switches = [row for row in controller if boolean(row, "switch_applied")]
    switch_latencies = {"high_to_low": [], "low_to_high": []}
    for switch in switches:
        index = controller.index(switch)
        requested_quality = switch["applied_quality"]
        first = index
        while (
            first > 0
            and controller[first - 1]["requested_quality"] == requested_quality
            and controller[first - 1]["applied_quality"] != requested_quality
        ):
            first -= 1
        latency = (
            float(switch["timestamp_monotonic"])
            - float(controller[first]["timestamp_monotonic"])
        )
        direction = (
            "high_to_low" if requested_quality == "low" else "low_to_high"
        )
        switch_latencies[direction].append(latency)
    unique_health = {}
    for row in controller:
        sequence = row.get("health_feedback_sequence", "")
        if sequence:
            unique_health[sequence] = row
    health_totals = {
        name: sum(int(number(row, column, 0)) for row in unique_health.values())
        for name, column in (
            ("total", "health_total_frames"),
            ("full", "health_full_misses"),
            ("partial", "health_partial_frames"),
            ("decode", "health_decode_failures"),
            ("reference_unavailable", "health_reference_unavailable"),
            ("frozen", "health_frozen_frames"),
        )
    }
    health_totals["aggregate_component_events"] = sum(
        health_totals[name]
        for name in (
            "full",
            "partial",
            "decode",
            "reference_unavailable",
            "frozen",
        )
    )
    probes = {}
    for row in probe_rows:
        probe_id = int(number(row, "probe_id", 0))
        if probe_id <= 0:
            continue
        record = probes.setdefault(probe_id, {
            "probe_id": probe_id,
            "start_timestamp": None,
            "end_timestamp": None,
            "state": None,
            "reason": None,
            "offered_bytes": 0,
            "confirmed_bytes": 0,
            "measured_probe_mbps": None,
            "authorization_expiry": None,
        })
        timestamp = number(row, "monotonic_timestamp", None)
        if row.get("probe_state") == "PROBING" and record["start_timestamp"] is None:
            record["start_timestamp"] = timestamp
        if row.get("probe_state") in ("SUCCEEDED", "FAILED", "ABORTED"):
            record["end_timestamp"] = timestamp
            record["state"] = row["probe_state"]
            record["reason"] = row.get("probe_reason")
        elif (
            row.get("event") == "normal_stream_shutdown"
            and row.get("probe_reason") == "normal_stream_shutdown"
        ):
            record["end_timestamp"] = timestamp
            record["state"] = "INCOMPLETE_AT_STREAM_END"
            record["reason"] = "normal_stream_shutdown"
        record["offered_bytes"] = max(
            record["offered_bytes"],
            int(number(row, "offered_probe_bytes", 0)),
        )
        record["confirmed_bytes"] = max(
            record["confirmed_bytes"],
            int(number(row, "confirmed_probe_bytes", 0)),
        )
        measured = number(row, "measured_probe_delivery_rate_mbps", None)
        if measured is not None:
            record["measured_probe_mbps"] = measured
        expiry = number(row, "authorization_expiry", None)
        if expiry is not None:
            record["authorization_expiry"] = expiry
    probe_summaries = list(probes.values())
    for record in probe_summaries:
        record["duration_s"] = (
            None
            if record["start_timestamp"] is None
            or record["end_timestamp"] is None
            else record["end_timestamp"] - record["start_timestamp"]
        )

    valid_measurements = [
        row
        for row in measurements
        if boolean(row, "sent") and start <= number(row, "timestamp_monotonic", -1) <= end
    ]
    transmitted_bytes = sum(
        int(number(row, "encoded_frame_size_bytes", 0))
        for row in valid_measurements
    )
    received_rows = [
        row
        for row in receiver_capacity
        if receiver_start
        <= number(row, "receiver_timestamp", -1)
        <= receiver_end
    ]
    delivered_bytes = sum(int(number(row, "packet_size_bytes", 0)) for row in received_rows)
    duration = max(1e-9, end - start)

    sender_resource = process_resources(sender_diagnostics)
    receiver_resource = process_resources(receiver_diagnostics)
    prior_summary_path = output_dir / "track_b_metrics.json"
    prior_summary = (
        json.loads(prior_summary_path.read_text())
        if metrics_only and prior_summary_path.exists()
        else {}
    )
    preserved_ssim = {
        key: value
        for key, value in prior_summary.items()
        if (
            key.startswith("rgb_ssim")
            or key.startswith("depth_ssim")
            or "missing_output_tail_frames" in key
        )
    }
    summary = {
        "run_dir": str(run_dir),
        "mode": controller[0]["adaptation_mode"],
        "source_frames": len(controller),
        "controlled_duration_s": duration,
        "receiver_controlled_window_start": receiver_start,
        "receiver_controlled_window_end": receiver_end,
        "receiver_controlled_window_method": receiver_window_method,
        "high_fraction": float(np.mean(qualities)),
        "low_fraction": float(np.mean(~qualities)),
        "switch_requests": sum(boolean(row, "switch_requested") for row in controller),
        "switches_applied": len(switches),
        "switch_events": [
            {
                "elapsed_s": float(row["timestamp_monotonic"]) - start,
                "frame_id": int(row["frame_id"]),
                "gop_id": int(row["gop_id"]),
                "from": row["current_quality"],
                "to": row["applied_quality"],
                "reason": row["primary_decision_reason"],
            }
            for row in switches
        ],
        "switch_request_to_apply_latency_s": switch_latencies,
        **capacity_availability_counts(
            receiver_capacity, sender_capacity, controller
        ),
        # Compatibility name retained for old comparison readers. Its meaning
        # is explicitly controller sampling, not receiver publication.
        "capacity_fresh_fraction": float(
            np.mean([boolean(row, "capacity_fresh") for row in controller])
        ),
        "receiver_health_fresh_fraction": float(np.mean(health_fresh)),
        "decision_reasons": dict(
            Counter(row["primary_decision_reason"] for row in controller)
        ),
        "buffer_median_bytes": float(np.median(buffers)),
        "buffer_p95_bytes": float(np.percentile(buffers, 95)),
        "buffer_p99_bytes": float(np.percentile(buffers, 99)),
        "buffer_max_bytes": int(np.max(buffers)),
        "delivered_application_mbps": delivered_bytes * 8 / duration / 1_000_000,
        "transmitted_encoded_payload_mbps": transmitted_bytes
        * 8
        / duration
        / 1_000_000,
        "receiver_messages": len(received_rows),
        "probe_count": len(probe_summaries),
        "probe_results": probe_summaries,
        "probe_successes": sum(
            row["state"] == "SUCCEEDED" for row in probe_summaries
        ),
        "probe_failures": sum(
            row["state"] == "FAILED" for row in probe_summaries
        ),
        "probe_aborts": sum(
            row["state"] == "ABORTED" for row in probe_summaries
        ),
        "probe_overhead_bytes": sum(
            row["offered_bytes"] for row in probe_summaries
        ),
        "probe_overhead_fraction_of_delivered_bytes": (
            0.0
            if delivered_bytes <= 0
            else sum(row["offered_bytes"] for row in probe_summaries)
            / delivered_bytes
        ),
        "health_counts": health_totals,
        "health_windows_100_percent": full_impairment_windows(controller),
        "high_quality_iframe_timing": media_deadline_audit(
            controller, measurements, receiver_media_timing
        ),
        "sender_resources": sender_resource,
        "receiver_resources": receiver_resource,
        "sender_nonempty_queue_longest_s": longest_true_duration(
            sender_diagnostics,
            lambda row: (
                number(row, "rgb_buffered_amount", 0) > 0
                or number(row, "depth_buffered_amount", 0) > 0
                or number(row, "sctp_data_channel_queue", 0) > 0
                or number(row, "sctp_outbound_queue", 0) > 0
            ),
        ),
        "trace_rows": len(applied_trace),
        "trace_elapsed_s": None
        if not applied_trace
        else number(applied_trace[-1], "elapsed_s"),
        **estimator_reference_metrics(
            receiver_capacity, receiver_start, receiver_end
        ),
        **preserved_ssim,
        **parse_receiver_summary(run_dir / "logs" / "receiver.log"),
    }
    statuses = {}
    for name in ("tc", "sender", "receiver"):
        path = run_dir / "metadata" / f"{name}_exit_status.txt"
        statuses[name] = None if not path.exists() else int(path.read_text().strip())
    summary["exit_status"] = statuses
    summary["complete"] = all(value == 0 for value in statuses.values())

    ssim_rows = []
    for stream, gt_path in (("rgb", gt_rgb), ("depth", gt_depth)):
        if metrics_only:
            continue
        if gt_path is None:
            continue
        values, missing_reconstructed = compare_video(
            gt_path, run_dir / f"receiver_{stream}.mp4", len(controller)
        )
        summary[f"{stream}_ssim_frames"] = len(values)
        summary[f"{stream}_missing_output_tail_frames"] = missing_reconstructed
        summary[f"{stream}_ssim_mean"] = None if not values else float(np.mean(values))
        summary[f"{stream}_ssim_median"] = percentile(values, 50)
        summary[f"{stream}_ssim_p05"] = percentile(values, 5)
        summary[f"{stream}_ssim_p95"] = percentile(values, 95)
        ssim_rows.extend(
            {
                "stream": stream,
                "frame_id": frame_id,
                "elapsed_s": frame_id / 25.0,
                "ssim": value,
            }
            for frame_id, value in enumerate(values)
        )
    if ssim_rows:
        with (output_dir / "ssim_per_frame.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ssim_rows[0]))
            writer.writeheader()
            writer.writerows(ssim_rows)

    with (output_dir / "track_b_metrics.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    goodput_x, goodput = one_second_rate(
        receiver_capacity, "receiver_timestamp", "packet_size_bytes",
        receiver_start, receiver_end,
    )
    transmit_x, transmit = one_second_rate(
        valid_measurements,
        "timestamp_monotonic",
        "encoded_frame_size_bytes",
        start,
        end,
    )
    figure, axes = plt.subplots(7, 1, figsize=(14, 16), sharex=True)
    axes[0].step(elapsed, qualities.astype(int), where="post", label="applied high")
    axes[0].step(elapsed, requested.astype(int), where="post", alpha=0.5, label="requested high")
    axes[0].set_ylabel("quality")
    axes[0].legend()
    axes[1].plot(elapsed, capacity, label="fresh published capacity")
    axes[1].plot(elapsed, demand_high, label="predicted high demand")
    axes[1].plot(elapsed, demand_low, label="predicted low demand")
    axes[1].set_ylabel("Mbps")
    axes[1].legend()
    axes[2].plot(goodput_x, goodput, label="delivered application goodput")
    axes[2].plot(transmit_x, transmit, label="transmitted encoded payload")
    axes[2].set_ylabel("Mbps")
    axes[2].legend()
    axes[3].plot(elapsed, buffers)
    axes[3].set_ylabel("buffered B")
    axes[4].plot(elapsed, health)
    axes[4].set_ylabel("impairment")
    axes[5].plot(
        elapsed,
        [number(row, "capacity_age_s", np.nan) for row in controller],
        label="capacity age",
    )
    axes[5].plot(
        elapsed,
        [number(row, "receiver_health_age_s", np.nan) for row in controller],
        label="health age",
    )
    axes[5].set_ylabel("age (s)")
    axes[5].legend()
    reason_ids = {
        reason: index
        for index, reason in enumerate(
            sorted({row["primary_decision_reason"] for row in controller})
        )
    }
    axes[6].scatter(
        elapsed,
        [reason_ids[row["primary_decision_reason"]] for row in controller],
        s=2,
    )
    axes[6].set_yticks(list(reason_ids.values()), list(reason_ids))
    axes[6].set_ylabel("reason")
    axes[6].set_xlabel("Elapsed controlled time (s)")
    figure.tight_layout()
    figure.savefig(output_dir / "track_b_timeline.png", dpi=160)
    plt.close(figure)

    if ssim_rows:
        figure, axes = plt.subplots(1, 2, figsize=(13, 5))
        for stream in ("rgb", "depth"):
            rows = [row for row in ssim_rows if row["stream"] == stream]
            values = np.asarray([row["ssim"] for row in rows])
            axes[0].plot(
                [row["elapsed_s"] for row in rows], values, label=stream, alpha=0.8
            )
            ordered = np.sort(values)
            axes[1].plot(ordered, np.arange(1, len(ordered) + 1) / len(ordered), label=stream)
        axes[0].set_xlabel("Elapsed source time (s)")
        axes[0].set_ylabel("SSIM")
        axes[1].set_xlabel("SSIM")
        axes[1].set_ylabel("CDF")
        for axis in axes:
            axis.legend()
            axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / "ssim_time_and_cdf.png", dpi=160)
        plt.close(figure)
    return summary


def compare_modes(root, output_dir):
    summaries = []
    for mode in MODES:
        path = root / mode / "analysis" / "track_b_metrics.json"
        if path.exists():
            summaries.append(json.loads(path.read_text()))
    if not summaries:
        raise RuntimeError(f"no analyzed modes under {root}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "mode_comparison.json").open("w") as handle:
        json.dump(summaries, handle, indent=2)
        handle.write("\n")
    labels = [summary["mode"] for summary in summaries]
    fields = (
        ("rgb_ssim_mean", "RGB SSIM"),
        ("receiver_decoded_frames", "Decoded frames"),
        ("delivered_application_mbps", "Delivered Mbps"),
        ("buffer_p99_bytes", "Buffer p99 (B)"),
        ("capacity_fresh_fraction", "Capacity fresh fraction"),
        ("switches_applied", "Switches"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for axis, (field, title) in zip(axes.flat, fields):
        axis.bar(labels, [summary.get(field, 0) or 0 for summary in summaries])
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "mode_comparison.png", dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gt-rgb", type=Path)
    parser.add_argument("--gt-depth", type=Path)
    parser.add_argument("--comparison-root", type=Path)
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Reuse existing SSIM fields while refreshing CSV-only metrics",
    )
    args = parser.parse_args()
    if bool(args.run_dir) == bool(args.comparison_root):
        parser.error("provide exactly one of --run-dir or --comparison-root")
    if args.comparison_root:
        compare_modes(args.comparison_root, args.output_dir)
    else:
        summary = analyze_run(
            args.run_dir,
            args.output_dir,
            args.gt_rgb,
            args.gt_depth,
            metrics_only=args.metrics_only,
        )
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
