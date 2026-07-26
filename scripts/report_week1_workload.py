#!/usr/bin/env python3
"""Report whether normal ReVo sender traffic can identify path capacity."""

import argparse
import csv
import statistics
from pathlib import Path


def mean(values):
    return statistics.fmean(values) if values else 0.0


def median(values):
    return statistics.median(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Summarize Week 1 normal ReVo offered workload")
    parser.add_argument("--measurements", required=True)
    parser.add_argument("--capacity", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    measurement_rows = list(csv.DictReader(open(args.measurements)))
    capacity_rows = list(csv.DictReader(open(args.capacity)))
    gt_rows = list(csv.DictReader(open(args.ground_truth)))

    if not measurement_rows:
        raise SystemExit("empty measurement CSV")

    by_stream = {"rgb": [], "depth": []}
    for row in measurement_rows:
        by_stream[row["stream"]].append(row)

    lines = []
    lines.append(f"measurement_rows: {len(measurement_rows)}")
    lines.append("video_resolution: 512x512")
    lines.append("frame_rate_fps: 30")

    total_bytes = 0
    first_send = None
    last_send = None
    for stream, rows in by_stream.items():
        i_sizes = [int(r["encoded_frame_size_bytes"]) for r in rows if r["frame_type"] == "I"]
        p_sizes = [int(r["encoded_frame_size_bytes"]) for r in rows if r["frame_type"] == "P"]
        i_chunks = [int(r["num_chunks"]) for r in rows if r["frame_type"] == "I"]
        p_chunks = [int(r["num_chunks"]) for r in rows if r["frame_type"] == "P"]
        total_bytes += sum(int(r["encoded_frame_size_bytes"]) for r in rows if r["sent"] == "1")
        times = [float(r["send_start_monotonic"]) for r in rows if r["send_start_monotonic"]]
        if times:
            first_send = min(times) if first_send is None else min(first_send, min(times))
            last_send = max(times) if last_send is None else max(last_send, max(times))
        lines.extend([
            f"{stream}_frames: {len(rows)}",
            f"{stream}_i_frame_avg_bytes: {mean(i_sizes):.3f}",
            f"{stream}_i_frame_max_bytes: {max(i_sizes) if i_sizes else 0}",
            f"{stream}_p_frame_avg_bytes: {mean(p_sizes):.3f}",
            f"{stream}_p_frame_max_bytes: {max(p_sizes) if p_sizes else 0}",
            f"{stream}_i_chunks_median: {median(i_chunks):.3f}",
            f"{stream}_i_chunks_max: {max(i_chunks) if i_chunks else 0}",
            f"{stream}_p_chunks_median: {median(p_chunks):.3f}",
            f"{stream}_p_chunks_max: {max(p_chunks) if p_chunks else 0}",
        ])

    duration = max(1e-9, (last_send - first_send) if first_send is not None and last_send is not None else 0.0)
    offered_mbps = total_bytes * 8.0 / duration / 1_000_000.0
    lines.append(f"total_encoded_offered_bitrate_mbps: {offered_mbps:.6f}")

    packet_rows = [r for r in capacity_rows if r.get("acknowledged_sequence_number") or r.get("latest_sequence_id")]
    outstanding_key = "packets_outstanding" if packet_rows and "packets_outstanding" in packet_rows[0] else "outstanding_packets"
    outstanding = [int(float(r[outstanding_key])) for r in packet_rows if r.get(outstanding_key)]
    lines.append(f"max_packets_outstanding: {max(outstanding) if outstanding else 0}")
    lines.append(f"median_packets_outstanding: {median(outstanding):.3f}")
    lines.append(f"sender_backlogged: {bool(outstanding and max(outstanding) > 10)}")

    gt_bandwidths = [float(r["bandwidth_mbps"]) for r in gt_rows]
    min_gt = min(gt_bandwidths) if gt_bandwidths else 0
    lines.append(f"min_ground_truth_bandwidth_mbps: {min_gt:.6f}")
    lines.append(
        "normal_workload_capacity_identifiable: "
        + str(offered_mbps >= min_gt and bool(outstanding and max(outstanding) > 10))
    )
    lines.append(
        "normal_workload_conclusion: "
        + ("can load bottleneck" if offered_mbps >= min_gt else "NOT IDENTIFIABLE - ESTIMATE IS A LOWER BOUND")
    )

    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
