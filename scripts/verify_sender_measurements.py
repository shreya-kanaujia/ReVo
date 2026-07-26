#!/usr/bin/env python3
"""Verify Week 1A sender measurement CSVs and print basic frame-size stats."""

import argparse
import csv
import statistics
from pathlib import Path


REQUIRED_COLUMNS = [
    "timestamp_monotonic",
    "frame_id",
    "stream",
    "frame_type",
    "encoded_frame_size_bytes",
    "num_chunks",
    "chunk_size",
    "buffered_amount_before_send",
    "buffered_watermark_hard",
    "sent",
    "drop_reason",
    "send_start_monotonic",
    "send_end_monotonic",
]


def _mean(values):
    return statistics.fmean(values) if values else 0.0


def main():
    parser = argparse.ArgumentParser(description="Verify sender frame measurement CSV")
    parser.add_argument("csv_path", help="Path to sender_frame_measurements.csv")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = [col for col in REQUIRED_COLUMNS if col not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(f"Missing required columns: {', '.join(missing)}")
        rows = list(reader)

    total_rows = len(rows)
    total_frame_ids = len({(row["stream"], row["frame_id"]) for row in rows})
    dropped = [row for row in rows if row["sent"] == "0"]
    drop_pct = (len(dropped) / total_rows * 100.0) if total_rows else 0.0

    i_sizes = [int(row["encoded_frame_size_bytes"]) for row in rows if row["frame_type"] == "I"]
    p_sizes = [int(row["encoded_frame_size_bytes"]) for row in rows if row["frame_type"] == "P"]

    print(f"csv: {csv_path}")
    print(f"total rows: {total_rows}")
    print(f"total stream/frame pairs: {total_frame_ids}")
    print(f"drop percentage: {drop_pct:.2f}%")
    print(f"maximum I-frame size: {max(i_sizes) if i_sizes else 0} bytes")
    print(f"average I-frame size: {_mean(i_sizes):.2f} bytes")
    print(f"average P-frame size: {_mean(p_sizes):.2f} bytes")


if __name__ == "__main__":
    main()
