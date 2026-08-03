#!/usr/bin/env python3
"""Measure QP20/QP30 RGB+depth demand without a network experiment."""

import argparse
import asyncio
import csv
import json
from pathlib import Path
import struct
import sys

import av
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sender"))

from quality_encoder_manager import (  # noqa: E402
    QualityEncoderManager,
    application_bytes_for_payload,
)

DESC_SIZE = struct.calcsize("<BBIIBHHHIQdd")


def frames(path, limit):
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if limit and index >= limit:
                break
            array = frame.to_ndarray(format="rgb24")
            yield torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)


async def measure(args):
    manager = QualityEncoderManager(
        high_qp=args.high_qp,
        low_qp=args.low_qp,
        intra_period=30,
        pool_threads=args.encoder_pool_threads or None,
        enable_affinity=True,
    )
    rows = []
    rgb_frames = frames(args.rgb, args.frames)
    depth_frames = frames(args.depth, args.frames)
    loop = asyncio.get_running_loop()
    try:
        for frame_id, (rgb, depth) in enumerate(zip(rgb_frames, depth_frames)):
            rgb_array = rgb[0, 0].permute(1, 2, 0).contiguous().numpy()
            depth_array = depth[0, 0].permute(1, 2, 0).contiguous().numpy()
            if frame_id == 0:
                await manager.start(
                    rgb_array.shape, depth_array.shape,
                    rgb_array.dtype, depth_array.dtype,
                )
            started = loop.time()
            await manager.submit_frame(
                frame_id, args.fps, rgb_array, depth_array
            )
            candidate_set, worker_record = await manager.wait_result(frame_id)
            wall_encode_s = loop.time() - started
            if candidate_set.high is None or candidate_set.low is None:
                continue
            row = {
                "frame_id": candidate_set.frame_id,
                "wall_encode_s": wall_encode_s,
                "worker_pid": worker_record["worker_pid"],
                "worker_started": worker_record["worker_started"],
                "worker_completed": worker_record["worker_completed"],
                "active_native_max": worker_record["active_native_max"],
                "worker_affinity": ",".join(
                    map(str, worker_record["worker_affinity"])
                ),
            }
            for quality, candidate in (
                ("high", candidate_set.high),
                ("low", candidate_set.low),
            ):
                aggregate = 0
                for stream in ("rgb", "depth"):
                    item = getattr(candidate, stream)
                    size = len(item["payload"])
                    row[f"{quality}_{stream}_encoded_bytes"] = size
                    row[f"{quality}_{stream}_encode_s"] = getattr(
                        candidate, f"{stream}_encode_s"
                    )
                    aggregate += application_bytes_for_payload(
                        size, 1024, bool(item["is_key"]), DESC_SIZE
                    )
                row[f"{quality}_application_bytes"] = aggregate
                row[f"{quality}_application_mbps"] = (
                    aggregate * 8 * args.fps / 1_000_000
                )
            rows.append(row)
    finally:
        await manager.aclose()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=0)
    parser.add_argument("--high-qp", type=int, default=20)
    parser.add_argument("--low-qp", type=int, default=30)
    parser.add_argument(
        "--encoder-pool-threads",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    if args.fps <= 0:
        with av.open(str(args.rgb)) as container:
            rate = container.streams.video[0].average_rate
            args.fps = max(1, int(round(float(rate)))) if rate else 30
    rows = asyncio.run(measure(args))
    if not rows:
        raise SystemExit("no aligned high/low RGB/depth outputs were produced")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    wall_values = np.asarray([row["wall_encode_s"] for row in rows])
    summary = {
        "frames": len(rows),
        "source_fps": args.fps,
        "encoder_pool_threads": args.encoder_pool_threads,
        "wall_encode_mean_ms": float(np.mean(wall_values) * 1000),
        "wall_encode_p95_ms": float(np.percentile(wall_values, 95) * 1000),
        "achieved_encode_fps": float(len(wall_values) / np.sum(wall_values)),
        "realtime_at_source_fps": bool(
            np.mean(wall_values) <= 1.0 / args.fps
        ),
        "worker_pids": sorted(set(row["worker_pid"] for row in rows)),
        "worker_affinity": sorted(set(row["worker_affinity"] for row in rows)),
        "max_native_concurrency": max(
            row["active_native_max"] for row in rows
        ),
        "strict_frame_order": [row["frame_id"] for row in rows]
        == list(range(len(rows))),
        "single_frame_at_a_time": all(
            rows[index]["worker_started"]
            >= rows[index - 1]["worker_completed"]
            for index in range(1, len(rows))
        ),
    }
    for quality in ("high", "low"):
        values = np.asarray([row[f"{quality}_application_mbps"] for row in rows])
        summary[quality] = {
            "median_mbps": float(np.median(values)),
            "p95_mbps": float(np.percentile(values, 95)),
            "mean_mbps": float(np.mean(values)),
            "total_application_bytes": int(
                sum(row[f"{quality}_application_bytes"] for row in rows)
            ),
            "rgb_encode_mean_ms": float(np.mean([
                row[f"{quality}_rgb_encode_s"] * 1000 for row in rows
            ])),
            "rgb_encode_p95_ms": float(np.percentile([
                row[f"{quality}_rgb_encode_s"] * 1000 for row in rows
            ], 95)),
            "depth_encode_mean_ms": float(np.mean([
                row[f"{quality}_depth_encode_s"] * 1000 for row in rows
            ])),
            "depth_encode_p95_ms": float(np.percentile([
                row[f"{quality}_depth_encode_s"] * 1000 for row in rows
            ], 95)),
        }
    summary["high_low_median_ratio"] = (
        summary["high"]["median_mbps"] / summary["low"]["median_mbps"]
    )
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
