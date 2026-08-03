#!/usr/bin/env python3
"""Real-x265 characterization of the bounded Track B producer, without a network."""

import argparse
import asyncio
import csv
import json
import pickle
from pathlib import Path
import sys
import time

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "sender"))

from quality_encoder_manager import (  # noqa: E402
    BoundedCandidateQueue,
    CandidateRefillCondition,
    QualityEncoderManager,
    get_ordered_source_pair,
    prefetch_source_pairs,
)


class SequentialPairLoader:
    def __init__(self, rgb_path, depth_path, start_frame=0):
        self.rgb_container = av.open(str(rgb_path))
        self.depth_container = av.open(str(depth_path))
        self.rgb_frames = self.rgb_container.decode(
            self.rgb_container.streams.video[0]
        )
        self.depth_frames = self.depth_container.decode(
            self.depth_container.streams.video[0]
        )
        self.expected = 0
        for _ in range(int(start_frame)):
            next(self.rgb_frames)
            next(self.depth_frames)

    def __call__(self, frame_id):
        if int(frame_id) != self.expected:
            raise RuntimeError(
                f"source loader order mismatch: expected {self.expected}, got {frame_id}"
            )
        self.expected += 1
        rgb = next(self.rgb_frames).to_ndarray(format="rgb24")
        depth = next(self.depth_frames).to_ndarray(format="rgb24")
        return np.ascontiguousarray(rgb), np.ascontiguousarray(depth)

    def close(self):
        self.rgb_container.close()
        self.depth_container.close()


def percentile(values, q):
    return float(np.percentile(np.asarray(values), q)) if values else None


async def characterize(args):
    raw_queue = asyncio.Queue(maxsize=3)
    completed = BoundedCandidateQueue(5)
    refill = CandidateRefillCondition()
    manager = QualityEncoderManager(
        pool_threads=1, max_inflight=2, completed_lead=5,
        sender_reserved_cpus=2, enable_affinity=True,
    )
    loader = SequentialPairLoader(args.rgb, args.depth, args.start_frame)
    prefetch = asyncio.create_task(
        prefetch_source_pairs(args.frames, loader, raw_queue)
    )
    records = {}
    candidates = {}
    producer = None
    try:
        first, first_wait = await get_ordered_source_pair(raw_queue, 0)
        await manager.start(
            first.rgb.shape, first.depth.shape, first.rgb.dtype, first.depth.dtype
        )

        async def produce():
            submitted = collected = 0
            first_pending = (first, first_wait)
            previous_submit = None
            try:
                while collected < args.frames:
                    while submitted < args.frames and submitted - completed.consumed_count < 7:
                        item = first_pending
                        first_pending = None
                        if item is None:
                            item = await get_ordered_source_pair(raw_queue, submitted)
                        source, raw_wait = item
                        request_ts = await manager.submit_frame(
                            submitted, args.fps, source.rgb, source.depth
                        )
                        records[submitted] = {
                            "frame_id": submitted,
                            "source_load_started": source.load_started,
                            "source_load_completed": source.load_completed,
                            "raw_pair_wait_s": raw_wait,
                            "request_submitted": request_ts,
                            "request_submission_interval_s": (
                                0.0 if previous_submit is None else request_ts - previous_submit
                            ),
                        }
                        previous_submit = request_ts
                        submitted += 1
                    if collected < submitted:
                        candidate, worker = await manager.wait_result(collected)
                        candidates[collected] = candidate
                        records[collected].update(worker)
                        await completed.put((
                            collected, [], worker["worker_started"],
                            worker["worker_completed"], 0.0, None,
                        ))
                        collected += 1
                        continue
                    await refill.wait_for_room(submitted, completed, 7)
            finally:
                if not prefetch.done():
                    prefetch.cancel()
                await asyncio.gather(prefetch, return_exceptions=True)

        producer = asyncio.create_task(produce())
        while completed.qsize() < 5:
            if producer.done():
                await producer
            await asyncio.sleep(0)

        origin = time.perf_counter()
        observed = []
        for frame_id in range(args.frames):
            deadline = origin + frame_id / args.fps
            record = await completed.get_before(deadline)
            if int(record[0]) != frame_id:
                raise RuntimeError(
                    f"consumer order mismatch: expected {frame_id}, got {record[0]}"
                )
            observed.append({
                "frame_id": frame_id,
                "consumed_at": time.perf_counter(),
                "completed_lead_after_get": completed.qsize(),
            })
            await refill.notify_consumed()
            if not args.unpaced:
                next_deadline = origin + (frame_id + 1) / args.fps
                await asyncio.sleep(
                    max(0.0, next_deadline - time.perf_counter())
                )
        await producer

        merged = [
            {
                **records[row["frame_id"]],
                **row,
                "source_frame_id": args.start_frame + row["frame_id"],
            }
            for row in observed
        ]
        intervals = [
            merged[index]["worker_completed"] - merged[index - 1]["worker_completed"]
            for index in range(1, len(merged))
        ]
        native = [row["total_production_s"] for row in merged]
        idle = [row["worker_idle_s"] for row in merged[1:]]
        spikes = [
            index for index, value in enumerate(native)
            if value > 1.0 / args.fps
        ]
        recovered = all(
            any(
                row["completed_lead_after_get"] == 4
                for row in merged[index + 1:min(len(merged), index + 31)]
            )
            for index in spikes
        )
        summary = {
            "mode": "unpaced_headroom" if args.unpaced else "paced_reserve",
            "frames": len(merged),
            "source_fps": args.fps,
            "achieved_completion_fps": (
                (len(merged) - 1)
                / (merged[-1]["worker_completed"] - merged[0]["worker_completed"])
            ),
            "native_mean_ms": float(np.mean(native) * 1000),
            "native_p95_ms": percentile(native, 95) * 1000,
            "native_max_ms": max(native) * 1000,
            "worker_idle_mean_ms": float(np.mean(idle) * 1000),
            "worker_idle_p95_ms": percentile(idle, 95) * 1000,
            "completion_interval_p95_ms": percentile(intervals, 95) * 1000,
            "lead_min_after_get": min(
                row["completed_lead_after_get"] for row in merged
            ),
            "lead_max_after_get": max(
                row["completed_lead_after_get"] for row in merged
            ),
            "production_spikes_over_period": len(spikes),
            "reserve_recovered_after_spikes": recovered,
            "strict_order": [row["frame_id"] for row in merged]
            == list(range(len(merged))),
            "raw_prefetch_capacity": raw_queue.maxsize,
            "completed_capacity": completed.max_frames,
            "max_inflight": manager.max_inflight,
            "total_work_bound": manager.completed_lead + manager.max_inflight,
            "worker_pid": manager.runtime_snapshot()["child_pid"],
            "sender_affinity": manager.runtime_snapshot()["sender_cpu_affinity"],
            "worker_affinity": manager.runtime_snapshot()["worker_cpu_affinity"],
        }
        retained = {
            "source_frame_offset": args.start_frame,
            "candidates": [candidates[index] for index in range(args.frames)],
            "records": [
                {"durations": records[index]["durations"]}
                for index in range(args.frames)
            ],
        }
        return merged, summary, retained
    finally:
        if producer is not None and not producer.done():
            producer.cancel()
            await asyncio.gather(producer, return_exceptions=True)
        if not prefetch.done():
            prefetch.cancel()
            await asyncio.gather(prefetch, return_exceptions=True)
        await manager.aclose()
        loader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=600)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--unpaced", action="store_true")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--retain-manifest", type=Path)
    args = parser.parse_args()
    rows, summary, retained = asyncio.run(characterize(args))
    args.output_csv.parent.mkdir(parents=True, exist_ok=False)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.retain_manifest:
        with args.retain_manifest.open("wb") as handle:
            pickle.dump(retained, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(json.dumps(summary, indent=2))
    failed = (
        (
            args.unpaced
            and (
                summary["achieved_completion_fps"] < args.fps * 1.10
                or summary["worker_idle_p95_ms"] >= 10.0
            )
        )
        or (
            not args.unpaced
            and (
                not summary["reserve_recovered_after_spikes"]
                or summary["lead_max_after_get"] != 4
            )
        )
        or not summary["strict_order"]
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
