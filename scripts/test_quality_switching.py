#!/usr/bin/env python3
"""Deterministic tests for Track B process-isolated candidate encoding."""

import asyncio
from multiprocessing import shared_memory
import os
import queue
import sys
import threading
import time
import unittest
from unittest import mock

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "sender"))

from adaptation_controller import Quality  # noqa: E402
from quality_encoder_manager import (  # noqa: E402
    BoundedCandidateQueue,
    CandidateRefillCondition,
    CandidateSet,
    CausalDemandPredictor,
    EncodedCandidate,
    QualityEncoderManager,
    SourceFrameRecord,
    get_ordered_source_pair,
    partition_track_b_cpus,
    prefetch_source_pairs,
    wait_for_candidate_priming,
)


_factory_serial = 0


class ProcessFakeCodec:
    """Pickle-safe codec whose output proves child ownership and persistence."""

    def __init__(self, qp, *, delay=0.0, fail_frame=None, missing=False):
        global _factory_serial
        _factory_serial += 1
        self.instance_serial = _factory_serial
        self.qp = int(qp)
        self.delay = float(delay)
        self.fail_frame = fail_frame
        self.missing = missing
        self.calls = 0

    def compress_prepared(self, array, frame_id, fps):
        if self.delay:
            time.sleep(self.delay)
        if self.fail_frame == frame_id:
            raise RuntimeError(f"intentional worker failure at {frame_id}")
        self.calls += 1
        if self.missing:
            return
        yield {
            "frame_id": frame_id,
            "is_key": frame_id % 30 == 0,
            "payload": bytes([self.qp]) * (100 - self.qp),
            "qp": self.qp,
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
            "worker_pid": os.getpid(),
            "codec_instance": self.instance_serial,
            "codec_call": self.calls,
        }


def fake_codec_factory(qp, intra_period, pool_threads):
    return ProcessFakeCodec(qp)


def slow_codec_factory(qp, intra_period, pool_threads):
    return ProcessFakeCodec(qp, delay=0.05)


def failing_codec_factory(qp, intra_period, pool_threads):
    return ProcessFakeCodec(qp, fail_frame=1)


def startup_failing_codec_factory(qp, intra_period, pool_threads):
    raise RuntimeError("intentional encoder startup failure")


def missing_low_depth_factory(qp, intra_period, pool_threads):
    global _factory_serial
    # Worker construction order is high RGB, low RGB, high depth, low depth.
    return ProcessFakeCodec(qp, missing=(_factory_serial == 3))


def frame(value=0):
    return np.full((8, 8, 3), value, dtype=np.uint8)


async def start_manager(factory=fake_codec_factory):
    manager = QualityEncoderManager(
        codec_factory=factory,
        enable_affinity=False,
        max_inflight=2,
        completed_lead=5,
    )
    sample = frame()
    await manager.start(sample.shape, sample.shape, sample.dtype, sample.dtype)
    return manager


async def encode_frames(manager, count):
    results = []
    submitted = 0
    consumed = 0
    while consumed < count:
        while submitted < count and submitted - consumed < 2:
            await manager.submit_frame(submitted, 25, frame(submitted), frame(submitted))
            submitted += 1
        candidate, record = await manager.wait_result(consumed)
        results.append((candidate, record))
        consumed += 1
    return results


def candidate_set(frame_id, *, low=True, key=False):
    def candidate(qp):
        encoded = {"frame_id": frame_id, "is_key": key, "payload": b"x", "qp": qp}
        return EncodedCandidate(frame_id, key, encoded, encoded, 1, 1, 0.01, 0.01)
    return CandidateSet(
        frame_id, candidate(20), candidate(30) if low else None, low,
        "low_candidate_missing_or_mismatched" if not low else "",
    )


class QualitySwitchingTests(unittest.TestCase):
    def test_linux_worker_is_ready_before_submission_thread_and_frame_zero_completes(self):
        async def exercise():
            manager = await start_manager()
            submitter = None
            try:
                self.assertTrue(manager._worker_ready.is_set())
                self.assertTrue(manager._process.is_alive())
                submitted = threading.Event()

                def submit_first():
                    manager.submit_frame_blocking(0, 25, frame(), frame())
                    submitted.set()

                submitter = threading.Thread(target=submit_first)
                submitter.start()
                await asyncio.to_thread(submitter.join, 2.0)
                self.assertTrue(submitted.is_set())
                candidate, record = await asyncio.wait_for(
                    manager.wait_result(0), timeout=2.0
                )
                return (
                    candidate.frame_id,
                    record["worker_pid"],
                    os.getpid(),
                    await manager.aclose(),
                    manager._reader.is_alive(),
                )
            finally:
                if submitter is not None and submitter.is_alive():
                    submitter.join(timeout=1.0)
                if manager._process is not None and manager._process.is_alive():
                    await manager.terminate_on_error()

        frame_id, child_pid, parent_pid, exitcode, reader_alive = asyncio.run(
            exercise()
        )
        self.assertEqual(frame_id, 0)
        self.assertNotEqual(child_pid, parent_pid)
        self.assertEqual(exitcode, 0)
        self.assertFalse(reader_alive)

    def test_worker_startup_failure_propagates_and_cleans_every_resource(self):
        async def exercise():
            manager = QualityEncoderManager(
                codec_factory=startup_failing_codec_factory,
                enable_affinity=False,
            )
            sample = frame()
            with self.assertRaisesRegex(
                RuntimeError, "intentional encoder startup failure"
            ):
                await asyncio.wait_for(
                    manager.start(
                        sample.shape, sample.shape, sample.dtype, sample.dtype
                    ),
                    timeout=3.0,
                )
            return (
                manager._process.is_alive(),
                manager._reader.is_alive(),
                len(manager._shms),
            )

        process_alive, reader_alive, shm_count = asyncio.run(exercise())
        self.assertFalse(process_alive)
        self.assertFalse(reader_alive)
        self.assertEqual(shm_count, 0)

    def test_sender_orders_worker_readiness_before_background_threads(self):
        source = (
            __import__("pathlib").Path(ROOT) / "src" / "sender" / "sender-3d.py"
        ).read_text()
        start_at = source.index("await self.quality_manager.start(")
        prefetch_at = source.index(
            "source_prefetch_task = asyncio.create_task(", start_at
        )
        submitter_at = source.index(
            "candidate_submission_thread.start()", prefetch_at
        )
        self.assertLess(start_at, prefetch_at)
        self.assertLess(prefetch_at, submitter_at)

    def test_process_owns_persistent_independent_codecs_and_order(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            try:
                results = await encode_frames(manager, 3)
                return manager, results
            finally:
                await manager.aclose()

        manager, results = asyncio.run(exercise())
        self.assertNotEqual(results[0][1]["worker_pid"], os.getpid())
        for candidates, record in results:
            self.assertEqual(candidates.frame_id, record["frame_id"])
            self.assertEqual(record["active_native_max"], 4)
            for encoded in (candidates.high.rgb, candidates.high.depth,
                            candidates.low.rgb, candidates.low.depth):
                self.assertEqual(encoded["worker_pid"], record["worker_pid"])
        first_instances = {
            results[0][0].high.rgb["codec_instance"],
            results[0][0].high.depth["codec_instance"],
            results[0][0].low.rgb["codec_instance"],
            results[0][0].low.depth["codec_instance"],
        }
        self.assertEqual(len(first_instances), 4)
        self.assertEqual(results[-1][0].high.rgb["codec_call"], 3)
        self.assertEqual(results[-1][0].low.depth["codec_call"], 3)
        self.assertEqual(manager._requests._maxsize, 2)
        self.assertEqual(manager._results._maxsize, 2)

    def test_asyncio_heartbeat_runs_during_slow_child_encode(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            ticks = 0
            try:
                await manager.submit_frame(0, 25, frame(), frame())
                task = asyncio.create_task(manager.wait_result(0))
                while not task.done():
                    ticks += 1
                    await asyncio.sleep(0.002)
                await task
                return ticks
            finally:
                await manager.aclose()

        self.assertGreater(asyncio.run(exercise()), 10)

    def test_shared_memory_slots_are_not_reused_before_completion(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            try:
                await manager.submit_frame(0, 25, frame(1), frame(1))
                await manager.submit_frame(1, 25, frame(2), frame(2))
                self.assertEqual(
                    manager.runtime_snapshot()["unfinished_ipc_requests"], 2
                )
                third = asyncio.create_task(
                    manager.submit_frame(2, 25, frame(3), frame(3))
                )
                await asyncio.sleep(0.01)
                self.assertFalse(third.done())
                await manager.wait_result(0)
                await third
                await manager.wait_result(1)
                await manager.wait_result(2)
                return manager.max_inflight
            finally:
                await manager.aclose()

        self.assertEqual(asyncio.run(exercise()), 2)

    def test_worker_exception_propagates_and_shutdown_unlinks_memory(self):
        async def exercise():
            manager = await start_manager(failing_codec_factory)
            names = [shm.name for pair in manager._shms for shm in pair]
            try:
                await manager.submit_frame(0, 25, frame(), frame())
                await manager.submit_frame(1, 25, frame(), frame())
                await manager.wait_result(0)
                with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
                    await manager.wait_result(1)
            finally:
                await manager.terminate_on_error()
            return names, manager._process.exitcode

        names, exitcode = asyncio.run(exercise())
        self.assertNotEqual(exitcode, 0)
        for name in names:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_worker_death_is_detected(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            try:
                await manager.submit_frame(0, 25, frame(), frame())
                manager._process.terminate()
                await asyncio.to_thread(manager._process.join, 1.0)
                with self.assertRaisesRegex(RuntimeError, "worker died"):
                    await asyncio.wait_for(manager.wait_result(0), 1.0)
            finally:
                await manager.terminate_on_error()

        asyncio.run(exercise())

    def test_normal_shutdown_has_zero_exit_and_unlinks_memory(self):
        async def exercise():
            manager = await start_manager()
            names = [shm.name for pair in manager._shms for shm in pair]
            await manager.submit_frame(0, 25, frame(), frame())
            await manager.wait_result(0)
            code = await manager.aclose()
            return names, code

        names, code = asyncio.run(exercise())
        self.assertEqual(code, 0)
        for name in names:
            with self.assertRaises(FileNotFoundError):
                shared_memory.SharedMemory(name=name)

    def test_affinity_partition_requires_four_disjoint_cpus(self):
        with self.assertRaisesRegex(RuntimeError, "at least 4 CPUs"):
            partition_track_b_cpus({0, 1, 2})
        with self.assertRaisesRegex(RuntimeError, "4 encoder-worker CPUs"):
            partition_track_b_cpus({0, 1, 2, 3, 4}, 2)
        sender, worker = partition_track_b_cpus({7, 3, 9, 5, 11, 13}, 2)
        self.assertEqual(sender, (3, 5))
        self.assertEqual(worker, (7, 9, 11, 13))
        self.assertFalse(set(sender) & set(worker))

    def test_only_one_frame_id_is_processed_at_a_time(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            try:
                results = await encode_frames(manager, 4)
                return [record for _candidate, record in results]
            finally:
                await manager.aclose()

        records = asyncio.run(exercise())
        self.assertTrue(all(
            records[index]["worker_started"]
            >= records[index - 1]["worker_completed"]
            for index in range(1, len(records))
        ))

    def test_rgb_depth_selection_is_atomic_and_gop_locked(self):
        manager = QualityEncoderManager(enable_affinity=False)
        selected, reason = manager.select(candidate_set(0, low=True, key=True), Quality.LOW)
        self.assertEqual((selected.rgb["qp"], selected.depth["qp"]), (30, 30))
        self.assertEqual(reason, "")
        selected, reason = manager.select(candidate_set(1, low=False), Quality.LOW)
        self.assertEqual((selected.rgb["qp"], selected.depth["qp"]), (20, 20))
        self.assertTrue(reason)
        selected, reason = manager.select(candidate_set(2, low=True), Quality.LOW)
        self.assertEqual((selected.rgb["qp"], selected.depth["qp"]), (20, 20))
        self.assertIn("gop_locked", reason)
        selected, _ = manager.select(candidate_set(30, low=True, key=True), Quality.LOW)
        self.assertEqual((selected.rgb["qp"], selected.depth["qp"]), (30, 30))

    def test_demand_predictor_is_causal_and_uses_bytes_once(self):
        predictor = CausalDemandPredictor(alpha=1.0)
        high, low = predictor.observe_completed_frame(
            high_application_bytes=1000,
            low_application_bytes=500,
            frame_interval_s=0.1,
        )
        self.assertAlmostEqual(high, 0.08)
        self.assertAlmostEqual(low, 0.04)

    def test_bounded_candidate_queue_preserves_order_and_buffers_collector(self):
        async def exercise():
            handoff = BoundedCandidateQueue(2)
            stop = threading.Event()
            record = lambda fid: (fid, [], 1.0, 2.0, 0.0, None)
            self.assertTrue(handoff.put_from_collector_until(record(0), stop))
            self.assertTrue(handoff.put_from_collector_until(record(1), stop))
            self.assertTrue(handoff.put_from_collector_until(record(2), stop))
            self.assertTrue(handoff.put_from_collector_until(record(3), stop))
            self.assertFalse(handoff.put_from_collector_until(record(4), stop))
            self.assertEqual(handoff.qsize(), 2)
            self.assertEqual(handoff.collector_buffer_qsize(), 2)
            self.assertEqual((await handoff.get())[0], 0)
            self.assertEqual(handoff.qsize(), 2)
            self.assertEqual(handoff.collector_buffer_qsize(), 1)
            self.assertEqual((await handoff.get())[0], 1)
            self.assertEqual((await handoff.get())[0], 2)
            self.assertEqual((await handoff.get())[0], 3)

        asyncio.run(exercise())

    def test_completed_handoff_drains_worker_results_when_completed_queue_is_full(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            records = {}
            state = {}
            manager = QualityEncoderManager(
                codec_factory=fake_codec_factory,
                enable_affinity=False,
                max_inflight=2,
                completed_lead=5,
            )
            manager.bind_completed_handoff(handoff, records, state)
            sample = frame()
            await manager.start(sample.shape, sample.shape, sample.dtype, sample.dtype)
            try:
                submitted = 0
                max_unfinished = 0
                for fid in range(7):
                    await asyncio.wait_for(
                        manager.submit_frame(fid, 25, frame(fid), frame(fid)),
                        2.0,
                    )
                    submitted += 1
                    max_unfinished = max(
                        max_unfinished,
                        manager.runtime_snapshot()["unfinished_ipc_requests"],
                    )
                deadline = time.perf_counter() + 2.0
                while len(records) < 7 and time.perf_counter() < deadline:
                    max_unfinished = max(
                        max_unfinished,
                        manager.runtime_snapshot()["unfinished_ipc_requests"],
                    )
                    await asyncio.sleep(0.005)
                snapshot = manager.runtime_snapshot()
                queue_depth = handoff.qsize()
                buffer_depth = handoff.collector_buffer_qsize()
                drained = []
                while True:
                    try:
                        drained.append((await handoff.get())[0])
                    except asyncio.QueueEmpty:
                        break
                    if len(drained) == 7:
                        break
                return {
                    "submitted": submitted,
                    "record_count": len(records),
                    "drained": drained,
                    "queue_depth": queue_depth,
                    "buffer_depth": buffer_depth,
                    "buffer_max": handoff.collector_buffer_max_frames,
                    "request_depth": snapshot["input_queue_depth"],
                    "result_depth": snapshot["output_queue_depth"],
                    "unfinished": snapshot["unfinished_ipc_requests"],
                    "max_unfinished": max_unfinished,
                    "max_frames": handoff.max_frames,
                    "reader_alive_before_close": manager._reader.is_alive(),
                }
            finally:
                await manager.aclose()
                self.assertFalse(manager._reader.is_alive())

        result = asyncio.run(exercise())
        self.assertEqual(result["submitted"], 7)
        self.assertEqual(result["record_count"], 7)
        self.assertEqual(result["drained"], list(range(7)))
        self.assertEqual(result["queue_depth"], 5)
        self.assertEqual(result["buffer_depth"], 2)
        self.assertEqual(result["buffer_max"], 2)
        self.assertLessEqual(result["max_frames"], 5)
        self.assertIn(result["request_depth"], (-1, 0))
        self.assertIn(result["result_depth"], (-1, 0))
        self.assertEqual(result["unfinished"], 0)
        self.assertLessEqual(result["max_unfinished"], 2)
        self.assertTrue(result["reader_alive_before_close"])

    def test_inflight_permit_and_shared_slot_release_before_sender_consumption(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            manager = QualityEncoderManager(
                codec_factory=fake_codec_factory,
                enable_affinity=False,
                max_inflight=2,
                completed_lead=5,
            )
            manager.bind_completed_handoff(handoff, {}, {})
            sample = frame()
            await manager.start(sample.shape, sample.shape, sample.dtype, sample.dtype)
            try:
                for fid in range(7):
                    await asyncio.wait_for(
                        manager.submit_frame(fid, 25, frame(fid), frame(fid)),
                        2.0,
                    )
                deadline = time.perf_counter() + 2.0
                while handoff.collector_buffer_qsize() < 2 and time.perf_counter() < deadline:
                    await asyncio.sleep(0.005)
                snapshot = manager.runtime_snapshot()
                return {
                    "unfinished": snapshot["unfinished_ipc_requests"],
                    "free_slots": manager._free_slots.qsize(),
                    "inflight_value": manager._inflight._value,
                    "completed_depth": handoff.qsize(),
                    "buffer_depth": handoff.collector_buffer_qsize(),
                }
            finally:
                await manager.aclose()

        result = asyncio.run(exercise())
        self.assertEqual(result["unfinished"], 0)
        self.assertEqual(result["free_slots"], 3)
        self.assertEqual(result["inflight_value"], 2)
        self.assertEqual(result["completed_depth"], 5)
        self.assertEqual(result["buffer_depth"], 2)

    def test_refill_runs_before_simulated_long_send_pacing_after_consume_yield(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            refill = CandidateRefillCondition()
            stop = threading.Event()
            record = lambda fid: (fid, [], 1.0, 2.0, 0.0, None)
            for fid in range(7):
                self.assertTrue(handoff.put_from_collector_until(record(fid), stop))
            producer_done = asyncio.Event()

            async def producer():
                await refill.wait_for_room(7, handoff, 7)
                self.assertTrue(handoff.put_from_collector_until(record(7), stop))
                producer_done.set()

            task = asyncio.create_task(producer())
            first = await handoff.get()
            await refill.notify_consumed()
            await asyncio.wait_for(producer_done.wait(), timeout=0.25)
            await asyncio.sleep(0.05)
            drained = [first[0]]
            while len(drained) < 8:
                drained.append((await handoff.get())[0])
            await task
            return {
                "drained": drained,
                "queue_depth": handoff.qsize(),
                "buffer_depth": handoff.collector_buffer_qsize(),
                "max_frames": handoff.max_frames,
                "buffer_max": handoff.collector_buffer_max_frames,
            }

        result = asyncio.run(exercise())
        self.assertEqual(result["drained"], list(range(8)))
        self.assertLessEqual(result["queue_depth"], 5)
        self.assertLessEqual(result["buffer_depth"], 2)
        self.assertEqual(result["max_frames"], 5)
        self.assertEqual(result["buffer_max"], 2)

    def test_refill_submission_thread_runs_while_asyncio_is_blocked(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            refill = CandidateRefillCondition()
            stop = threading.Event()
            record = lambda fid: (fid, [], 1.0, 2.0, 0.0, None)
            for fid in range(7):
                self.assertTrue(handoff.put_from_collector_until(record(fid), stop))
            submitted = threading.Event()

            def submitter():
                self.assertTrue(
                    refill.wait_for_room_blocking(7, handoff, 7, stop)
                )
                self.assertTrue(
                    handoff.put_from_collector_until(record(7), stop)
                )
                submitted.set()

            thread = threading.Thread(target=submitter)
            thread.start()
            self.assertEqual((await handoff.get())[0], 0)
            refill.notify_consumed_sync()
            # Model a synchronous DataChannel send which monopolizes asyncio.
            time.sleep(0.05)
            self.assertTrue(submitted.is_set())
            drained = []
            while len(drained) < 7:
                drained.append((await handoff.get())[0])
            thread.join(1.0)
            return drained, handoff.max_frames, handoff.collector_buffer_max_frames

        drained, completed, collector = asyncio.run(exercise())
        self.assertEqual(drained, list(range(1, 8)))
        self.assertEqual(completed, 5)
        self.assertEqual(collector, 2)

    def test_native_source_prefetch_remains_bounded_and_ordered(self):
        async def exercise():
            raw = queue.Queue(maxsize=3)
            task = asyncio.create_task(prefetch_source_pairs(
                5, lambda fid: (frame(fid), frame(fid)), raw
            ))
            observed = []
            for fid in range(5):
                record, _ = await get_ordered_source_pair(raw, fid)
                observed.append(record.frame_id)
                self.assertLessEqual(raw.qsize(), 3)
            await task
            return observed

        self.assertEqual(asyncio.run(exercise()), list(range(5)))

    def test_candidate_queue_rejects_duplicate_frame(self):
        async def exercise():
            handoff = BoundedCandidateQueue(2)
            await handoff.put((0, [], 0.0, 0.0, 0.0, None))
            with self.assertRaises(RuntimeError):
                await handoff.put((0, [], 0.0, 0.0, 0.0, None))

        asyncio.run(exercise())

    def test_candidate_wait_accepts_result_before_deadline_without_failing_early(self):
        async def exercise():
            handoff = BoundedCandidateQueue(2)
            started = time.perf_counter()

            async def delayed_result():
                await asyncio.sleep(0.025)
                await handoff.put((0, [], 0.0, time.perf_counter(), 0.0, None))

            producer = asyncio.create_task(delayed_result())
            heartbeat = 0

            async def count_heartbeat():
                nonlocal heartbeat
                while not producer.done():
                    heartbeat += 1
                    await asyncio.sleep(0.001)

            beat = asyncio.create_task(count_heartbeat())
            record = await handoff.get_before(started + 0.1)
            await producer
            await beat
            return record, time.perf_counter() - started, heartbeat

        record, elapsed, heartbeat = asyncio.run(exercise())
        self.assertEqual(record[0], 0)
        self.assertGreaterEqual(elapsed, 0.02)
        self.assertGreater(heartbeat, 5)

    def test_candidate_wait_fails_only_at_actual_deadline(self):
        async def exercise():
            handoff = BoundedCandidateQueue(2)
            started = time.perf_counter()
            with self.assertRaises(asyncio.QueueEmpty):
                await handoff.get_before(started + 0.03)
            return time.perf_counter() - started, handoff.consumed_count

        elapsed, consumed = asyncio.run(exercise())
        self.assertGreaterEqual(elapsed, 0.025)
        self.assertEqual(consumed, 0)

    def test_timeout_does_not_consume_late_or_future_candidate(self):
        async def exercise():
            handoff = BoundedCandidateQueue(2)
            with self.assertRaises(asyncio.QueueEmpty):
                await handoff.get_before(time.perf_counter() + 0.005)
            await handoff.put((0, [], 0.0, 0.0, 0.0, None))
            first = await handoff.get_before(time.perf_counter() + 0.1)
            await handoff.put((1, [], 0.0, 0.0, 0.0, None))
            second = await handoff.get_before(time.perf_counter() + 0.1)
            return first[0], second[0]

        self.assertEqual(asyncio.run(exercise()), (0, 1))

    def test_ready_queue_succeeds_with_zero_remaining_timeout(self):
        async def exercise():
            handoff = BoundedCandidateQueue(3)
            await handoff.put((0, [], 0.0, 0.0, 0.0, None))
            return await handoff.get_before(time.perf_counter())

        self.assertEqual(asyncio.run(exercise())[0], 0)

    def test_empty_queue_fails_with_zero_remaining_timeout(self):
        async def exercise():
            handoff = BoundedCandidateQueue(3)
            with self.assertRaises(asyncio.QueueEmpty):
                await handoff.get_before(time.perf_counter())

        asyncio.run(exercise())

    def test_deadline_boundary_arrival_succeeds_on_final_check(self):
        async def exercise():
            handoff = BoundedCandidateQueue(3)
            deadline = time.perf_counter() + 0.02
            collector = threading.Thread(
                target=lambda: (
                    time.sleep(0.019),
                    handoff.put_from_collector(
                        (0, [], 0.0, time.perf_counter(), 0.0, None)
                    ),
                )
            )
            collector.start()
            result = await handoff.get_before(deadline)
            collector.join()
            return result

        self.assertEqual(asyncio.run(exercise())[0], 0)

    def test_collector_result_survives_event_loop_stall_past_deadline(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            deadline = time.perf_counter() + 0.02
            collector = threading.Thread(
                target=lambda: (
                    time.sleep(0.005),
                    handoff.put_from_collector(
                        (0, [], 0.0, time.perf_counter(), 0.0, None)
                    ),
                )
            )
            collector.start()
            # Deliberately stall asyncio beyond the media deadline. The native
            # collector still exposes the completed record.
            time.sleep(0.04)
            result = await handoff.get_before(deadline)
            collector.join()
            return result

        self.assertEqual(asyncio.run(exercise())[0], 0)

    def test_deadline_get_preserves_strict_order(self):
        async def exercise():
            handoff = BoundedCandidateQueue(3)
            await handoff.put((0, [], 0.0, 0.0, 0.0, None))
            await handoff.put((1, [], 0.0, 0.0, 0.0, None))
            first = await handoff.get_before(time.perf_counter())
            second = await handoff.get_before(time.perf_counter())
            return first[0], second[0], handoff.consumed_count

        self.assertEqual(asyncio.run(exercise()), (0, 1, 2))

    def test_pacing_waits_for_all_five_primed_frames(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            primed = asyncio.Event()
            for fid in range(4):
                await handoff.put((fid, [], 0.0, 0.0, 0.0, None))
            gate = asyncio.create_task(
                wait_for_candidate_priming(handoff, primed, required=5)
            )
            await asyncio.sleep(0.01)
            waiting_after_four = not gate.done()
            await handoff.put((4, [], 0.0, 0.0, 0.0, None))
            primed.set()
            origin = await gate
            return waiting_after_four, origin, handoff.qsize(), handoff.max_frames

        waiting, origin, queued, capacity = asyncio.run(exercise())
        self.assertTrue(waiting)
        self.assertGreater(origin, 0)
        self.assertEqual(queued, 5)
        self.assertEqual(capacity, 5)

    def test_slow_startup_finishes_before_pacing_origin(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            primed = asyncio.Event()
            started = time.perf_counter()

            async def slow_prime():
                for fid in range(5):
                    await asyncio.sleep(0.015)
                    await handoff.put((fid, [], 0.0, 0.0, 0.0, None))
                primed.set()

            task = asyncio.create_task(slow_prime())
            origin = await wait_for_candidate_priming(
                handoff, primed, required=5
            )
            await task
            return started, origin

        started, origin = asyncio.run(exercise())
        self.assertGreaterEqual(origin - started, 0.065)

    def test_worker_failure_during_priming_fails_cleanly(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            primed = asyncio.Event()
            await handoff.put_error(RuntimeError("startup worker failed"))
            primed.set()
            with self.assertRaisesRegex(RuntimeError, "startup worker failed"):
                await wait_for_candidate_priming(
                    handoff, primed, required=5
                )

        asyncio.run(exercise())

    def test_pacing_waits_for_five_completed_candidates(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            handoff = BoundedCandidateQueue(5)
            primed = asyncio.Event()
            try:
                await manager.submit_frame(0, 25, frame(0), frame(0))
                await manager.submit_frame(1, 25, frame(1), frame(1))
                first, _ = await manager.wait_result(0)
                second, _ = await manager.wait_result(1)
                await handoff.put((0, [first], 0.0, 0.0, 0.0, None))
                await handoff.put((1, [second], 0.0, 0.0, 0.0, None))
                gate = asyncio.create_task(
                    wait_for_candidate_priming(handoff, primed, required=5)
                )
                last_record = None
                for fid in range(2, 5):
                    await manager.submit_frame(fid, 25, frame(fid), frame(fid))
                    candidate, last_record = await manager.wait_result(fid)
                    self.assertFalse(gate.done())
                    await handoff.put((fid, [candidate], 0.0, 0.0, 0.0, None))
                primed.set()
                origin = await gate
                return last_record, origin, handoff.qsize(), manager
            finally:
                await manager.aclose()

        fifth, origin, queued, manager = asyncio.run(exercise())
        self.assertEqual(fifth["frame_id"], 4)
        self.assertLessEqual(fifth["worker_completed"], origin)
        self.assertEqual(queued, 5)
        self.assertEqual(manager.max_inflight, 2)
        self.assertEqual(manager.completed_lead, 5)
        self.assertEqual(len(manager._shms), 0)

    def test_pacing_stays_unset_with_only_four_completed_candidates(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            primed = asyncio.Event()
            for fid in range(4):
                await handoff.put((fid, [], 0.0, 0.0, 0.0, None))
            gate = asyncio.create_task(
                wait_for_candidate_priming(handoff, primed, required=5)
            )
            await asyncio.sleep(0.01)
            waiting = not gate.done()
            gate.cancel()
            await asyncio.gather(gate, return_exceptions=True)
            return waiting

        self.assertTrue(asyncio.run(exercise()))

    def test_incomplete_request_bound_remains_two_with_five_completed_lead(self):
        async def exercise():
            manager = await start_manager(slow_codec_factory)
            try:
                await manager.submit_frame(0, 25, frame(), frame())
                await manager.submit_frame(1, 25, frame(), frame())
                third_submit = asyncio.create_task(
                    manager.submit_frame(2, 25, frame(), frame())
                )
                await asyncio.sleep(0.01)
                blocked = not third_submit.done()
                await manager.wait_result(0)
                await third_submit
                await manager.wait_result(1)
                await manager.wait_result(2)
                return blocked, manager._requests._maxsize, manager._results._maxsize
            finally:
                await manager.aclose()

        blocked, request_bound, result_bound = asyncio.run(exercise())
        self.assertTrue(blocked)
        self.assertEqual(request_bound, 2)
        self.assertEqual(result_bound, 2)

    def test_five_frame_lead_absorbs_one_long_production_spike_in_order(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            record = lambda fid: (fid, [], 0.0, 0.0, 0.0, None)
            for fid in range(5):
                await handoff.put(record(fid))

            async def delayed_sixth():
                # Longer than one 25-FPS frame period; existing completed
                # candidates continue to be consumed while production stalls.
                await asyncio.sleep(0.05)
                await handoff.put(record(5))

            producer = asyncio.create_task(delayed_sixth())
            consumed = []
            for _ in range(4):
                consumed.append((await handoff.get())[0])
                await asyncio.sleep(0.01)
            consumed.append((await handoff.get())[0])
            await producer
            consumed.append((await handoff.get())[0])
            return consumed, handoff.max_frames

        consumed, capacity = asyncio.run(exercise())
        self.assertEqual(consumed, list(range(6)))
        self.assertEqual(capacity, 5)

    def test_sender_refill_bound_is_completed_lead_plus_inflight(self):
        sender_source = (
            __import__("pathlib").Path(ROOT) / "src" / "sender" / "sender-3d.py"
        ).read_text()
        self.assertIn(
            "self.quality_manager.completed_lead\n"
            "                            + self.quality_manager.max_inflight",
            sender_source,
        )

        completed_lead = 5
        max_inflight = 2
        consumed = 0
        collected = 5
        submitted = 7
        self.assertEqual(collected - consumed, completed_lead)
        self.assertEqual(submitted - collected, max_inflight)
        self.assertEqual(submitted - consumed, 7)

    def test_refill_restores_reserve_and_lookahead_after_consumption(self):
        completed_lead = 5
        max_inflight = 2
        limit = completed_lead + max_inflight
        consumed = 0
        collected = 5
        submitted = 7

        # Consuming one completed frame creates one unit of total-work room;
        # the producer can submit immediately even though two older results
        # are still outside the completed handoff queue.
        consumed += 1
        self.assertLess(submitted - consumed, limit)
        submitted += 1
        self.assertEqual(submitted - consumed, limit)
        self.assertEqual(submitted - collected, 3)

        # The oldest lookahead completion refills the five-frame reserve. The
        # worker-facing semaphore independently keeps actual unfinished IPC at
        # two; this accounting represents ordered submitted/uncollected work.
        collected += 1
        self.assertEqual(collected - consumed, completed_lead)
        self.assertLessEqual(submitted - consumed, limit)

    def test_refill_model_never_exceeds_queue_or_total_work_bounds(self):
        completed_lead = 5
        max_inflight = 2
        limit = completed_lead + max_inflight
        consumed = collected = submitted = 0
        completed_order = []

        for frame_id in range(40):
            while submitted < 40 and submitted - consumed < limit:
                submitted += 1
            while collected < submitted and collected - consumed < completed_lead:
                completed_order.append(collected)
                collected += 1
            self.assertLessEqual(submitted - consumed, limit)
            self.assertLessEqual(collected - consumed, completed_lead)
            self.assertLessEqual(submitted - collected, max_inflight)
            self.assertEqual(completed_order[consumed], frame_id)
            consumed += 1

        self.assertEqual(completed_order, list(range(40)))

    def test_source_prefetch_overlaps_next_load_with_current_encode(self):
        second_load_started = threading.Event()
        release_second_load = threading.Event()

        def loader(frame_id):
            if frame_id == 1:
                second_load_started.set()
                release_second_load.wait(timeout=1.0)
            return frame(frame_id), frame(frame_id)

        async def exercise():
            queue = asyncio.Queue(maxsize=2)
            task = asyncio.create_task(prefetch_source_pairs(2, loader, queue))
            first, _ = await get_ordered_source_pair(queue, 0)
            # Model frame 0 native encoding while the independent prefetch task
            # is already loading frame 1.
            observed_overlap = await asyncio.to_thread(
                second_load_started.wait, 0.5
            )
            release_second_load.set()
            second, _ = await get_ordered_source_pair(queue, 1)
            await task
            return first.frame_id, second.frame_id, observed_overlap

        self.assertEqual(asyncio.run(exercise()), (0, 1, True))

    def test_prefetched_next_request_is_ready_after_prior_result(self):
        async def exercise():
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(
                prefetch_source_pairs(
                    3, lambda fid: (frame(fid), frame(fid)), queue
                )
            )
            first, _ = await get_ordered_source_pair(queue, 0)
            while queue.qsize() < 2:
                await asyncio.sleep(0)
            second, waited = await get_ordered_source_pair(queue, 1)
            third, _ = await get_ordered_source_pair(queue, 2)
            await task
            return first.frame_id, second.frame_id, third.frame_id, waited

        first, second, third, waited = asyncio.run(exercise())
        self.assertEqual((first, second, third), (0, 1, 2))
        self.assertLess(waited, 0.01)

    def test_raw_prefetch_queue_capacity_is_three(self):
        loaded = []

        def loader(frame_id):
            loaded.append(frame_id)
            return frame(frame_id), frame(frame_id)

        async def exercise():
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(prefetch_source_pairs(20, loader, queue))
            while len(loaded) < 4:
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            state = queue.qsize(), queue.maxsize, task.done()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            return state, task.done()

        state, cleaned = asyncio.run(exercise())
        self.assertEqual(state, (3, 3, False))
        self.assertTrue(cleaned)

    def test_source_prefetch_order_and_decode_failure_propagate(self):
        def loader(frame_id):
            if frame_id == 2:
                raise RuntimeError("source decode failed")
            return frame(frame_id), frame(frame_id)

        async def exercise():
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(prefetch_source_pairs(5, loader, queue))
            first, _ = await get_ordered_source_pair(queue, 0)
            second, _ = await get_ordered_source_pair(queue, 1)
            with self.assertRaisesRegex(RuntimeError, "source decode failed"):
                await get_ordered_source_pair(queue, 2)
            await task
            return first.frame_id, second.frame_id

        self.assertEqual(asyncio.run(exercise()), (0, 1))

    def test_source_prefetch_rejects_unexpected_frame_id(self):
        async def exercise():
            queue = asyncio.Queue(maxsize=3)
            await queue.put(SourceFrameRecord(
                1, frame(1), frame(1), 1.0, 2.0
            ))
            with self.assertRaisesRegex(RuntimeError, "expected 0, got 1"):
                await get_ordered_source_pair(queue, 0)

        asyncio.run(exercise())

    def test_consumption_notification_cannot_be_lost_before_wait(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            refill = CandidateRefillCondition()
            for fid in range(5):
                await handoff.put((fid, [], 0.0, 0.0, 0.0, None))
            await handoff.get()
            # Notify before the producer begins waiting. The predicate is true,
            # so no edge-triggered wakeup is required or lost.
            await refill.notify_consumed()
            await asyncio.wait_for(
                refill.wait_for_room(7, handoff, 7), timeout=0.05
            )
            return handoff.consumed_count

        self.assertEqual(asyncio.run(exercise()), 1)

    def test_encoder_failure_cleanup_cancels_prefetch_task(self):
        async def exercise():
            queue = asyncio.Queue(maxsize=3)
            task = asyncio.create_task(prefetch_source_pairs(
                100, lambda fid: (frame(fid), frame(fid)), queue
            ))
            try:
                raise RuntimeError("encoder failed")
            except RuntimeError:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            return task.done(), task.cancelled()

        self.assertEqual(asyncio.run(exercise()), (True, True))

    def test_worker_failure_before_five_frame_priming_fails_cleanly(self):
        async def exercise():
            manager = await start_manager(failing_codec_factory)
            try:
                await manager.submit_frame(0, 25, frame(), frame())
                await manager.submit_frame(1, 25, frame(), frame())
                await manager.wait_result(0)
                with self.assertRaisesRegex(RuntimeError, "intentional worker failure"):
                    await manager.wait_result(1)
            finally:
                await manager.terminate_on_error()

        asyncio.run(exercise())

    def test_strict_candidate_miss_remains_fail_fast(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            with self.assertRaises(asyncio.QueueEmpty):
                await handoff.get_before(time.perf_counter())
        asyncio.run(exercise())

    def test_diagnostic_late_result_is_discarded_in_order(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            missing, initially_discarded = await handoff.get_frame_before(
                0, time.perf_counter()
            )
            self.assertIsNone(missing)
            self.assertEqual(initially_discarded, [])
            await handoff.put((0, ["late-zero"], 0, 1, 0, None))
            await handoff.put((1, ["one"], 1, 2, 0, None))
            current, discarded = await handoff.get_frame_before(
                1, time.perf_counter() + 0.1
            )
            return current, discarded, handoff.consumed_count
        current, discarded, consumed = asyncio.run(exercise())
        self.assertEqual(current[0], 1)
        self.assertEqual([record[0] for record in discarded], [0])
        self.assertEqual(consumed, 2)

    def test_diagnostic_multiple_misses_do_not_fake_frames(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            missed = []
            sent = []
            for fid in (0, 1):
                current, _ = await handoff.get_frame_before(
                    fid, time.perf_counter()
                )
                if current is None:
                    missed.append(fid)
            await handoff.put((0, ["late-zero"], 0, 1, 0, None))
            await handoff.put((1, ["late-one"], 1, 2, 0, None))
            await handoff.put((2, ["real-two"], 2, 3, 0, None))
            current, discarded = await handoff.get_frame_before(
                2, time.perf_counter() + 0.1
            )
            sent.extend(current[1])
            return missed, [r[0] for r in discarded], sent
        missed, discarded, sent = asyncio.run(exercise())
        self.assertEqual(missed, [0, 1])
        self.assertEqual(discarded, [0, 1])
        self.assertEqual(sent, ["real-two"])
        self.assertNotIn("late-zero", sent)
        self.assertNotIn("late-one", sent)

    def test_diagnostic_returns_drained_late_records_on_another_miss(self):
        async def exercise():
            handoff = BoundedCandidateQueue(5)
            await handoff.put((0, ["late-zero"], 0, 1, 0, None))
            current, discarded = await handoff.get_frame_before(
                1, time.perf_counter()
            )
            return current, discarded, handoff.consumed_count
        current, discarded, consumed = asyncio.run(exercise())
        self.assertIsNone(current)
        self.assertEqual([record[0] for record in discarded], [0])
        self.assertEqual(consumed, 1)


if __name__ == "__main__":
    unittest.main()
