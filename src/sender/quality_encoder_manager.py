"""Dual-quality H.265 candidate encoding and atomic RGB/depth selection."""

from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing
from multiprocessing import shared_memory
import os
import queue
import resource
import threading
import time
import traceback
from typing import Any, Optional

import numpy as np

from adaptation_controller import Quality


class CausalEncodeLead:
    """Predicts the next dual-candidate encode cost from completed encodes only."""

    def __init__(self, alpha: float = 0.2):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.ewma_s: Optional[float] = None

    def predict(self, frame_interval_s: float) -> float:
        """Return a bounded causal lead; never consumes the current encode."""
        if frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")
        if self.ewma_s is None:
            return 0.0
        return min(float(frame_interval_s), max(0.0, self.ewma_s))

    def observe(self, completed_encode_s: float) -> float:
        completed_encode_s = float(completed_encode_s)
        if completed_encode_s < 0:
            raise ValueError("completed_encode_s must be nonnegative")
        self.ewma_s = (
            completed_encode_s
            if self.ewma_s is None
            else self.alpha * completed_encode_s
            + (1.0 - self.alpha) * self.ewma_s
        )
        return self.ewma_s


def packet_pacing_interval(
    send_budget_s: float, packet_count: int, *, is_keyframe: bool
) -> float:
    """Urgent keyframe FEC is enqueued immediately; P-frames retain pacing."""
    if packet_count <= 0:
        raise ValueError("packet_count must be positive")
    if is_keyframe:
        return 0.0
    return max(0.0, float(send_budget_s)) / int(packet_count)


def sender_frame_deadline(
    origin_s: float,
    frame_id: int,
    frame_period_s: float,
    causal_encode_lead_s: float,
) -> float:
    """Track B enqueue deadline using only a prior causal encode prediction."""
    if frame_id < 0:
        raise ValueError("frame_id must be nonnegative")
    if frame_period_s <= 0:
        raise ValueError("frame_period_s must be positive")
    return (
        float(origin_s)
        + int(frame_id) * float(frame_period_s)
        - max(0.0, float(causal_encode_lead_s))
    )


def application_bytes_for_payload(
    payload_size: int, chunk_size: int, is_keyframe: bool, descriptor_size: int
) -> int:
    """Application bytes offered by ReVo chunking, FEC, and P-header retry."""
    payload_size = int(payload_size)
    chunk_size = int(chunk_size)
    descriptor_size = int(descriptor_size)
    chunks = max(1, (payload_size + chunk_size - 1) // chunk_size)
    if is_keyframe:
        total_chunks = chunks + (chunks + 1) // 2
        shard_size = (payload_size + chunks - 1) // chunks
        return total_chunks * (descriptor_size + shard_size)
    first_size = min(chunk_size, payload_size)
    return payload_size + chunks * descriptor_size + first_size + descriptor_size


@dataclass(frozen=True)
class EncodedCandidate:
    frame_id: int
    is_keyframe: bool
    rgb: dict
    depth: dict
    rgb_bytes: int
    depth_bytes: int
    rgb_encode_s: float
    depth_encode_s: float


@dataclass(frozen=True)
class CandidateSet:
    frame_id: int
    high: Optional[EncodedCandidate]
    low: Optional[EncodedCandidate]
    low_usable: bool
    fallback_reason: str = ""


class BoundedCandidateQueue:
    """Frame-ordered bounded handoff between candidate production and sending."""

    def __init__(self, max_frames: int, collector_buffer_frames: int = 2):
        if int(max_frames) <= 0:
            raise ValueError("max_frames must be positive")
        if int(collector_buffer_frames) < 0:
            raise ValueError("collector_buffer_frames must be nonnegative")
        # A native Queue is intentional: the encoder IPC collector owns the
        # production side, while asyncio owns consumption. Completed x265 work
        # must not wait for a delayed event-loop callback before becoming
        # visible to the paced sender.
        self._queue = queue.Queue(maxsize=int(max_frames))
        self._collector_buffer = deque()
        self._collector_buffer_max = int(collector_buffer_frames)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._next_produced = 0
        self._next_consumed = 0

    @property
    def max_frames(self):
        return self._queue.maxsize

    def qsize(self):
        return self._queue.qsize()

    def collector_buffer_qsize(self):
        with self._lock:
            return len(self._collector_buffer)

    @property
    def collector_buffer_max_frames(self):
        return self._collector_buffer_max

    @property
    def consumed_count(self):
        return self._next_consumed

    async def put(self, record):
        await asyncio.to_thread(self.put_from_collector, record)

    def put_from_collector(self, record):
        if not self._put_from_collector_nowait(record):
            raise queue.Full("completed candidate collector buffer is full")

    def put_from_collector_until(self, record, stop_event):
        if stop_event.is_set():
            return False
        return self._put_from_collector_nowait(record)

    def _put_from_collector_nowait(self, record):
        frame_id = int(record[0])
        with self._changed:
            if frame_id != self._next_produced:
                raise RuntimeError(
                    "candidate production order mismatch: "
                    f"expected {self._next_produced}, got {frame_id}"
                )
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                if len(self._collector_buffer) >= self._collector_buffer_max:
                    return False
                self._collector_buffer.append(record)
            self._next_produced += 1
            self._changed.notify_all()
            return True

    async def put_error(self, error: BaseException):
        await asyncio.to_thread(
            self._queue.put, (-1, [], 0.0, 0.0, 0.0, error)
        )

    def put_error_from_collector(self, error: BaseException):
        record = (-1, [], 0.0, 0.0, 0.0, error)
        with self._changed:
            try:
                self._queue.put_nowait(record)
            except queue.Full:
                if len(self._collector_buffer) < self._collector_buffer_max:
                    self._collector_buffer.append(record)
                else:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        pass
                    self._queue.put_nowait(record)
            self._changed.notify_all()

    async def wait_until_size(self, required):
        required = int(required)
        await asyncio.to_thread(self._wait_until_size_blocking, required)

    def _wait_until_size_blocking(self, required):
        with self._changed:
            self._changed.wait_for(
                lambda: self._queue.qsize() >= required or self._failure_queued_locked()
            )

    def _failure_queued(self):
        with self._changed:
            return self._failure_queued_locked()

    def _failure_queued_locked(self):
        with self._queue.mutex:
            if any(record[-1] is not None for record in self._queue.queue):
                return True
        return any(record[-1] is not None for record in self._collector_buffer)

    def _drain_collector_buffer_locked(self):
        moved = False
        while self._collector_buffer and not self._queue.full():
            self._queue.put_nowait(self._collector_buffer.popleft())
            moved = True
        if moved:
            self._changed.notify_all()

    async def get(self):
        return await asyncio.to_thread(self._get_blocking)

    def _get_blocking(self):
        with self._changed:
            self._changed.wait_for(lambda: self._queue.qsize() > 0)
            record = self._queue.get_nowait()
            return self._consume_locked(record)

    def get_nowait(self):
        with self._changed:
            self._drain_collector_buffer_locked()
            try:
                record = self._queue.get_nowait()
            except queue.Empty as exc:
                raise asyncio.QueueEmpty from exc
            return self._consume_locked(record)

    def _consume_locked(self, record):
        if record[-1] is not None:
            self._drain_collector_buffer_locked()
            return record
        frame_id = int(record[0])
        if frame_id != self._next_consumed:
            raise RuntimeError(
                "candidate consumption order mismatch: "
                f"expected {self._next_consumed}, got {frame_id}"
            )
        self._next_consumed += 1
        self._drain_collector_buffer_locked()
        return record

    async def get_before(self, deadline_monotonic):
        """Wait asynchronously for the next ordered candidate until its deadline."""
        try:
            return self.get_nowait()
        except asyncio.QueueEmpty:
            pass

        timeout = float(deadline_monotonic) - time.perf_counter()
        if timeout <= 0.0:
            # Close the check/use race at an already-reached deadline.
            return self.get_nowait()

        try:
            return await asyncio.to_thread(self._get_with_timeout, timeout)
        except queue.Empty:
            # A result can arrive as wait_for cancels its waiter at the exact
            # boundary. Give that already-produced record one ordered dequeue.
            return self.get_nowait()

    def _get_with_timeout(self, timeout):
        deadline = time.perf_counter() + max(0.0, float(timeout))
        with self._changed:
            while self._queue.qsize() <= 0:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise queue.Empty
                self._changed.wait(remaining)
            record = self._queue.get_nowait()
            return self._consume_locked(record)

    async def get_frame_before(self, expected_frame_id, deadline_monotonic):
        """Return one frame while draining already-late ordered results.

        The returned ``discarded`` records have frame IDs below
        ``expected_frame_id``. This is used only by diagnostic continue-on-miss
        runs; strict validation continues to call :meth:`get_before` directly.
        """
        discarded = []
        while True:
            try:
                record = await self.get_before(deadline_monotonic)
            except (asyncio.QueueEmpty, asyncio.TimeoutError):
                return None, discarded
            if record[-1] is not None:
                return record, discarded
            frame_id = int(record[0])
            if frame_id < int(expected_frame_id):
                discarded.append(record)
                continue
            if frame_id != int(expected_frame_id):
                raise RuntimeError(
                    "candidate diagnostic order mismatch: "
                    f"expected at most {expected_frame_id}, got {frame_id}"
                )
            return record, discarded


@dataclass(frozen=True)
class SourceFrameRecord:
    frame_id: int
    rgb: object
    depth: object
    load_started: float
    load_completed: float
    error: BaseException | None = None


async def prefetch_source_pairs(
    frame_count, load_source_pair, output_queue, *, start_frame_id=0
):
    """Load source pairs in order without sharing candidate-production work."""
    for frame_id in range(int(start_frame_id), int(frame_count)):
        started = time.perf_counter()
        try:
            rgb, depth = await asyncio.to_thread(load_source_pair, frame_id)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            await _put_source_record(output_queue, SourceFrameRecord(
                frame_id, None, None, started, time.perf_counter(), exc
            ))
            return
        await _put_source_record(output_queue, SourceFrameRecord(
            frame_id, rgb, depth, started, time.perf_counter()
        ))


async def _put_source_record(output_queue, record):
    if isinstance(output_queue, queue.Queue):
        await asyncio.to_thread(output_queue.put, record)
    else:
        await output_queue.put(record)


async def get_ordered_source_pair(input_queue, expected_frame_id):
    """Return one prefetched pair or fail closed on decode/order errors."""
    wait_started = time.perf_counter()
    if isinstance(input_queue, queue.Queue):
        record = await asyncio.to_thread(input_queue.get)
    else:
        record = await input_queue.get()
    waited = time.perf_counter() - wait_started
    if record.error is not None:
        raise record.error
    if int(record.frame_id) != int(expected_frame_id):
        raise RuntimeError(
            "source prefetch order mismatch: "
            f"expected {expected_frame_id}, got {record.frame_id}"
        )
    return record, waited


class CandidateRefillCondition:
    """Predicate-based producer wakeup; consumption notifications cannot be lost."""

    def __init__(self):
        self._condition = threading.Condition()

    async def wait_for_room(self, submitted, candidate_queue, total_limit):
        await asyncio.to_thread(
            self.wait_for_room_blocking,
            submitted, candidate_queue, total_limit, None,
        )

    def wait_for_room_blocking(
        self, submitted, candidate_queue, total_limit, stop_event=None
    ):
        with self._condition:
            self._condition.wait_for(lambda: (
                int(submitted) - candidate_queue.consumed_count < int(total_limit)
                or (stop_event is not None and stop_event.is_set())
            ))
            return stop_event is None or not stop_event.is_set()

    async def notify_consumed(self):
        self.notify_consumed_sync()

    def notify_consumed_sync(self):
        with self._condition:
            self._condition.notify_all()


async def wait_for_candidate_priming(candidate_queue, primed_event, required=2):
    """Set the pacing origin only after the bounded startup lead is complete."""
    if primed_event is None:
        await candidate_queue.wait_until_size(required)
    else:
        await primed_event.wait()
    if candidate_queue.qsize() < int(required):
        # Producer failures also wake the gate, preventing a startup hang.
        try:
            record = candidate_queue.get_nowait()
        except asyncio.QueueEmpty as exc:
            raise RuntimeError(
                "candidate priming ended early: "
                f"required={required} queued={candidate_queue.qsize()}"
            ) from exc
        if record[-1] is not None:
            raise record[-1]
        raise RuntimeError(
            "candidate priming ended early: "
            f"required={required} queued={candidate_queue.qsize()}"
        )
    return time.perf_counter()


def partition_track_b_cpus(available, sender_reserved=2):
    cpus = sorted(int(cpu) for cpu in available)
    if len(cpus) < 4:
        raise RuntimeError(
            "Track B encoder process isolation requires at least 4 CPUs; "
            f"available={cpus}"
        )
    reserved = max(2, int(sender_reserved))
    if reserved >= len(cpus):
        raise RuntimeError("sender CPU reservation leaves no encoder CPUs")
    sender_cpus = tuple(cpus[:reserved])
    worker_cpus = tuple(cpus[reserved:])
    if len(worker_cpus) < 4:
        raise RuntimeError(
            "Track B concurrency=4 requires at least 4 encoder-worker CPUs; "
            f"sender={sender_cpus} worker={worker_cpus}"
        )
    return sender_cpus, worker_cpus


def _make_h265_codec(qp, intra_period, pool_threads):
    from H265_wrapper import H265VideoCodec
    return H265VideoCodec(
        qp=int(qp), intra_period=int(intra_period),
        pool_threads=pool_threads,
    )


def _encode_candidate(codec, array, frame_id, fps, active, lock):
    with lock:
        active["current"] += 1
        active["maximum"] = max(active["maximum"], active["current"])
        if active["shared"] is not None:
            active["shared"].value = active["current"]
    started = time.perf_counter()
    try:
        outputs = list(codec.compress_prepared(array, frame_id, fps=fps))
        return outputs, time.perf_counter() - started
    finally:
        with lock:
            active["current"] -= 1
            if active["shared"] is not None:
                active["shared"].value = active["current"]


def _encoder_worker_main(config, requests, results, shm_specs):
    shms = []
    executor = None
    try:
        affinity = tuple(config.get("worker_affinity", ()))
        if affinity:
            os.sched_setaffinity(0, set(affinity))
        factory = config.get("codec_factory") or _make_h265_codec
        codecs = {
            "high_rgb": factory(config["high_qp"], config["intra_period"], config["pool_threads"]),
            "low_rgb": factory(config["low_qp"], config["intra_period"], config["pool_threads"]),
            "high_depth": factory(config["high_qp"], config["intra_period"], config["pool_threads"]),
            "low_depth": factory(config["low_qp"], config["intra_period"], config["pool_threads"]),
        }
        for spec in shm_specs:
            shms.append((
                shared_memory.SharedMemory(name=spec["rgb_name"]),
                shared_memory.SharedMemory(name=spec["depth_name"]),
            ))
        executor = ThreadPoolExecutor(
            max_workers=int(config["max_native_concurrency"]),
            thread_name_prefix="revo-x265",
        )
        results.put({
            "type": "ready",
            "worker_pid": os.getpid(),
            "worker_affinity": (
                list(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else []
            ),
        })
        expected = 0
        pending = {}
        previous_completed = None
        active = {
            "current": 0,
            "maximum": 0,
            "shared": config["active_native_shared"],
        }
        active_lock = threading.Lock()
        while True:
            request = requests.get()
            if request.get("type") == "shutdown":
                results.put({"type": "shutdown", "worker_pid": os.getpid()})
                return
            frame_id = int(request["frame_id"])
            if frame_id != expected:
                raise RuntimeError(
                    f"encoder request order mismatch: expected {expected}, got {frame_id}"
                )
            expected += 1
            slot = int(request["slot"])
            spec = shm_specs[slot]
            rgb = np.ndarray(
                tuple(request["rgb_shape"]), dtype=np.dtype(request["rgb_dtype"]),
                buffer=shms[slot][0].buf,
            )
            depth = np.ndarray(
                tuple(request["depth_shape"]), dtype=np.dtype(request["depth_dtype"]),
                buffer=shms[slot][1].buf,
            )
            worker_started = time.perf_counter()
            usage_started = resource.getrusage(resource.RUSAGE_SELF)
            cpu_started = usage_started.ru_utime + usage_started.ru_stime
            worker_idle_s = (
                0.0
                if previous_completed is None
                else max(0.0, worker_started - previous_completed)
            )
            futures = {
                label: executor.submit(
                    _encode_candidate, codec,
                    rgb if "rgb" in label else depth,
                    frame_id, int(request["fps"]), active, active_lock,
                )
                for label, codec in codecs.items()
            }
            durations = {}
            for label, future in futures.items():
                outputs, duration = future.result()
                durations[label] = duration
                for output in outputs:
                    pending[(label, int(output["frame_id"]))] = output
            high_rgb = pending.pop(("high_rgb", frame_id), None)
            high_depth = pending.pop(("high_depth", frame_id), None)
            low_rgb = pending.pop(("low_rgb", frame_id), None)
            low_depth = pending.pop(("low_depth", frame_id), None)
            if high_rgb is None or high_depth is None:
                raise RuntimeError(
                    f"high candidate missing for frame {frame_id}"
                )
            low_usable = (
                low_rgb is not None and low_depth is not None
                and bool(low_rgb["is_key"]) == bool(low_depth["is_key"])
                and int(low_rgb["frame_id"]) == int(low_depth["frame_id"])
            )
            if not low_usable:
                low_rgb = low_depth = None
            completed = time.perf_counter()
            usage_completed = resource.getrusage(resource.RUSAGE_SELF)
            previous_completed = completed
            results.put({
                "type": "result", "frame_id": frame_id, "slot": slot,
                "worker_pid": os.getpid(),
                "worker_affinity": list(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
                "worker_started": worker_started,
                "worker_completed": completed,
                "worker_idle_s": worker_idle_s,
                "worker_cpu_s": (
                    usage_completed.ru_utime + usage_completed.ru_stime
                    - cpu_started
                ),
                "worker_voluntary_context_switches": (
                    usage_completed.ru_nvcsw - usage_started.ru_nvcsw
                ),
                "worker_involuntary_context_switches": (
                    usage_completed.ru_nivcsw - usage_started.ru_nivcsw
                ),
                "total_production_s": completed - worker_started,
                "active_native_max": active["maximum"],
                "high_rgb": high_rgb, "high_depth": high_depth,
                "low_rgb": low_rgb, "low_depth": low_depth,
                "durations": durations,
                "fallback_reason": "" if low_usable else "low_candidate_missing_or_mismatched",
            })
            active["maximum"] = 0
    except BaseException as exc:
        try:
            results.put({
                "type": "error", "worker_pid": os.getpid(),
                "error": repr(exc), "traceback": traceback.format_exc(),
            })
        finally:
            raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for rgb_shm, depth_shm in shms:
            rgb_shm.close()
            depth_shm.close()


class QualityEncoderManager:
    def __init__(
        self,
        high_qp=20, low_qp=30, intra_period=30, pool_threads=1, *,
        max_inflight=2, completed_lead=5, sender_reserved_cpus=2,
        enable_affinity=True, codec_factory=None,
        diagnostic_live_worker_state=False,
    ):
        if int(max_inflight) != 2 or int(completed_lead) != 5:
            raise ValueError(
                "Track B process isolation requires max_inflight=2 and "
                "completed_lead=5"
            )
        self.high_qp = int(high_qp)
        self.low_qp = int(low_qp)
        self.intra_period = int(intra_period)
        self.pool_threads = pool_threads
        self.max_inflight = int(max_inflight)
        self.completed_lead = int(completed_lead)
        self.sender_reserved_cpus = int(sender_reserved_cpus)
        self.enable_affinity = bool(enable_affinity)
        self.codec_factory = codec_factory
        self._force_high_until_keyframe = False
        self._ctx = multiprocessing.get_context("spawn")
        self._requests = self._ctx.Queue(maxsize=2)
        self._results = self._ctx.Queue(maxsize=2)
        self._process = None
        self._reader = None
        self._reader_stop = threading.Event()
        self._worker_ready = threading.Event()
        self._completed_handoff = None
        self._owns_completed_handoff = False
        self._producer_records = None
        self._producer_state = None
        self._shms = []
        self._free_slots = queue.Queue(maxsize=3)
        self._inflight = threading.BoundedSemaphore(self.max_inflight)
        self._next_submitted = 0
        self._next_received = 0
        self._next_consumed = 0
        self._failure = None
        self._shutdown_seen = False
        self._last_runtime = {}
        self._candidate_not_ready = 0
        self._worker_failures = 0
        self._active_native_shared = (
            self._ctx.Value("i", 0) if diagnostic_live_worker_state else None
        )
        self.sender_affinity = ()
        self.worker_affinity = ()
        self._rgb_nbytes = 0
        self._depth_nbytes = 0

    async def start(self, rgb_shape, depth_shape, rgb_dtype, depth_dtype):
        if self._process is not None:
            return
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError(
                "encoder worker must be started from the sender main thread"
            )
        if not hasattr(shared_memory, "SharedMemory"):
            raise RuntimeError("multiprocessing shared memory is unavailable")
        if self._completed_handoff is None:
            self._completed_handoff = BoundedCandidateQueue(self.completed_lead)
            self._producer_records = {}
            self._producer_state = {}
            self._owns_completed_handoff = True
        if self.enable_affinity:
            if not hasattr(os, "sched_getaffinity"):
                raise RuntimeError("CPU affinity is unavailable for Track B isolation")
            self.sender_affinity, self.worker_affinity = partition_track_b_cpus(
                os.sched_getaffinity(0), self.sender_reserved_cpus
            )
            os.sched_setaffinity(0, set(self.sender_affinity))
        rgb_nbytes = int(np.prod(rgb_shape)) * np.dtype(rgb_dtype).itemsize
        depth_nbytes = int(np.prod(depth_shape)) * np.dtype(depth_dtype).itemsize
        self._rgb_nbytes = rgb_nbytes
        self._depth_nbytes = depth_nbytes
        specs = []
        for slot in range(3):
            rgb_shm = shared_memory.SharedMemory(create=True, size=rgb_nbytes)
            depth_shm = shared_memory.SharedMemory(create=True, size=depth_nbytes)
            self._shms.append((rgb_shm, depth_shm))
            specs.append({"rgb_name": rgb_shm.name, "depth_name": depth_shm.name})
            self._free_slots.put_nowait(slot)
        config = {
            "high_qp": self.high_qp, "low_qp": self.low_qp,
            "intra_period": self.intra_period, "pool_threads": self.pool_threads,
            "max_native_concurrency": 4,
            "worker_affinity": self.worker_affinity,
            "codec_factory": self.codec_factory,
            "active_native_shared": self._active_native_shared,
        }
        self._process = self._ctx.Process(
            target=_encoder_worker_main,
            args=(config, self._requests, self._results, specs),
            name="revo-encoder-worker",
        )
        self._process.start()
        self._reader = threading.Thread(
            target=self._read_results, name="revo-encoder-ipc", daemon=True
        )
        self._reader.start()
        try:
            await asyncio.to_thread(self._wait_until_worker_ready, 15.0)
        except BaseException:
            await self.terminate_on_error()
            raise

    def _wait_until_worker_ready(self, timeout_s):
        if not self._worker_ready.wait(float(timeout_s)):
            raise RuntimeError("encoder worker startup timed out")
        if self._failure is not None:
            raise self._failure
        if self._process is None or not self._process.is_alive():
            exitcode = None if self._process is None else self._process.exitcode
            raise RuntimeError(
                f"encoder worker exited before startup completed: {exitcode}"
            )

    def bind_completed_handoff(self, handoff, producer_records, producer_state):
        """Bind the collector's bounded, ordered destination before start."""
        self._completed_handoff = handoff
        self._producer_records = producer_records
        self._producer_state = producer_state

    def _read_results(self):
        while not self._reader_stop.is_set():
            try:
                message = self._results.get(timeout=0.1)
            except queue.Empty:
                if self._process is not None and not self._process.is_alive():
                    self._handle_worker_death(self._process.exitcode)
                    return
                continue
            self._handle_message(message)
            if message.get("type") in ("shutdown", "error"):
                if message.get("type") == "shutdown":
                    # Prevent the process-exit poll from racing the shutdown
                    # callback queued onto the asyncio loop.
                    self._shutdown_seen = True
                return

    def _handle_worker_death(self, exitcode):
        if self._shutdown_seen:
            return
        self._fail_all(RuntimeError(f"encoder worker died with exit code {exitcode}"))

    def _fail_all(self, error):
        self._failure = error
        self._worker_failures += 1
        self._worker_ready.set()
        if self._completed_handoff is not None:
            self._completed_handoff.put_error_from_collector(error)

    def _handle_message(self, message):
        kind = message.get("type")
        if kind == "ready":
            self._last_runtime.update(message)
            self._worker_ready.set()
            return
        if kind == "shutdown":
            self._shutdown_seen = True
            return
        if kind == "error":
            self._fail_all(RuntimeError(
                f"encoder worker error: {message.get('error')}\n{message.get('traceback')}"
            ))
            return
        frame_id = int(message["frame_id"])
        if frame_id != self._next_received:
            self._fail_all(RuntimeError(
                f"encoder result order mismatch: expected {self._next_received}, got {frame_id}"
            ))
            return
        self._next_received += 1
        slot = int(message["slot"])
        self._free_slots.put_nowait(slot)
        self._inflight.release()
        message["result_received"] = time.perf_counter()
        self._last_runtime = message
        if self._completed_handoff is None:
            self._fail_all(RuntimeError("encoder completed handoff is not bound"))
            return
        record = (self._candidate_set(message), message)
        details = self._producer_records.get(frame_id, {})
        details.update(message)
        self._producer_records[frame_id] = details
        if not self._completed_handoff.put_from_collector_until((
            frame_id, [record[0]], message["worker_started"],
            message["worker_completed"], details.get("causal_lead", 0.0), None,
        ), self._reader_stop):
            self._fail_all(RuntimeError("completed candidate collector buffer is full"))
            return
        if self._producer_state is not None:
            self._producer_state["collected"] = self._next_received

    async def submit_frame(self, frame_id, fps, rgb, depth):
        return await asyncio.to_thread(
            self.submit_frame_blocking, frame_id, fps, rgb, depth
        )

    def submit_frame_blocking(self, frame_id, fps, rgb, depth):
        if self._failure is not None:
            raise self._failure
        frame_id = int(frame_id)
        if frame_id != self._next_submitted:
            raise RuntimeError(
                f"encoder submit order mismatch: expected {self._next_submitted}, got {frame_id}"
            )
        self._inflight.acquire()
        slot = self._free_slots.get()
        rgb = np.ascontiguousarray(rgb)
        depth = np.ascontiguousarray(depth)
        if rgb.nbytes != self._rgb_nbytes:
            self._free_slots.put_nowait(slot)
            self._inflight.release()
            raise ValueError(
                f"RGB frame size changed: expected {self._rgb_nbytes}, "
                f"got {rgb.nbytes}"
            )
        if depth.nbytes != self._depth_nbytes:
            self._free_slots.put_nowait(slot)
            self._inflight.release()
            raise ValueError(
                f"depth frame size changed: expected {self._depth_nbytes}, "
                f"got {depth.nbytes}"
            )
        np.ndarray(rgb.shape, rgb.dtype, buffer=self._shms[slot][0].buf)[:] = rgb
        np.ndarray(depth.shape, depth.dtype, buffer=self._shms[slot][1].buf)[:] = depth
        submitted = time.perf_counter()
        request = {
            "type": "frame", "frame_id": frame_id, "fps": int(fps),
            "slot": slot, "rgb_shape": rgb.shape, "depth_shape": depth.shape,
            "rgb_dtype": rgb.dtype.str, "depth_dtype": depth.dtype.str,
            "request_submitted": submitted,
        }
        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._requests.put(request)
        self._next_submitted += 1
        return submitted

    async def get_result(self, frame_id):
        frame_id = int(frame_id)
        try:
            record = self._completed_handoff.get_nowait()
        except queue.Empty as exc:
            self._candidate_not_ready += 1
            raise RuntimeError(
                f"candidate_not_ready frame={frame_id} {self.runtime_snapshot()}"
            ) from exc
        return self._unwrap_direct_result(frame_id, record)

    async def wait_result(self, frame_id):
        record = await self._completed_handoff.get()
        return self._unwrap_direct_result(int(frame_id), record)

    def _unwrap_direct_result(self, frame_id, record):
        if record[-1] is not None:
            raise record[-1]
        if int(record[0]) != int(frame_id):
            raise RuntimeError(
                f"encoder consume order mismatch: expected {frame_id}, got {record[0]}"
            )
        message = self._producer_records[int(frame_id)]
        return record[1][0], message

    @staticmethod
    def _candidate_set(message):
        durations = message["durations"]
        high = EncodedCandidate(
            frame_id=message["frame_id"],
            is_keyframe=bool(message["high_rgb"]["is_key"]),
            rgb=message["high_rgb"], depth=message["high_depth"],
            rgb_bytes=len(message["high_rgb"]["payload"]),
            depth_bytes=len(message["high_depth"]["payload"]),
            rgb_encode_s=durations["high_rgb"],
            depth_encode_s=durations["high_depth"],
        )
        low = None
        if message["low_rgb"] is not None and message["low_depth"] is not None:
            low = EncodedCandidate(
                frame_id=message["frame_id"],
                is_keyframe=bool(message["low_rgb"]["is_key"]),
                rgb=message["low_rgb"], depth=message["low_depth"],
                rgb_bytes=len(message["low_rgb"]["payload"]),
                depth_bytes=len(message["low_depth"]["payload"]),
                rgb_encode_s=durations["low_rgb"],
                depth_encode_s=durations["low_depth"],
            )
        return CandidateSet(
            frame_id=message["frame_id"], high=high, low=low,
            low_usable=low is not None,
            fallback_reason=message["fallback_reason"],
        )


    def runtime_snapshot(self):
        process = self._process
        return {
            "child_pid": "" if process is None else process.pid,
            "sender_cpu_affinity": ",".join(map(str, self.sender_affinity)),
            "worker_cpu_affinity": ",".join(map(str, self.worker_affinity)),
            "active_encoder_workers": (
                int(self._active_native_shared.value)
                if self._active_native_shared is not None
                else self._last_runtime.get("active_native_max", 0)
            ),
            "configured_encoder_workers": 4,
            "cancelled_candidate_jobs": 0,
            "candidate_not_ready_count": self._candidate_not_ready,
            "worker_failures": self._worker_failures,
            "worker_exit_code": "" if process is None else process.exitcode,
            "input_queue_depth": self._safe_qsize(self._requests),
            "output_queue_depth": self._safe_qsize(self._results),
            "collector_buffer_depth": (
                self._completed_handoff.collector_buffer_qsize()
                if self._completed_handoff is not None
                else 0
            ),
            "collector_buffer_max_frames": (
                self._completed_handoff.collector_buffer_max_frames
                if self._completed_handoff is not None
                else 0
            ),
            "unfinished_ipc_requests": max(
                0, self._next_submitted - self._next_received
            ),
        }

    @staticmethod
    def _safe_qsize(q):
        try:
            return q.qsize()
        except (NotImplementedError, AttributeError):
            return -1

    async def aclose(self):
        process = self._process
        if process is None:
            return 0
        if process.is_alive():
            try:
                await asyncio.to_thread(self._requests.put, {"type": "shutdown"}, True, 1.0)
            except queue.Full:
                process.terminate()
            await asyncio.to_thread(process.join, 5.0)
        if process.is_alive():
            process.terminate()
            await asyncio.to_thread(process.join, 2.0)
        exitcode = process.exitcode
        self._reader_stop.set()
        if self._reader is not None:
            await asyncio.to_thread(self._reader.join, 1.0)
        self._release_ipc_resources()
        if exitcode != 0:
            raise RuntimeError(f"encoder worker exit code {exitcode}")
        return exitcode

    async def terminate_on_error(self):
        if self._process is not None and self._process.is_alive():
            self._process.terminate()
            await asyncio.to_thread(self._process.join, 2.0)
        self._reader_stop.set()
        if self._reader is not None:
            await asyncio.to_thread(self._reader.join, 1.0)
        self._release_ipc_resources()

    def _release_ipc_resources(self):
        for rgb_shm, depth_shm in self._shms:
            rgb_shm.close()
            depth_shm.close()
            try:
                rgb_shm.unlink()
            except FileNotFoundError:
                pass
            try:
                depth_shm.unlink()
            except FileNotFoundError:
                pass
        self._shms.clear()
        for ipc_queue in (self._requests, self._results):
            try:
                ipc_queue.close()
                ipc_queue.join_thread()
            except (AttributeError, ValueError):
                pass

    def select(self, candidates: CandidateSet, quality: Quality) -> tuple[EncodedCandidate, str]:
        if candidates.high is None:
            raise ValueError("high-quality candidate pair is required")
        if (
            candidates.high.is_keyframe
            and candidates.low_usable
            and candidates.low is not None
        ):
            self._force_high_until_keyframe = False
        if self._force_high_until_keyframe and not candidates.high.is_keyframe:
            return candidates.high, "gop_locked_high_after_low_candidate_failure"
        if quality == Quality.LOW and candidates.low_usable and candidates.low:
            return candidates.low, ""
        if quality == Quality.LOW:
            self._force_high_until_keyframe = True
            return candidates.high, candidates.fallback_reason
        return candidates.high, ""


class PreencodedQualityEncoderManager(QualityEncoderManager):
    """Diagnostic-only manager replaying retained candidates without codecs."""

    def __init__(self, manifest_path, **kwargs):
        import pickle

        kwargs.pop("enable_affinity", None)
        super().__init__(enable_affinity=False, **kwargs)
        with open(manifest_path, "rb") as handle:
            manifest = pickle.load(handle)
        self._preencoded = list(manifest["candidates"])
        self._preencoded_records = list(manifest["records"])
        self._started = False

    async def start(self, *_args, **_kwargs):
        self._loop = asyncio.get_running_loop()
        self._started = True

    async def submit_frame(self, frame_id, _fps, _rgb, _depth):
        frame_id = int(frame_id)
        if frame_id != self._next_submitted:
            raise RuntimeError(
                f"preencoded submit order mismatch: expected "
                f"{self._next_submitted}, got {frame_id}"
            )
        if frame_id >= len(self._preencoded):
            raise RuntimeError(f"preencoded candidate unavailable for frame {frame_id}")
        self._next_submitted += 1
        return time.perf_counter()

    async def wait_result(self, frame_id):
        frame_id = int(frame_id)
        if frame_id != self._next_consumed:
            raise RuntimeError(
                f"preencoded consume order mismatch: expected "
                f"{self._next_consumed}, got {frame_id}"
            )
        self._next_consumed += 1
        now = time.perf_counter()
        retained = self._preencoded_records[frame_id]
        record = {
            **retained,
            "frame_id": frame_id,
            "request_submitted": now,
            "worker_started": now,
            "worker_completed": now,
            "result_received": now,
            "worker_idle_s": 0.0,
            "total_production_s": 0.0,
            "active_native_max": 0,
            "slot": -1,
            "durations": retained["durations"],
        }
        return self._preencoded[frame_id], record

    def runtime_snapshot(self):
        return {
            "child_pid": "preencoded",
            "sender_cpu_affinity": "",
            "worker_cpu_affinity": "",
            "active_encoder_workers": 0,
            "configured_encoder_workers": 0,
            "cancelled_candidate_jobs": 0,
            "candidate_not_ready_count": 0,
            "worker_failures": 0,
            "worker_exit_code": 0,
            "input_queue_depth": 0,
            "output_queue_depth": 0,
            "unfinished_ipc_requests": 0,
        }

    async def aclose(self):
        self._started = False
        return 0


class CausalDemandPredictor:
    """EWMA of completed aggregate application bytes, expressed in Mbps."""

    def __init__(self, alpha: float = 0.2):
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.high_mbps: Optional[float] = None
        self.low_mbps: Optional[float] = None

    def observe_completed_frame(
        self,
        *,
        high_application_bytes: int,
        low_application_bytes: int,
        frame_interval_s: float,
    ) -> tuple[float, float]:
        if frame_interval_s <= 0:
            raise ValueError("frame_interval_s must be positive")
        high = high_application_bytes * 8.0 / frame_interval_s / 1_000_000.0
        low = low_application_bytes * 8.0 / frame_interval_s / 1_000_000.0
        self.high_mbps = (
            high
            if self.high_mbps is None
            else self.alpha * high + (1 - self.alpha) * self.high_mbps
        )
        self.low_mbps = (
            low
            if self.low_mbps is None
            else self.alpha * low + (1 - self.alpha) * self.low_mbps
        )
        return self.high_mbps, self.low_mbps
