"""Causal frame-deadline helpers and validation logging."""

from __future__ import annotations

import csv
import os
import threading


def ready_before_deadline(ready_timestamp, deadline: float) -> bool:
    """Use recorded arrival time so worker scheduling delay cannot reject media."""
    return (
        ready_timestamp is not None
        and float(ready_timestamp) <= float(deadline)
    )


class FrameDeadlinePolicy:
    """Maps frame IDs to guarded assembly and display deadlines."""

    def __init__(self, frame_period_s: float, decode_guard_s: float):
        self.configure(frame_period_s, decode_guard_s)

    def configure(self, frame_period_s: float, decode_guard_s: float) -> None:
        frame_period_s = float(frame_period_s)
        decode_guard_s = float(decode_guard_s)
        if frame_period_s <= 0:
            raise ValueError("frame period must be positive")
        if decode_guard_s < 0 or decode_guard_s >= frame_period_s:
            raise ValueError(
                "decode guard must be nonnegative and shorter than one frame"
            )
        self.frame_period_s = frame_period_s
        self.decode_guard_s = decode_guard_s

    def display_deadline(self, clock_t0: float, frame_id: int) -> float:
        return (
            float(clock_t0)
            + (int(frame_id) + 1) * self.frame_period_s
        )

    def assembly_deadline(self, clock_t0: float, frame_id: int) -> float:
        return (
            self.display_deadline(clock_t0, frame_id)
            - self.decode_guard_s
        )


MEDIA_TIMING_FIELDNAMES = (
    "timestamp_monotonic",
    "event",
    "frame_id",
    "gop_id",
    "is_keyframe",
    "first_chunk_arrival",
    "last_chunk_arrival",
    "fec_ready_timestamp",
    "assembly_deadline",
    "display_deadline",
    "decode_guard_s",
    "assembly_ready",
    "assembly_reason",
    "decode_start",
    "decode_end",
    "display_timestamp",
    "frozen_display",
)


class MediaTimingCSV:
    """Thread-safe event CSV used only when an output path is supplied."""

    def __init__(self, path: str):
        self._lock = threading.Lock()
        self._file = None
        self._writer = None
        if not path:
            return
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._file = open(path, "w", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=MEDIA_TIMING_FIELDNAMES
        )
        self._writer.writeheader()
        self._file.flush()

    def write(self, **values) -> None:
        if self._writer is None:
            return
        unknown = set(values) - set(MEDIA_TIMING_FIELDNAMES)
        if unknown:
            raise ValueError(
                f"unknown media timing fields: {sorted(unknown)}"
            )
        row = {field: values.get(field, "") for field in MEDIA_TIMING_FIELDNAMES}
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
                self._writer = None
