"""Opt-in WebRTC validation diagnostics; inactive during normal ReVo runs."""

import asyncio
import csv
import logging
import math
import os
import time


class _MonotonicLogFilter(logging.Filter):
    def filter(self, record):
        record.monotonic = time.perf_counter()
        return True


def enable_ice_debug_logging():
    """Timestamp aioice STUN requests/responses without enabling app debug spam."""
    formatter = logging.Formatter(
        "%(monotonic).9f %(levelname)s:%(name)s:%(message)s"
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_MonotonicLogFilter())
        handler.setFormatter(formatter)
    logging.getLogger("aioice.ice").setLevel(logging.DEBUG)


def apply_ice_consent_timeout(timeout_s, role):
    """Apply an explicit validation-only consent timeout after ICE completes."""
    if timeout_s is None:
        return
    import aioice.stun

    aioice.stun.RETRY_RTO = float(timeout_s)
    logging.warning(
        "[%s] Applied validation-only ICE consent timeout: %.3f s",
        role,
        timeout_s,
    )


class WebRTCDiagnostics:
    FIELDNAMES = [
        "timestamp_monotonic",
        "role",
        "event_loop_delay_s",
        "feedback_events_total",
        "feedback_events_per_s",
        "feedback_delay_mean_s",
        "feedback_delay_max_s",
        "rgb_buffered_amount",
        "depth_buffered_amount",
        "outstanding_packets",
        "last_sent_sequence",
        "last_feedback_sequence",
        "connection_state",
        "ice_state",
    ]

    def __init__(self, path, role):
        self.enabled = bool(path)
        self.role = role
        self.feedback_total = 0
        self.feedback_interval = 0
        self.feedback_delays = []
        self.send_times = {}
        self.last_feedback_sequence = 0
        self._task = None
        self._stop = False
        self._file = None
        self._writer = None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._file = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()
            self._file.flush()

    def packet_sent(self, seq_id, timestamp):
        if self.enabled:
            self.send_times[int(seq_id)] = float(timestamp)

    def feedback_event(self, seq_id, timestamp=None):
        if not self.enabled:
            return
        now = time.perf_counter() if timestamp is None else float(timestamp)
        seq_id = int(seq_id)
        self.feedback_total += 1
        self.feedback_interval += 1
        self.last_feedback_sequence = max(self.last_feedback_sequence, seq_id)
        sent = self.send_times.pop(seq_id, None)
        if sent is not None:
            self.feedback_delays.append(max(0.0, now - sent))

    def start(self, snapshot):
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._monitor(snapshot))

    async def _monitor(self, snapshot):
        loop = asyncio.get_running_loop()
        previous = time.perf_counter()
        while not self._stop:
            deadline = loop.time() + 1.0
            await asyncio.sleep(1.0)
            now = time.perf_counter()
            elapsed = max(1e-9, now - previous)
            previous = now
            state = snapshot()
            delays = self.feedback_delays
            mean_delay = sum(delays) / len(delays) if delays else None
            max_delay = max(delays) if delays else None
            row = {
                "timestamp_monotonic": f"{now:.9f}",
                "role": self.role,
                "event_loop_delay_s": f"{max(0.0, loop.time() - deadline):.9f}",
                "feedback_events_total": self.feedback_total,
                "feedback_events_per_s": f"{self.feedback_interval / elapsed:.6f}",
                "feedback_delay_mean_s": "" if mean_delay is None else f"{mean_delay:.9f}",
                "feedback_delay_max_s": "" if max_delay is None else f"{max_delay:.9f}",
                "last_feedback_sequence": self.last_feedback_sequence,
            }
            for field in self.FIELDNAMES:
                if field not in row:
                    value = state.get(field, "")
                    if isinstance(value, float) and not math.isfinite(value):
                        value = ""
                    row[field] = value
            self._writer.writerow(row)
            self._file.flush()
            self.feedback_interval = 0
            self.feedback_delays = []

    async def stop(self):
        if not self.enabled:
            return
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._file is not None:
            self._file.close()
            self._file = None
