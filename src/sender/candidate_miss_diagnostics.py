"""Diagnostic-only per-frame accounting for Track B candidate misses."""

from __future__ import annotations

import csv
import os
from pathlib import Path


FIELDNAMES = (
    "frame_id", "missed_candidate", "first_miss_already_occurred",
    "diagnostic_only_after_first_miss", "scheduled_deadline",
    "actual_check_time", "deadline_lateness_s", "candidate_ready_timestamp",
    "source_load_start", "source_load_end", "prefetch_wait_s",
    "request_submit", "worker_start", "worker_end",
    "high_rgb_encode_s", "low_rgb_encode_s", "high_depth_encode_s",
    "low_depth_encode_s", "worker_idle_s", "completed_queue_depth",
    "completed_lead", "source_prefetch_depth", "inflight_count",
    "submitted_count", "collected_count", "consumed_count",
    "total_submitted_minus_consumed", "input_ipc_queue_depth",
    "output_ipc_queue_depth", "active_native_calls", "worker_alive",
    "worker_exit_code", "sender_cpu_s", "worker_cpu_s",
    "worker_process_cpu_s",
    "sender_voluntary_context_switches",
    "sender_involuntary_context_switches",
    "worker_voluntary_context_switches",
    "worker_involuntary_context_switches", "event_loop_delay_s",
    "worker_process_voluntary_context_switches",
    "worker_process_involuntary_context_switches",
    "send_start", "send_end", "rgb_buffered_amount",
    "depth_buffered_amount", "sctp_outbound_queue", "sctp_sent_queue",
    "sctp_outstanding", "sctp_flight_size_bytes", "drop_skip_reason",
)


def child_process_snapshot(pid):
    """Read cumulative Linux process CPU/context counters without signalling it."""
    result = {
        "worker_alive": False,
        "worker_cpu_s": "",
        "worker_voluntary_context_switches": "",
        "worker_involuntary_context_switches": "",
    }
    try:
        pid = int(pid)
        stat = Path(f"/proc/{pid}/stat").read_text().split()
        ticks = float(os.sysconf(os.sysconf_names["SC_CLK_TCK"]))
        result["worker_cpu_s"] = (int(stat[13]) + int(stat[14])) / ticks
        status = Path(f"/proc/{pid}/status").read_text().splitlines()
        values = dict(line.split(":", 1) for line in status if ":" in line)
        result["worker_voluntary_context_switches"] = int(
            values.get("voluntary_ctxt_switches", "0").strip()
        )
        result["worker_involuntary_context_switches"] = int(
            values.get("nonvoluntary_ctxt_switches", "0").strip()
        )
        result["worker_alive"] = True
    except (OSError, ValueError, IndexError):
        pass
    return result


class CandidateDiagnosticLog:
    """Collect exactly one final row per attempted source frame."""

    def __init__(self, path):
        self.path = Path(path) if path else None
        self.rows = {}

    def record(self, frame_id, **values):
        row = self.rows.setdefault(int(frame_id), {"frame_id": int(frame_id)})
        row.update(values)

    def close(self):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, extrasaction="raise")
            writer.writeheader()
            for frame_id in sorted(self.rows):
                row = {field: self.rows[frame_id].get(field, "") for field in FIELDNAMES}
                writer.writerow(row)
