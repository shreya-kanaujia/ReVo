#!/usr/bin/env python3
"""Event-driven, fail-closed Track B local validation gate."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

DEFAULT_EVENT_TIMEOUT_S = 60.0


def read_rows(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with path.open(newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def truthy(value):
    return str(value).lower() in ("1", "true")


class EventGateState:
    def __init__(self, gop_size=30):
        self.gop_size = int(gop_size)
        self.probe_success_ts = None
        self.switch_frame = None
        self.switch_ts = None
        self.target_gop_end = None
        self.failure = None
        self.complete = False

    def update(self, probe_rows, controller_rows, timing_rows, measurement_rows):
        if self.failure or self.complete:
            return
        successful = [
            row for row in probe_rows
            if row.get("probe_state") == "SUCCEEDED"
            and truthy(row.get("authorization_created"))
        ]
        if successful and self.probe_success_ts is None:
            self.probe_success_ts = min(
                float(row["monotonic_timestamp"]) for row in successful
            )

        if self.probe_success_ts is not None and self.switch_frame is None:
            applied = [
                row for row in controller_rows
                if truthy(row.get("switch_applied"))
                and row.get("applied_quality") == "high"
                and truthy(row.get("keyframe"))
                and int(row.get("frame_id", -1)) % self.gop_size == 0
                and float(row.get("timestamp_monotonic", 0))
                >= self.probe_success_ts
            ]
            if applied:
                first = min(applied, key=lambda row: int(row["frame_id"]))
                self.switch_frame = int(first["frame_id"])
                self.switch_ts = float(first["timestamp_monotonic"])
                self.target_gop_end = self.switch_frame + self.gop_size - 1

        assembly = {
            int(row["frame_id"]): row for row in timing_rows
            if row.get("event") == "assembly_decode" and row.get("frame_id")
        }
        display = {
            int(row["frame_id"]): row for row in timing_rows
            if row.get("event") == "display" and row.get("frame_id")
        }
        for frame_id, row in assembly.items():
            if row.get("assembly_ready") == "0":
                self.failure = (
                    f"media_health_failure: frame {frame_id} "
                    f"assembly={row.get('assembly_reason', '')}"
                )
                return
        for frame_id, row in display.items():
            if truthy(row.get("frozen_display")):
                self.failure = f"media_health_failure: frame {frame_id} frozen"
                return

        if self.switch_frame is None:
            return
        target = range(self.switch_frame, self.target_gop_end + 1)
        sent = {}
        for row in measurement_rows:
            if not row.get("frame_id") or row.get("stream") not in ("rgb", "depth"):
                continue
            sent[(int(row["frame_id"]), row["stream"])] = row
        for frame_id in target:
            for stream in ("rgb", "depth"):
                row = sent.get((frame_id, stream))
                if row is None or row.get("sent") != "1":
                    return
            if frame_id not in assembly or frame_id not in display:
                return
            if assembly[frame_id].get("assembly_ready") != "1":
                return
            if truthy(display[frame_id].get("frozen_display")):
                return
        self.complete = True

    def timeout_reason(self):
        if self.probe_success_ts is None:
            return "timeout_no_successful_probe"
        if self.switch_frame is None:
            return "timeout_no_applied_high_switch"
        return "timeout_incomplete_high_gop"

    def report(self, status, reason, elapsed_s):
        return {
            "status": status,
            "reason": reason,
            "elapsed_s": elapsed_s,
            "probe_success_timestamp": self.probe_success_ts,
            "switch_frame": self.switch_frame,
            "switch_timestamp": self.switch_ts,
            "target_gop_end": self.target_gop_end,
            "gop_size": self.gop_size,
        }


def monitor(args):
    state = EventGateState(args.gop_size)
    started = time.monotonic()
    report_path = args.run_dir / "metadata" / "event_gate_live.json"
    while True:
        state.update(
            read_rows(args.run_dir / "capacity_probe.csv"),
            read_rows(args.run_dir / "controller_decisions.csv"),
            read_rows(args.run_dir / "receiver_media_timing.csv"),
            read_rows(args.run_dir / "sender_measurements.csv"),
        )
        elapsed = time.monotonic() - started
        if state.failure:
            report = state.report("failed", state.failure, elapsed)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            return 1
        if state.complete:
            report = state.report("complete", "event_complete", elapsed)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            args.stop_signal.touch(exist_ok=False)
            return 0
        if elapsed >= args.timeout_s:
            report = state.report("timeout", state.timeout_reason(), elapsed)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            return 1
        time.sleep(args.poll_s)


def validate_run(run_dir: Path):
    issues = []
    live_path = run_dir / "metadata" / "event_gate_live.json"
    try:
        live = json.loads(live_path.read_text())
    except (OSError, json.JSONDecodeError):
        live = {}
        issues.append("missing_or_invalid_event_gate_live_report")
    if live.get("status") != "complete":
        issues.append(f"event_gate_status={live.get('status', 'missing')}")

    measurements = read_rows(run_dir / "sender_measurements.csv")
    source_fids = {
        int(row["frame_id"]) for row in measurements if row.get("frame_id")
    }
    timing = read_rows(run_dir / "receiver_media_timing.csv")
    assembly = {
        int(row["frame_id"]): row for row in timing
        if row.get("event") == "assembly_decode" and row.get("frame_id")
    }
    display = {
        int(row["frame_id"]): row for row in timing
        if row.get("event") == "display" and row.get("frame_id")
    }
    for frame_id in sorted(source_fids):
        if frame_id not in assembly or assembly[frame_id].get("assembly_ready") != "1":
            issues.append(f"source_frame_{frame_id}_not_assembly_ready")
        if frame_id not in display or truthy(display[frame_id].get("frozen_display")):
            issues.append(f"source_frame_{frame_id}_missing_or_frozen_display")

    for name in ("sender_diagnostics.csv", "receiver_diagnostics.csv"):
        rows = read_rows(run_dir / name)
        if not rows:
            issues.append(f"missing_{name}")
            continue
        final = rows[-1]
        fields = (
            "rgb_buffered_amount", "depth_buffered_amount",
            "sctp_data_channel_queue", "sctp_outbound_queue",
            "sctp_sent_queue", "sctp_sent_outstanding",
        )
        if name.startswith("sender"):
            fields += ("outstanding_packets",)
        for field in fields:
            value = final.get(field, "")
            if value not in ("", None) and int(float(value)) != 0:
                issues.append(f"queue_drain_failure:{name}:{field}={value}")

    transport_events = read_rows(run_dir / "sender_sctp_events.csv")
    if any(row.get("event") == "t3_expire" for row in transport_events):
        issues.append("unexplained_sender_sctp_t3_expiry")

    report = {
        "valid": not issues,
        "issues": issues,
        "source_frame_count": len(source_fids),
        "source_frame_start": min(source_fids) if source_fids else None,
        "source_frame_end": max(source_fids) if source_fids else None,
        "event_gate": live,
    }
    path = run_dir / "metadata" / "event_gate_validation.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if issues:
        for issue in issues:
            print(f"event_gate_failure: {issue}", flush=True)
        return 1
    print(f"Track B event gate validated: {run_dir}")
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stop-signal", type=Path)
    parser.add_argument(
        "--timeout-s", type=float, default=DEFAULT_EVENT_TIMEOUT_S,
        help="Hard fail-safe only; successful event completion exits earlier",
    )
    parser.add_argument("--poll-s", type=float, default=0.02)
    parser.add_argument("--gop-size", type=int, default=30)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        return validate_run(args.run_dir)
    if args.stop_signal is None:
        parser.error("--stop-signal is required while monitoring")
    return monitor(args)


if __name__ == "__main__":
    raise SystemExit(main())
