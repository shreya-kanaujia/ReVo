#!/usr/bin/env python3
"""Fail closed when a Track B run has crashed or produced unusable artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2


FATAL_LOG_MARKERS = (
    "Traceback (most recent call last):",
    "[Sender] Error in stream_video:",
    "[Sender] feedback handler failed",
    "[Receiver] error handling message",
    "[Receiver] decode worker failed",
    "[Receiver] display worker failed",
    "[Receiver] Failed to write video",
)

REQUIRED_CSVS = (
    "controller_decisions.csv",
    "receiver_capacity.csv",
    "sender_capacity_feedback.csv",
    "sender_measurements.csv",
    "sender_diagnostics.csv",
    "receiver_diagnostics.csv",
)


def csv_data_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for row in reader if any(value.strip() for value in row))


def video_frame_count(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        return 0
    count = 0
    try:
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            count += 1
    finally:
        capture.release()
    return count


def inspect_run(
    run_dir: Path,
    mode: str = "combined",
    *,
    require_media_timing: bool = False,
    diagnostic_only: bool = False,
) -> dict:
    issues = []
    statuses = {}
    for process in ("tc", "sender", "receiver"):
        path = run_dir / "metadata" / f"{process}_exit_status.txt"
        if not path.is_file():
            statuses[process] = None
            issues.append({
                "category": "setup_failure",
                "artifact": str(path),
                "reason": "missing process exit status",
            })
            continue
        try:
            status = int(path.read_text().strip())
        except ValueError:
            status = None
            issues.append({
                "category": "setup_failure",
                "artifact": str(path),
                "reason": "invalid process exit status",
            })
        statuses[process] = status
        if status not in (None, 0):
            issues.append({
                "category": "process_failure",
                "artifact": process,
                "reason": f"process exited with status {status}",
            })

    csv_counts = {}
    required_csvs = list(REQUIRED_CSVS)
    if mode == "combined":
        required_csvs.append("capacity_probe.csv")
    if require_media_timing:
        required_csvs.append("receiver_media_timing.csv")
    if diagnostic_only:
        required_csvs.append("candidate_diagnostics.csv")
    for name in required_csvs:
        path = run_dir / name
        count = csv_data_rows(path)
        csv_counts[name] = count
        if not path.is_file():
            reason = "missing required CSV"
        elif path.stat().st_size == 0:
            reason = "empty required CSV"
        elif count == 0:
            reason = "header-only required CSV"
        else:
            continue
        issues.append({
            "category": "artifact_failure",
            "artifact": str(path),
            "reason": reason,
        })

    video_counts = {}
    for name in ("receiver_rgb.mp4", "receiver_depth.mp4"):
        path = run_dir / name
        count = video_frame_count(path)
        video_counts[name] = count
        if count == 0:
            issues.append({
                "category": "artifact_failure",
                "artifact": str(path),
                "reason": "missing, empty, or unreadable output video",
            })

    fatal_log_hits = []
    for name in ("sender.log", "receiver.log"):
        path = run_dir / "logs" / name
        if not path.is_file():
            issues.append({
                "category": "setup_failure",
                "artifact": str(path),
                "reason": "missing required process log",
            })
            continue
        text = path.read_text(errors="replace")
        for marker in FATAL_LOG_MARKERS:
            if marker in text:
                fatal_log_hits.append({"log": str(path), "marker": marker})
                issues.append({
                    "category": "runtime_failure",
                    "artifact": str(path),
                    "reason": f"fatal marker found: {marker}",
                })

    artifact_complete = not issues
    return {
        "run_dir": str(run_dir),
        "mode": mode,
        "valid": artifact_complete and not diagnostic_only,
        "artifact_complete": artifact_complete,
        "diagnostic_only": bool(diagnostic_only),
        "classification": (
            "DIAGNOSTIC_ONLY" if diagnostic_only else
            ("PASS" if artifact_complete else "FAIL")
        ),
        "statuses": statuses,
        "csv_data_rows": csv_counts,
        "video_frames": video_counts,
        "fatal_log_hits": fatal_log_hits,
        "issues": issues,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", default="combined")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-media-timing", action="store_true")
    parser.add_argument("--diagnostic-only", action="store_true")
    args = parser.parse_args()
    result = inspect_run(
        args.run_dir,
        args.mode,
        require_media_timing=args.require_media_timing,
        diagnostic_only=args.diagnostic_only,
    )
    report = args.report or args.run_dir / "metadata" / "artifact_validation.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2) + "\n")
    if not result["artifact_complete"]:
        for issue in result["issues"]:
            print(
                f"{issue['category']}: {issue['artifact']}: {issue['reason']}",
                flush=True,
            )
        return 1
    if result["diagnostic_only"]:
        print(f"DIAGNOSTIC_ONLY artifacts complete: {args.run_dir}")
    else:
        print(f"Track B artifacts validated: {args.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
