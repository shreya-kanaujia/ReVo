#!/usr/bin/env python3
"""Deterministic contracts for Track B logging and process failure handling."""

from __future__ import annotations

import ast
import csv
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sender.capacity_probe_controller import (  # noqa: E402
    CapacityProbeController,
    ProbeConfig,
    ProbeInputs,
)
from sender.adaptation_controller import (  # noqa: E402
    AdaptationController,
    AdaptationMode,
    ControllerInputs,
)
from sender.track_b_logging import (  # noqa: E402
    CAPACITY_PROBE_FIELDNAMES,
    CONTROLLER_FIELDNAMES,
    PROBE_SNAPSHOT_FIELDS,
    neutral_probe_snapshot,
    require_exact_keys,
    write_strict_row,
)
from validate_track_b_run import inspect_run  # noqa: E402
from sender.candidate_miss_diagnostics import (  # noqa: E402
    CandidateDiagnosticLog,
    FIELDNAMES as CANDIDATE_DIAGNOSTIC_FIELDS,
)
from partition_track_b_cpus import partition  # noqa: E402


def literal_writer_contracts(path):
    tree = ast.parse(path.read_text())
    fieldnames = {}
    rows = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "DictWriter"
        ):
            value = next(
                (
                    keyword.value
                    for keyword in node.value.keywords
                    if keyword.arg == "fieldnames"
                ),
                None,
            )
            if isinstance(value, (ast.List, ast.Tuple)):
                fields = ast.literal_eval(value)
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        fieldnames[target.attr] = fields
                    elif isinstance(target, ast.Name):
                        fieldnames[target.id] = fields
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "writerow"
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            target = node.func.value
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else target.id if isinstance(target, ast.Name) else None
            )
            if name and all(
                isinstance(key, ast.Constant) and isinstance(key.value, str)
                for key in node.args[0].keys
            ):
                rows.setdefault(name, []).append(
                    [key.value for key in node.args[0].keys]
                )
    return fieldnames, rows


def write_video(path, frames=2):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16)
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer unavailable")
    for _ in range(frames):
        writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


def write_csv(path, rows=1):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["value"])
        writer.writeheader()
        for index in range(rows):
            writer.writerow({"value": index})


class CsvContractTests(unittest.TestCase):
    def test_track_b_cpu_partition_is_disjoint_and_complete(self):
        result = partition(range(20, 29))
        roles = ("sender", "encoder", "receiver", "signaling", "trace")
        for index, left in enumerate(roles):
            for right in roles[index + 1:]:
                self.assertTrue(set(result[left]).isdisjoint(result[right]))
        self.assertEqual(result["sender_container"], [20, 21, 22, 23, 24, 25])

    def test_track_b_cpu_partition_fails_when_host_is_too_small(self):
        with self.assertRaisesRegex(RuntimeError, "requires 9 CPUs"):
            partition(range(8))

    def test_runner_logs_and_applies_every_critical_cpuset(self):
        source = (ROOT / "scripts" / "run_track_b_two_container.sh").read_text()
        for role in ("SENDER_CONTAINER", "RECEIVER", "SIGNALING", "TRACE"):
            self.assertIn(f"TRACK_B_{role}_CPUS", source)
        self.assertIn("effective_container_cpusets.txt", source)
        self.assertIn("cpu_partition.json", source)

    def test_runner_prioritizes_and_preflights_isolated_sender_process_tree(self):
        source = (ROOT / "scripts" / "run_track_b_two_container.sh").read_text()
        self.assertIn('TRACK_B_CRITICAL_NICE="${TRACK_B_CRITICAL_NICE:--10}"', source)
        self.assertIn("scheduler_priority_preflight.txt", source)
        self.assertIn(
            "exec nice -n '$TRACK_B_CRITICAL_NICE' /usr/local/bin/python3 "
            "src/sender/sender-3d.py",
            source,
        )
        self.assertIn("track_b_critical_nice=%s", source)

    def test_every_literal_writer_row_matches_its_schema_and_writes(self):
        files = (
            ROOT / "src" / "sender" / "sender-3d.py",
            ROOT / "src" / "receiver" / "receiver-3d.py",
            ROOT / "src" / "sender" / "run_loss_trace.py",
        )
        checked = 0
        for path in files:
            fields_by_writer, rows_by_writer = literal_writer_contracts(path)
            for writer_name, row_variants in rows_by_writer.items():
                if writer_name not in fields_by_writer:
                    continue
                for keys in row_variants:
                    with self.subTest(path=path.name, writer=writer_name):
                        self.assertEqual(
                            set(fields_by_writer[writer_name]), set(keys)
                        )
                        output = io.StringIO()
                        writer = csv.DictWriter(
                            output, fieldnames=fields_by_writer[writer_name]
                        )
                        writer.writeheader()
                        writer.writerow({key: "test" for key in keys})
                        self.assertEqual(
                            len(list(csv.DictReader(io.StringIO(output.getvalue())))),
                            1,
                        )
                        checked += 1
        self.assertGreaterEqual(checked, 5)

    def test_passive_deterioration_field_is_in_both_real_rows(self):
        self.assertIn(
            "passive_deterioration_ignored", CONTROLLER_FIELDNAMES
        )
        self.assertIn(
            "passive_deterioration_ignored", CAPACITY_PROBE_FIELDNAMES
        )
        sender_tree = ast.parse(
            (ROOT / "src" / "sender" / "sender-3d.py").read_text()
        )
        function_rows = {}
        for node in ast.walk(sender_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for child in ast.walk(node):
                    if (
                        isinstance(child, ast.Assign)
                        and any(
                            isinstance(target, ast.Name)
                            and target.id == "row"
                            for target in child.targets
                        )
                        and isinstance(child.value, ast.Dict)
                        and all(
                            isinstance(key, ast.Constant)
                            and isinstance(key.value, str)
                            for key in child.value.keys
                        )
                    ):
                        function_rows[node.name] = {
                            key.value for key in child.value.keys
                        }
        self.assertEqual(
            function_rows["_log_controller_decision"],
            set(CONTROLLER_FIELDNAMES),
        )
        self.assertEqual(
            function_rows["_log_capacity_probe"],
            set(CAPACITY_PROBE_FIELDNAMES),
        )

    def test_all_modes_share_complete_controller_and_probe_contracts(self):
        non_probe_modes = tuple(
            mode for mode in AdaptationMode
            if mode != AdaptationMode.COMBINED
        )
        for mode in AdaptationMode:
            with self.subTest(mode=mode.value):
                probe = neutral_probe_snapshot(
                    "probe_disabled_for_mode"
                    if mode != AdaptationMode.COMBINED
                    else "combined_test_snapshot"
                )
                require_exact_keys(
                    probe, PROBE_SNAPSHOT_FIELDS, "probe snapshot"
                )
                self.assertIn("passive_deterioration_ignored", probe)
                row = {field: "" for field in CONTROLLER_FIELDNAMES}
                row.update({
                    "adaptation_mode": mode.value,
                    "probe_state": probe["state"],
                    "probe_authorization_valid": "0",
                    "probe_result_age_s": "",
                    "probe_measured_rate_mbps": "",
                    "probe_failure_reason": "",
                    "passive_deterioration_ignored": "0",
                })
                output = io.StringIO()
                writer = csv.DictWriter(
                    output, fieldnames=CONTROLLER_FIELDNAMES
                )
                writer.writeheader()
                write_strict_row(
                    writer, row, CONTROLLER_FIELDNAMES, "controller CSV"
                )
                written = list(csv.DictReader(
                    io.StringIO(output.getvalue())
                ))
                self.assertEqual(len(written), 1)
                self.assertEqual(
                    set(written[0]), set(CONTROLLER_FIELDNAMES)
                )
                if mode in non_probe_modes:
                    self.assertEqual(probe["state"], "IDLE")
                    self.assertEqual(probe["probe_id"], 0)
                    self.assertEqual(probe["offered_bytes"], 0)
                    self.assertFalse(probe["authorization_valid"])
                    self.assertFalse(
                        probe["passive_deterioration_ignored"]
                    )
        combined = CapacityProbeController().snapshot(1.0)
        require_exact_keys(
            combined, PROBE_SNAPSHOT_FIELDS, "combined probe snapshot"
        )
        self.assertEqual(
            set(combined), set(neutral_probe_snapshot())
        )

    def test_every_mode_executes_one_simulated_decision(self):
        inputs = ControllerInputs(
            timestamp=1.0,
            frame_id=30,
            gop_id=1,
            is_keyframe=True,
            capacity_raw_mbps=5.0,
            capacity_filtered_mbps=5.0,
            capacity_published_mbps=5.0,
            capacity_age_s=0.01,
            capacity_fresh=True,
            rgb_buffered_amount=0,
            depth_buffered_amount=0,
            buffer_trend_bytes=0,
            receiver_impairment_rate=0.0,
            receiver_health_age_s=0.01,
            receiver_health_fresh=True,
            predicted_high_demand_mbps=4.0,
            predicted_low_demand_mbps=2.0,
            current_offered_video_mbps=2.0,
            probe_state="IDLE",
            probe_authorization_valid=False,
        )
        for mode in AdaptationMode:
            with self.subTest(mode=mode.value):
                controller = AdaptationController(mode)
                decision = controller.decide(inputs)
                self.assertEqual(decision.decision_seq, 1)
                if mode == AdaptationMode.FIXED_HIGH:
                    self.assertEqual(decision.applied_quality.value, "high")
                elif mode == AdaptationMode.FIXED_LOW:
                    self.assertEqual(decision.applied_quality.value, "low")
                elif mode == AdaptationMode.LEGACY:
                    self.assertEqual(decision.primary_reason, "legacy")

    def test_sender_and_receiver_failure_paths_are_not_swallowed(self):
        sender = (ROOT / "src" / "sender" / "sender-3d.py").read_text()
        self.assertIn(
            'logging.exception(f"[Sender] Error in stream_video: {e}")\n'
            "            raise",
            sender,
        )
        self.assertIn(
            'logging.exception(f"[Sender] Error during execution: {e}")\n'
            "            raise",
            sender,
        )
        self.assertIn("self._capacity_probe_task.result()", sender)
        self.assertIn("def _handle_feedback_callback(self, msg):", sender)
        self.assertIn("raise self._fatal_error", sender)
        receiver = (ROOT / "src" / "receiver" / "receiver-3d.py").read_text()
        self.assertIn("self._record_fatal_error(", receiver)
        self.assertIn("raise self._fatal_error", receiver)


class ProbeShutdownTests(unittest.TestCase):
    def test_normal_shutdown_with_active_probe_is_neutral(self):
        controller = CapacityProbeController(
            ProbeConfig(max_duration_s=0.3, min_duration_s=0.2)
        )
        state = controller.evaluate(ProbeInputs(
            now=1.0,
            current_quality="low",
            rgb_buffer=0,
            depth_buffer=0,
            buffer_trend=0,
            receiver_impairment=0.0,
            receiver_health_fresh=True,
            dwell_complete=True,
            transport_connected=True,
            seconds_to_keyframe=1.0,
            current_offered_video_mbps=2.0,
            predicted_low_demand_mbps=2.0,
            predicted_high_demand_mbps=4.0,
        ))
        self.assertEqual(state["state"], "PROBING")
        stopped = controller.stop_for_stream_shutdown(1.1)
        self.assertEqual(stopped["state"], "IDLE")
        self.assertEqual(stopped["reason"], "normal_stream_shutdown")
        self.assertFalse(stopped["authorization_valid"])


class ArtifactValidatorTests(unittest.TestCase):
    def make_run(self, root):
        run = Path(root)
        (run / "metadata").mkdir()
        (run / "logs").mkdir()
        for process in ("tc", "sender", "receiver"):
            (run / "metadata" / f"{process}_exit_status.txt").write_text("0\n")
        for name in (
            "controller_decisions.csv",
            "receiver_capacity.csv",
            "sender_capacity_feedback.csv",
            "sender_measurements.csv",
            "sender_diagnostics.csv",
            "receiver_diagnostics.csv",
            "receiver_media_timing.csv",
            "capacity_probe.csv",
        ):
            write_csv(run / name)
        write_video(run / "receiver_rgb.mp4")
        write_video(run / "receiver_depth.mp4")
        (run / "logs" / "sender.log").write_text("clean\n")
        (run / "logs" / "receiver.log").write_text("clean\n")
        return run

    def test_complete_run_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            result = inspect_run(
                self.make_run(directory), require_media_timing=True
            )
            self.assertTrue(result["valid"], result["issues"])

    def test_new_deadline_run_requires_media_timing_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            (run / "receiver_media_timing.csv").write_text("value\n")
            result = inspect_run(run, require_media_timing=True)
            self.assertFalse(result["valid"])

    def test_artifact_validator_accepts_every_mode(self):
        for mode in AdaptationMode:
            with self.subTest(mode=mode.value), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(directory)
                result = inspect_run(run, mode.value)
                self.assertTrue(result["valid"], result["issues"])
                self.assertGreater(
                    result["csv_data_rows"]["controller_decisions.csv"], 0
                )

    def test_sender_and_receiver_crashes_fail(self):
        for process in ("sender", "receiver"):
            with self.subTest(process=process), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(directory)
                (run / "metadata" / f"{process}_exit_status.txt").write_text("1\n")
                result = inspect_run(run)
                self.assertFalse(result["valid"])
                self.assertTrue(any(
                    issue["category"] == "process_failure"
                    and issue["artifact"] == process
                    for issue in result["issues"]
                ))

    def test_missing_video_empty_csv_and_header_only_csv_fail(self):
        variants = ("missing_video", "empty_csv", "header_only_csv")
        for variant in variants:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                run = self.make_run(directory)
                if variant == "missing_video":
                    os.unlink(run / "receiver_rgb.mp4")
                elif variant == "empty_csv":
                    (run / "controller_decisions.csv").write_text("")
                else:
                    (run / "controller_decisions.csv").write_text("value\n")
                result = inspect_run(run)
                self.assertFalse(result["valid"])
                self.assertTrue(any(
                    issue["category"] == "artifact_failure"
                    for issue in result["issues"]
                ))

    def test_hidden_traceback_fails_even_with_zero_status(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            (run / "logs" / "sender.log").write_text(
                "Traceback (most recent call last):\nboom\n"
            )
            result = inspect_run(run)
            self.assertFalse(result["valid"])
            self.assertTrue(result["fatal_log_hits"])

    def test_runner_uses_status_and_artifact_validator(self):
        runner = (
            ROOT / "scripts" / "run_track_b_two_container.sh"
        ).read_text()
        self.assertIn('wait "$SENDER_PID"; SENDER_STATUS=$?', runner)
        self.assertIn('wait "$RECEIVER_PID"; RECEIVER_STATUS=$?', runner)
        self.assertIn("validate_track_b_run.py", runner)
        self.assertIn("--require-media-timing", runner)
        self.assertIn('if [[ -e "$OUT_DIR" ]]', runner)

    def test_diagnostic_run_can_never_be_validation_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self.make_run(directory)
            write_csv(run / "candidate_diagnostics.csv")
            result = inspect_run(
                run, require_media_timing=True, diagnostic_only=True
            )
            self.assertTrue(result["artifact_complete"], result["issues"])
            self.assertFalse(result["valid"])
            self.assertEqual(result["classification"], "DIAGNOSTIC_ONLY")

    def test_candidate_diagnostic_csv_has_one_complete_row_per_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidate.csv"
            log = CandidateDiagnosticLog(path)
            log.record(1, missed_candidate=1, drop_skip_reason="candidate_not_ready")
            log.record(1, candidate_ready_timestamp=12.5)
            log.record(2, missed_candidate=0, send_start=13.0, send_end=13.01)
            log.close()
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(tuple(rows[0]), CANDIDATE_DIAGNOSTIC_FIELDS)
            self.assertEqual(rows[0]["frame_id"], "1")
            self.assertEqual(rows[0]["candidate_ready_timestamp"], "12.5")


if __name__ == "__main__":
    unittest.main()
