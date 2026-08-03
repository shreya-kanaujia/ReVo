#!/usr/bin/env python3
from pathlib import Path
import csv
import sys
import tempfile
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_track_b import (  # noqa: E402
    analyze_run,
    capacity_availability_counts,
    frame_ssim,
    estimator_reference_metrics,
    longest_true_duration,
    media_deadline_audit,
    one_second_rate,
    process_resources,
    receiver_controlled_window,
    full_impairment_windows,
)


def write_rows(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_test_video(path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (16, 16)
    )
    if not writer.isOpened():
        raise RuntimeError("test video writer unavailable")
    writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
    writer.release()


class AnalyzeTrackBTests(unittest.TestCase):
    def test_cross_host_receiver_window_uses_paired_feedback_clocks(self):
        rows = [
            {"timestamp": "9.9", "receiver_timestamp": "4099.9"},
            {"timestamp": "10.1", "receiver_timestamp": "4100.1"},
            {"timestamp": "11.9", "receiver_timestamp": "4101.9"},
            {"timestamp": "12.1", "receiver_timestamp": "4102.1"},
        ]
        start, end, method = receiver_controlled_window(rows, 10.0, 12.0)
        self.assertEqual(method, "paired_capacity_feedback")
        self.assertEqual(start, 4100.1)
        self.assertEqual(end, 4101.9)

    def test_cross_host_receiver_rows_are_counted_in_analyzer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._complete_synthetic_run(root / "run")
            write_rows(run / "sender_capacity_feedback.csv", [
                {
                    "timestamp": "1.1",
                    "receiver_timestamp": "4001.1",
                    "capacity_state_update_status": "accepted_fresh",
                },
                {
                    "timestamp": "1.9",
                    "receiver_timestamp": "4001.9",
                    "capacity_state_update_status": "accepted_fresh",
                },
            ])
            write_rows(run / "receiver_capacity.csv", [
                {
                    "receiver_timestamp": "4001.1",
                    "packet_size_bytes": "1000",
                    "corrected_interarrival_time": "0.01",
                    "published_estimated_capacity_mbps": "0.8",
                    "estimate_fresh": "1",
                    "feedback_sent": "1",
                },
                {
                    "receiver_timestamp": "4001.9",
                    "packet_size_bytes": "1000",
                    "corrected_interarrival_time": "0.01",
                    "published_estimated_capacity_mbps": "0.8",
                    "estimate_fresh": "1",
                    "feedback_sent": "1",
                },
            ])
            summary = analyze_run(run, root / "analysis", metrics_only=True)
            self.assertEqual(summary["receiver_messages"], 2)
            self.assertGreater(summary["delivered_application_mbps"], 0)
            self.assertGreater(summary["estimator_fresh_one_second_bins"], 0)
            self.assertEqual(
                summary["receiver_controlled_window_method"],
                "paired_capacity_feedback",
            )

    def test_media_timing_join_and_reference_health_window(self):
        controller = [{
            "frame_id": "30",
            "keyframe": "1",
            "applied_quality": "high",
            "health_feedback_sequence": "2",
            "receiver_impairment_rate": "1.0",
            "health_window_start_frame": "30",
            "health_window_end_frame": "59",
            "health_full_misses": "1",
            "health_partial_frames": "0",
            "health_decode_failures": "0",
            "health_reference_unavailable": "29",
            "health_frozen_frames": "30",
        }]
        measurements = [{
            "frame_id": "30",
            "stream": "rgb",
            "frame_type": "I",
            "production_timestamp_monotonic": "1.0",
            "send_start_monotonic": "1.1",
            "send_end_monotonic": "1.2",
            "causal_encode_lead_s": "0.02",
        }]
        timing = [{
            "frame_id": "30",
            "event": "assembly_decode",
            "first_chunk_arrival": "1.11",
            "last_chunk_arrival": "1.19",
            "fec_ready_timestamp": "1.18",
            "assembly_deadline": "1.25",
            "decode_start": "1.18",
            "decode_end": "1.20",
            "display_deadline": "1.26",
            "assembly_ready": "1",
            "assembly_reason": "ready",
        }, {
            "frame_id": "30",
            "event": "display",
            "frozen_display": "0",
        }]
        audit = media_deadline_audit(controller, measurements, timing)
        self.assertEqual(len(audit), 1)
        self.assertAlmostEqual(audit[0]["enqueue_interval_s"], 0.1)
        self.assertTrue(audit[0]["assembly_ready"])
        windows = full_impairment_windows(controller)
        self.assertEqual(windows[0]["reference_unavailable"], 29)
        self.assertEqual(windows[0]["codec_decode_failures"], 0)

    def test_identical_frame_ssim_is_one(self):
        frame = np.arange(32 * 32, dtype=np.uint8).reshape(32, 32)
        self.assertAlmostEqual(frame_ssim(frame, frame), 1.0, places=12)

    def test_rate_uses_same_explicit_window(self):
        rows = [
            {"ts": "10.1", "bytes": "125000"},
            {"ts": "11.1", "bytes": "250000"},
            {"ts": "12.1", "bytes": "999999"},
        ]
        _, rates = one_second_rate(rows, "ts", "bytes", 10.0, 12.0)
        self.assertEqual(rates.tolist(), [1.0, 2.0])

    def test_process_cpu_and_rss(self):
        rows = [
            {
                "timestamp_monotonic": "1",
                "process_cpu_time_s": "2",
                "process_max_rss_bytes": "100",
            },
            {
                "timestamp_monotonic": "3",
                "process_cpu_time_s": "3",
                "process_max_rss_bytes": "200",
            },
        ]
        result = process_resources(rows)
        self.assertEqual(result["cpu_mean_percent"], 50.0)
        self.assertEqual(result["rss_max_bytes"], 200)

    def test_longest_episode_uses_observed_timestamps(self):
        rows = [
            {"timestamp_monotonic": "0", "active": "1"},
            {"timestamp_monotonic": "1", "active": "1"},
            {"timestamp_monotonic": "3", "active": "0"},
            {"timestamp_monotonic": "4", "active": "1"},
            {"timestamp_monotonic": "5", "active": "0"},
        ]
        self.assertEqual(
            longest_true_duration(rows, lambda row: row["active"] == "1"), 3
        )

    def test_estimator_reference_uses_matching_bins(self):
        rows = [
            {
                "receiver_timestamp": "10.1",
                "packet_size_bytes": "125000",
                "corrected_interarrival_time": "1",
                "published_estimated_capacity_mbps": "1",
            },
            {
                "receiver_timestamp": "11.1",
                "packet_size_bytes": "250000",
                "corrected_interarrival_time": "1",
                "published_estimated_capacity_mbps": "2",
            },
        ]
        result = estimator_reference_metrics(rows, 10, 12)
        self.assertEqual(result["estimator_fresh_one_second_bins"], 2)
        self.assertEqual(result["estimator_active_service_mae_mbps"], 0)
        self.assertEqual(result["estimator_goodput_rmse_mbps"], 0)

    def test_capacity_availability_stages_are_not_conflated(self):
        result = capacity_availability_counts(
            [
                {
                    "estimate_fresh": "1", "feedback_sent": "1",
                    "filter_reason": "sample_accepted",
                },
                {
                    "estimate_fresh": "1", "feedback_sent": "0",
                    "filter_reason": "no_valid_spacing_sample",
                },
                {"estimate_fresh": "0", "feedback_sent": "1"},
            ],
            [
                {"capacity_state_update_status": "accepted_fresh"},
                {"capacity_state_update_status": "accepted_sequence_gap"},
            ],
            [
                {"capacity_fresh": "1"},
                {"capacity_fresh": "0"},
                {"capacity_fresh": "0"},
            ],
        )
        self.assertEqual(result["receiver_fresh_publications"], 1)
        self.assertEqual(result["receiver_fresh_feedback_rows"], 2)
        self.assertEqual(result["receiver_capacity_feedback_sent"], 2)
        self.assertEqual(result["sender_capacity_feedback_received"], 2)
        self.assertEqual(result["sender_accepted_fresh_feedback"], 1)
        self.assertEqual(result["controller_fresh_capacity_decisions"], 1)
        self.assertAlmostEqual(
            result["controller_fresh_capacity_fraction"], 1 / 3
        )

    def _complete_synthetic_run(self, root, mode="combined"):
        run = Path(root)
        (run / "metadata").mkdir(parents=True)
        (run / "logs").mkdir()
        for process in ("tc", "sender", "receiver"):
            (run / "metadata" / f"{process}_exit_status.txt").write_text("0\n")
        base = {
            "timestamp_monotonic": "1.0",
            "adaptation_mode": mode,
            "applied_quality": "low",
            "requested_quality": "low",
            "switch_applied": "0",
            "switch_requested": "0",
            "primary_decision_reason": "hold",
            "frame_id": "0",
            "gop_id": "0",
            "current_quality": "low",
            "capacity_fresh": "0",
            "receiver_health_fresh": "1",
            "max_bufferedAmount": "0",
            "receiver_impairment_rate": "0",
            "capacity_age_s": "",
            "receiver_health_age_s": "0.1",
            "predicted_high_demand_mbps": "4",
            "predicted_low_demand_mbps": "2",
            "health_feedback_sequence": "1",
            "health_total_frames": "1",
            "health_full_misses": "0",
            "health_partial_frames": "0",
            "health_decode_failures": "0",
            "health_reference_unavailable": "0",
            "health_frozen_frames": "0",
        }
        second = dict(base, timestamp_monotonic="2.0", frame_id="1")
        write_rows(run / "controller_decisions.csv", [base, second])
        write_rows(run / "receiver_capacity.csv", [{
            "receiver_timestamp": "1.5",
            "packet_size_bytes": "1000",
            "corrected_interarrival_time": "0.01",
            "published_estimated_capacity_mbps": "",
            "estimate_fresh": "0",
            "feedback_sent": "1",
        }])
        for name in (
            "sender_capacity_feedback.csv",
            "sender_measurements.csv",
            "sender_diagnostics.csv",
            "receiver_diagnostics.csv",
            "receiver_media_timing.csv",
            "capacity_probe.csv",
        ):
            write_rows(run / name, [{"timestamp_monotonic": "1", "value": "1"}])
        (run / "applied_trace.csv").write_text(
            "elapsed_s\n0\n"
        )
        (run / "logs" / "sender.log").write_text("clean\n")
        (run / "logs" / "receiver.log").write_text("clean\n")
        write_test_video(run / "receiver_rgb.mp4")
        write_test_video(run / "receiver_depth.mp4")
        return run

    def test_analyzer_accepts_complete_run(self):
        for mode in (
            "legacy", "fixed_high", "fixed_low", "buffer_health",
            "estimator_only", "combined",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run = self._complete_synthetic_run(root / "run", mode)
                summary = analyze_run(
                    run, root / "analysis", metrics_only=True
                )
                self.assertTrue(summary["complete"])
                self.assertEqual(summary["mode"], mode)
                self.assertTrue(
                    (root / "analysis" / "track_b_metrics.json").is_file()
                )

    def test_analyzer_rejects_failed_run_without_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = self._complete_synthetic_run(root / "run")
            (run / "metadata" / "sender_exit_status.txt").write_text("1\n")
            with self.assertRaisesRegex(RuntimeError, "incomplete Track B run"):
                analyze_run(run, root / "analysis", metrics_only=True)
            self.assertTrue((root / "analysis" / "analysis_failure.json").is_file())
            self.assertFalse((root / "analysis" / "track_b_metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
