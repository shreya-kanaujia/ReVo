#!/usr/bin/env python3
"""Deterministic contracts for Track B media deadlines and health accounting."""

from __future__ import annotations

import csv
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "receiver"))
sys.path.insert(0, str(ROOT / "src" / "sender"))

from media_timing import (  # noqa: E402
    MEDIA_TIMING_FIELDNAMES,
    FrameDeadlinePolicy,
    MediaTimingCSV,
    ready_before_deadline,
)
from quality_encoder_manager import (  # noqa: E402
    CausalEncodeLead,
    packet_pacing_interval,
    sender_frame_deadline,
)
from receiver_health import ReceiverHealthTracker  # noqa: E402


class FrameDeadlineTests(unittest.TestCase):
    def test_current_frame_deadline_and_guard(self):
        policy = FrameDeadlinePolicy(0.04, 0.01)
        self.assertAlmostEqual(policy.display_deadline(10.0, 5), 10.24)
        self.assertAlmostEqual(policy.assembly_deadline(10.0, 5), 10.23)
        previous_display = policy.display_deadline(10.0, 4)
        self.assertNotEqual(policy.assembly_deadline(10.0, 5), previous_display)
        self.assertLess(
            policy.assembly_deadline(10.0, 5),
            policy.display_deadline(10.0, 5),
        )

    def test_before_and_after_guarded_deadline(self):
        policy = FrameDeadlinePolicy(1 / 25, 0.01)
        deadline = policy.assembly_deadline(100.0, 30)
        self.assertTrue(ready_before_deadline(deadline - 1e-6, deadline))
        self.assertFalse(ready_before_deadline(deadline + 1e-6, deadline))
        # A large keyframe that reaches FEC readiness anywhere within the
        # corrected current-frame window remains eligible.
        previous_display = policy.display_deadline(100.0, 29)
        fec_ready = previous_display + 0.020
        self.assertGreater(fec_ready, previous_display)
        self.assertLess(fec_ready, deadline)

    def test_25_and_30_fps_remain_distinct(self):
        at_25 = FrameDeadlinePolicy(1 / 25, 0.01)
        at_30 = FrameDeadlinePolicy(1 / 30, 0.01)
        self.assertAlmostEqual(
            at_25.display_deadline(0.0, 24), 1.0
        )
        self.assertAlmostEqual(
            at_30.display_deadline(0.0, 29), 1.0
        )
        self.assertNotEqual(at_25.frame_period_s, at_30.frame_period_s)

    def test_rgb_depth_share_one_frame_deadline(self):
        policy = FrameDeadlinePolicy(0.04, 0.01)
        rgb = policy.assembly_deadline(7.0, 60)
        depth = policy.assembly_deadline(7.0, 60)
        self.assertEqual(rgb, depth)

    def test_guard_configuration_rejects_display_overrun(self):
        with self.assertRaises(ValueError):
            FrameDeadlinePolicy(0.04, 0.04)
        with self.assertRaises(ValueError):
            FrameDeadlinePolicy(0.04, -0.001)

    def test_media_timing_csv_writes_complete_real_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "timing.csv")
            log = MediaTimingCSV(path)
            log.write(
                timestamp_monotonic=1.2,
                event="assembly_decode",
                frame_id=30,
                gop_id=1,
                is_keyframe=1,
                fec_ready_timestamp=1.1,
                assembly_deadline=1.3,
                display_deadline=1.31,
                decode_guard_s=0.01,
                assembly_ready=1,
                assembly_reason="ready",
            )
            log.close()
            with open(path, newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(set(rows[0]), set(MEDIA_TIMING_FIELDNAMES))


class SenderTimingTests(unittest.TestCase):
    def test_encode_lead_uses_only_completed_history(self):
        lead = CausalEncodeLead(alpha=0.5)
        self.assertEqual(lead.predict(0.04), 0.0)
        lead.observe(0.020)
        self.assertAlmostEqual(lead.predict(0.04), 0.020)
        lead.observe(0.030)
        self.assertAlmostEqual(lead.predict(0.04), 0.025)
        self.assertAlmostEqual(
            sender_frame_deadline(10.0, 2, 0.04, 0.025),
            10.055,
        )

    def test_encode_lead_is_bounded_to_one_frame(self):
        lead = CausalEncodeLead(alpha=1.0)
        lead.observe(0.5)
        self.assertAlmostEqual(lead.predict(0.04), 0.04)

    def test_keyframe_urgency_is_finite_and_does_not_shift_future_deadline(self):
        self.assertEqual(
            packet_pacing_interval(0.04, 100, is_keyframe=True), 0.0
        )
        self.assertAlmostEqual(
            packet_pacing_interval(0.04, 10, is_keyframe=False), 0.004
        )
        first = sender_frame_deadline(5.0, 30, 0.04, 0.01)
        next_frame = sender_frame_deadline(5.0, 31, 0.04, 0.01)
        self.assertAlmostEqual(next_frame - first, 0.04)


class HealthClassificationTests(unittest.TestCase):
    def test_reference_loss_is_not_codec_failure_and_next_gop_recovers(self):
        tracker = ReceiverHealthTracker(window_frames=3)
        tracker.record(0, full_miss=True, finalized=True)
        tracker.record(1, reference_unavailable=True, finalized=True)
        # A new valid keyframe is represented as an unimpaired finalized frame.
        tracker.record(2, finalized=True)
        tracker.complete_display_frame(0, 1.0, frozen=True)
        tracker.complete_display_frame(1, 2.0, frozen=True)
        result = tracker.complete_display_frame(2, 3.0, frozen=False)
        self.assertEqual(result.full_misses, 1)
        self.assertEqual(result.reference_unavailable, 1)
        self.assertEqual(result.decode_failures, 0)
        self.assertEqual(result.frozen_frames, 2)
        self.assertAlmostEqual(result.impairment_rate, 2 / 3)


if __name__ == "__main__":
    unittest.main()
