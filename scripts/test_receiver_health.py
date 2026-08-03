#!/usr/bin/env python3
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "receiver"))

from receiver_health import (  # noqa: E402
    ReceiverHealthTracker,
    lacks_required_reference,
)


class ReceiverHealthTests(unittest.TestCase):
    def test_reference_dependency_recovers_at_matching_keyframe(self):
        self.assertTrue(lacks_required_reference(
            is_keyframe=False,
            gop_id=30,
            last_decoded_keyframe_id=None,
        ))
        self.assertFalse(lacks_required_reference(
            is_keyframe=True,
            gop_id=30,
            last_decoded_keyframe_id=None,
        ))
        self.assertFalse(lacks_required_reference(
            is_keyframe=False,
            gop_id=30,
            last_decoded_keyframe_id=30,
        ))
        self.assertTrue(lacks_required_reference(
            is_keyframe=False,
            gop_id=30,
            last_decoded_keyframe_id=0,
        ))

    def test_display_clock_emits_without_decode_progress(self):
        tracker = ReceiverHealthTracker(window_frames=3)
        tracker.record(0, full_miss=True, finalized=True)
        tracker.record(1, finalized=True)
        tracker.record(2, finalized=True)
        self.assertIsNone(tracker.complete_display_frame(0, 1.0, frozen=True))
        self.assertIsNone(tracker.complete_display_frame(1, 2.0, frozen=False))
        result = tracker.complete_display_frame(2, 3.0, frozen=True)
        self.assertEqual((result.window_start_frame, result.window_end_frame), (0, 2))
        self.assertEqual(result.full_misses, 1)
        self.assertEqual(result.frozen_frames, 2)
        self.assertEqual(result.feedback_seq, 1)
        self.assertAlmostEqual(result.impairment_rate, 2.0 / 3.0)

    def test_component_events_are_merged_per_frame(self):
        tracker = ReceiverHealthTracker(window_frames=2)
        tracker.record(0, partial=True)
        tracker.record(0, partial=True, decode_failure=True, finalized=True)
        tracker.record(1, finalized=True)
        tracker.complete_display_frame(0, 1.0, frozen=False)
        result = tracker.complete_display_frame(1, 2.0, frozen=False)
        self.assertEqual(result.partial_frames, 1)
        self.assertEqual(result.decode_failures, 1)

    def test_reference_unavailable_is_distinct_and_union_counted_once(self):
        tracker = ReceiverHealthTracker(window_frames=2)
        tracker.record(
            0,
            reference_unavailable=True,
            finalized=True,
        )
        tracker.record(1, finalized=True)
        tracker.complete_display_frame(0, 1.0, frozen=True)
        result = tracker.complete_display_frame(1, 2.0, frozen=False)
        self.assertEqual(result.reference_unavailable, 1)
        self.assertEqual(result.decode_failures, 0)
        self.assertEqual(result.frozen_frames, 1)
        self.assertEqual(result.impairment_rate, 0.5)

    def test_window_waits_for_decode_finalization(self):
        tracker = ReceiverHealthTracker(window_frames=2)
        tracker.record(0, finalized=True)
        tracker.complete_display_frame(0, 1.0, frozen=False)
        self.assertIsNone(tracker.complete_display_frame(1, 2.0, frozen=True))
        tracker.record(1, full_miss=True, finalized=True)
        result = tracker.complete_display_frame(2, 3.0, frozen=False)
        self.assertEqual(result.full_misses, 1)
        self.assertEqual(result.frozen_frames, 1)


if __name__ == "__main__":
    unittest.main()
