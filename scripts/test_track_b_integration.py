#!/usr/bin/env python3
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "src", "sender"))

from adaptation_controller import (  # noqa: E402
    AdaptationController,
    ControllerConfig,
    ControllerInputs,
    Quality,
)
from receiver_health_feedback import (  # noqa: E402
    ReceiverHealthFeedback,
    decode_receiver_health,
    encode_receiver_health,
)


class TrackBIntegrationTests(unittest.TestCase):
    def test_health_round_trip_drives_combined_safety_override(self):
        health = ReceiverHealthFeedback(
            feedback_seq=1,
            receiver_ts=1.0,
            window_start_frame=0,
            window_end_frame=29,
            total_frames=30,
            full_misses=6,
            partial_frames=0,
            decode_failures=0,
            reference_unavailable=0,
            frozen_frames=0,
            impairment_rate=0.2,
        )
        decoded = decode_receiver_health(encode_receiver_health(health))
        controller = AdaptationController(
            "combined",
            ControllerConfig(
                buffer_soft_bytes=100,
                buffer_hard_bytes=200,
                buffer_growth_bytes=10,
            ),
        )
        decision = controller.decide(
            ControllerInputs(
                timestamp=1.1,
                frame_id=30,
                gop_id=1,
                is_keyframe=True,
                capacity_raw_mbps=10,
                capacity_filtered_mbps=10,
                capacity_published_mbps=10,
                capacity_age_s=0.01,
                capacity_fresh=True,
                rgb_buffered_amount=0,
                depth_buffered_amount=0,
                buffer_trend_bytes=0,
                receiver_impairment_rate=decoded.impairment_rate,
                receiver_health_age_s=0.1,
                receiver_health_fresh=True,
                predicted_high_demand_mbps=5,
                predicted_low_demand_mbps=2,
            )
        )
        self.assertEqual(decision.applied_quality, Quality.LOW)
        self.assertEqual(decision.primary_reason, "safety_override")

    def test_no_trace_or_ground_truth_api_exists(self):
        fields = ControllerInputs.__dataclass_fields__
        self.assertFalse(any("trace" in name or "ground" in name for name in fields))


if __name__ == "__main__":
    unittest.main()
