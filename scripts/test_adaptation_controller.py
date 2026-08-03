#!/usr/bin/env python3
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "sender"))

from adaptation_controller import (  # noqa: E402
    AdaptationController,
    AdaptationMode,
    ControllerConfig,
    ControllerInputs,
    Quality,
)


def inputs(**updates):
    values = dict(
        timestamp=1.0,
        frame_id=30,
        gop_id=1,
        is_keyframe=True,
        capacity_raw_mbps=8.0,
        capacity_filtered_mbps=7.5,
        capacity_published_mbps=7.5,
        capacity_age_s=0.01,
        capacity_fresh=True,
        rgb_buffered_amount=0,
        depth_buffered_amount=0,
        buffer_trend_bytes=0,
        receiver_impairment_rate=0.0,
        receiver_health_age_s=0.1,
        receiver_health_fresh=True,
        predicted_high_demand_mbps=5.0,
        predicted_low_demand_mbps=2.0,
        current_offered_video_mbps=5.0,
    )
    values.update(updates)
    return ControllerInputs(**values)


class AdaptationControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ControllerConfig(
            buffer_soft_bytes=100,
            buffer_hard_bytes=200,
            buffer_growth_bytes=10,
            downgrade_confirmations=1,
            upgrade_confirmations=3,
            min_dwell_gops=2,
        )

    def test_fresh_capacity_supports_high(self):
        decision = AdaptationController("combined", self.cfg).decide(inputs())
        self.assertEqual(decision.applied_quality, Quality.HIGH)

    def test_low_passive_service_alone_holds_high(self):
        decision = AdaptationController("combined", self.cfg).decide(
            inputs(capacity_published_mbps=3.0)
        )
        self.assertEqual(decision.applied_quality, Quality.HIGH)
        self.assertEqual(
            decision.primary_reason, "passive_service_uncorroborated_hold"
        )
        self.assertIn("passive_service_deterioration", decision.reason_flags)
        self.assertIn(
            "passive_only_downgrade_suppressed", decision.reason_flags
        )

    def test_stale_capacity_blocks_upgrade_but_does_not_force_low(self):
        controller = AdaptationController("combined", self.cfg)
        held = controller.decide(
            inputs(capacity_fresh=False, capacity_published_mbps=None)
        )
        self.assertEqual(held.applied_quality, Quality.HIGH)
        controller.current_quality = Quality.LOW
        for gop in range(1, 6):
            decision = controller.decide(
                inputs(
                    frame_id=gop * 30,
                    gop_id=gop,
                    capacity_fresh=False,
                    capacity_published_mbps=None,
                )
            )
        self.assertEqual(decision.applied_quality, Quality.LOW)

    def test_stale_plus_growing_buffer_requests_low(self):
        decision = AdaptationController("combined", self.cfg).decide(
            inputs(
                capacity_fresh=False,
                capacity_published_mbps=None,
                rgb_buffered_amount=120,
                buffer_trend_bytes=20,
            )
        )
        self.assertEqual(decision.applied_quality, Quality.LOW)

    def test_buffer_and_health_override_optimistic_capacity(self):
        for update in (
            dict(rgb_buffered_amount=201),
            dict(receiver_impairment_rate=0.25),
        ):
            with self.subTest(update=update):
                decision = AdaptationController("combined", self.cfg).decide(
                    inputs(**update)
                )
                self.assertEqual(decision.applied_quality, Quality.LOW)
                self.assertEqual(decision.primary_reason, "safety_override")

    def test_short_buffer_spike_does_not_switch(self):
        controller = AdaptationController("combined", self.cfg)
        decision = controller.decide(
            inputs(rgb_buffered_amount=120, buffer_trend_bytes=-20)
        )
        self.assertEqual(decision.applied_quality, Quality.HIGH)

    def test_upgrade_needs_more_evidence_and_dwell(self):
        controller = AdaptationController("buffer_health", self.cfg)
        controller.decide(inputs(receiver_impairment_rate=0.10, gop_id=1))
        self.assertEqual(controller.current_quality, Quality.LOW)
        for gop in (2, 3):
            decision = controller.decide(
                inputs(
                    gop_id=gop, frame_id=gop * 30,
                    probe_authorization_valid=True,
                )
            )
            self.assertEqual(decision.applied_quality, Quality.LOW)
        decision = controller.decide(
            inputs(
                gop_id=4, frame_id=120,
                probe_authorization_valid=True,
            )
        )
        self.assertEqual(decision.applied_quality, Quality.HIGH)

    def test_stale_resets_upgrade_votes(self):
        controller = AdaptationController("buffer_health", self.cfg)
        controller.current_quality = Quality.LOW
        controller.decide(inputs(gop_id=2, probe_authorization_valid=True))
        controller.decide(
            inputs(
                gop_id=3,
                receiver_health_fresh=False,
                capacity_fresh=False,
                capacity_published_mbps=None,
            )
        )
        decision = controller.decide(
            inputs(gop_id=4, probe_authorization_valid=True)
        )
        self.assertEqual(decision.applied_quality, Quality.LOW)

    def test_switch_only_at_keyframe_and_once_per_gop(self):
        controller = AdaptationController("combined", self.cfg)
        pending = controller.decide(
            inputs(
                is_keyframe=False,
                capacity_published_mbps=2.0,
                rgb_buffered_amount=120,
                buffer_trend_bytes=20,
            )
        )
        self.assertTrue(pending.switch_requested)
        self.assertFalse(pending.switch_applied)
        applied = controller.decide(
            inputs(
                capacity_published_mbps=2.0,
                rgb_buffered_amount=120,
                buffer_trend_bytes=20,
            )
        )
        self.assertTrue(applied.switch_applied)
        again = controller.decide(inputs(capacity_published_mbps=10.0))
        self.assertFalse(again.switch_applied)

    def test_downgrade_is_latched_until_safe_keyframe(self):
        controller = AdaptationController("combined", self.cfg)
        pending = controller.decide(
            inputs(
                frame_id=31, gop_id=1, is_keyframe=False,
                capacity_published_mbps=2.0,
                rgb_buffered_amount=120,
                buffer_trend_bytes=20,
            )
        )
        self.assertTrue(pending.switch_requested)
        held = controller.decide(
            inputs(
                frame_id=32, gop_id=1, is_keyframe=False,
                capacity_fresh=False, capacity_published_mbps=None,
            )
        )
        self.assertFalse(held.switch_requested)
        self.assertIn("downgrade_latched", held.reason_flags)
        applied = controller.decide(
            inputs(
                frame_id=60, gop_id=2, is_keyframe=True,
                capacity_fresh=False, capacity_published_mbps=None,
            )
        )
        self.assertTrue(applied.switch_applied)
        self.assertEqual(applied.applied_quality, Quality.LOW)

    def test_latched_downgrade_has_explicit_safe_cancellation(self):
        controller = AdaptationController("combined", self.cfg)
        controller.decide(
            inputs(
                frame_id=31, gop_id=1, is_keyframe=False,
                capacity_published_mbps=2.0,
                rgb_buffered_amount=120,
                buffer_trend_bytes=20,
            )
        )
        cancelled = controller.decide(
            inputs(
                frame_id=60, gop_id=2, is_keyframe=True,
                probe_authorization_valid=True,
            )
        )
        self.assertFalse(cancelled.switch_requested)
        self.assertFalse(cancelled.switch_applied)
        self.assertEqual(
            cancelled.primary_reason,
            "latched_downgrade_cancelled_at_keyframe",
        )

    def test_mode_signal_isolation(self):
        estimator = AdaptationController("estimator_only", self.cfg)
        self.assertEqual(
            estimator.decide(inputs(rgb_buffered_amount=500)).applied_quality,
            Quality.HIGH,
        )
        buffer_health = AdaptationController("buffer_health", self.cfg)
        self.assertEqual(
            buffer_health.decide(inputs(capacity_published_mbps=1.0)).applied_quality,
            Quality.HIGH,
        )

    def test_passive_service_alone_never_authorizes_upgrade(self):
        controller = AdaptationController("combined", self.cfg)
        controller.current_quality = Quality.LOW
        for gop in range(1, 6):
            decision = controller.decide(
                inputs(
                    frame_id=gop * 30, gop_id=gop,
                    capacity_published_mbps=100.0,
                    current_offered_video_mbps=2.0,
                    probe_authorization_valid=False,
                )
            )
        self.assertEqual(decision.applied_quality, Quality.LOW)

    def test_passive_service_requires_direct_corroboration(self):
        cases = (
            (
                "sustained_soft_buffer_growth",
                dict(rgb_buffered_amount=120, buffer_trend_bytes=10),
            ),
            (
                "bad_receiver_health",
                dict(receiver_impairment_rate=0.05),
            ),
        )
        for name, update in cases:
            with self.subTest(name=name):
                decision = AdaptationController(
                    "combined", self.cfg
                ).decide(
                    inputs(
                        capacity_published_mbps=2.0,
                        current_offered_video_mbps=5.0,
                        **update,
                    )
                )
                self.assertEqual(decision.applied_quality, Quality.LOW)
                self.assertIn(
                    "passive_service_corroborated",
                    decision.reason_flags,
                )

    def test_transient_keyframe_buffer_growth_does_not_corroborate_passive(self):
        decision = AdaptationController("combined", self.cfg).decide(
            inputs(
                frame_id=31, gop_id=1, is_keyframe=False,
                capacity_published_mbps=1.0,
                current_offered_video_mbps=5.0,
                rgb_buffered_amount=99,
                buffer_trend_bytes=99,
            )
        )
        self.assertEqual(decision.applied_quality, Quality.HIGH)
        self.assertEqual(
            decision.primary_reason, "passive_service_uncorroborated_hold"
        )

    def test_probe_upgrade_survives_passive_only_post_upgrade_period(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        upgraded = controller.decide(
            inputs(
                frame_id=60,
                gop_id=2,
                is_keyframe=True,
                probe_authorization_valid=True,
            )
        )
        self.assertTrue(upgraded.switch_applied)
        self.assertEqual(upgraded.applied_quality, Quality.HIGH)
        self.assertIn("post_upgrade_validation", upgraded.reason_flags)

        held = controller.decide(
            inputs(
                frame_id=90,
                gop_id=3,
                is_keyframe=True,
                capacity_published_mbps=1.0,
                current_offered_video_mbps=5.0,
                probe_authorization_valid=False,
            )
        )
        self.assertEqual(held.applied_quality, Quality.HIGH)
        self.assertFalse(held.switch_requested)
        self.assertIn("post_upgrade_validation", held.reason_flags)
        self.assertIn(
            "passive_only_downgrade_suppressed", held.reason_flags
        )

        after_validation = controller.decide(
            inputs(
                frame_id=120,
                gop_id=4,
                is_keyframe=True,
                capacity_published_mbps=1.0,
                current_offered_video_mbps=5.0,
                probe_authorization_valid=False,
            )
        )
        self.assertEqual(after_validation.applied_quality, Quality.HIGH)
        self.assertNotIn(
            "post_upgrade_validation", after_validation.reason_flags
        )

    def test_post_upgrade_validation_does_not_block_severe_downgrade(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        controller.decide(
            inputs(
                frame_id=60,
                gop_id=2,
                is_keyframe=True,
                probe_authorization_valid=True,
            )
        )
        decision = controller.decide(
            inputs(
                frame_id=90,
                gop_id=3,
                is_keyframe=True,
                rgb_buffered_amount=201,
                capacity_published_mbps=1.0,
            )
        )
        self.assertTrue(decision.switch_applied)
        self.assertEqual(decision.applied_quality, Quality.LOW)
        self.assertEqual(decision.primary_reason, "safety_override")

    def test_post_upgrade_hold_blocks_non_emergency_keyframe_transient(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        controller.decide(inputs(
            frame_id=60, gop_id=2, is_keyframe=True,
            probe_authorization_valid=True,
        ))
        held = controller.decide(inputs(
            frame_id=61, gop_id=2, is_keyframe=False,
            capacity_published_mbps=1.0,
            current_offered_video_mbps=5.0,
            rgb_buffered_amount=120,
            buffer_trend_bytes=20,
        ))
        self.assertEqual(held.applied_quality, Quality.HIGH)
        self.assertEqual(held.primary_reason, "post_upgrade_validation_hold")
        self.assertFalse(held.switch_requested)

    def test_non_emergency_downgrade_resumes_after_bounded_hold(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        controller.decide(inputs(
            frame_id=60, gop_id=2, is_keyframe=True,
            probe_authorization_valid=True,
        ))
        held = controller.decide(inputs(
            frame_id=61, gop_id=2, is_keyframe=False,
            capacity_published_mbps=1.0,
            rgb_buffered_amount=120, buffer_trend_bytes=20,
        ))
        self.assertEqual(held.applied_quality, Quality.HIGH)
        pending = controller.decide(inputs(
            frame_id=121, gop_id=4, is_keyframe=False,
            capacity_published_mbps=1.0,
            rgb_buffered_amount=120, buffer_trend_bytes=20,
        ))
        self.assertTrue(pending.switch_requested)
        duplicate = controller.decide(inputs(
            frame_id=122, gop_id=4, is_keyframe=False,
            capacity_published_mbps=1.0,
            rgb_buffered_amount=120, buffer_trend_bytes=20,
        ))
        self.assertFalse(duplicate.switch_requested)
        applied = controller.decide(inputs(
            frame_id=150, gop_id=5, is_keyframe=True,
            capacity_published_mbps=1.0,
            rgb_buffered_amount=120, buffer_trend_bytes=20,
        ))
        self.assertTrue(applied.switch_applied)
        self.assertEqual(applied.applied_quality, Quality.LOW)

    def test_probe_authorization_upgrades_only_at_keyframe(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        pending = controller.decide(
            inputs(
                frame_id=59, gop_id=1, is_keyframe=False,
                probe_authorization_valid=True,
            )
        )
        self.assertEqual(pending.applied_quality, Quality.LOW)
        applied = controller.decide(
            inputs(
                frame_id=60, gop_id=2, is_keyframe=True,
                probe_authorization_valid=True,
            )
        )
        self.assertTrue(applied.switch_applied)
        self.assertEqual(applied.applied_quality, Quality.HIGH)

    def test_expired_authorization_cannot_upgrade(self):
        controller = AdaptationController(
            "combined", self.cfg, initial_quality=Quality.LOW
        )
        for gop in range(2, 6):
            result = controller.decide(
                inputs(
                    frame_id=gop * 30, gop_id=gop,
                    probe_authorization_valid=False,
                )
            )
        self.assertEqual(result.applied_quality, Quality.LOW)

    def test_fixed_and_legacy_modes(self):
        self.assertEqual(
            AdaptationController("fixed_high", self.cfg).decide(
                inputs(rgb_buffered_amount=999)
            ).applied_quality,
            Quality.HIGH,
        )
        self.assertEqual(
            AdaptationController("fixed_low", self.cfg).decide(inputs()).applied_quality,
            Quality.LOW,
        )
        self.assertEqual(
            AdaptationController("legacy", self.cfg).decide(
                inputs(capacity_published_mbps=1.0, rgb_buffered_amount=999)
            ).applied_quality,
            Quality.HIGH,
        )

    def test_causal_decision_does_not_accept_future_input(self):
        controller_a = AdaptationController("combined", self.cfg)
        first_a = controller_a.decide(inputs())
        controller_a.decide(inputs(timestamp=99, capacity_published_mbps=1.0))
        controller_b = AdaptationController("combined", self.cfg)
        first_b = controller_b.decide(inputs())
        self.assertEqual(first_a, first_b)


if __name__ == "__main__":
    unittest.main()
