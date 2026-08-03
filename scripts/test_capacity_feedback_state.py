#!/usr/bin/env python3
"""Deterministic sender-side capacity-feedback freshness tests."""

import os
import sys
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from sender.adaptation_controller import (  # noqa: E402
    AdaptationController,
    ControllerConfig,
    ControllerInputs,
)
from sender.capacity_feedback_state import CapacityFeedbackState  # noqa: E402


def accept(
    store,
    *,
    seq=1,
    receiver_ts=100.0,
    age=0.01,
    fresh=True,
    threshold=0.05,
    received_ts=100.2,
    packet_rtt=2.0,
    filtered=8.0,
    feedback_seq=None,
    app_seq=None,
    recovering=False,
    unavailable=False,
):
    feedback_seq = seq if feedback_seq is None else feedback_seq
    app_seq = seq if app_seq is None else app_seq
    stale = not fresh and not recovering and not unavailable
    return store.accept(
        feedback_sequence_id=feedback_seq,
        application_sequence_id=app_seq,
        receiver_timestamp=receiver_ts,
        receiver_raw_update_age_s=age,
        receiver_published_estimate_age_s=age,
        receiver_estimate_fresh=fresh,
        receiver_estimate_stale=stale,
        receiver_estimate_recovering=recovering,
        receiver_estimate_unavailable=unavailable,
        estimate_state_reason=(
            "published" if fresh else
            "startup_warmup" if unavailable else
            "sequence_gap" if recovering else
            "publication_expired"
        ),
        freshness_threshold_s=threshold,
        raw_capacity_mbps=7.5,
        filtered_capacity_mbps=filtered,
        last_valid_update_ts=receiver_ts - (age or 0.0),
        last_published_estimate_ts=(
            receiver_ts - age if age is not None else None
        ),
        feedback_received_ts=received_ts,
        packet_rtt_s=packet_rtt,
    )


class CapacityFeedbackStateTests(unittest.TestCase):
    def test_immediate_receipt_uses_receiver_age_not_packet_rtt(self):
        store = CapacityFeedbackState()
        self.assertEqual(accept(store), "accepted_fresh")
        state = store.snapshot(100.2)
        self.assertAlmostEqual(state["estimate_age_s"], 0.01)
        self.assertAlmostEqual(state["packet_rtt_s"], 2.0)
        self.assertTrue(state["estimate_fresh"])

    def test_age_grows_only_by_sender_local_elapsed_time(self):
        store = CapacityFeedbackState()
        accept(store, received_ts=50.0, receiver_ts=1.0, age=0.01)
        self.assertAlmostEqual(store.snapshot(50.02)["estimate_age_s"], 0.03)
        self.assertAlmostEqual(store.snapshot(50.04)["estimate_age_s"], 0.05)
        self.assertFalse(store.snapshot(50.041)["estimate_fresh"])

    def test_reverse_delivery_delay_is_added_when_clocks_are_shared(self):
        store = CapacityFeedbackState(shared_monotonic_clock=True)
        accept(
            store, receiver_ts=10.0, received_ts=10.02, age=0.01,
            threshold=0.05, packet_rtt=1.5
        )
        state = store.snapshot(10.02)
        self.assertAlmostEqual(state["feedback_delivery_delay_s"], 0.02)
        self.assertAlmostEqual(state["estimate_age_s"], 0.03)
        self.assertTrue(state["estimate_fresh"])

    def test_unshared_clocks_are_never_subtracted(self):
        store = CapacityFeedbackState(shared_monotonic_clock=False)
        accept(store, receiver_ts=5000.0, received_ts=10.0)
        state = store.snapshot(10.0)
        self.assertIsNone(state["feedback_delivery_delay_s"])
        self.assertEqual(
            state["feedback_clock_status"],
            "not_comparable_age_uncertainty",
        )

    def test_reordered_and_duplicate_feedback_cannot_overwrite_newer_state(self):
        store = CapacityFeedbackState()
        accept(
            store, feedback_seq=3, app_seq=10,
            received_ts=1.0, receiver_ts=0.9, filtered=6.0
        )
        self.assertEqual(
            accept(
                store, feedback_seq=2, app_seq=11,
                received_ts=1.1, receiver_ts=1.0, filtered=2.0
            ),
            "reordered",
        )
        self.assertEqual(
            accept(
                store, feedback_seq=3, app_seq=12,
                received_ts=1.2, receiver_ts=1.1, filtered=3.0
            ),
            "duplicate",
        )
        state = store.snapshot(1.2)
        self.assertEqual(state["feedback_sequence_id"], 3)
        self.assertEqual(state["filtered_estimated_capacity_mbps"], 6.0)
        self.assertEqual(state["application_sequence_id"], 10)

    def test_application_ack_sequence_is_independent_of_feedback_order(self):
        store = CapacityFeedbackState()
        self.assertEqual(
            accept(store, feedback_seq=1, app_seq=50), "accepted_fresh"
        )
        self.assertEqual(
            accept(store, feedback_seq=2, app_seq=49), "accepted_fresh"
        )
        state = store.snapshot(100.2)
        self.assertEqual(state["feedback_sequence_id"], 2)
        self.assertEqual(state["application_sequence_id"], 49)

    def test_newer_recovery_feedback_explicitly_invalidates_publication(self):
        store = CapacityFeedbackState()
        accept(store, seq=1, received_ts=1.0, receiver_ts=0.9)
        self.assertTrue(store.snapshot(1.0)["estimate_fresh"])
        result = accept(
            store,
            seq=2,
            received_ts=1.01,
            receiver_ts=1.0,
            age=0.0,
            fresh=False,
            filtered=None,
            recovering=True,
        )
        self.assertEqual(result, "accepted_sequence_gap")
        state = store.snapshot(1.01)
        self.assertFalse(state["estimate_fresh"])
        self.assertIsNone(state["published_estimated_capacity_mbps"])
        self.assertTrue(state["estimate_recovering"])
        self.assertEqual(state["last_published_capacity_mbps"], 8.0)

    def test_fresh_state_survives_until_threshold_for_controller_sampling(self):
        store = CapacityFeedbackState()
        accept(store, received_ts=1.0, receiver_ts=0.5, age=0.0)
        state = store.snapshot(1.04)
        controller = AdaptationController("combined", ControllerConfig())
        inputs = ControllerInputs(
            timestamp=1.04,
            frame_id=1,
            gop_id=0,
            is_keyframe=False,
            capacity_raw_mbps=state["raw_estimated_capacity_mbps"],
            capacity_filtered_mbps=state["filtered_estimated_capacity_mbps"],
            capacity_published_mbps=state[
                "published_estimated_capacity_mbps"
            ],
            capacity_age_s=state["estimate_age_s"],
            capacity_fresh=state["estimate_fresh"],
            rgb_buffered_amount=0,
            depth_buffered_amount=0,
            buffer_trend_bytes=0,
            receiver_impairment_rate=0.0,
            receiver_health_age_s=0.0,
            receiver_health_fresh=True,
            predicted_high_demand_mbps=1.0,
            predicted_low_demand_mbps=0.5,
        )
        decision = controller.decide(inputs)
        self.assertTrue(inputs.capacity_fresh)
        self.assertAlmostEqual(inputs.capacity_age_s, 0.04)
        self.assertAlmostEqual(inputs.capacity_published_mbps, 8.0)
        self.assertAlmostEqual(decision.capacity_safe_mbps, 6.8)

    def test_invalid_values_and_clock_regression_are_rejected(self):
        store = CapacityFeedbackState()
        self.assertEqual(accept(store, seq=0), "invalid")
        self.assertEqual(accept(store, age=-0.1), "invalid")
        self.assertEqual(accept(store, filtered=float("nan")), "invalid")
        accept(store, received_ts=2.0, receiver_ts=1.0)
        with self.assertRaises(ValueError):
            store.snapshot(1.9)

    def test_reset_clears_connection_state(self):
        store = CapacityFeedbackState()
        accept(store)
        store.reset()
        state = store.snapshot(200.0)
        self.assertEqual(state["feedback_sequence_id"], 0)
        self.assertFalse(state["estimate_fresh"])
        self.assertIsNone(state["estimate_age_s"])


if __name__ == "__main__":
    unittest.main()
