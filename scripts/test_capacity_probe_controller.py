#!/usr/bin/env python3
import inspect
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from probe_feedback import (  # noqa: E402
    PROBE_FLAG_END,
    PROBE_HEADER_SIZE,
    ProbeAck,
)
from sender.capacity_probe_controller import (  # noqa: E402
    CapacityProbeController,
    CausalOfferedRate,
    ProbeBufferGrowthBaseline,
    ProbeConfig,
    ProbeInputs,
    ProbeState,
    frames_until_keyframe,
)


def healthy(**updates):
    values = dict(
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
        severe_downgrade=False,
    )
    values.update(updates)
    return ProbeInputs(**values)


class CapacityProbeControllerTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ProbeConfig(
            payload_bytes=1000,
            min_duration_s=0.2,
            max_duration_s=0.3,
            max_total_bytes=200000,
            max_additional_rate_mbps=10,
            max_rate_multiplier=8,
            min_confirmed_fraction=0.8,
            ack_timeout_s=0.1,
            cooldown_s=0.5,
            authorization_ttl_s=0.4,
            buffer_abort_bytes=1000,
            buffer_growth_abort_bytes=100,
            boundary_guard_s=0.05,
        )

    def started(self):
        controller = CapacityProbeController(self.cfg)
        state = controller.evaluate(healthy())
        self.assertEqual(state["state"], ProbeState.PROBING.value)
        return controller

    def test_healthy_low_is_eligible_but_high_is_not(self):
        self.assertEqual(
            CapacityProbeController(self.cfg).evaluate(healthy())["state"],
            "PROBING",
        )
        result = CapacityProbeController(self.cfg).evaluate(
            healthy(current_quality="high")
        )
        self.assertEqual(result["state"], "WAITING_FOR_ELIGIBILITY")

    def test_current_keyframe_is_zero_frames_away_and_cannot_start_probe(self):
        self.assertEqual(frames_until_keyframe(60, 30), 0)
        self.assertEqual(frames_until_keyframe(61, 30), 29)
        state = CapacityProbeController(self.cfg).evaluate(
            healthy(keyframe=True, frame_id=60, seconds_to_keyframe=0.0)
        )
        self.assertEqual(state["state"], "WAITING_FOR_ELIGIBILITY")
        self.assertEqual(state["offered_bytes"], 0)

    def test_pre_probe_keyframe_buffer_is_the_activation_baseline(self):
        tracker = ProbeBufferGrowthBaseline(8300, 0)
        state = tracker.observe(8300, 0)
        self.assertEqual(state["media_buffer_baseline_bytes"], 8300)
        self.assertEqual(state["media_buffer_growth_bytes"], 0)
        config = ProbeConfig(**{
            **self.cfg.__dict__,
            "buffer_abort_bytes": 20000,
        })
        controller = CapacityProbeController(config)
        result = controller.evaluate(healthy(
            rgb_buffer=8300,
            buffer_trend=state["media_buffer_growth_bytes"],
        ))
        self.assertEqual(result["state"], "PROBING")

    def test_media_growth_after_activation_still_aborts(self):
        tracker = ProbeBufferGrowthBaseline(200, 0)
        controller = CapacityProbeController(self.cfg)
        controller.evaluate(healthy(rgb_buffer=200, buffer_trend=0))
        state = tracker.observe(350, 0)
        result = controller.evaluate(healthy(
            now=1.01,
            rgb_buffer=350,
            buffer_trend=state["media_buffer_growth_bytes"],
        ))
        self.assertEqual(state["media_buffer_growth_bytes"], 150)
        self.assertEqual(result["state"], "ABORTED")
        self.assertEqual(result["reason"], "buffer_growth_abort")

    def test_probe_enqueue_credit_is_not_media_growth(self):
        tracker = ProbeBufferGrowthBaseline(0, 0)
        tracker.record_probe_enqueue(0, 1000)
        probe_only = tracker.observe(1000, 0)
        self.assertEqual(probe_only["probe_buffer_credit_bytes"], 1000)
        self.assertEqual(probe_only["media_buffer_growth_bytes"], 0)
        with_media = tracker.observe(1150, 0)
        self.assertEqual(with_media["media_buffer_bytes"], 150)
        self.assertEqual(with_media["media_buffer_growth_bytes"], 150)

    def test_buffer_and_health_gate_start(self):
        for update in (
            {"buffer_trend": 100},
            {"receiver_impairment": 0.05},
            {"receiver_health_fresh": False},
            {"transport_connected": False},
            {"dwell_complete": False},
        ):
            with self.subTest(update=update):
                state = CapacityProbeController(self.cfg).evaluate(
                    healthy(**update)
                )
                self.assertEqual(state["state"], "WAITING_FOR_ELIGIBILITY")

    def test_unaligned_passive_deterioration_does_not_abort_healthy_probe(self):
        controller = self.started()
        state = controller.evaluate(healthy(
            now=1.01,
            passive_deterioration_unaligned=True,
            rgb_buffer=0,
            depth_buffer=0,
            buffer_trend=0,
            receiver_impairment=0.0,
            receiver_health_fresh=True,
        ))
        self.assertEqual(state["state"], "PROBING")
        self.assertTrue(state["passive_deterioration_ignored"])
        self.assertEqual(state["reason"], "probing")

    def test_target_is_causal_difference_and_bounded(self):
        controller = CapacityProbeController(self.cfg)
        state = controller.evaluate(healthy())
        self.assertAlmostEqual(state["target_high_mbps"], 4)
        self.assertAlmostEqual(
            state["target_additional_mbps"], 4 / 0.85 - 2
        )
        bounded = CapacityProbeController(
            ProbeConfig(**{**self.cfg.__dict__, "max_additional_rate_mbps": 1})
        ).evaluate(healthy())
        self.assertEqual(bounded["target_additional_mbps"], 1)

    def test_only_one_probe_and_cooldown(self):
        controller = self.started()
        first = controller.snapshot(1.0)["probe_id"]
        controller.evaluate(healthy(now=1.01))
        self.assertEqual(controller.snapshot(1.01)["probe_id"], first)
        controller.cancel_authorization("test_abort", 1.02)
        self.assertEqual(controller.evaluate(healthy(now=1.03))["state"], "COOLDOWN")
        self.assertEqual(
            controller.evaluate(healthy(now=1.40))["state"], "COOLDOWN"
        )

    def test_duplicate_reordered_and_future_ack(self):
        controller = self.started()
        packet1 = controller.next_packet(1.0)
        packet2 = controller.next_packet(1.01)
        ack2 = ProbeAck(packet2.probe_id, packet2.sequence, 0, 1.01, 1000)
        self.assertEqual(controller.accept_ack(ack2, 1.01), "accepted")
        self.assertEqual(controller.accept_ack(ack2, 1.02), "duplicate")
        ack1 = ProbeAck(
            packet1.probe_id,
            packet1.sequence,
            packet1.flags,
            1.0,
            1000,
        )
        self.assertEqual(
            controller.accept_ack(ack1, 1.03), "accepted_reordered"
        )
        self.assertEqual(controller.snapshot(1.03)["confirmed_bytes"], 2000)
        self.assertEqual(
            controller.snapshot(1.03)["confirmed_application_bytes"],
            2000 + 2 * PROBE_HEADER_SIZE,
        )
        future = ProbeAck(packet2.probe_id, 99, 0, 1.0, 1000)
        self.assertEqual(
            controller.accept_ack(future, 1.04), "invalid_future_sequence"
        )

    def test_ack_must_match_the_exact_sent_packet(self):
        controller = self.started()
        packet = controller.next_packet(1.0)
        wrong_bytes = ProbeAck(
            packet.probe_id,
            packet.sequence,
            packet.flags,
            1.0,
            packet.payload_bytes + 1,
        )
        self.assertEqual(
            controller.accept_ack(wrong_bytes, 1.01),
            "payload_or_flags_mismatch",
        )
        wrong_flags = ProbeAck(
            packet.probe_id,
            packet.sequence,
            PROBE_FLAG_END,
            1.0,
            packet.payload_bytes,
        )
        self.assertEqual(
            controller.accept_ack(wrong_flags, 1.02),
            "payload_or_flags_mismatch",
        )
        self.assertEqual(controller.snapshot(1.02)["confirmed_bytes"], 0)

    def test_probe_application_rate_counts_typed_header_exactly_once(self):
        controller = self.started()
        packet = controller.next_packet(1.0)
        snapshot = controller.snapshot(1.0)
        self.assertEqual(snapshot["offered_bytes"], packet.payload_bytes)
        self.assertEqual(
            snapshot["offered_application_bytes"],
            packet.payload_bytes + PROBE_HEADER_SIZE,
        )
        controller.accept_ack(
            ProbeAck(
                packet.probe_id, packet.sequence, packet.flags, 1.0,
                packet.payload_bytes,
            ),
            1.001,
        )
        snapshot = controller.snapshot(1.001)
        self.assertEqual(
            snapshot["confirmed_application_bytes"],
            packet.payload_bytes + PROBE_HEADER_SIZE,
        )

    def test_absolute_pacing_deadline_does_not_accumulate_scheduler_delay(self):
        controller = self.started()
        self.assertIsNotNone(controller.next_packet(1.0))
        # At this late turn more than one nominal deadline has passed. Each
        # subsequent call can catch up without shifting the schedule to now.
        self.assertIsNotNone(controller.next_packet(1.010))
        self.assertIsNotNone(controller.next_packet(1.0101))

    def test_end_ack_does_not_discard_earlier_reordered_acks(self):
        controller = self.started()
        first = controller.next_packet(1.0)
        second = controller.next_packet(1.01)
        end = None
        for _ in range(1000):
            packet = controller.next_packet(1.31)
            if packet is not None and packet.flags & PROBE_FLAG_END:
                end = packet
                break
        self.assertIsNotNone(end)
        self.assertTrue(end.flags & PROBE_FLAG_END)
        self.assertEqual(
            controller.accept_ack(
                ProbeAck(end.probe_id, end.sequence, end.flags, 1.0, 0),
                1.001,
            ),
            "accepted",
        )
        still_probing = controller.evaluate(healthy(now=1.002))
        self.assertEqual(still_probing["state"], "PROBING")
        for packet in (second, first):
            controller.accept_ack(
                ProbeAck(
                    packet.probe_id,
                    packet.sequence,
                    packet.flags,
                    1.0,
                    packet.payload_bytes,
                ),
                1.003,
            )
        self.assertEqual(controller.snapshot(1.003)["confirmed_bytes"], 2000)

    def test_buffer_health_and_severe_abort_immediately(self):
        for update, reason in (
            ({"rgb_buffer": 1000}, "buffer_abort"),
            ({"buffer_trend": 100}, "buffer_growth_abort"),
            ({"receiver_health_fresh": False}, "receiver_health_stale"),
            ({"receiver_impairment": 0.05}, "receiver_health_bad"),
            ({"severe_downgrade": True}, "severe_downgrade"),
        ):
            with self.subTest(update=update):
                controller = self.started()
                state = controller.evaluate(healthy(now=1.01, **update))
                self.assertEqual(state["state"], "ABORTED")
                self.assertEqual(state["reason"], reason)
                self.assertFalse(state["authorization_valid"])

    def test_missing_ack_fails(self):
        controller = self.started()
        controller.next_packet(1.0)
        state = controller.evaluate(healthy(now=1.11))
        self.assertEqual(state["state"], "FAILED")
        self.assertEqual(state["reason"], "ack_timeout")

    def test_ack_stall_timeout_fails_safely(self):
        controller = self.started()
        packet = controller.next_packet(1.0)
        controller.accept_ack(
            ProbeAck(
                packet.probe_id, packet.sequence, packet.flags, 1.0,
                packet.payload_bytes,
            ),
            1.001,
        )
        state = controller.evaluate(healthy(now=1.102))
        self.assertEqual(state["state"], "FAILED")
        self.assertEqual(state["reason"], "ack_stalled")
        self.assertFalse(state["authorization_valid"])

    def _complete(self, *, confirm_fraction=1.0, offered_rate_ok=True):
        inputs = healthy(
            current_offered_video_mbps=(2.0 if offered_rate_ok else 0.1),
            predicted_low_demand_mbps=(2.0 if offered_rate_ok else 0.1),
            predicted_high_demand_mbps=4.0,
        )
        controller = CapacityProbeController(self.cfg)
        controller.evaluate(inputs)
        now = 1.0
        end = None
        while now <= 1.32:
            packet = controller.next_packet(now)
            if packet is not None:
                should_ack = (
                    confirm_fraction >= 1.0
                    or packet.flags & PROBE_FLAG_END
                    or packet.sequence % 2 == 0
                )
                if should_ack:
                    ack = ProbeAck(
                        packet.probe_id,
                        packet.sequence,
                        packet.flags,
                        now,
                        packet.payload_bytes,
                    )
                    controller.accept_ack(ack, now + 0.001)
                if packet.flags & PROBE_FLAG_END:
                    end = packet
                    break
            now += 0.001
        self.assertIsNotNone(end)
        decision_time = (
            now + 0.002
            if confirm_fraction >= 1.0
            else now + self.cfg.ack_timeout_s + 0.002
        )
        return controller, controller.evaluate(ProbeInputs(
            **{**inputs.__dict__, "now": decision_time}
        ))

    def test_success_needs_duration_bytes_rate_and_creates_expiry(self):
        controller, result = self._complete()
        self.assertEqual(result["state"], "SUCCEEDED")
        self.assertTrue(result["authorization_valid"])
        self.assertGreater(result["confirmed_bytes"], 0)
        self.assertGreater(
            result["confirmed_application_bytes"], result["confirmed_bytes"]
        )
        self.assertGreater(result["measured_probe_mbps"], 0)
        self.assertFalse(controller.snapshot(result["authorization_expiry"] + 0.01)[
            "authorization_valid"
        ])

    def test_insufficient_confirmations_never_authorize(self):
        _controller, result = self._complete(confirm_fraction=0.5)
        self.assertEqual(result["state"], "FAILED")
        self.assertFalse(result["authorization_valid"])
        self.assertIn("insufficient_confirmed_bytes", result["reason"])

    def test_rate_failure_never_authorizes(self):
        _controller, result = self._complete(offered_rate_ok=False)
        self.assertEqual(result["state"], "FAILED")
        self.assertFalse(result["authorization_valid"])

    def test_offered_rate_accounting_is_causal(self):
        rate = CausalOfferedRate(alpha=1.0)
        self.assertAlmostEqual(rate.observe(125000, 1.0), 1.0)
        before = rate.mbps
        self.assertEqual(before, 1.0)

    def test_no_trace_or_future_inputs(self):
        names = set(ProbeInputs.__dataclass_fields__)
        self.assertFalse(any("trace" in name or "ground" in name for name in names))
        source = inspect.getsource(CapacityProbeController)
        self.assertNotIn("ground_truth", source)


if __name__ == "__main__":
    unittest.main()
