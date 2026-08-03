#!/usr/bin/env python3
import os
import struct
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from receiver_health_feedback import (  # noqa: E402
    FORMAT,
    MSG_RECEIVER_HEALTH,
    RECEIVER_HEALTH_VERSION,
    V1_SIZE,
    ReceiverHealthFeedback,
    ReceiverHealthInbox,
    decode_receiver_health,
    encode_receiver_health,
)


class ReceiverHealthFeedbackTests(unittest.TestCase):
    def sample(self):
        return ReceiverHealthFeedback(
            feedback_seq=9,
            receiver_ts=12.5,
            window_start_frame=30,
            window_end_frame=59,
            total_frames=30,
            full_misses=1,
            partial_frames=1,
            decode_failures=0,
            reference_unavailable=0,
            frozen_frames=1,
            impairment_rate=0.1,
        )

    def test_round_trip_preserves_every_field(self):
        original = self.sample()
        self.assertEqual(decode_receiver_health(encode_receiver_health(original)), original)

    def test_wrong_type_version_or_size_is_rejected(self):
        payload = bytearray(encode_receiver_health(self.sample()))
        payload[0] = MSG_RECEIVER_HEALTH + 1
        self.assertIsNone(decode_receiver_health(payload))
        payload = bytearray(encode_receiver_health(self.sample()))
        payload[1] = RECEIVER_HEALTH_VERSION + 1
        self.assertIsNone(decode_receiver_health(payload))
        self.assertIsNone(decode_receiver_health(b"short"))

    def test_out_of_range_rate_is_rejected(self):
        values = list(struct.unpack(FORMAT, encode_receiver_health(self.sample())))
        values[-1] = 1.1
        self.assertIsNone(decode_receiver_health(struct.pack(FORMAT, *values)))

    def test_old_or_reordered_feedback_is_rejected_and_freshness_expires(self):
        inbox = ReceiverHealthInbox(freshness_s=2.0)
        payload = encode_receiver_health(self.sample())
        self.assertTrue(inbox.accept(payload, received_ts=10.0))
        self.assertFalse(inbox.accept(payload, received_ts=11.0))
        self.assertTrue(inbox.snapshot(11.9)["health_fresh"])
        self.assertFalse(inbox.snapshot(12.1)["health_fresh"])

    def test_shared_clock_feedback_delay_is_diagnostic_only(self):
        payload = encode_receiver_health(self.sample())
        ordinary = ReceiverHealthInbox(freshness_s=2.0)
        shared = ReceiverHealthInbox(
            freshness_s=2.0, shared_monotonic_clock=True
        )
        self.assertTrue(ordinary.accept(payload, received_ts=12.75))
        self.assertTrue(shared.accept(payload, received_ts=12.75))
        self.assertIsNone(ordinary.snapshot(13.0)["feedback_delay_s"])
        self.assertAlmostEqual(
            shared.snapshot(13.0)["feedback_delay_s"], 0.25
        )

    def test_v1_decode_remains_compatible(self):
        legacy = ReceiverHealthFeedback(
            **{
                **self.sample().to_dict(),
                "version": 1,
                "reference_unavailable": 0,
            }
        )
        payload = encode_receiver_health(legacy)
        self.assertEqual(len(payload), V1_SIZE)
        decoded = decode_receiver_health(payload)
        self.assertEqual(decoded.version, 1)
        self.assertEqual(decoded.reference_unavailable, 0)


if __name__ == "__main__":
    unittest.main()
