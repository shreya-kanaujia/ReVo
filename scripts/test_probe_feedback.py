#!/usr/bin/env python3
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from probe_feedback import (  # noqa: E402
    PROBE_FLAG_END,
    PROBE_FLAG_START,
    ProbeAck,
    decode_probe_ack,
    decode_probe_data,
    encode_probe_ack,
    encode_probe_data,
)


class ProbeFeedbackTests(unittest.TestCase):
    def test_probe_data_round_trip_and_distinct_payload(self):
        encoded = encode_probe_data(
            9, 3, PROBE_FLAG_START | PROBE_FLAG_END, 12.5, b"x" * 37
        )
        decoded = decode_probe_data(encoded)
        self.assertEqual(decoded.probe_id, 9)
        self.assertEqual(decoded.sequence, 3)
        self.assertEqual(decoded.payload_bytes, 37)
        self.assertEqual(decoded.payload, b"x" * 37)
        self.assertNotEqual(encoded[0], 1)  # video descriptor type

    def test_probe_ack_round_trip(self):
        ack = ProbeAck(7, 11, PROBE_FLAG_END, 22.25, 1024)
        self.assertEqual(decode_probe_ack(encode_probe_ack(ack)), ack)

    def test_malformed_messages_are_rejected(self):
        data = encode_probe_data(1, 1, 0, 1.0, b"abc")
        self.assertIsNone(decode_probe_data(data[:-1]))
        ack = bytearray(encode_probe_ack(ProbeAck(1, 1, 0, 1.0, 3)))
        ack[1] = 255
        self.assertIsNone(decode_probe_ack(bytes(ack)))


if __name__ == "__main__":
    unittest.main()
