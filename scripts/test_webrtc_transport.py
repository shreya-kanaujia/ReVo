#!/usr/bin/env python3
"""Deterministic checks for the validation-only aiortc SCTP workaround."""

import asyncio
import os
import sys
import csv
import tempfile
from collections import deque
from types import SimpleNamespace

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
)

import webrtc_diagnostics


class FakeSctp:
    def __init__(self, first):
        self._sent_queue = deque([first])
        self.samples = []

    def _update_rto(self, sample_s):
        self.samples.append(sample_s)

    async def _receive_sack_chunk(self, _chunk):
        self._update_rto(45.0)


async def exercise(first_acked):
    first = SimpleNamespace(tsn=100, _acked=first_acked, _sent_count=1)
    sctp = FakeSctp(first)
    peer_connection = SimpleNamespace(sctp=sctp)

    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "aiortc.rtcsctptransport":
            return SimpleNamespace(uint32_gte=lambda left, right: left >= right)
        return original_import(name, *args, **kwargs)

    import builtins

    builtins.__import__ = fake_import
    try:
        webrtc_diagnostics.apply_sctp_gap_rtt_fix(peer_connection, "Test")
    finally:
        builtins.__import__ = original_import

    await sctp._receive_sack_chunk(SimpleNamespace(cumulative_tsn=100))
    return sctp


def test_gap_acked_sample_is_suppressed():
    sctp = asyncio.run(exercise(True))
    assert sctp.samples == []
    assert sctp._revo_suppressed_rtt_samples == 1
    assert sctp._revo_last_suppressed_rtt_s == 45.0


def test_unambiguous_sample_updates_rto():
    sctp = asyncio.run(exercise(False))
    assert sctp.samples == [45.0]
    assert sctp._revo_suppressed_rtt_samples == 0


def test_message_classifier_identifies_media_and_probe():
    import struct
    media = struct.pack(
        "<BBIIBHHHIQdd", 2, 3, 150, 150, 30, 4, 12, 8, 5000,
        999, 1.0, 0.0,
    ) + b"payload"
    result = webrtc_diagnostics.classify_revo_message(media)
    assert result["message_type"] == "rgb_media"
    assert result["frame_id"] == 150
    assert result["media_chunk_index"] == 4
    assert result["media_sequence"] == 999
    probe = struct.pack("<BBQQBdI", 7, 1, 22, 7, 0, 1.0, 4) + b"data"
    result = webrtc_diagnostics.classify_revo_message(probe)
    assert result["message_type"] == "capacity_probe"
    assert result["probe_id"] == 22
    assert result["probe_sequence"] == 7


def test_event_diagnostics_maps_message_to_tsn_and_logs_t3():
    class EventSctp:
        def __init__(self):
            self._local_tsn = 100
            self._sent_queue = deque()
            self._outbound_queue = deque()
            self._data_channel_queue = deque()
            self._data_channels = {0: SimpleNamespace(label="rgb_payload")}
            self._flight_size = 0
            self._cwnd = 4800
            self._rto = 1.0
            self._t3_handle = None
            self._advanced_peer_ack_tsn = 99

        async def _send_chunk(self, chunk):
            pass

        async def _send(self, stream_id, pp_id, user_data, **kwargs):
            chunk = SimpleNamespace(
                tsn=self._local_tsn, stream_id=stream_id, flags=7,
                user_data=user_data, _sent_count=1,
            )
            self._local_tsn += 1
            self._sent_queue.append(chunk)
            await self._send_chunk(chunk)

        async def _receive_sack_chunk(self, chunk):
            self._advanced_peer_ack_tsn = chunk.cumulative_tsn

        def _t3_expired(self):
            self._t3_handle = None

        async def _receive_data_chunk(self, chunk):
            pass

        async def _receive(self, stream_id, pp_id, data):
            pass

    async def exercise(path):
        sctp = EventSctp()
        logger = webrtc_diagnostics.SctpEventDiagnostics(path, "test")
        logger.instrument(SimpleNamespace(sctp=sctp))
        media = __import__("struct").pack(
            "<BBIIBHHHIQdd", 2, 3, 150, 150, 30, 0, 1, 1, 10,
            42, 1.0, 0.0,
        ) + b"x"
        await sctp._send(0, 53, media, max_retransmits=0, ordered=False)
        await sctp._receive_sack_chunk(
            SimpleNamespace(cumulative_tsn=100, gaps=[])
        )
        sctp._t3_expired()
        logger.close()

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "events.csv")
        asyncio.run(exercise(path))
        with open(path, newline="") as handle:
            rows = list(csv.DictReader(handle))
    events = [row["event"] for row in rows]
    assert "message_submit" in events
    assert "data_send" in events
    assert "sack_receive" in events
    assert "t3_expire" in events
    sent = next(row for row in rows if row["event"] == "data_send")
    assert sent["message_type"] == "rgb_media"
    assert sent["frame_id"] == "150"
    assert sent["tsn"] == "100"


def main():
    tests = [
        test_gap_acked_sample_is_suppressed,
        test_unambiguous_sample_updates_rto,
        test_message_classifier_identifies_media_and_probe,
        test_event_diagnostics_maps_message_to_tsn_and_logs_t3,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} WebRTC transport checks passed.")


if __name__ == "__main__":
    main()
