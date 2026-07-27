#!/usr/bin/env python3
"""Deterministic checks for the validation-only aiortc SCTP workaround."""

import asyncio
import os
import sys
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


def main():
    tests = [
        test_gap_acked_sample_is_suppressed,
        test_unambiguous_sample_updates_rto,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} WebRTC transport checks passed.")


if __name__ == "__main__":
    main()
