#!/usr/bin/env python3
import csv
import json
from pathlib import Path
import tempfile
import unittest

from track_b_event_gate import (
    DEFAULT_EVENT_TIMEOUT_S,
    EventGateState,
    validate_run,
)


def probe_success(ts=1.0):
    return [{
        "probe_state": "SUCCEEDED", "authorization_created": "1",
        "monotonic_timestamp": str(ts),
    }]


def applied_switch(frame=60, ts=2.0):
    return [{
        "switch_applied": "1", "applied_quality": "high",
        "keyframe": "1", "frame_id": str(frame),
        "timestamp_monotonic": str(ts),
    }]


def complete_rows(start=60, count=30):
    timing = []
    measurements = []
    for fid in range(start, start + count):
        timing.extend((
            {"event": "assembly_decode", "frame_id": str(fid),
             "assembly_ready": "1", "assembly_reason": "ready"},
            {"event": "display", "frame_id": str(fid),
             "frozen_display": "0"},
        ))
        for stream in ("rgb", "depth"):
            measurements.append({
                "frame_id": str(fid), "stream": stream, "sent": "1",
            })
    return timing, measurements


class EventGateTests(unittest.TestCase):
    def test_sixty_seconds_is_only_the_hard_upper_bound(self):
        self.assertEqual(DEFAULT_EVENT_TIMEOUT_S, 60.0)
        state = EventGateState()
        timing, sent = complete_rows()
        state.update(probe_success(), applied_switch(), timing, sent)
        self.assertTrue(state.complete)

    def test_probe_success_alone_does_not_complete(self):
        state = EventGateState()
        state.update(probe_success(), [], [], [])
        self.assertIsNotNone(state.probe_success_ts)
        self.assertFalse(state.complete)

    def test_requested_high_without_applied_switch_does_not_complete(self):
        state = EventGateState()
        requested = [{
            "switch_applied": "0", "applied_quality": "low",
            "requested_quality": "high", "keyframe": "0",
            "frame_id": "59", "timestamp_monotonic": "2.0",
        }]
        state.update(probe_success(), requested, [], [])
        self.assertIsNone(state.switch_frame)

    def test_applied_switch_waits_for_all_thirty_frames(self):
        state = EventGateState()
        timing, sent = complete_rows(count=29)
        state.update(probe_success(), applied_switch(), timing, sent)
        self.assertEqual(state.switch_frame, 60)
        self.assertEqual(state.target_gop_end, 89)
        self.assertFalse(state.complete)

    def test_gate_waits_for_final_receiver_display_classification(self):
        state = EventGateState()
        timing, sent = complete_rows()
        timing = [row for row in timing if not (
            row["frame_id"] == "89" and row["event"] == "display"
        )]
        state.update(probe_success(), applied_switch(), timing, sent)
        self.assertFalse(state.complete)
        timing.append({"event": "display", "frame_id": "89",
                       "frozen_display": "0"})
        state.update(probe_success(), applied_switch(), timing, sent)
        self.assertTrue(state.complete)

    def test_timeout_reasons_are_phase_specific(self):
        state = EventGateState()
        self.assertEqual(state.timeout_reason(), "timeout_no_successful_probe")
        state.update(probe_success(), [], [], [])
        self.assertEqual(state.timeout_reason(), "timeout_no_applied_high_switch")
        state.update(probe_success(), applied_switch(), [], [])
        self.assertEqual(state.timeout_reason(), "timeout_incomplete_high_gop")

    def test_timeout_never_becomes_success(self):
        state = EventGateState()
        self.assertFalse(state.complete)
        self.assertEqual(state.timeout_reason(), "timeout_no_successful_probe")

    def test_timeout_runner_requests_clean_validation_stop(self):
        runner = (Path(__file__).resolve().parent /
                  "run_track_b_two_container.sh").read_text()
        timeout_block = runner.split(
            'if [[ "$GATE_STATUS" -ne 0 ]]', 1
        )[1].split("\n  fi", 1)[0]
        self.assertIn('touch "$STOP_SIGNAL"', timeout_block)
        self.assertNotIn('kill "$SENDER_PID"', timeout_block)
        self.assertIn("artifact_validator_exit_status.txt", runner)
        self.assertIn("event_gate_validator_exit_status.txt", runner)

    def test_media_health_failure_fails_immediately(self):
        state = EventGateState()
        state.update([], [], [{
            "event": "assembly_decode", "frame_id": "12",
            "assembly_ready": "0", "assembly_reason": "no_media",
        }], [])
        self.assertIn("media_health_failure", state.failure)

    def test_complete_gop_succeeds_only_after_transmission_and_evaluation(self):
        state = EventGateState()
        timing, sent = complete_rows()
        state.update(probe_success(), applied_switch(), timing, sent)
        self.assertTrue(state.complete)
        self.assertIsNone(state.failure)

    def test_queue_drain_failure_fails_postvalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            (run / "metadata").mkdir()
            (run / "metadata" / "event_gate_live.json").write_text(
                json.dumps({"status": "complete"})
            )
            self._write(run / "sender_measurements.csv", [
                {"frame_id": "0", "stream": "rgb", "sent": "1"},
                {"frame_id": "0", "stream": "depth", "sent": "1"},
            ])
            self._write(run / "receiver_media_timing.csv", [
                {"event": "assembly_decode", "frame_id": "0",
                 "assembly_ready": "1", "frozen_display": ""},
                {"event": "display", "frame_id": "0",
                 "assembly_ready": "", "frozen_display": "0"},
            ])
            diagnostic = {
                "rgb_buffered_amount": "0", "depth_buffered_amount": "0",
                "sctp_data_channel_queue": "0", "sctp_outbound_queue": "0",
                "sctp_sent_queue": "1", "sctp_sent_outstanding": "1",
                "outstanding_packets": "0",
            }
            self._write(run / "sender_diagnostics.csv", [diagnostic])
            diagnostic = dict(diagnostic, sctp_sent_queue="0",
                              sctp_sent_outstanding="0")
            self._write(run / "receiver_diagnostics.csv", [diagnostic])
            self._write(run / "sender_sctp_events.csv", [{"event": "ok"}])
            self.assertEqual(validate_run(run), 1)
            report = json.loads(
                (run / "metadata" / "event_gate_validation.json").read_text()
            )
            self.assertTrue(any("queue_drain_failure" in issue
                                for issue in report["issues"]))

    @staticmethod
    def _write(path, rows):
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
