#!/usr/bin/env python3
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gcc_controller import ReceiverGccEstimator, SenderGccController
import wifi_eval_runner
from sender.H265_wrapper_v2 import H265VideoCodec


class GccWifiIntegrationTests(unittest.TestCase):
    def test_gcc_is_a_distinct_runner_method(self):
        config = wifi_eval_runner.method_config("gcc")
        self.assertEqual(config["REVO_GCC_MODE"], "1")
        self.assertEqual(config["SALSIFY_MODE"], "0")
        self.assertEqual(config["SALSIFY_DUP_DELAY_MS"], "33")
        self.assertEqual(wifi_eval_runner.method_config("abr")["REVO_GCC_MODE"], "0")
        self.assertEqual(wifi_eval_runner.method_config("baseline")["REVO_GCC_MODE"], "0")

    def test_delay_gradient_does_not_require_clock_synchronization(self):
        a = ReceiverGccEstimator()
        b = ReceiverGccEstimator()
        for seq, fid, send, recv in ((1, 0, 10.0, 100.0),
                                     (2, 1, 10.04, 100.05),
                                     (3, 2, 10.08, 100.11)):
            a.observe(seq=seq, fid=fid, sender_time=send,
                      receiver_time=recv, size=1000)
            b.observe(seq=seq, fid=fid, sender_time=send + 5000.0,
                      receiver_time=recv - 7000.0, size=1000)
        fa = a.feedback(now=100.12)
        fb = b.feedback(now=-6899.88)
        self.assertAlmostEqual(fa["delay_gradient_ms"],
                               fb["delay_gradient_ms"], places=9)

    def test_duplicate_packet_is_excluded_from_loss_and_delay(self):
        estimator = ReceiverGccEstimator()
        self.assertTrue(estimator.observe(seq=1, fid=0, sender_time=1.0,
                                          receiver_time=2.0, size=100))
        self.assertFalse(estimator.observe(seq=1, fid=0, sender_time=1.0,
                                           receiver_time=2.033, size=100))
        self.assertEqual(len(estimator.loss_seqs), 1)
        self.assertEqual(estimator.frames[0]["t_recv_last"], 2.0)

    def test_loss_window_is_only_consumed_when_feedback_is_sent(self):
        estimator = ReceiverGccEstimator()
        estimator.observe(seq=1, fid=0, sender_time=1.0,
                          receiver_time=2.0, size=100)
        estimator.feedback(now=2.1, consume_loss=False)
        self.assertEqual(estimator.loss_seqs, [1])
        self.assertEqual(estimator.last_seq, -1)
        estimator.feedback(now=3.1, consume_loss=True)
        self.assertEqual(estimator.loss_seqs, [])
        self.assertEqual(estimator.last_seq, 1)

    def test_nonconsuming_ticks_preserve_cross_window_loss(self):
        estimator = ReceiverGccEstimator()
        estimator.observe(seq=1, fid=0, sender_time=1.0,
                          receiver_time=2.0, size=100)
        estimator.feedback(now=2.1, consume_loss=True)
        # Sequences 2..5 were lost before the first arrival in this window.
        estimator.observe(seq=6, fid=1, sender_time=2.0,
                          receiver_time=3.0, size=100)
        peek = estimator.feedback(now=3.05, consume_loss=False)
        self.assertAlmostEqual(peek["loss_rate"], 4 / 5)
        self.assertEqual(estimator.last_seq, 1)
        report = estimator.feedback(now=4.0, consume_loss=True)
        self.assertAlmostEqual(report["loss_rate"], 4 / 5)
        self.assertEqual(estimator.last_seq, 6)

    def test_delay_overuse_can_decrease_remote_rate(self):
        estimator = ReceiverGccEstimator()
        estimator.rate_bps = 1_000_000.0
        estimator.last_rate_update = 0.0
        # Increasing receiver spacing relative to a fixed 40 ms sender cadence
        # drives the filtered delay gradient over its adaptive threshold long
        # enough to enter DECREASE.
        recv = 10.0
        seq = 1
        for fid in range(140):
            recv += 0.065
            estimator.observe(seq=seq, fid=fid, sender_time=fid * 0.040,
                              receiver_time=recv, size=1000)
            seq += 1
            estimator.feedback(now=recv, consume_loss=False)
        feedback = estimator.feedback(now=recv + 0.2, consume_loss=True)
        self.assertEqual(feedback["state"], "DECREASE")
        self.assertLess(feedback["A_r"], 1_000_000.0)

    def test_feedback_changes_qp_request(self):
        sender = SenderGccController()
        changed = sender.update({"A_r": 150_000, "loss_rate": 0.5}, now=3.0)
        self.assertTrue(changed)
        self.assertGreater(sender.target_qp, 30)
        self.assertLess(sender.target_rate_bps, 800_000)

    def test_increase_then_decrease_reaches_both_codecs_at_boundary(self):
        sender = SenderGccController()
        self.assertTrue(sender.update(
            {"A_r": 1_200_000, "loss_rate": 0.0}, now=3.0))
        increased_quality_qp = sender.target_qp
        self.assertLess(increased_quality_qp, 30)
        before = sender.target_rate_bps
        for now in (6.0, 7.0, 8.0):
            sender.update({"A_r": 150_000, "loss_rate": 0.5}, now=now)
        self.assertLess(sender.target_rate_bps, before)
        self.assertGreater(sender.target_qp, increased_quality_qp)

        rgb = H265VideoCodec(qp=increased_quality_qp, intra_period=30)
        depth = H265VideoCodec(qp=increased_quality_qp, intra_period=30)
        rgb.set_qp(sender.target_qp)
        depth.set_qp(sender.target_qp)
        with rgb.lock, depth.lock:
            rgb._apply_pending_qp_locked(29, force_keyframe=False)
            depth._apply_pending_qp_locked(29, force_keyframe=False)
            self.assertEqual(rgb.qp_p, increased_quality_qp)
            self.assertEqual(depth.qp_p, increased_quality_qp)
            rgb._apply_pending_qp_locked(30, force_keyframe=True)
            depth._apply_pending_qp_locked(30, force_keyframe=True)
        self.assertEqual(rgb.qp_p, sender.target_qp)
        self.assertEqual(depth.qp_p, sender.target_qp)

    def test_qp_actuation_is_deferred_to_a_real_keyframe_boundary(self):
        codec = H265VideoCodec(qp=30, intra_period=30)
        codec.set_qp(36)
        with codec.lock:
            codec._apply_pending_qp_locked(7, force_keyframe=False)
        self.assertEqual(codec.qp_p, 30)
        with codec.lock:
            codec._apply_pending_qp_locked(7, force_keyframe=True)
        self.assertEqual(codec.qp_p, 36)
        self.assertIsNone(codec._pending_qp)

    def test_runtime_uses_source_fps_and_delayed_duplicate(self):
        sender = (ROOT / "src/sender/sender-3d.py").read_text()
        receiver = (ROOT / "src/receiver/receiver-3d.py").read_text()
        self.assertIn("decoder.metadata.average_fps", sender)
        self.assertIn("SALSIFY_DUP_DELAY_MS / 1000.0", sender)
        self.assertIn("frame_deadline - time.perf_counter()", sender)
        self.assertIn('feedback["A_r"] < 0.97 * self.gcc_last_sent_rate',
                      receiver)
        self.assertIn('feedback["loss_rate"] = self.gcc._loss(consume=True)',
                      receiver)
        self.assertNotIn("self.data_channel_rgb.send(first_rgb_packet)\n"
                         "                            self.data_channel_depth.send(first_depth_packet)",
                         sender.split("if SALSIFY_DUP_DELAY_MS > 0:", 1)[1].split("else:", 1)[0])

    def test_gcc_manifest_configuration_is_frozen(self):
        self.assertEqual(wifi_eval_runner.method_config("gcc")["REVO_GCC_MODE"], "1")
        self.assertIn("REVO_EXPECTED_GCC_SHA256",
                      wifi_eval_runner.RUNTIME_IDENTITY_FILES)


if __name__ == "__main__":
    unittest.main()
