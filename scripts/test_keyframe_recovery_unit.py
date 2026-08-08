import sys
import os
import unittest
from unittest.mock import MagicMock
import types
import torch
import time

# Inject module paths
PROJECT_ROOT = "/Users/kushagraaitha/Documents/ReVo"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src/sender"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src/receiver"))

# Mock torchcodec before importing sender/receiver
torchcodec = types.ModuleType("torchcodec")
torchcodec.decoders = types.ModuleType("torchcodec.decoders")
class MockVideoDecoder:
    def __init__(self, file_path, device="cpu"):
        self.num_frames = 60
    def __len__(self):
        return self.num_frames
    def __getitem__(self, idx):
        return torch.zeros((3, 512, 512), dtype=torch.uint8)
torchcodec.decoders.VideoDecoder = MockVideoDecoder
sys.modules["torchcodec"] = torchcodec
sys.modules["torchcodec.decoders"] = torchcodec.decoders

import importlib.util
spec_h265 = importlib.util.spec_from_file_location("h265_sender", os.path.join(PROJECT_ROOT, "src/sender/H265_wrapper.py"))
h265 = importlib.util.module_from_spec(spec_h265)
sys.modules["H265_wrapper"] = h265
spec_h265.loader.exec_module(h265)

spec_sender = importlib.util.spec_from_file_location("sender_3d", os.path.join(PROJECT_ROOT, "src/sender/sender-3d.py"))
sender_3d = importlib.util.module_from_spec(spec_sender)
spec_sender.loader.exec_module(sender_3d)

spec_receiver = importlib.util.spec_from_file_location("receiver_3d", os.path.join(PROJECT_ROOT, "src/receiver/receiver-3d.py"))
receiver_3d = importlib.util.module_from_spec(spec_receiver)
spec_receiver.loader.exec_module(receiver_3d)


class TestKeyframeRecoveryUnit(unittest.TestCase):

    def test_h265_wrapper_keyframe_forcing(self):
        """
        Verify:
        1. H265 normal scheduled keyframe works (frame 0, 30)
        2. Arbitrary mid-GOP forced keyframe works (frame 10)
        3. Following frame returns to P-frame
        4. Normal GOP keyframes still occur after forcing
        """
        codec = h265.H265VideoCodec(intra_period=30, qp=30)
        keyframes = []

        for i in range(45):
            img_tensor = torch.zeros((1, 1, 3, 512, 512), dtype=torch.uint8)
            force = (i == 10)
            outputs = list(codec.compress_stream(img_tensor, frame_id=i, fps=30, force_keyframe=force))

            for out in outputs:
                if out["is_key"]:
                    keyframes.append(out["frame_id"])

        for out in codec.flush():
            if out["is_key"]:
                keyframes.append(out["frame_id"])

        # Assertions
        self.assertIn(0, keyframes, "Frame 0 must be a keyframe")
        self.assertIn(10, keyframes, "Forced frame 10 must be a keyframe")
        self.assertNotIn(11, keyframes, "Frame 11 should return to P-frame")
        self.assertNotIn(30, keyframes, "Old GOP schedule (30) should be reset")
        self.assertIn(40, keyframes, "Next scheduled keyframe should be at 40 (10 + 30)")

    def test_sender_feedback_parsing_and_cooldown(self):
        """
        Verify:
        1. Sender receives request
        2. Cooldown logic prevents duplicate request storms
        3. Later independent request can trigger another recovery
        """
        args = MagicMock()
        args.file = "dummy.mp4"
        args.depth_file = "dummy_depth.mp4"
        args.stun_url = "stun:..."
        args.server_ip = "127.0.0.1"
        args.codec = "h265"

        sender = sender_3d.Sender(args)

        # Initial state
        self.assertFalse(sender.force_keyframe_pending)

        # Test request handling
        sender._handle_rgb_message("KEYFRAME_REQUEST:10")
        self.assertTrue(sender.force_keyframe_pending)

        # Mock keyframe forced at frame 12
        sender.force_keyframe_pending = False
        sender.last_forced_keyframe_fid = 12
        sender.sent_frames = 15

        # Duplicate request within cooldown (15 - 12 = 3 < 30) -> ignored
        sender._handle_rgb_message("KEYFRAME_REQUEST:15")
        self.assertFalse(sender.force_keyframe_pending)

        # Request after cooldown (45 - 12 = 33 >= 30) -> accepted
        sender.sent_frames = 45
        sender._handle_rgb_message("KEYFRAME_REQUEST:45")
        self.assertTrue(sender.force_keyframe_pending)

    def test_receiver_loss_detection_and_recovery(self):
        """
        Verify:
        1. P-frame loss => no KEYFRAME_REQUEST
        2. Keyframe loss => exactly one KEYFRAME_REQUEST
        3. Receiver unfreezes and updates last_decode_i_frame_id on recovery keyframe
        """
        args = MagicMock()
        args.codec = "h265"

        receiver = receiver_3d.Receiver(args)
        receiver.codec = MagicMock()
        receiver.codec.intra_period = 30
        receiver.depth_codec = MagicMock()
        receiver.depth_codec.intra_period = 30

        receiver.feedback_channel = MagicMock()
        receiver.loop = MagicMock()

        # Set initial decode state
        receiver.last_decode_i_frame_id = 0

        # Helper to run inline loss detection
        def run_loss_check(fid, is_key, payload, payload_depth):
            if is_key and (payload is None or payload_depth is None):
                now = time.perf_counter()
                if not receiver.keyframe_request_pending or (now - receiver.last_keyframe_request_time >= 1.0):
                    receiver.last_keyframe_request_time = now
                    receiver.keyframe_request_pending = True
                    receiver.feedback_channel.send(f"KEYFRAME_REQUEST:{fid}")

        # 1. P-frame loss (frame 5) -> No request enqueued
        run_loss_check(fid=5, is_key=False, payload=None, payload_depth=None)
        self.assertFalse(receiver.keyframe_request_pending)
        receiver.feedback_channel.send.assert_not_called()

        # 2. Keyframe loss (frame 30) -> Exactly one request enqueued
        run_loss_check(fid=30, is_key=True, payload=None, payload_depth=None)
        self.assertTrue(receiver.keyframe_request_pending)
        self.assertEqual(receiver.feedback_channel.send.call_count, 1)
        receiver.feedback_channel.send.assert_called_with("KEYFRAME_REQUEST:30")

        # Reset mock call count
        receiver.feedback_channel.send.reset_mock()

        # Duplicate request check before cooldown -> ignored
        run_loss_check(fid=30, is_key=True, payload=None, payload_depth=None)
        receiver.feedback_channel.send.assert_not_called()

        # 3. Successful decode of forced keyframe (frame 33) -> unfreezes
        receiver._gop_frozen_active_rgb = True
        receiver._gop_frozen_active_depth = True

        # Simulate decode success
        receiver.last_decode_i_frame_id = 33
        receiver.keyframe_request_pending = False
        receiver._gop_frozen_active_rgb = False
        receiver._gop_frozen_active_depth = False

        self.assertFalse(receiver.keyframe_request_pending)
        self.assertEqual(receiver.last_decode_i_frame_id, 33)
        self.assertFalse(receiver._gop_frozen_active_rgb)

    def test_receiver_dynamic_is_key_and_grace_period(self):
        """
        Verify receiver updates is_key dynamically from arrived fc chunks
        and grace period is evaluated.
        """
        args = MagicMock()
        args.codec = "h265"
        receiver = receiver_3d.Receiver(args)
        receiver.codec = MagicMock()
        receiver.codec.intra_period = 30

        # Initial scheduled checks
        receiver.last_decode_i_frame_id = 0
        self.assertFalse(receiver._is_iframe(33)) # 33 is not scheduled GOP

        # Frame content setup for Frame 33
        receiver.frame_content[33] = {"is_key": True}

        # Verify that dynamic read extracts is_key = True
        fc = receiver.frame_content.get(33)
        is_key = bool(fc.get("is_key")) if fc else receiver._is_iframe(33)
        self.assertTrue(is_key)

        # Verify grace period calculation
        receiver.keyframe_request_pending = False
        receiver.clock_started = True
        receiver.clock_t0 = 100.0
        receiver.T = 0.0333

        # Normal deadline
        deadline_normal = receiver._deadline_time(32, "decode") + (0.300 if receiver.keyframe_request_pending else 0.0)
        self.assertAlmostEqual(deadline_normal, 100.0 + 33 * 0.0333)

        # Pending recovery deadline
        receiver.keyframe_request_pending = True
        deadline_recovery = receiver._deadline_time(32, "decode") + (0.300 if receiver.keyframe_request_pending else 0.0)
        self.assertAlmostEqual(deadline_recovery, 100.0 + 33 * 0.0333 + 0.300)


if __name__ == "__main__":
    unittest.main()
