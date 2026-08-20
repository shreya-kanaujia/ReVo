#!/usr/bin/env python3
"""Regression test: receiver output is bounded and finalized incrementally."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import sys

import av
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_receiver_module():
    sys.path.insert(0, str(ROOT / "src/receiver"))
    spec = importlib.util.spec_from_file_location(
        "receiver_3d", ROOT / "src/receiver/receiver-3d.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReceiverStreamingOutputTests(unittest.TestCase):
    def test_incremental_writer_does_not_retain_frame_lists(self):
        module = load_receiver_module()
        receiver = object.__new__(module.Receiver)
        with tempfile.TemporaryDirectory() as tmp:
            receiver.media_file = str(Path(tmp) / "rgb.mp4")
            receiver.media_file_depth = str(Path(tmp) / "depth.mp4")
            receiver.fps = 25
            receiver.codec = SimpleNamespace(intra_period=30)
            receiver.output_containers = None
            receiver.output_streams = None
            receiver.output_frame_count = 0
            frame = np.zeros((64, 64, 3), dtype=np.uint8)
            for _ in range(300):
                receiver._write_output_pair(frame, frame)
            receiver._finalize_output_writers()
            self.assertEqual(receiver.output_frame_count, 300)
            self.assertFalse(hasattr(receiver, "saved_frames"))
            for name in ("rgb.mp4", "depth.mp4"):
                with av.open(Path(tmp) / name) as container:
                    self.assertEqual(sum(1 for _ in container.decode(video=0)), 300)


if __name__ == "__main__":
    unittest.main()
