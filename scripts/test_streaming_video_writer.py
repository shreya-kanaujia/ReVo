#!/usr/bin/env python3
import os
from pathlib import Path
import sys
import tempfile
import unittest

import av
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "receiver"))

from streaming_video_writer import StreamingVideoPairWriter  # noqa: E402


class StreamingVideoWriterTests(unittest.TestCase):
    def test_writes_synchronized_frames_and_releases_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            rgb_path = os.path.join(tmp, "rgb.mp4")
            depth_path = os.path.join(tmp, "depth.mp4")
            writer = StreamingVideoPairWriter(
                rgb_path, depth_path, fps=25, intra_period=30
            )
            for value in range(5):
                frame = np.full((32, 32, 3), value * 20, dtype=np.uint8)
                writer.append(frame, frame)
            writer.close()
            writer.close()
            self.assertEqual(writer.frame_count, 5)
            for path in (rgb_path, depth_path):
                with av.open(path) as container:
                    self.assertEqual(
                        sum(1 for _ in container.decode(video=0)), 5
                    )

    def test_rejects_mismatched_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            writer = StreamingVideoPairWriter(
                os.path.join(tmp, "rgb.mp4"),
                os.path.join(tmp, "depth.mp4"),
                fps=25,
                intra_period=30,
            )
            with self.assertRaises(ValueError):
                writer.append(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    np.zeros((16, 16, 3), dtype=np.uint8),
                )


if __name__ == "__main__":
    unittest.main()
