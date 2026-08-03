"""Memory-bounded RGB/depth video output for long validation runs."""

from __future__ import annotations

import os

import av
import numpy as np


class StreamingVideoPairWriter:
    """Incrementally encode synchronized RGB/depth frames without retaining them."""

    def __init__(
        self,
        rgb_path: str,
        depth_path: str,
        *,
        fps: int,
        intra_period: int,
        crf: int = 0,
        preset: str = "ultrafast",
    ):
        if fps <= 0:
            raise ValueError("fps must be positive")
        self.rgb_path = rgb_path
        self.depth_path = depth_path
        self.fps = int(fps)
        self.intra_period = int(intra_period)
        self.crf = int(crf)
        self.preset = preset
        self._rgb_container = None
        self._depth_container = None
        self._rgb_stream = None
        self._depth_stream = None
        self.frame_count = 0
        self.closed = False

    @staticmethod
    def _uint8(frame):
        if frame.dtype == np.uint8:
            return frame
        return np.clip(frame * 255, 0, 255).astype(np.uint8)

    def _open_stream(self, path: str, frame):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        container = av.open(path, mode="w")
        stream = container.add_stream("libx264", rate=self.fps)
        stream.width = int(frame.shape[1])
        stream.height = int(frame.shape[0])
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": str(self.crf),
            "preset": self.preset,
            "g": str(self.intra_period),
            "threads": "1",
        }
        return container, stream

    @staticmethod
    def _append(container, stream, frame):
        video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(video_frame):
            container.mux(packet)

    def append(self, rgb, depth) -> None:
        if self.closed:
            raise RuntimeError("cannot append to a closed video writer")
        rgb = self._uint8(rgb)
        depth = self._uint8(depth)
        if rgb.shape != depth.shape:
            raise ValueError("RGB and depth frames must have matching shapes")
        if self._rgb_container is None:
            self._rgb_container, self._rgb_stream = self._open_stream(
                self.rgb_path, rgb
            )
            self._depth_container, self._depth_stream = self._open_stream(
                self.depth_path, depth
            )
        self._append(self._rgb_container, self._rgb_stream, rgb)
        self._append(self._depth_container, self._depth_stream, depth)
        self.frame_count += 1

    @staticmethod
    def _finish(container, stream):
        if container is None:
            return
        for packet in stream.encode(None):
            container.mux(packet)
        container.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self._finish(self._rgb_container, self._rgb_stream)
        self._finish(self._depth_container, self._depth_stream)
