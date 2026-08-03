import time
import torch
import numpy as np
import av
from fractions import Fraction
import collections


class H265VideoCodec:
    """
    Simple H.265 (HEVC) streaming wrapper using PyAV.

    Interface is compatible with your sender/receiver:
      - compress_stream(frames, frame_id, fps) -> yields dict with
        {frame_id, is_key, payload, qp, height, width}
      - decompress_stream(frame_dict, ...) -> yields dict with
        {frame_id, is_key, decoded_frame}
    """

    def __init__(self,
                 qp: int = 30,#22,#30,
                 intra_period: int = 30,
                 pool_threads: int | None = None):
        self.qp_i = qp
        self.qp_p = qp
        self.intra_period = intra_period
        self.pool_threads = (
            None if pool_threads is None else max(1, int(pool_threads))
        )
        self.enc = None
        self.dec = None
        self.width = None
        self.height = None
        self.fps = 30
        # Queue of frame_ids that are "in flight" in x265's pipeline
        self._inflight_ids = collections.deque()

    # Internal helpers
    def _ensure_encoder(self, width: int, height: int, fps: int):
        """
        Lazily create and configure the libx265 encoder.
        We keep it zero-latency and constant-QP.
        """
        if self.enc is not None:
            return

        self.width = width
        self.height = height
        self.fps = fps
        ctx = av.CodecContext.create("libx265", "w")
        ctx.width = width
        ctx.height = height
        ctx.pix_fmt = "yuv420p"
        ctx.time_base = Fraction(1, fps)
        # Choose one QP for the whole stream.
        qp_stream = self.qp_p

        x265_params = (
            f"qp={qp_stream}:"
            f"keyint={self.intra_period}:"
            f"min-keyint={self.intra_period}:"
            "scenecut=0:bframes=0:rc-lookahead=0:no-scenecut=1:frame-threads=1"
        )
        if self.pool_threads is not None:
            x265_params += f":pools={self.pool_threads}"
        ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",  # avoid latency / lookahead
            # constant-QP, no B-frames, fixed keyint, minimal buffering
            "x265-params": x265_params,
        }
        ctx.open()
        self.enc = ctx

    def _ensure_decoder(self):
        """
        Lazily create the HEVC (H.265) decoder.
        """
        if self.dec is not None:
            return

        ctx = av.CodecContext.create("hevc", "r")  # decoder for H.265
        # [ADD THIS] Allow the decoder to output incomplete/corrupt frames
        # This corresponds to AV_CODEC_FLAG_OUTPUT_CORRUPT
        ctx.options = {"flags": "output_corrupt"}
        
        ctx.open()
        self.dec = ctx

    def compress_stream(self, frames: torch.Tensor, frame_id: int, fps: int = 30):
        """
        frames: (1, 1, C, H, W) torch tensor in [0,1] or [0,255], RGB
        Yields at most ONE dict per call, but possibly zero (if encoder is buffering):

            {
              "frame_id": int,   # output id, matched via _inflight_ids
              ...
            }
        """
        x = self.prepare_input(frames)
        yield from self.compress_prepared(x, frame_id, fps=fps)

    @staticmethod
    def prepare_input(frames: torch.Tensor):
        """Convert a source tensor once for reuse by independent QP encoders."""
        _, _, C, _, _ = frames.shape
        assert C == 3, "Expected RGB (C=3)"
        x = frames[0, 0]  # (C, H, W)
        if x.dtype != torch.uint8:
            x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        return x.permute(1, 2, 0).contiguous().cpu().numpy()

    def compress_prepared(self, rgb_array, frame_id: int, fps: int = 30):
        """Encode a prepared, read-only RGB ndarray with this codec context."""
        H, W, C = rgb_array.shape
        assert C == 3, "Expected RGB (H,W,3)"
        self._ensure_encoder(W, H, fps)

        # remember which *input* frame this call corresponds to
        self._inflight_ids.append(frame_id)

        frame = av.VideoFrame.from_ndarray(rgb_array, format="rgb24")

        # Encode this frame; libx265 may or may not output a packet yet.
        packets = self.enc.encode(frame)
        if not packets:
            return  # no output yet; keep inflight_ids as-is

        # x265 can emit multiple packets for one frame; we concatenate them
        payload = b"".join(bytes(p) for p in packets)
        if not payload:
            return

        # [DM] H265 can buffer frames internally. So, don't dequeue frame_ids, 
        # till actual payload is encoded by H265.

        # The earliest "inflight" frame now gets its payload
        out_id = self._inflight_ids.popleft()

        is_key = (out_id == 0 or
                  (self.intra_period > 0 and out_id % self.intra_period == 0))
        qp = self.qp_i if is_key else self.qp_p

        yield {
            "frame_id": out_id,
            "is_key": is_key,
            "payload": payload,
            "qp": qp,
            "height": H,
            "width": W,
        }

    def flush(self, fps: int = 30):
        """
        Flush any remaining frames from the encoder pipeline.
        Yields zero or more outputs in the same format as compress_stream().
        """
        if self.enc is None:
            return

        while True:
            packets = self.enc.encode(None)
            if not packets:
                break

            payload = b"".join(bytes(p) for p in packets)
            if not payload:
                continue

            if not self._inflight_ids:
                # Safety: encoder produced more frames than we tracked.
                # Just drop them or assign -1.
                out_id = -1
            else:
                out_id = self._inflight_ids.popleft()

            H, W = self.height, self.width
            is_key = (out_id == 0 or
                      (self.intra_period > 0 and out_id % self.intra_period == 0))
            qp = self.qp_i if is_key else self.qp_p

            yield {
                "frame_id": out_id,
                "is_key": is_key,
                "payload": payload,
                "qp": qp,
                "height": H,
                "width": W,
            }

    def decompress_stream(self, frame, pic_height=None, pic_width=None, fps: int = 30):
        """
        frame: dict from sender:
          {
            "frame_id": int,
            "is_key": bool,
            "payload": bytes,
            "qp": int,
            ...
          }

        Yields:
          {
            "frame_id": int,
            "is_key": bool,
            "decoded_frame": np.ndarray (H, W, 3) uint8
          }
        """
        frame_id = frame["frame_id"]
        is_key = frame["is_key"]
        payload = frame["payload"]

        self._ensure_decoder()

        # payload -> Packet
        pkt = av.packet.Packet(payload)

        # decode; in our low-latency config, this should give at most 1 frame
        decoded_frames = self.dec.decode(pkt)
        for f in decoded_frames:
            # convert to RGB24 ndarray (H, W, 3) uint8
            rgb = f.to_ndarray(format="rgb24")
            yield {
                "frame_id": frame_id,
                "is_key": is_key,
                "decoded_frame": rgb,
            }
            break  # only one frame per payload is expected
