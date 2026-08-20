import time
import torch
import numpy as np
import av
from fractions import Fraction
import collections

import logging
logging.basicConfig(level=logging.INFO)

import threading

RED = "\033[31m"
GREEN = '\033[32m'
BLUE = '\033[34m'
RESET = '\033[0m'

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
                 intra_period: int = 30):
        self.qp_i = qp
        self.qp_p = qp
        self.intra_period = intra_period
        self.enc = None
        self.dec = None
        self.width = None
        self.height = None
        self.fps = 30
        # Queue of frame_ids that are "in flight" in x265's pipeline
        self._inflight_ids = collections.deque()

        # QP change requested by ABR/GCC, applied only at the next GOP boundary
        self._pending_qp = None

        self.lock = threading.Lock()
    
    def set_qp(self, new_qp: int):
        """
        Request a QP change. The change is NOT applied immediately: it is
        deferred until the next GOP boundary (frame_id % intra_period == 0)
        inside compress_stream().

        Rationale: re-initializing libx265 mid-GOP forces an IDR at an
        arbitrary frame position, but both sender and receiver label
        keyframes POSITIONALLY (fid % intra_period == 0). A mid-GOP re-init
        therefore (a) ships a real IDR labeled as a P-frame (no FEC,
        zero-fill on loss -> corrupt GOP), (b) shifts the encoder's GOP
        phase away from the positional labels, and (c) can leave stale ids
        in _inflight_ids, misaligning every subsequent frame id.
        Deferring the re-init to a boundary makes the fresh IDR land exactly
        where both sides already expect an I-frame.
        """
        with self.lock:
            if new_qp == self.qp_p and self._pending_qp is None:
                return
            self._pending_qp = int(new_qp)
            logging.info(f"{GREEN}[H265VideoCodec] QP change to {new_qp} queued; "
                         f"will apply at next GOP boundary.{RESET}")

    def _apply_pending_qp_locked(self, frame_id: int, force_keyframe: bool = False):
        """
        Called with self.lock held, BEFORE encoding frame_id.
        Applies a pending QP change only at a GOP boundary by rebuilding the
        encoder, so the resulting IDR coincides with the positional I-frame.
        """
        if self._pending_qp is None:
            return
        is_boundary = (force_keyframe or
                       (self.intra_period > 0 and frame_id % self.intra_period == 0))
        if not is_boundary:
            return

        self.qp_i = self._pending_qp
        self.qp_p = self._pending_qp
        self._pending_qp = None

        # Any frames still tracked as in-flight belong to the old encoder
        # context and will never be emitted by the new one. Drop them so
        # frame-id <-> payload matching stays aligned. (With
        # zerolatency/bframes=0/rc-lookahead=0 this deque should already be
        # empty here; the warning flags the unexpected case.)
        if self._inflight_ids:
            logging.warning(f"{RED}[H265VideoCodec] Dropping {len(self._inflight_ids)} "
                            f"in-flight frame ids on GOP-boundary re-init: "
                            f"{list(self._inflight_ids)}{RESET}")
            self._inflight_ids.clear()

        self.enc = None  # _ensure_encoder() rebuilds with the new QP
        logging.info(f"{GREEN}[H265VideoCodec] Applied QP {self.qp_p} at GOP boundary "
                     f"(frame {frame_id}); encoder re-init aligned with I-frame.{RESET}")

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

        ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",  # avoid latency / lookahead
            # constant-QP, no B-frames, fixed keyint, minimal buffering
            "x265-params": (
                f"qp={qp_stream}:"
                f"keyint={self.intra_period}:"
                f"min-keyint={self.intra_period}:"
                "scenecut=0:bframes=0:rc-lookahead=0:no-scenecut=1:frame-threads=1"
            ),
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

    def compress_stream(self, frames: torch.Tensor, frame_id: int, fps: int = 30,
                        force_keyframe: bool = False):
        """
        frames: (1, 1, C, H, W) torch tensor in [0,1] or [0,255], RGB
        Yields at most ONE dict per call, but possibly zero (if encoder is buffering):

            {
              "frame_id": int,   # output id, matched via _inflight_ids
              ...
            }
        """
        _, _, C, H, W = frames.shape
        assert C == 3, "Expected RGB (C=3)"

        # Apply any deferred QP change first (only takes effect at a GOP
        # boundary), and only then register this frame as in-flight so a
        # boundary re-init never orphans the current frame's id.
        with self.lock:
            self._apply_pending_qp_locked(frame_id, force_keyframe)

        # remember which *input* frame this call corresponds to
        self._inflight_ids.append(frame_id)

        # torch -> uint8 RGB ndarray (H, W, 3)
        x = frames[0, 0]  # (C, H, W)
        if x.dtype != torch.uint8:
            x = (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        x = x.permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        frame = av.VideoFrame.from_ndarray(x, format="rgb24")
        if force_keyframe:
            frame.pict_type = av.video.frame.PictureType.I
            frame.key_frame = True

        # --- CRITICAL SECTION START ---
        with self.lock:
            self._ensure_encoder(W, H, fps)
            # Encode frame safely
            packets = self.enc.encode(frame)
        # --- CRITICAL SECTION END ---

        # Encode this frame; libx265 may or may not output a packet yet.
        # packets = self.enc.encode(frame)
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

        is_key = any(getattr(p, "is_keyframe", False) for p in packets)

        # Sanity check: the positional label must match the actual bitstream.
        # If these ever disagree, GOP alignment between sender and receiver is
        # broken (this was the mid-GOP re-init failure mode).
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
