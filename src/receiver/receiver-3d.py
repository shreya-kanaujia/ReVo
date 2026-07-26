"""
receiver-3d.py  —  ReVo

WebRTC receiver for synchronized RGB + depth video streams.

Pipeline overview:
  Signaling server (WebSocket)
      │
      ▼
  RTCPeerConnection (WebRTC)
      │
      ▼
  DataChannel  ──►  on_message()          (async, network thread)
                        │
                        ▼
                   frame_content[]         (shared dict, fc_lock)
                        │
                        ▼
              _decode_worker_thread        (background thread)
                        │
                        ▼
                   display_buf[]           (shared dict, display_lock)
                        │
                        ▼
              _display_worker_thread       (background thread)
                        │
                        ▼
                 saved_frames[]  ──►  write_video_pyav()
"""

import argparse, asyncio, json, logging, sys
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiortc.contrib.media import MediaRecorder
from aiohttp import ClientSession
import aiohttp
import numpy as np
import DCVCRT_wrapper as dcvc
import H265_wrapper as h265
import H264_wrapper as h264
import time
import torch, gc
import av
import numpy as np
import cv2
import struct
import threading
from zfec import Decoder
import os
from capacity_estimator import ArrivalCapacityEstimator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from webrtc_diagnostics import (
    WebRTCDiagnostics,
    apply_ice_consent_timeout,
    enable_ice_debug_logging,
)

logging.basicConfig(level=logging.INFO)

# ANSI color codes for log readability
RED   = "\033[31m"
GREEN = '\033[32m'
BLUE  = '\033[34m'
RESET = '\033[0m'

# ---------------------------------------------------------------------------
# Control-plane message types sent over the reliable DataChannel
# ---------------------------------------------------------------------------
MSG_INIT  = 1   # stream parameters (width, height, fps, chunk sizes)
MSG_DESC  = 2   # per-chunk descriptor (frame id, chunk index, FEC params, …)
MSG_CHUNK = 3   # (unused; payload is appended directly after MSG_DESC)
MSG_PROBE = 5   # validation-only packet-train probe
MSG_CAPACITY_FEEDBACK = 6

# ---------------------------------------------------------------------------
# Binary wire formats (little-endian)
#
# INIT  – type:u8 | width:u16 | height:u16 | fps:u16
#           | chunk_size_rgb:u16 | chunk_size_depth:u16
#
# DESC  – type:u8 | frame_type:u8 | frame_id:u32 | gop_id:u32
#           | qp:u8 | chunk_idx:u16 | num_chunks(n):u16
#           | k_data:u16 | total_size:u32 | seq_id:u64
#           | sender_send_ts:f64 | sender_grace_period:f64
#         (raw shard bytes follow immediately after the fixed header)
# ---------------------------------------------------------------------------
FMT_INIT = "<BHHHHH"
FMT_DESC = "<BBIIBHHHIQdd"
FMT_PROBE = "<BQIIHd"
FMT_CAPACITY_FEEDBACK = "<BQIddddd"

# Frame-type tags that travel in the DESC header
FRAME_TYPE_RGB   = 3
FRAME_TYPE_DEPTH = 4

SZ_INIT = struct.calcsize(FMT_INIT)
SZ_DESC = struct.calcsize(FMT_DESC)
SZ_PROBE = struct.calcsize(FMT_PROBE)
SZ_CAPACITY_FEEDBACK = struct.calcsize(FMT_CAPACITY_FEEDBACK)

# Hard cap on in-flight frame slots to prevent unbounded memory growth
MAX_FRAME_CHUNK_LIMIT = 2000


class Receiver():
    """
    Receives a dual-stream (RGB + depth) video over a WebRTC DataChannel,
    reassembles chunked / FEC-protected frames, decodes them with the
    chosen codec, and writes the result to two MP4 files.

    Threading model
    ───────────────
    • asyncio event loop  – handles WebRTC signaling and DataChannel messages
    • _decode_worker_thread – waits for frame assembly, runs the codec
    • _display_worker_thread – paces output to wall-clock time, saves frames
    """

    def __init__(self, args):
        # ── Output paths ────────────────────────────────────────────────────
        self.media_file = args.out
        if args.out_depth is not None:
            self.media_file_depth = args.out_depth
        else:
            # Auto-derive depth path: e.g. "out.mp4" → "out_depth.mp4"
            stem, ext = os.path.splitext(self.media_file)
            self.media_file_depth = stem + "_depth" + ext

        # ── Network / WebRTC ────────────────────────────────────────────────
        self.stun_url          = args.stun
        self.signalling_server = f"ws://{args.server_ip}:8080/ws/demo"
        self.cfg               = None
        self.pc                = None
        self.ice_consent_timeout_s = args.ice_consent_timeout_s
        self._ice_consent_timeout_applied = False

        # ── Stream parameters (overwritten by INIT message) ─────────────────
        self.pic_height        = 512
        self.pic_width         = 512
        self.fps               = 30
        self.chunk_size        = 1024   # RGB chunk size in bytes
        self.chunk_size_depth  = 512    # depth chunk size in bytes

        # ── Codec selection ──────────────────────────────────────────────────
        # RGB codec
        self.codec = h265.H265VideoCodec(intra_period=30)
        if args.codec == "dcvcrt":
            self.codec = dcvc.DCVCVideoCodec()
        if args.codec == "h264":
            self.codec = h264.H264VideoCodec(intra_period=30)

        # Depth codec (mirrors RGB codec choice)
        self.depth_codec = h265.H265VideoCodec(intra_period=30)
        if args.codec == "dcvcrt":
            self.depth_codec = dcvc.DCVCVideoCodec()
        if args.codec == "h264":
            self.depth_codec = h264.H264VideoCodec(intra_period=30)

        # ── Frame-assembly state ─────────────────────────────────────────────
        self.stream_inited = False
        self.done_fids     = set()    # frame ids already consumed by decode thread

        # frame_content[fid] holds all metadata and received chunk shards for
        # a frame that is still being assembled.  See init_frame_content().
        self.frame_content = {}

        # ── Deadline clock ───────────────────────────────────────────────────
        # The clock starts when the first I-frame is submitted for decode.
        # Every frame fid has a display deadline:
        #   deadline(fid) = clock_t0 + (fid + 1) * T
        self.T             = 1.0 / float(self.fps)   # seconds per frame
        self.p_slack       = 0.010                   # safety margin (s)
        self.clock_started = False
        self.clock_t0      = 0.0
        self.clock_fid0    = 0
        self.first_packet_clock = None               # wall time of first packet

        # ── Decode/display pipeline ──────────────────────────────────────────
        self.stop_event    = asyncio.Event()
        self.stop_threads  = threading.Event()

        # fc_lock / fc_cv protect frame_content and expected_frame
        self.fc_lock = threading.Lock()
        self.fc_cv   = threading.Condition(self.fc_lock)

        # display_lock / display_cv protect display_buf
        self.display_lock = threading.Lock()
        self.display_cv   = threading.Condition(self.display_lock)

        # display_buf[fid] = (rgb_ndarray | None, depth_ndarray | None)
        # None means the frame was dropped → display thread freezes on last good frame
        self.display_buf           = {}
        self.display_next_fid      = 0
        self.last_displayed_frame  = None
        self.last_displayed_frame_depth = None

        # Ordered lists of frames written to disk (same order as display)
        self.saved_frames       = []
        self.saved_frames_depth = []

        # ── Counters / metrics ───────────────────────────────────────────────
        self.total_bytes_received       = 0
        self.total_bytes_received_depth = 0
        self.decode_times               = {}    # fid → decode latency (ms)
        self.decode_times_depth         = {}
        self.decoded_frames             = 0
        self.decoded_frames_depth       = 0
        self.lost_frames                = 0
        self.total_frames               = 0
        self.lost_frames_full           = 0     # no chunks arrived at all
        self.lost_frames_partial        = 0     # DESC arrived but some chunks missing

        # ── GOP / P-frame continuity ─────────────────────────────────────────
        # Tracks which I-frame the decoder last successfully decoded.
        # P-frames that belong to an older GOP are discarded to avoid artifacts.
        self.last_decode_i_frame_id = None
        self.pending_p_by_gop       = {}        # gop_id → {fid: (is_key, qp, payload)}

        # Next frame the decode thread expects to process
        self.expected_frame = 0

        # ── Per-GOP quality masks (written alongside saved frames) ───────────
        # corrupted_frame_list_*: True if a frame is known-corrupted (codec got
        #   bad data).  Propagates through the rest of the GOP once set.
        # frozen_mask_list_*: True if a frame was totally lost, causing the
        #   display to freeze on the previous good frame.
        self.corrupted_frame_list_rgb   = []
        self.corrupted_frame_list_depth = []
        self._gop_corrupted_active_rgb   = False
        self._gop_corrupted_active_depth = False

        self.frozen_mask_list_rgb    = []
        self.frozen_mask_list_depth  = []
        self._gop_frozen_active_rgb   = False
        self._gop_frozen_active_depth = False

        # Worker threads (started in run())
        self.decode_thread  = None
        self.display_thread = None

        self._last_chunk_gc = time.perf_counter()
        self.capacity_estimator = ArrivalCapacityEstimator(alpha=args.estimator_alpha)
        if args.diagnostic_ice:
            enable_ice_debug_logging()
        self.diagnostics = WebRTCDiagnostics(args.diagnostic_csv, "receiver")
        self.diagnostic_channels = {}

    # ────────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────────

    def _is_iframe(self, fid: int) -> bool:
        """Return True if fid is an intra (I) frame position in the GOP."""
        return (fid % int(self.codec.intra_period) == 0)

    def init_frame_content(self, fid):
        """
        Allocate a fresh assembly slot for frame fid.

        Fields:
          parts / parts_depth – list of received shards (None = not yet received)
          missing             – count of shards still awaited
          recv_count          – count of shards already received (used for FEC)
          k_data              – minimum shards needed to reconstruct (FEC parameter k)
          total_size          – unpadded payload length in bytes
          shard_len           – byte length of each FEC shard (uniform)
        """
        self.frame_content[fid] = {
            # shared
            "is_key": False,
            "gop_id": None,
            # RGB stream
            "qp":           self.codec.qp_p,
            "num_chunks":   0,
            "k_data":       None,
            "total_size":   0,
            "parts":        {},     # replaced with [None]*n on first chunk
            "missing":      0,
            "shard_len":    None,
            "recv_count":   0,
            # depth stream
            "qp_depth":         self.depth_codec.qp_p,
            "num_chunks_depth": 0,
            "k_data_depth":     None,
            "total_size_depth": 0,
            "parts_depth":      {},
            "missing_depth":    0,
            "shard_len_depth":  None,
            "recv_count_depth": 0,
        }

    def _deadline_time(self, fid: int, caller=None) -> float:
        """
        Wall-clock deadline for frame fid: the moment it should be displayed.
        deadline(fid) = clock_t0 + (fid + 1) * T
        """
        return self.clock_t0 + (fid + 1) * self.T

    def _send_capacity_feedback(self, channel, *, seq_id, packet_size,
                                receiver_ts, sender_grace_period):
        sample = self.capacity_estimator.observe(
            receiver_ts,
            sender_grace_period,
            sequence_id=seq_id,
            packet_size_bytes=packet_size,
        )
        def encode_optional(value):
            return -1.0 if value is None else float(value)
        feedback = struct.pack(
            FMT_CAPACITY_FEEDBACK,
            MSG_CAPACITY_FEEDBACK,
            int(seq_id),
            int(packet_size),
            float(receiver_ts),
            encode_optional(sample["raw_interarrival"]),
            float(sample["sender_grace_period"]),
            encode_optional(sample["corrected_interarrival"]),
            encode_optional(sample["ewma_interarrival"]),
        )
        try:
            channel.send(feedback)
            self.diagnostics.feedback_event(seq_id, receiver_ts)
        except Exception:
            logging.debug("[Receiver] capacity feedback send failed", exc_info=True)

    # ────────────────────────────────────────────────────────────────────────
    # Best-effort payload builder (P-frames with missing chunks)
    # ────────────────────────────────────────────────────────────────────────

    def _build_best_effort_payload(self, fid: int):
        """
        For a P-frame that has some but not all chunks, substitute missing
        chunks with zero-filled bytes of the correct size.  This lets the
        codec attempt a decode rather than dropping the frame entirely.

        Returns (rgb_payload, depth_payload) or (None, None) on unrecoverable loss.

        Hard rule: if chunk 0 is missing the frame is dropped unconditionally
        because most codecs require the first NAL/slice header to be intact.
        """
        f_content = self.frame_content.get(fid)
        if f_content is None:
            logging.warning(f"{RED}[FID: {fid}] Whole frame missing. lost.{RESET}")
            return None, None

        parts       = f_content.get("parts")
        parts_depth = f_content.get("parts_depth")
        try:
            num_chunks            = int(f_content.get("num_chunks", 0))
            total_size            = int(f_content.get("total_size", 0))
            num_missing_chunks    = int(f_content.get("missing", 0))
            num_chunks_depth      = int(f_content.get("num_chunks_depth", 0))
            total_size_depth      = int(f_content.get("total_size_depth", 0))
            num_missing_chunks_depth = int(f_content.get("missing_depth", 0))
        except Exception as e:
            logging.error(f"{RED}[FID: {fid}] Error parsing metadata: {e} | Content: {f_content}{RESET}")
            return None, None

        # Bail out if we have nothing at all for either stream
        if (len(parts) == 0 or num_chunks <= 0 or num_chunks == num_missing_chunks or
                len(parts_depth) == 0 or num_chunks_depth <= 0 or
                num_chunks_depth == num_missing_chunks_depth):
            logging.warning(f"{RED}[FID: {fid}] Whole frame missing.{RESET}")
            return None, None

        def _pad_chunks(chunk_list, total, chunk_sz, label):
            out = []
            for i in range(len(chunk_list)):
                if chunk_list[i] is None:
                    logging.warning(f"{RED}[FID: {fid}] chunk {i} of {label} frame is missing{RESET}")
                    if i == 0:
                        # Cannot recover without the first chunk
                        logging.warning(f"{RED}[FID: {fid}] First chunk ({label}) missing – dropping.{RESET}")
                        return None
                    # Last chunk may be shorter than chunk_sz
                    pad_len = (total % chunk_sz) if (i == len(chunk_list) - 1 and total % chunk_sz != 0) else chunk_sz
                    out.append(b"\x00" * pad_len)
                else:
                    out.append(chunk_list[i])
            return b"".join(out)

        rgb_payload   = _pad_chunks(parts,       total_size,       self.chunk_size,       "RGB")
        depth_payload = _pad_chunks(parts_depth, total_size_depth, self.chunk_size_depth, "depth")
        if rgb_payload is None or depth_payload is None:
            return None, None
        return rgb_payload, depth_payload

    # ────────────────────────────────────────────────────────────────────────
    # Decode worker thread
    # ────────────────────────────────────────────────────────────────────────

    def _decode_worker_thread(self):
        """
        Processes frames in strict display order (expected_frame, expected_frame+1, …).

        For each frame:
          1. Wait until the frame is fully assembled OR its deadline passes.
          2. Reconstruct payload:
               – I-frame: Reed-Solomon / zfec FEC decode from k-of-n shards
               – P-frame: complete payload, or best-effort zero-padded payload
          3. Call the codec to decode the payload → numpy frame.
          4. Publish result to display_buf so the display thread can pick it up.
          5. Update corruption and frozen-frame masks.
        """
        print(f"{RED}Decode worker thread start{RESET}")
        while not self.stop_threads.is_set():
            fid    = int(self.expected_frame)
            with self.fc_cv:
                fc = self.frame_content.get(fid)
            is_key = bool(fc.get("is_key")) if fc else self._is_iframe(fid)

            # Deadline for this frame = display time of the previous frame
            deadline = self._deadline_time(fid - 1, "decode")
            if not self.clock_started:
                deadline = 9999999999  # block indefinitely until first I-frame arrives

            # ── Wait for full assembly or deadline ───────────────────────────
            while True:
                now = time.perf_counter()
                if now >= deadline:
                    break

                with self.fc_cv:
                    fc         = self.frame_content.get(fid)
                    full_ready = False
                    if fc is not None:
                        if is_key:
                            # I-frame: ready as soon as k shards received (FEC can reconstruct)
                            full_ready = (
                                fc.get("k_data") is not None and
                                fc.get("recv_count", 0) >= int(fc["k_data"]) and
                                fc.get("k_data_depth") is not None and
                                fc.get("recv_count_depth", 0) >= int(fc["k_data_depth"])
                            )
                        else:
                            # P-frame: all chunks must arrive (no FEC on P-frames)
                            full_ready = (
                                fc.get("num_chunks") > 0 and fc.get("missing") == 0 and
                                fc.get("num_chunks_depth") > 0 and fc.get("missing_depth") == 0
                            )

                    if full_ready:
                        # Start the deadline clock on the very first decoded I-frame
                        if (not self.clock_started) and is_key:
                            print(f"{RED}Starting deadline clock.{RESET}")
                            self.clock_started  = True
                            # Give ~1 frame of buffer before the first deadline fires
                            self.clock_t0       = max(self.first_packet_clock + self.T,
                                                      time.perf_counter() + 0.066)
                            self.display_next_fid = fid
                        break

                    # Sleep briefly; wake early if a new packet arrives via fc_cv.notify
                    timeout = max(0.0, min(0.01, deadline - now))
                    self.fc_cv.wait(timeout=timeout)

            # ── Build payload ────────────────────────────────────────────────
            with self.fc_lock:
                fc         = self.frame_content.get(fid)
                full_ready = False
                if fc is not None:
                    if is_key:
                        full_ready = (
                            fc.get("k_data") is not None and
                            fc.get("recv_count", 0) >= int(fc["k_data"]) and
                            fc.get("k_data_depth") is not None and
                            fc.get("recv_count_depth", 0) >= int(fc["k_data_depth"])
                        )
                    else:
                        full_ready = (
                            fc.get("num_chunks") > 0 and fc.get("missing") == 0 and
                            fc.get("num_chunks_depth") > 0 and fc.get("missing_depth") == 0
                        )

                payload       = None
                payload_depth = None
                gop_id  = int(fc.get("gop_id", -1))      if fc else -1
                qp      = int(fc.get("qp",      self.codec.qp_p))       if fc else int(self.codec.qp_p)
                qp_depth= int(fc.get("qp_depth", self.depth_codec.qp_p)) if fc else int(self.depth_codec.qp_p)

                if fc is None:
                    logging.warning(f"{RED}[FID: {fid}] content is None. Nothing arrived within time!{RESET}")
                    self.lost_frames_full += 1
                elif full_ready:
                    if is_key:
                        # ── FEC reconstruction for I-frames ─────────────────
                        # Collect the first k available shards (any k of the n
                        # transmitted shards are sufficient for zfec to recover all k
                        # original data shards).
                        def _fec_reconstruct(parts_dict, k, n, total):
                            idxs, shards = [], []
                            for i, b in enumerate(parts_dict):
                                if b is not None:
                                    idxs.append(i)
                                    shards.append(b)
                                    if len(shards) == k:
                                        break
                            dec = Decoder(k, n)
                            data_shards = dec.decode(shards, idxs)
                            return b"".join(data_shards)[:total]

                        payload = _fec_reconstruct(
                            fc["parts"], int(fc["k_data"]), int(fc["num_chunks"]), int(fc["total_size"])
                        )
                        payload_depth = _fec_reconstruct(
                            fc["parts_depth"], int(fc["k_data_depth"]), int(fc["num_chunks_depth"]), int(fc["total_size_depth"])
                        )
                    else:
                        # P-frame: all chunks present, simple concatenation
                        payload       = b"".join(fc["parts"])
                        payload_depth = b"".join(fc["parts_depth"])
                else:
                    # Deadline expired before full assembly
                    if is_key:
                        # I-frames cannot be partially reconstructed without FEC threshold
                        logging.warning(f"{RED}[FID: {fid}] I-frame not fully ready. Dropping.{RESET}")
                        self.lost_frames_full += 1
                    else:
                        # Attempt best-effort P-frame with zero-padded missing chunks
                        payload, payload_depth = self._build_best_effort_payload(fid)
                        if not payload or not payload_depth:
                            payload = payload_depth = None
                            logging.warning(f"{RED}[FID: {fid}] Could not build best-effort payload for P-frame{RESET}")
                            self.lost_frames_full += 1
                        else:
                            logging.warning(f"{RED}[FID: {fid}] Built best-effort payload for P-frame{RESET}")
                            self.lost_frames_partial += 1

                # Release the assembly slot; we no longer need the raw chunks
                self.done_fids.add(fid)
                self.frame_content.pop(fid, None)

            # ── Decode ───────────────────────────────────────────────────────
            frame_rgb   = None
            frame_depth = None
            if payload is not None and payload_depth is not None:
                if (not is_key) and (self.last_decode_i_frame_id is not None) and (gop_id != self.last_decode_i_frame_id):
                    # P-frame belongs to an expired GOP; skip to avoid visual corruption
                    self.lost_frames_full += 1
                else:
                    frame_rgb   = self._decode_frame_sync(fid, FRAME_TYPE_RGB,   is_key, qp,       payload)
                    frame_depth = self._decode_frame_sync(fid, FRAME_TYPE_DEPTH, is_key, qp_depth, payload_depth)
                    with self.fc_lock:
                        if frame_rgb is not None and frame_depth is not None and is_key:
                            # Record successful I-frame so future P-frames can verify GOP membership
                            self.last_decode_i_frame_id = fid

            # ── Publish decoded frames to display thread ─────────────────────
            with self.display_cv:
                self.display_buf[fid] = (frame_rgb, frame_depth)
                self.display_cv.notify_all()

            # ── Update corruption mask ───────────────────────────────────────
            # corrupted = True means the codec received syntactically bad data.
            # Corruption propagates forward through the GOP because each P-frame
            # depends on all previous frames.
            gop = int(self.codec.intra_period)
            if fid % gop == 0:
                # Start of GOP: reset propagation state
                self._gop_corrupted_active_rgb   = frame_rgb   is None
                self._gop_corrupted_active_depth = frame_depth is None
                corrupted_rgb   = False  # never mark an I-frame as corrupted itself
                corrupted_depth = False
            else:
                if payload is None or frame_rgb is None:
                    self._gop_corrupted_active_rgb = True
                if payload_depth is None or frame_depth is None:
                    self._gop_corrupted_active_depth = True
                corrupted_rgb   = self._gop_corrupted_active_rgb
                corrupted_depth = self._gop_corrupted_active_depth

            self.corrupted_frame_list_rgb.append(bool(corrupted_rgb))
            self.corrupted_frame_list_depth.append(bool(corrupted_depth))

            # ── Update frozen mask ───────────────────────────────────────────
            # frozen = True means the display thread will repeat the last good frame.
            # It triggers when a frame is totally lost (payload is None), and
            # propagates for the rest of the GOP since subsequent P-frames can't
            # reference a frame that was never decoded.
            if fid % gop == 0:
                self._gop_frozen_active_rgb   = frame_rgb   is None
                self._gop_frozen_active_depth = frame_depth is None
            else:
                if not self._gop_frozen_active_rgb and payload is None:
                    self._gop_frozen_active_rgb = True
                if not self._gop_frozen_active_depth and payload_depth is None:
                    self._gop_frozen_active_depth = True

            self.frozen_mask_list_rgb.append(bool(self._gop_frozen_active_rgb))
            self.frozen_mask_list_depth.append(bool(self._gop_frozen_active_depth))

            # Always advance; we process every frame id exactly once
            self.expected_frame += 1

    # ────────────────────────────────────────────────────────────────────────
    # Codec decode helper (called from decode thread)
    # ────────────────────────────────────────────────────────────────────────

    def _decode_frame_sync(self, frame_id: int, frame_type: int, is_key: bool,
                           qp: int, payload: bytes):
        """
        Run the codec decompressor for one frame and return the decoded numpy
        array, or None on failure.  Updates per-stream byte/timing counters.
        """
        frame_iter = {"frame_id": frame_id, "is_key": is_key, "qp": qp, "payload": payload}
        frame_rgb   = None
        frame_depth = None

        with torch.no_grad():
            t0 = time.perf_counter()
            if frame_type == FRAME_TYPE_RGB:
                for result in self.codec.decompress_stream(frame_iter, self.pic_height, self.pic_width, self.fps):
                    frame_rgb = result.get("decoded_frame", None)
            else:
                for result in self.depth_codec.decompress_stream(frame_iter, self.pic_height, self.pic_width, self.fps):
                    frame_depth = result.get("decoded_frame", None)
            t1 = time.perf_counter()

        if frame_type == FRAME_TYPE_RGB:
            if frame_rgb is not None:
                self.decode_times[frame_id]    = (t1 - t0) * 1000.0
                self.total_bytes_received     += len(payload)
                self.decoded_frames           += 1
            else:
                logging.warning(f"{RED}[FID: {frame_id}] frame_rgb is None after decode{RESET}")
            return frame_rgb
        else:
            if frame_depth is not None:
                self.decode_times_depth[frame_id]    = (t1 - t0) * 1000.0
                self.total_bytes_received_depth     += len(payload)
                self.decoded_frames_depth           += 1
            else:
                logging.warning(f"{RED}[FID: {frame_id}] frame_depth is None after decode{RESET}")
            return frame_depth

    # ────────────────────────────────────────────────────────────────────────
    # Display worker thread
    # ────────────────────────────────────────────────────────────────────────

    def _display_worker_thread(self):
        """
        Paces frame output to match wall-clock time.

        • Blocks until the deadline clock has started (first I-frame decoded).
        • Sleeps until the next frame's display time.
        • If behind (e.g. after a burst decode stall), fast-forwards by
          displaying skipped frames as quickly as possible.
        • If a frame is missing from display_buf, freezes on the last good frame.
        """
        print(f"{GREEN}Display worker thread start{RESET}")

        # Wait for the deadline clock to start
        with self.fc_cv:
            while (not self.stop_threads.is_set()) and (not self.clock_started):
                self.fc_cv.wait(timeout=0.05)

        if self.stop_threads.is_set():
            return

        with self.fc_lock:
            clock_t0 = self.clock_t0

        T   = float(self.T)
        gop = int(self.codec.intra_period)

        while not self.stop_threads.is_set():
            now        = time.perf_counter()
            target_fid = int((now - clock_t0) / T)   # frame we "should" be at right now

            # Sleep until the next frame is due
            next_time = self._deadline_time(self.display_next_fid, "display")
            if now < next_time:
                time.sleep(min(0.01, next_time - now))
                continue

            # Fast-forward: display any frames we're behind on
            while self.display_next_fid < target_fid and not self.stop_threads.is_set():
                logging.info(f"{GREEN} Displaying frame {self.display_next_fid}, target: {target_fid}{RESET}")
                self._display_one(self.display_next_fid)
                self.display_next_fid += 1

    def _display_one(self, fid: int):
        """
        Consume frame fid from display_buf and append it to saved_frames.
        If the frame is not in the buffer (dropped / not yet decoded) the last
        successfully displayed frame is repeated (freeze-frame strategy).
        """
        with self.display_lock:
            frame_rgb, frame_depth = self.display_buf.pop(fid, (None, None))

        if frame_rgb is None and frame_depth is None:
            # Frame lost or not decoded in time – freeze on last good frame
            logging.info(f"{GREEN} Frame {fid} frozen{RESET}")
            if self.last_displayed_frame is None or self.last_displayed_frame_depth is None:
                frame_rgb   = np.zeros((self.pic_height, self.pic_width, 3), dtype=np.uint8)
                frame_depth = np.zeros((self.pic_height, self.pic_width, 3), dtype=np.uint8)
            else:
                frame_rgb   = self.last_displayed_frame
                frame_depth = self.last_displayed_frame_depth
        else:
            self.last_displayed_frame       = frame_rgb
            self.last_displayed_frame_depth = frame_depth

        self.saved_frames.append(frame_rgb)
        self.saved_frames_depth.append(frame_depth)

    # ────────────────────────────────────────────────────────────────────────
    # Main async entry point
    # ────────────────────────────────────────────────────────────────────────

    async def run(self):
        """
        Connect to the signaling server, negotiate WebRTC, receive the stream,
        and save both RGB and depth videos on completion.
        """
        self.cfg = RTCConfiguration([RTCIceServer(urls=[self.stun_url])])
        self.pc  = RTCPeerConnection(configuration=self.cfg)
        self.diagnostics.start(lambda: {
            "rgb_buffered_amount": int(self.diagnostic_channels["rgb_payload"].bufferedAmount)
            if "rgb_payload" in self.diagnostic_channels else "",
            "depth_buffered_amount": int(self.diagnostic_channels["depth_payload"].bufferedAmount)
            if "depth_payload" in self.diagnostic_channels else "",
            "outstanding_packets": "",
            "last_sent_sequence": "",
            "connection_state": self.pc.connectionState,
            "ice_state": self.pc.iceConnectionState,
        })

        # Start background worker threads
        self.stop_threads.clear()
        self.decode_thread  = threading.Thread(target=self._decode_worker_thread,  daemon=True)
        self.display_thread = threading.Thread(target=self._display_worker_thread, daemon=True)
        self.decode_thread.start()
        self.display_thread.start()

        @self.pc.on("connectionstatechange")
        async def on_state_change():
            logging.info(f"[Receiver] Connection state: {self.pc.connectionState}")

        @self.pc.on("iceconnectionstatechange")
        async def on_ice_state_change():
            logging.info(f"[Receiver] ICE state: {self.pc.iceConnectionState}")
            if (self.pc.iceConnectionState == "completed"
                    and self.ice_consent_timeout_s is not None
                    and not self._ice_consent_timeout_applied):
                apply_ice_consent_timeout(self.ice_consent_timeout_s, "Receiver")
                self._ice_consent_timeout_applied = True

        @self.pc.on("datachannel")
        def on_datachannel(channel):
            logging.info("Receiver: DataChannel %s created", channel.label)
            self.diagnostic_channels[channel.label] = channel

            @channel.on("message")
            async def on_message(msg):
                try:
                    receiver_ts = time.perf_counter()
                    if isinstance(msg, str):
                        msg = msg.encode("utf-8")

                    # ── INIT message: learn stream parameters ────────────────
                    if len(msg) == SZ_INIT and msg[0] == MSG_INIT:
                        _, w, h, fps, chunk_size, chunk_size_depth = struct.unpack(FMT_INIT, msg)
                        self.pic_width        = int(w)
                        self.pic_height       = int(h)
                        self.fps              = int(fps)
                        self.T                = 1.0 / float(self.fps)
                        self.chunk_size       = int(chunk_size)
                        self.chunk_size_depth = int(chunk_size_depth)
                        self.stream_inited    = True
                        logging.info(
                            f"[Receiver] INIT {self.pic_width}x{self.pic_height} @ {self.fps} fps, "
                            f"chunk_size={self.chunk_size}, chunk_size_depth={self.chunk_size_depth}"
                        )
                        return

                    # ── DESC + shard payload ─────────────────────────────────
                    if len(msg) >= SZ_PROBE and msg[0] == MSG_PROBE:
                        (_mtype, seq_id, _burst_id, _packet_idx,
                         _packets_in_burst, sender_grace_period) = struct.unpack(
                            FMT_PROBE, msg[:SZ_PROBE]
                        )
                        self._send_capacity_feedback(
                            channel,
                            seq_id=int(seq_id),
                            packet_size=len(msg),
                            receiver_ts=receiver_ts,
                            sender_grace_period=sender_grace_period,
                        )
                        await asyncio.sleep(0)
                        return

                    if len(msg) < SZ_DESC:
                        return  # too short to be a valid DESC packet; discard

                    (mtype, frame_type, fid, gop_id, qp,
                     chunk_idx, num_chunks, k_data, total_size,
                     seq_id, _sender_send_ts, sender_grace_period) = struct.unpack(
                        FMT_DESC, msg[:SZ_DESC]
                    )

                    if mtype != MSG_DESC:
                        return  # unexpected message type

                    fid        = int(fid)
                    is_key     = self._is_iframe(fid)
                    gop_id     = int(gop_id)
                    qp         = int(qp)
                    chunk_idx  = int(chunk_idx)
                    num_chunks = int(num_chunks)
                    k_data     = int(k_data)
                    total_size = int(total_size)
                    frame_type = int(frame_type)
                    seq_id     = int(seq_id)

                    self._send_capacity_feedback(
                        channel,
                        seq_id=seq_id,
                        packet_size=len(msg),
                        receiver_ts=receiver_ts,
                        sender_grace_period=sender_grace_period,
                    )

                    with self.fc_cv:
                        # Discard stale P-frames that belong to an already-decoded (older) GOP
                        if (not is_key) and (self.last_decode_i_frame_id is not None) and (gop_id < self.last_decode_i_frame_id):
                            if fid in self.frame_content:
                                self.done_fids.add(fid)
                                self.frame_content.pop(fid, None)
                            self.fc_cv.notify_all()
                            return

                        # Initialize assembly slot on first chunk for this fid
                        if fid not in self.frame_content:
                            self.init_frame_content(fid)
                        elif fid in self.done_fids:
                            # Redundant packet for an already-completed frame; discard
                            self.fc_cv.notify_all()
                            return

                        self.frame_content[fid]["is_key"] = is_key
                        self.frame_content[fid]["gop_id"] = gop_id

                        if frame_type == FRAME_TYPE_RGB:
                            self.frame_content[fid]["qp"]         = qp
                            self.frame_content[fid]["num_chunks"]  = num_chunks
                            self.frame_content[fid]["k_data"]      = k_data
                            self.frame_content[fid]["total_size"]  = total_size

                            # Lazily initialize the shard list on the first arriving chunk
                            if isinstance(self.frame_content[fid]["parts"], dict):
                                self.frame_content[fid]["parts"]   = [None] * num_chunks
                                self.frame_content[fid]["missing"] = num_chunks
                                self.frame_content[fid]["recv_count"] = 0

                            if is_key and self.frame_content[fid]["shard_len"] is None:
                                self.frame_content[fid]["shard_len"] = len(msg[SZ_DESC:])

                            # Store shard (guard against duplicates)
                            if self.frame_content[fid]["parts"][chunk_idx] is None:
                                self.frame_content[fid]["parts"][chunk_idx]  = msg[SZ_DESC:]
                                self.frame_content[fid]["missing"]           -= 1
                                self.frame_content[fid]["recv_count"]        += 1

                        else:  # FRAME_TYPE_DEPTH
                            self.frame_content[fid]["qp_depth"]          = qp
                            self.frame_content[fid]["num_chunks_depth"]  = num_chunks
                            self.frame_content[fid]["k_data_depth"]      = k_data
                            self.frame_content[fid]["total_size_depth"]  = total_size

                            if isinstance(self.frame_content[fid]["parts_depth"], dict):
                                self.frame_content[fid]["parts_depth"]       = [None] * num_chunks
                                self.frame_content[fid]["missing_depth"]     = num_chunks
                                self.frame_content[fid]["recv_count_depth"]  = 0

                            if is_key and self.frame_content[fid]["shard_len_depth"] is None:
                                self.frame_content[fid]["shard_len_depth"] = len(msg[SZ_DESC:])

                            if self.frame_content[fid]["parts_depth"][chunk_idx] is None:
                                self.frame_content[fid]["parts_depth"][chunk_idx]  = msg[SZ_DESC:]
                                self.frame_content[fid]["missing_depth"]           -= 1
                                self.frame_content[fid]["recv_count_depth"]        += 1

                            # Record arrival time of first ever packet to anchor the deadline clock
                            if self.first_packet_clock is None:
                                self.first_packet_clock = time.perf_counter()

                        # Wake the decode thread in case this shard completes the frame
                        self.fc_cv.notify_all()

                except Exception as e:
                    logging.exception(f"[Receiver] error handling message on {channel.label}: {e}")

        # ── WebSocket signaling loop ─────────────────────────────────────────
        async with ClientSession() as session:
            async with session.ws_connect(self.signalling_server, heartbeat=5) as ws:
                await ws.send_json({"type": "join", "role": "answer"})
                logging.info("Connected to signaling server as receiver")

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data     = msg.json()
                        msg_type = data.get("type")

                        if msg_type == "offer":
                            # Standard WebRTC offer/answer exchange
                            logging.info("Offer received, creating answer...")
                            await self.pc.setRemoteDescription(
                                RTCSessionDescription(sdp=data["sdp"], type="offer")
                            )
                            answer = await self.pc.createAnswer()
                            await self.pc.setLocalDescription(answer)
                            await asyncio.sleep(1)  # give ICE a moment to gather candidates
                            await ws.send_json({
                                "type": "answer",
                                "sdp":  self.pc.localDescription.sdp,
                                "role": "answer"
                            })
                            logging.info("Answer sent; ready to receive frames")

                        elif msg_type == "bye":
                            # Sender has finished; drain remaining in-flight frames
                            logging.info("[Receiver] Received bye — draining remaining frames (100 ms)")
                            await asyncio.sleep(0.1)
                            self.stop_threads.set()
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                            gc.collect()
                            logging.info("Closing WebSocket and stopping")
                            break

                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break

                await ws.close()

                if self.pc.connectionState != "closed":
                    await self.pc.close()

                # ── Session summary ──────────────────────────────────────────
                lost_total  = self.lost_frames_full + self.lost_frames_partial
                self.total_frames = self.decoded_frames + lost_total
                loss_rate   = (lost_total / max(1, self.total_frames)) * 100
                avg_decode  = (np.mean(np.array(list(self.decode_times.values())))
                               if self.decode_times else 0)
                logging.info(
                    f"[Receiver Summary] Total={self.total_frames} Decoded={self.decoded_frames} "
                    f"LostFull={self.lost_frames_full} LostPartial={self.lost_frames_partial} "
                    f"LostFrames={lost_total} Loss={loss_rate:.2f}% "
                    f"Bytes={self.total_bytes_received/1e6:.2f} MB "
                    f"AvgDecode={avg_decode:.1f} ms"
                )

                # ── Save video ───────────────────────────────────────────────
                if len(self.saved_frames) > 0:
                    os.makedirs(os.path.dirname(os.path.abspath(self.media_file)),       exist_ok=True)
                    os.makedirs(os.path.dirname(os.path.abspath(self.media_file_depth)), exist_ok=True)
                    try:
                        write_video_pyav(
                            self.saved_frames, self.saved_frames_depth,
                            self.media_file, self.media_file_depth,
                            self.fps, self.codec.intra_period,
                            crf=0, preset="veryslow"
                        )
                        logging.info(f"Receiver: saved video to {self.media_file}")
                    except Exception:
                        logging.exception("[Receiver] Failed to write video with PyAV")
                else:
                    logging.warning("[Receiver] No frames decoded; nothing to write")

                # Wake any blocked threads so they can exit cleanly
                with self.fc_cv:
                    self.fc_cv.notify_all()
                with self.display_cv:
                    self.display_cv.notify_all()

                if self.decode_thread:
                    self.decode_thread.join(timeout=1.0)
                if self.display_thread:
                    self.display_thread.join(timeout=1.0)

                await self.diagnostics.stop()
                cv2.destroyAllWindows()
                logging.info("[Receiver] Graceful shutdown complete")


# ────────────────────────────────────────────────────────────────────────────
# Video writer
# ────────────────────────────────────────────────────────────────────────────

def write_video_pyav(frames, frames_depth, media_file, media_file_depth,
                     fps, intra_period, crf=0, preset="slow"):
    """
    Write RGB and depth frame lists to separate MP4 files using libx264 via PyAV.

    Args:
        frames       : list of (H, W, 3) uint8 or float32[0,1] RGB arrays
        frames_depth : corresponding depth frames
        media_file   : output path for RGB video
        media_file_depth : output path for depth video
        fps          : playback frame rate
        intra_period : GOP length (controls keyframe interval in output)
        crf          : constant rate factor (0 = lossless)
        preset       : libx264 speed/quality preset
    """
    def _write(path, frame_list, label):
        if not frame_list:
            raise ValueError(f"write_video_pyav: no {label} frames to write")
        container = av.open(path, mode="w")
        stream    = container.add_stream("libx264", rate=fps)
        stream.width   = frame_list[0].shape[1]
        stream.height  = frame_list[0].shape[0]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": preset, "g": str(intra_period)}
        for frame in frame_list:
            if frame.dtype != np.uint8:
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            frame_av = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(frame_av):
                container.mux(packet)
        for packet in stream.encode(None):  # flush encoder
            container.mux(packet)
        container.close()
        logging.info(f"Saved {len(frame_list)} {label} frames to {path} ({fps} FPS, CRF={crf}, preset={preset})")

    _write(media_file,       frames,       "RGB")
    _write(media_file_depth, frames_depth, "depth")


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="WebRTC RGB+depth video receiver")
    p.add_argument("--out",       default="out_video.mp4", help="Output path for RGB video")
    p.add_argument("--out_depth", default=None,            help="Output path for depth video (default: <out>_depth.mp4)")
    p.add_argument("--server_ip", required=True,           help="Signaling server IP address")
    p.add_argument("--stun",      default="stun:stun.l.google.com:19302", help="STUN server URL")
    p.add_argument("--rtd",       default=True,            help="Real-time display (currently unused)")
    p.add_argument("--codec",     required=True,           default="h265",
                   choices=["dcvcrt", "h265", "h264"],
                   help="Video codec for both RGB and depth streams")
    p.add_argument("--estimator_alpha", type=float, default=0.1,
                   help="EWMA alpha for passive receiver inter-arrival estimator")
    p.add_argument("--diagnostic_csv", default="",
                   help="Validation only: one-second receiver event-loop/feedback diagnostics")
    p.add_argument("--diagnostic_ice", action="store_true",
                   help="Validation only: timestamp aioice STUN consent traffic")
    p.add_argument("--ice_consent_timeout_s", type=float, default=None,
                   help="Validation only: aioice consent timeout applied after ICE completes")

    args = p.parse_args()
    r    = Receiver(args)
    asyncio.run(r.run())
