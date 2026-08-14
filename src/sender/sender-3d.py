"""
sender-3d.py  —  ReVo

WebRTC sender for synchronized RGB + depth video streams.

Pipeline overview:
  VideoDecoder (torchcodec)
      │  raw frames (tensor)
      ▼
  codec.compress_stream()          (runs in asyncio thread pool, both streams in parallel)
      │  compressed payload bytes
      ▼
  _make_iframe_chunks() / slice    (FEC for I-frames; plain slicing for P-frames)
      │  shards / chunks
      ▼
  DataChannel send()               (two unreliable unordered channels: rgb_payload, depth_payload)
      │  packets interleaved RGB ↔ depth, paced over the frame interval
      ▼
  Receiver (receiver-3d.py)

Signaling:
  Sender connects to ws://<server_ip>:8080/ws/demo as role="offer",
  performs the standard WebRTC offer/answer exchange, then starts streaming
  once both DataChannels are open.
"""

import argparse, asyncio, logging
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer
from aiohttp import ClientSession
import aiohttp


from torchcodec.decoders import VideoDecoder
import torch
import DCVCRT_wrapper as dcvc
import H265_wrapper as h265
import H264_wrapper as h264
import time, json
import math
import random
import struct
from zfec import Encoder
import os
import subprocess
import sys
import bisect

# ANSI color codes for log readability
RED   = "\033[31m"
GREEN = '\033[32m'
BLUE  = '\033[34m'
RESET = '\033[0m'

logging.basicConfig(level=logging.INFO)

# Pause sending when the DataChannel's internal send buffer exceeds this limit.
# Prevents memory bloat if the network is slower than the encode rate.
BUFFERED_WATERMARK_HARD = 128 * 1024  # 128 KB

# ── Adaptive quality ─────────────────────────────────────────────────────────
# When enabled, every frame is encoded at three quality levels and the sender
# chooses RGB and depth quality independently while the link is under stress.
# Settings are read from the environment so a run can be configured without
# editing the source.
#
#   SALSIFY_MODE=0       reproduces the unmodified drop-on-full behaviour
#   SALSIFY_RGB_QP_HI/MID/LO      RGB QP ladder
#   SALSIFY_DEPTH_QP_HI/MID/LO    depth QP ladder
#   SALSIFY_SOFT_FRAC    fraction of BUFFERED_WATERMARK_HARD above which the
#                        sender degrades (sender-side congestion)
#   SALSIFY_MISS_THRESH  receiver-reported deadline-miss rate, in percent,
#                        above which the sender degrades (loss or lateness)
# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS ARE THE TUNED CONFIGURATION. Running run_sender_eval.py /
# run_receiver_eval.py with no environment variables reproduces the best
# measured setup; every value below is only an override.
#
#   ladder            QP 25 / 30 / 35   (MID = 30 = stock, so a calm link sends
#                                        exactly what the baseline sends)
#   GOP               30                (per-tier GOP 15 measured worse)
#   thresholds        degrade >8% miss, recover <4%   (trace14 p90 / p75)
#   keyframe FEC      adaptive 0.5-1.5, packet-capped (~29% fewer keyframe losses)
#   duplicate spacing 100 ms                          (+0.069 SSIM, t=6.2)
#   playout buffer    1000 ms  (receiver: REVO_PLAYOUT_MS)
#
# CAVEAT on the last two: the delayed duplicate is only usable while the playout
# buffer exceeds the spacing. 1000 ms of buffer is NOT conferencing-realistic --
# it reproduces the buffer stock ReVo was inheriting by accident. For a latency
# budget, set e.g. SALSIFY_DUP_DELAY_MS=33 with REVO_PLAYOUT_MS=66, or disable
# spacing with SALSIFY_DUP_DELAY_MS=0 (stock behaviour).
# ─────────────────────────────────────────────────────────────────────────────
SALSIFY_MODE        = int(os.environ.get("SALSIFY_MODE", "1"))
SALSIFY_RGB_QP_HI   = int(os.environ.get("SALSIFY_RGB_QP_HI", "25"))
SALSIFY_RGB_QP_MID  = int(os.environ.get("SALSIFY_RGB_QP_MID", "30"))
SALSIFY_RGB_QP_LO   = int(os.environ.get("SALSIFY_RGB_QP_LO", "35"))
SALSIFY_DEPTH_QP_HI = int(os.environ.get("SALSIFY_DEPTH_QP_HI", "25"))
SALSIFY_DEPTH_QP_MID = int(os.environ.get("SALSIFY_DEPTH_QP_MID", "30"))
SALSIFY_DEPTH_QP_LO = int(os.environ.get("SALSIFY_DEPTH_QP_LO", "35"))
SALSIFY_SOFT_FRAC   = float(os.environ.get("SALSIFY_SOFT_FRAC", "0.5"))
SALSIFY_MISS_THRESH = float(os.environ.get("SALSIFY_MISS_THRESH", "8.0"))
# Upgrade threshold, deliberately below MISS_THRESH. The gap between the two is
# a dead band in which the level holds, which is what stops the controller
# oscillating without making it react slowly.
SALSIFY_MISS_CLEAR  = float(os.environ.get("SALSIFY_MISS_CLEAR", "4.0"))
# Keyframe interval. Default 30 matches stock. The receiver must be told the
# same value (REVO_GOP) or its I-frame position fallback disagrees with the
# sender's schedule.
SALSIFY_GOP         = int(os.environ.get("SALSIFY_GOP", "30"))
# Frames that must pass between two forced recovery keyframes. Guards against a
# request storm without making recovery slow: the receiver already rate-limits
# its own retries, so this only needs to stop pathological feedback loops.
SALSIFY_KF_COOLDOWN = int(os.environ.get("SALSIFY_KF_COOLDOWN", "10"))
# Minimum GOPs a committed level must be held before another change. With the
# clock fix the feedback loop is 6 frames instead of 36, and the old 1-second lag
# had been acting as accidental damping: switches jumped 45-57 -> 156 and frozen
# went 5.8% -> 13.5% purely from the controller reacting to noise it could not
# previously see. Damping has to be explicit once the delay is gone.
SALSIFY_MIN_DWELL   = int(os.environ.get("SALSIFY_MIN_DWELL", "2"))
# Skip a recovery keyframe when a scheduled one is already this close: the
# recovery cannot arrive meaningfully sooner, so it is pure added load at the
# worst moment. Measured: with fast feedback only 51% of forced keyframes
# survived (vs 82% when they were rare), i.e. failed recoveries were feeding
# the congestion that killed them.
SALSIFY_KF_SKIP_NEAR = int(os.environ.get("SALSIFY_KF_SKIP_NEAR", "10"))
# ── Loss-adaptive keyframe protection ────────────────────────────────────────
# The useful adaptation on this link is not fidelity but PROTECTION. Every trace
# holds a 2.50 Mbps floor while the top tier needs ~0.4 Mbps, so bandwidth never
# binds and choosing a QP cannot buy anything -- proven by the raised-ladder
# runs, where spending the headroom on fidelity made quality significantly WORSE
# (-0.04 SSIM, t=-3.3..-3.8). Meanwhile keyframe cascades cause 63-89% of all
# frozen frames. So scale keyframe parity with measured loss and spend the idle
# bandwidth there instead.
#
# The cap is not optional: parity packets all leave inside one frame slot, so n
# packets is an instantaneous n*chunk*8*fps bitrate. Uncapped, this previously
# overran the link and caused the very loss it was added to survive.
SALSIFY_FEC_ADAPT = int(os.environ.get("SALSIFY_FEC_ADAPT", "1"))
SALSIFY_FEC_MIN   = float(os.environ.get("SALSIFY_FEC_MIN", "0.5"))
SALSIFY_FEC_MAX   = float(os.environ.get("SALSIFY_FEC_MAX", "1.5"))
SALSIFY_FEC_MAXPK = int(os.environ.get("SALSIFY_FEC_MAXPK", "24"))
# Compensate the degrade threshold for parity we chose to add.
#
# Adaptive FEC raises I-frame load ~14%, which costs unprotected P-frames
# (measured: lost_P 192 -> 255) and therefore inflates the receiver's miss rate.
# The controller cannot tell "the link got worse" from "we chose to send more
# redundancy", so it degrades in response to our own parity: HI residency fell
# 81.6% -> 75.4% and MID rose 9.9% -> 16.9%. That self-inflicted fidelity loss
# is what cancelled the SSIM gain the freeze reduction should have produced.
#
# Scale the degrade threshold with the parity currently in use, so only miss
# ABOVE what our own redundancy explains counts as network degradation.
SALSIFY_FEC_COMPENSATE = float(os.environ.get("SALSIFY_FEC_COMPENSATE", "0"))
# Per-tier GOP. "" keeps the single static SALSIFY_GOP; "15,30,15" gives
# HI=15, MID=30, LO=15.
#
# Rationale (measured): a P-frame loss poisons every frame from the loss to the
# END of its GOP, and that is the dominant damage -- 1892 P-corrupt frames vs
# 301 I-corrupt, with a mean poisoned span of 17.7 frames. Capping the GOP caps
# that span. It costs twice as many keyframes, whose extra bytes buy back some
# P-loss, so the net is an empirical question.
# Delay the P-frame duplicate by this many milliseconds instead of sending it at
# the end of the same frame window.
#
# Measured on trace14: loss arrives in bursts, p50 15 ms / p75 30 ms / p90 60 ms,
# and 80% of bursts are shorter than one 33 ms frame slot. The duplicate is
# currently emitted ~16-33 ms after the original, so it clears the median burst
# but not the p75 one -- both copies die together and the redundancy is wasted.
# Pushing separation past ~40 ms clears roughly 85% of bursts instead of ~50%,
# at zero extra bandwidth (the copy is already being sent).
SALSIFY_DUP_DELAY_MS = float(os.environ.get("SALSIFY_DUP_DELAY_MS", "100"))

SALSIFY_GOP_TIERS = os.environ.get("SALSIFY_GOP_TIERS", "").strip()
_GOP_BY_TIER = ([int(x) for x in SALSIFY_GOP_TIERS.split(",")]
                if SALSIFY_GOP_TIERS else None)

FPS_FALLBACK = 30  # used when the video file has no metadata fps

# ---------------------------------------------------------------------------
# Control-plane message types (shared with receiver)
# ---------------------------------------------------------------------------
MSG_INIT         = 1   # one-time stream parameters
MSG_DESC         = 2   # per-chunk descriptor (precedes every data shard)
FRAME_TYPE_RGB   = 3
FRAME_TYPE_DEPTH = 4

# ---------------------------------------------------------------------------
# Binary wire formats (little-endian)
#
# INIT  – type:u8 | width:u16 | height:u16 | fps:u16
#           | chunk_size_rgb:u16 | chunk_size_depth:u16
#
# DESC  – type:u8 | frame_type:u8 | frame_id:u32 | gop_id:u32
#           | qp:u8 | chunk_idx:u16 | num_chunks(n):u16
#           | k_data:u16 | total_size:u32
#         (raw shard bytes follow immediately after)
# ---------------------------------------------------------------------------
FMT_INIT = "<BHHHHH"
FMT_DESC = "<BBIIBHHHI"
SZ_INIT  = struct.calcsize(FMT_INIT)
SZ_DESC  = struct.calcsize(FMT_DESC)


class Sender():
    """
    Reads pre-encoded RGB and depth video files, compresses each frame with
    the chosen codec, and streams both over separate WebRTC DataChannels.

    I-frames use Reed-Solomon FEC (zfec k-of-n) so the receiver can
    reconstruct the frame from any k of the n transmitted shards.
    P-frames are split into plain chunks with no FEC; the first chunk of
    each P-frame is retransmitted once for extra reliability.

    RGB and depth chunks are interleaved within each frame's send window to
    spread the impact of burst packet loss across both streams equally.
    """

    def __init__(self, args):
        self.args          = args
        self.trace_process = None  # subprocess handle for the TC network trace

        # ── Chunk sizes ──────────────────────────────────────────────────────
        self.chunk_size       = 1024   # bytes per RGB chunk / FEC shard
        self.chunk_size_depth = 1024   # bytes per depth chunk / FEC shard

        # ── Input files ──────────────────────────────────────────────────────
        self.media_file       = args.file
        self.media_file_depth = args.depth_file

        # ── Network / WebRTC ─────────────────────────────────────────────────
        self.stun_url          = args.stun_url
        self.signalling_server = f"ws://{args.server_ip}:8080/ws/demo"
        self.cfg               = None
        self.pc                = None
        self.data_channel_rgb  = None
        self.data_channel_depth= None

        # ── Adaptive-quality state ───────────────────────────────────────────
        self.recv_miss_rate = 0.0  # deadline-miss rate last reported by the receiver (%)
        self.feedback_seen = False
        # Starting tier (0=hi, 1=mid, 2=lo). Starting at MID is right when MID is
        # deliverable, but on a link whose capacity sits below MID the controller
        # begins above capacity and has to fight its way down, losing frames the
        # whole way. Measured on trace14 with the 8/12/16 ladder: it ended up at
        # LO for 84% of the run, but the descent cost 15.4-16.8% frozen against
        # fixed QP16's 12.4%. Starting low and climbing only on demonstrated
        # headroom pays that cost once, in the safe direction.
        _start = int(os.environ.get("SALSIFY_START", "1"))
        self.quality_levels = {"rgb": _start, "depth": _start}
        # Level the controller currently wants. Recomputed every frame; copied
        # into quality_levels only at an IDR, where the rebuild is free.
        self.desired_levels = {"rgb": _start, "depth": _start}
        self.current_frame_id = 0   # loop index; the clock for all cooldowns
        self.qp_switches      = 0   # committed level changes, for the summary
        self.last_commit_fid  = -10**9   # fid of the last committed level change
        self.kf_skipped       = 0   # recoveries suppressed as pointless
        self._cur_fec_r       = SALSIFY_FEC_MIN   # parity ratio currently in use
        # Predicted fid of the next scheduled keyframe. Tracked from the
        # encoder's ACTUAL output rather than arithmetic on frame_id, so it stays
        # correct across both forced keyframes and GOP changes.
        # Duplicates held back for delayed transmission: (due_time, chan, packet).
        self._dup_queue       = []
        self.dup_sent         = 0   # delayed duplicates actually transmitted
        self._next_key_fid    = 0
        self._cur_gop         = SALSIFY_GOP
        self.kf_sent_count    = 0   # keyframes actually emitted by the encoder
        self.kf_forced_count  = 0   # of those, ones triggered by a request
        self.force_keyframe_pending = False
        self.last_forced_keyframe_fid = -999

        # ── Codec selection ──────────────────────────────────────────────────
        # RGB codec
        self.codec = h265.H265VideoCodec(intra_period=SALSIFY_GOP, qp=SALSIFY_RGB_QP_MID)
        if args.codec == "dcvcrt":
            self.codec = dcvc.DCVCVideoCodec(intra_period=30)
        if args.codec == "h264":
            self.codec = h264.H264VideoCodec(intra_period=30)

        # Depth codec (mirrors RGB codec choice)
        self.depth_codec = h265.H265VideoCodec(intra_period=SALSIFY_GOP, qp=SALSIFY_DEPTH_QP_MID)
        if args.codec == "dcvcrt":
            self.depth_codec = dcvc.DCVCVideoCodec(intra_period=30)
        if args.codec == "h264":
            self.depth_codec = h264.H264VideoCodec(intra_period=30)

        # Keep one active codec per stream and swap its QP only when the sender
        # has enough feedback to justify a change. This removes the overload from
        # running three encoders in parallel for every frame.

        # ── Byte / frame counters ────────────────────────────────────────────
        self.total_bytes_sent           = 0
        self.total_bytes_depth_sent     = 0
        self.sent_frames                = 0
        self.sent_frames_depth          = 0

        # Breakdown for the session summary log
        self.i_bytes_sent               = 0   # total I-frame bytes (data + parity)
        self.i_bytes_payload            = 0   # I-frame data shards only
        self.i_bytes_parity             = 0   # I-frame parity shards only
        self.p_bytes_sent               = 0
        self.reliable_bytes_meta        = 0   # DESC / INIT header bytes (RGB)

        self.i_bytes_depth_sent         = 0
        self.i_bytes_depth_payload      = 0
        self.i_bytes_depth_parity       = 0
        self.p_bytes_depth_sent         = 0
        self.reliable_bytes_meta_depth  = 0   # DESC / INIT header bytes (depth)

        # ── GOP tracking ─────────────────────────────────────────────────────
        # gop_id is set to the frame_id of the most recent I-frame.
        # The receiver uses it to discard P-frames from expired GOPs.
        self.sent_init    = False
        self.gop_id       = 0
        self.gop_id_depth = 0

        # ── Network trace (optional adaptive loss emulation) ─────────────────
        self.trace_losses      = []    # list of (t_sec, loss_fraction)
        self.trace_ts          = []    # timestamps for bisect lookup
        self.trace_duration    = None
        self.trace_t0_sender   = None  # wall time when first data chunk is sent

    # ────────────────────────────────────────────────────────────────────────
    # Network trace (tc qdisc) control
    # ────────────────────────────────────────────────────────────────────────

    def start_trace(self):
        """
        Launch the TC (traffic control) script as a subprocess to emulate
        real-world network loss / bandwidth conditions.  Does nothing if
        --trace_path was not provided.
        """
        if self.trace_process is None and self.args.trace_path:
            logging.info(f"Starting network trace: {self.args.trace_path}")
            cmd = [
                sys.executable,
                self.args.tc_script,
                "--trace",     self.args.trace_path,
                "--interface", self.args.interface,
            ]
            self.trace_process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)

        if self.trace_process is not None:
            logging.info(f"{GREEN}[Sender] Started loss trace, PID={self.trace_process.pid}{RESET}")

    def stop_trace(self):
        """
        Terminate the TC subprocess and flush any leftover qdisc rules so the
        network interface is returned to a clean state.
        """
        if self.trace_process:
            logging.info("Stopping network trace and cleaning tc rules...")
            self.trace_process.terminate()
            try:
                self.trace_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.trace_process.kill()
            self.trace_process = None

        # Belt-and-suspenders: always attempt to delete the root qdisc
        subprocess.run(
            f"sudo tc qdisc del dev {self.args.interface} root",
            shell=True, stderr=subprocess.DEVNULL
        )

    # ────────────────────────────────────────────────────────────────────────
    # Protocol helpers
    # ────────────────────────────────────────────────────────────────────────

    def _send_init(self, width: int, height: int, fps: int):
        """Send the one-time INIT message carrying stream parameters."""
        if self.sent_init:
            return
        pkt = struct.pack(
            FMT_INIT, MSG_INIT,
            width, height, int(fps),
            int(self.chunk_size), int(self.chunk_size_depth)
        )
        self.data_channel_rgb.send(pkt)
        self.reliable_bytes_meta += len(pkt)
        self.sent_init = True
        logging.info(f"[Sender] INIT sent: {width}x{height} @ {fps} fps")

    def _make_iframe_chunks(self, payload: bytes, k_data: int, n_total: int):
        """
        Apply Reed-Solomon / zfec erasure coding to an I-frame payload.

        The payload is padded to a multiple of k_data, split into k_data equal
        data shards, and then encoded into n_total shards (k_data data + parity).
        The receiver can reconstruct the original payload from any k_data of the
        n_total shards, tolerating up to (n_total - k_data) shard losses.

        Returns:
            chunks    – list of n_total equal-length shard bytes
            chunk_len – byte length of each shard
        """
        chunk_len = (len(payload) + k_data - 1) // k_data
        # Pad payload so it's exactly chunk_len * k_data bytes
        padded      = payload + b"\x00" * (chunk_len * k_data - len(payload))
        data_shards = [padded[i * chunk_len:(i + 1) * chunk_len] for i in range(k_data)]
        enc         = Encoder(k_data, n_total)
        chunks      = enc.encode(data_shards)  # length n_total
        return chunks, chunk_len

    # ────────────────────────────────────────────────────────────────────────
    # Adaptive quality helpers
    # ────────────────────────────────────────────────────────────────────────

    def _qp_for_level(self, stream: str, level: int) -> int:
        if stream == "rgb":
            if level <= 0:
                return SALSIFY_RGB_QP_HI
            if level >= 2:
                return SALSIFY_RGB_QP_LO
            return SALSIFY_RGB_QP_MID
        if level <= 0:
            return SALSIFY_DEPTH_QP_HI
        if level >= 2:
            return SALSIFY_DEPTH_QP_LO
        return SALSIFY_DEPTH_QP_MID

    def _activate_codec_qp(self, stream: str, level: int) -> None:
        """Swap the encoder for one at a different QP. No-op if QP is unchanged.

        Replacing the encoder throws away anything still inside x265's pipeline,
        which would silently drop those frames. bframes=0 and rc-lookahead=0
        should mean nothing is ever buffered, but that is an assumption about the
        encoder's behaviour, so it is checked and reported rather than trusted.
        """
        if not SALSIFY_MODE or self.args.codec != "h265":
            return
        qp  = self._qp_for_level(stream, level)
        gop = self._gop_for_level(level)
        old = self.codec if stream == "rgb" else self.depth_codec
        # Rebuild when EITHER qp or GOP changes -- with a per-tier GOP, two tiers
        # can share a qp while needing different keyframe intervals.
        if (getattr(old, "qp_p", None) == qp
                and int(getattr(old, "intra_period", -1)) == gop):
            return
        inflight = len(getattr(old, "_inflight_ids", ()) or ())
        if inflight:
            logging.warning(
                f"[Adaptive] rebuilding {stream} encoder at fid "
                f"{self.current_frame_id} with {inflight} frame(s) still in the "
                f"x265 pipeline -- those frames will be dropped"
            )
        new = h265.H265VideoCodec(intra_period=gop, qp=qp)
        if stream == "rgb":
            self.codec = new
        else:
            self.depth_codec = new

    def _gop_for_level(self, level: int) -> int:
        """GOP length for a tier. Falls back to the static value when no
        per-tier mapping is configured."""
        if not _GOP_BY_TIER:
            return SALSIFY_GOP
        return _GOP_BY_TIER[max(0, min(level, len(_GOP_BY_TIER) - 1))]

    def _fec_n(self, k: int) -> int:
        """Total shards for a keyframe of k data shards.

        Stock is ceil(1.5*k). With SALSIFY_FEC_ADAPT the parity ratio rises with
        the receiver-reported miss rate, bounded by a hard packet cap so a
        keyframe can never become a burst large enough to congest the link.
        """
        if not SALSIFY_FEC_ADAPT:
            return k + (k + 1) // 2
        m = self.recv_miss_rate
        if   m <= 1.0:  r = SALSIFY_FEC_MIN
        elif m >= 10.0: r = SALSIFY_FEC_MAX
        else:
            f = (m - 1.0) / 9.0
            r = SALSIFY_FEC_MIN + f * (SALSIFY_FEC_MAX - SALSIFY_FEC_MIN)
        self._cur_fec_r = r          # what the compensation reads
        n = max(k + 1, int(-(-k * (1.0 + r) // 1)))
        # The cap limits the burst, but must never protect a keyframe LESS than
        # stock would: for a large k, min(n, MAXPK) would otherwise fall back to
        # near-zero parity and make big keyframes more fragile than baseline.
        stock = k + (k + 1) // 2
        return min(n, max(stock, SALSIFY_FEC_MAXPK))

    def _decide_quality(self, *, frame_id: int, ba: int) -> None:
        """Recompute the wanted level from the freshest feedback. Runs EVERY frame.

        Deciding only at GOP boundaries meant acting on the network as it was up
        to a second ago, and the two-report streak counter added another second
        on top. Both were there to stop the level oscillating -- but oscillation
        is better fixed with hysteresis than with delay: separate degrade and
        upgrade thresholds leave a dead band where the level simply holds, so the
        decision can be instant without flapping.

        This only chooses a level; applying it is deferred to an IDR, because a
        QP change rebuilds the encoder and therefore forces one.
        """
        if not SALSIFY_MODE or self.args.codec != "h265" or not self.feedback_seen:
            return

        for stream in ("rgb", "depth"):
            current = self.quality_levels[stream]
            # Only miss beyond what our own added parity explains is evidence
            # the network degraded.
            _excess = max(0.0, getattr(self, "_cur_fec_r", SALSIFY_FEC_MIN) - SALSIFY_FEC_MIN)
            _thr    = SALSIFY_MISS_THRESH * (1.0 + SALSIFY_FEC_COMPENSATE * _excess)
            if (ba > BUFFERED_WATERMARK_HARD * SALSIFY_SOFT_FRAC
                    or self.recv_miss_rate > _thr):
                want = min(current + 1, 2)          # under pressure -> cheaper
            elif (ba < BUFFERED_WATERMARK_HARD * SALSIFY_SOFT_FRAC * 0.5
                    and self.recv_miss_rate < SALSIFY_MISS_CLEAR):
                want = max(current - 1, 0)          # clearly calm -> richer
            else:
                want = current                      # dead band -> hold
            self.desired_levels[stream] = want

    def _apply_pending_quality(self, frame_id: int) -> bool:
        """Commit any wanted level change. Caller must only invoke this on a
        frame that is already going to be an IDR.

        A QP change means a new encoder, and a new encoder emits an IDR. Applied
        on an arbitrary frame that costs an extra intra frame on a link where
        intra frames are what dies; applied on a frame that was going to be an
        IDR anyway it costs nothing. Returns True if anything changed.
        """
        changed = False
        # Hold a committed level for a minimum number of GOPs. Degrades bypass
        # the hold: sitting above capacity is the expensive mistake, so the guard
        # must never delay going DOWN -- only going back up.
        held = (frame_id - self.last_commit_fid) // max(1, SALSIFY_GOP)
        for stream in ("rgb", "depth"):
            want = self.desired_levels[stream]
            going_up = want < self.quality_levels[stream]   # lower index = richer
            if want != self.quality_levels[stream] and going_up and held < SALSIFY_MIN_DWELL:
                continue
            if want != self.quality_levels[stream]:
                logging.info(
                    f"[Adaptive] fid={frame_id} {stream} level "
                    f"{self.quality_levels[stream]} -> {want} "
                    f"(miss={self.recv_miss_rate:.1f}%)"
                )
                self.quality_levels[stream] = want
                self.last_commit_fid = frame_id
                changed = True
            self._activate_codec_qp(stream, self.quality_levels[stream])
        return changed

    def _log_quality(self, frame_id: int, ba: int) -> None:
        """Per-GOP state line. Kept so a run can be reconstructed from the log
        alone: level, buffer and the miss rate the level was chosen from."""
        logging.info(
            f"[Adaptive] GOP {frame_id}: levels={self.quality_levels} "
            f"wanted={self.desired_levels} "
            f"(buffered={ba} B, miss_rate={self.recv_miss_rate:.1f}%)"
        )

    # ────────────────────────────────────────────────────────────────────────
    # Channel warm-up
    # ────────────────────────────────────────────────────────────────────────

    async def send_garbage(self, dc, duration_s=1.0, pps=200, size=1200):
        """
        Probe the path capacity by sending zero-filled packets before streaming
        begins.  This helps the WebRTC congestion controller estimate available
        bandwidth before real video data arrives.
        """
        interval = 1.0 / pps
        t_end    = time.perf_counter() + duration_s
        zero_buf = b"\x00" * size
        while dc.readyState != "open":
            await asyncio.sleep(0.01)
        while time.perf_counter() < t_end:
            dc.send(zero_buf)
            await asyncio.sleep(interval)

    # ────────────────────────────────────────────────────────────────────────
    # Main streaming loop
    # ────────────────────────────────────────────────────────────────────────

    async def stream_video(self):
        """
        Encode and send every frame from the input video files.

        For each frame:
          1. Decode the raw tensor from both RGB and depth files.
          2. Compress RGB and depth concurrently in the asyncio thread pool.
          3. Compute a per-chunk send pacing interval that spreads all chunks
             (RGB + depth) evenly across the frame's time slot.
          4. Send I-frames with FEC shards interleaved RGB ↔ depth.
          5. Send P-frames as plain chunks interleaved RGB ↔ depth,
             then retransmit chunk 0 of each stream for extra resilience.
        """
        try:
            decoder       = VideoDecoder(self.media_file,       device="cpu")
            decoder_depth = VideoDecoder(self.media_file_depth, device="cpu")
            fps           = FPS_FALLBACK
            logging.info(f"[Sender] Starting stream: {len(decoder)} frames @ {fps} FPS")

            # Codec wrappers are not async, so run them in a thread pool.
            def encode(codec, raw_tensor, fid, current_fps, force_keyframe=False):
                return list(codec.compress_stream(raw_tensor, fid, fps=current_fps, force_keyframe=force_keyframe))

            self._send_init(512, 512, fps)

            # Start network trace after INIT is confirmed sent
            if self.sent_init and self.trace_process is None:
                self.start_trace()

            t0             = None          # wall time of frame 1 (used for pacing)
            frame_interval = 1.0 / fps

            for frame_id in range(len(decoder)):
                # Anchor the pacing clock at frame 1 (frame 0 may have a long
                # first-encode warm-up that would skew all subsequent deadlines)
                if frame_id == 1:
                    t0 = time.perf_counter()

                # ── Decode raw tensors ───────────────────────────────────────
                raw = decoder[frame_id].unsqueeze(0).unsqueeze(0).float().mul_(1.0 / 255.0)
                raw_depth = decoder_depth[frame_id].unsqueeze(0).unsqueeze(0).float().mul_(1.0 / 255.0)

                self.current_frame_id = frame_id

                # ── Encode one active quality per stream only. ─────────────────
                ba_rgb = self.data_channel_rgb.bufferedAmount
                ba_depth = self.data_channel_depth.bufferedAmount
                ba = max(ba_rgb, ba_depth)

                # Decide every frame, from whatever feedback has arrived by now.
                if SALSIFY_MODE and self.args.codec == "h265":
                    self._decide_quality(frame_id=frame_id, ba=ba)

                # A requested recovery keyframe is honoured on the very next
                # frame -- the point of the request is to end the freeze now.
                force_this_frame = False
                if self.force_keyframe_pending:
                    # How far to the next scheduled keyframe? If it is imminent the
                    # recovery frame cannot help enough to justify its cost.
                    _to_next = (-frame_id) % max(1, SALSIFY_GOP)
                    if _to_next != 0 and _to_next <= SALSIFY_KF_SKIP_NEAR:
                        self.force_keyframe_pending = False
                        self.kf_skipped += 1
                        logging.info(f"[Sender] Skipping recovery at {frame_id}: "
                                     f"scheduled keyframe in {_to_next} frames")
                    else:
                        force_this_frame = True
                        self.force_keyframe_pending = False
                        self.last_forced_keyframe_fid = frame_id
                        self.kf_forced_count += 1
                        logging.info(f"[Sender] Forcing keyframe for frame {frame_id}!")

                # frame_id % GOP is only valid while the GOP is static and no
                # keyframe has been forced. Track the encoder's own cadence.
                self._cur_gop = self._gop_for_level(self.quality_levels["rgb"])
                scheduled_key = (frame_id == 0) or (frame_id >= self._next_key_fid)
                is_key = scheduled_key or force_this_frame

                # Commit any wanted level change here and only here. Both cases
                # are frames that are already IDRs, so the rebuild the QP change
                # forces is one we were paying for regardless -- and a recovery
                # keyframe carries the change for free.
                if SALSIFY_MODE and self.args.codec == "h265" and is_key:
                    if self._apply_pending_quality(frame_id):
                        self.qp_switches += 1
                    self._log_quality(frame_id, ba)
                packet_list, packet_list_depth = await asyncio.gather(
                    asyncio.to_thread(encode, self.codec, raw, frame_id, fps, force_this_frame),
                    asyncio.to_thread(encode, self.depth_codec, raw_depth, frame_id, fps, force_this_frame),
                )
                
                t_send0 = time.perf_counter()
                if self.trace_t0_sender is None:
                    self.trace_t0_sender = t_send0

                # ── Compute send budget (time remaining in this frame slot) ──
                send_budget   = frame_interval
                frame_deadline = None
                if frame_id > 0:
                    frame_deadline = t0 + frame_interval * frame_id
                    send_budget    = max(0.0, frame_deadline - t_send0)

                # ── Extract compressed outputs ───────────────────────────────
                for out in packet_list:
                    payload       = out["payload"]
                    out_fid       = out["frame_id"]
                    is_key        = out["is_key"]
                    qp            = out["qp"]

                    out_depth     = next((item for item in packet_list_depth if item["frame_id"] == out_fid), None)
                    if out_depth is None:
                        logging.warning(f"[Sender] Missing depth output for frame {out_fid}")
                        continue
                    payload_depth = out_depth["payload"]
                    qp_depth      = out_depth["qp"]

                    # Update GOP id on I-frame so receiver can drop stale P-frames.
                    # gop_id == fid is also how the receiver identifies a
                    # keyframe on the wire, so this must follow the encoder's
                    # actual output and not a predicted schedule.
                    if is_key:
                        self.kf_sent_count += 1
                        self._next_key_fid = out_fid + max(1, self._cur_gop)
                        self.gop_id       = out_fid
                        self.gop_id_depth = out_fid
                    gop_id = self.gop_id

                    # ── Back-pressure / quality selection ────────────────────
                    if not SALSIFY_MODE:
                        # BASELINE: drop the frame if either buffer is full.
                        if self.data_channel_rgb.bufferedAmount > BUFFERED_WATERMARK_HARD:
                            logging.warning(f"[Sender] RGB buffer full — dropping frame {out_fid}")
                            break
                        if self.data_channel_depth.bufferedAmount > BUFFERED_WATERMARK_HARD:
                            logging.warning(f"[Sender] Depth buffer full — dropping frame {out_fid}")
                            break
                    else:
                        if ba > BUFFERED_WATERMARK_HARD:
                            logging.warning(f"[Sender] send buffer full — dropping frame {out_fid}")
                            break

                        logging.info(
                            f"[Adaptive] frame {out_fid} rgb_qp={qp} depth_qp={qp_depth} buffered={ba} B"
                        )

                    # ── Chunk / shard preparation ────────────────────────────
                    if is_key:
                        # I-frame: encode with FEC (50% parity overhead)
                        k_data   = max(1, (len(payload)       + self.chunk_size       - 1) // self.chunk_size)
                        n_total  = self._fec_n(k_data)
                        chunks, _ = self._make_iframe_chunks(payload, k_data, n_total)
                        num_chunks = n_total

                        k_data_depth  = max(1, (len(payload_depth) + self.chunk_size_depth - 1) // self.chunk_size_depth)
                        n_total_depth = self._fec_n(k_data_depth)
                        chunks_depth, _ = self._make_iframe_chunks(payload_depth, k_data_depth, n_total_depth)
                        num_chunks_depth = n_total_depth
                    else:
                        # P-frame: plain chunking, no FEC; k_data == num_chunks signals "no FEC"
                        num_chunks   = max(1, (len(payload)       + self.chunk_size       - 1) // self.chunk_size)
                        k_data       = num_chunks
                        num_chunks_depth = max(1, (len(payload_depth) + self.chunk_size_depth - 1) // self.chunk_size_depth)
                        k_data_depth     = num_chunks_depth

                    # Distribute the frame's send budget evenly across all packets
                    per_pkt_dt = send_budget / (num_chunks + num_chunks_depth)

                    # ── Send chunks ──────────────────────────────────────────
                    # Interleave RGB and depth chunks so burst loss affects
                    # both streams equally rather than wiping out one entirely.
                    cursor       = 0
                    cursor_depth = 0
                    chunk_idx       = 0
                    chunk_idx_depth = 0
                    idx             = 0  # global packet counter for pacing

                    # Stash first P-frame chunk for retransmission after the loop
                    first_rgb_packet   = None
                    first_depth_packet = None

                    def _build_rgb_hdr():
                        return struct.pack(
                            FMT_DESC, MSG_DESC, FRAME_TYPE_RGB,
                            int(out_fid), int(gop_id), int(qp),
                            int(chunk_idx), int(num_chunks), int(k_data), int(len(payload))
                        )

                    def _build_depth_hdr():
                        return struct.pack(
                            FMT_DESC, MSG_DESC, FRAME_TYPE_DEPTH,
                            int(out_fid), int(gop_id), int(qp_depth),
                            int(chunk_idx_depth), int(num_chunks_depth), int(k_data_depth), int(len(payload_depth))
                        )

                    async def _pace():
                        """Sleep until the next pacing slot, flushing any duplicates
                        whose delay has elapsed. Draining here (rather than in a
                        separate task) keeps the copies inside the normal pacing
                        rhythm instead of bursting them out together."""
                        nonlocal idx
                        target_t = t_send0 + per_pkt_dt * (idx + 1)
                        idx += 1
                        await asyncio.sleep(max(0.0, target_t - time.perf_counter()))
                        if self._dup_queue:
                            now_d = time.perf_counter()
                            keep = []
                            for due, chan, pkt in self._dup_queue:
                                if due <= now_d:
                                    try:
                                        (self.data_channel_rgb if chan == "rgb"
                                         else self.data_channel_depth).send(pkt)
                                        self.dup_sent += 1
                                    except Exception:
                                        pass
                                else:
                                    keep.append((due, chan, pkt))
                            self._dup_queue = keep

                    # Phase 1: interleaved RGB + depth (while both streams still have chunks)
                    while chunk_idx < num_chunks and chunk_idx_depth < num_chunks_depth:
                        # RGB chunk
                        hdr    = _build_rgb_hdr()
                        shard  = chunks[chunk_idx] if is_key else payload[cursor:cursor + self.chunk_size]
                        if not is_key:
                            cursor += len(shard)
                        packet = hdr + shard
                        if chunk_idx == 0 and not is_key:
                            first_rgb_packet = packet
                        self.data_channel_rgb.send(packet)
                        logging.debug(f"rgb  frame {out_fid} chunk {chunk_idx} sent")
                        self.reliable_bytes_meta += len(hdr)
                        if is_key:
                            self.i_bytes_sent += len(packet)
                            if chunk_idx >= k_data:
                                self.i_bytes_parity  += len(packet)
                            else:
                                self.i_bytes_payload += len(packet)
                        else:
                            self.p_bytes_sent += len(packet)
                        self.total_bytes_sent += len(packet)
                        chunk_idx += 1
                        await _pace()

                        # Depth chunk
                        hdr_d   = _build_depth_hdr()
                        shard_d = chunks_depth[chunk_idx_depth] if is_key else payload_depth[cursor_depth:cursor_depth + self.chunk_size_depth]
                        if not is_key:
                            cursor_depth += len(shard_d)
                        packet_d = hdr_d + shard_d
                        if chunk_idx_depth == 0 and not is_key:
                            first_depth_packet = packet_d
                        self.data_channel_depth.send(packet_d)
                        logging.debug(f"depth frame {out_fid} chunk {chunk_idx_depth} sent")
                        self.reliable_bytes_meta_depth += len(hdr_d)
                        if is_key:
                            self.i_bytes_depth_sent += len(packet_d)
                            if chunk_idx_depth >= k_data_depth:
                                self.i_bytes_depth_parity  += len(packet_d)
                            else:
                                self.i_bytes_depth_payload += len(packet_d)
                        else:
                            self.p_bytes_depth_sent += len(packet_d)
                        self.total_bytes_depth_sent += len(packet_d)
                        chunk_idx_depth += 1
                        await _pace()

                    # Phase 2: drain any remaining RGB chunks (if RGB had more than depth)
                    while chunk_idx < num_chunks:
                        hdr   = _build_rgb_hdr()
                        shard = chunks[chunk_idx] if is_key else payload[cursor:cursor + self.chunk_size]
                        if not is_key:
                            cursor += len(shard)
                        packet = hdr + shard
                        self.data_channel_rgb.send(packet)
                        self.reliable_bytes_meta += len(hdr)
                        if is_key:
                            self.i_bytes_sent += len(packet)
                            if chunk_idx >= k_data:
                                self.i_bytes_parity  += len(packet)
                            else:
                                self.i_bytes_payload += len(packet)
                        else:
                            self.p_bytes_sent += len(packet)
                        self.total_bytes_sent += len(packet)
                        chunk_idx += 1
                        await _pace()

                    # Phase 3: drain any remaining depth chunks
                    while chunk_idx_depth < num_chunks_depth:
                        hdr_d   = _build_depth_hdr()
                        shard_d = chunks_depth[chunk_idx_depth] if is_key else payload_depth[cursor_depth:cursor_depth + self.chunk_size_depth]
                        if not is_key:
                            cursor_depth += len(shard_d)
                        packet_d = hdr_d + shard_d
                        self.data_channel_depth.send(packet_d)
                        self.reliable_bytes_meta_depth += len(hdr_d)
                        if is_key:
                            self.i_bytes_depth_sent += len(packet_d)
                            if chunk_idx_depth >= k_data_depth:
                                self.i_bytes_depth_parity  += len(packet_d)
                            else:
                                self.i_bytes_depth_payload += len(packet_d)
                        else:
                            self.p_bytes_depth_sent += len(packet_d)
                        self.total_bytes_depth_sent += len(packet_d)
                        chunk_idx_depth += 1
                        await _pace()

                    # Phase 4 (P-frames only): retransmit chunk 0 of both streams.
                    # The first chunk carries the slice header that the codec needs
                    # to begin decoding, so one extra copy improves delivery odds.
                    if not is_key and first_rgb_packet and first_depth_packet:
                        if SALSIFY_DUP_DELAY_MS > 0:
                            # Hold the copy so it lands outside the burst that may
                            # be swallowing the original. Same bytes, later slot.
                            due = time.perf_counter() + SALSIFY_DUP_DELAY_MS / 1000.0
                            self._dup_queue.append((due, "rgb",   first_rgb_packet))
                            self._dup_queue.append((due, "depth", first_depth_packet))
                        else:
                            self.data_channel_rgb.send(first_rgb_packet)
                            self.data_channel_depth.send(first_depth_packet)
                        self.p_bytes_sent        += len(first_rgb_packet)
                        self.p_bytes_depth_sent  += len(first_depth_packet)
                        self.total_bytes_sent     += len(first_rgb_packet)
                        self.total_bytes_depth_sent += len(first_depth_packet)

                    self.sent_frames       += 1
                    self.sent_frames_depth += 1
                    if is_key or (out_fid % 15 == 0):
                        logging.info(
                            "Sent frame %04d (%s) RGB=%d B depth=%d B",
                            out_fid, "I" if is_key else "P", len(payload), len(payload_depth)
                        )

            # ── Encoder flush ────────────────────────────────────────────────
            # H.265 and H264 codecs may buffer a few frames internally; flush them.
            if isinstance(self.codec, (h265.H265VideoCodec, h264.H264VideoCodec)):
                for out in self.codec.flush(fps=fps):
                    payload = out["payload"]
                    out_fid = out["frame_id"]
                    is_key  = out["is_key"]
                    qp      = out["qp"]
                    hdr = struct.pack(FMT_DESC, MSG_DESC, FRAME_TYPE_RGB,
                                      int(out_fid), int(gop_id), int(qp), 0, 1, 1, int(len(payload)))
                    self.data_channel_rgb.send(hdr + payload)
                    self.i_bytes_sent        += len(hdr) + len(payload)
                    self.total_bytes_sent    += len(hdr) + len(payload)
                    self.reliable_bytes_meta += len(hdr)
                    self.sent_frames         += 1
                    logging.info("Sent frame %04d (%s, %d B RGB) [flush]",
                                 out_fid, "I" if is_key else "P", len(payload))

                for out_d in self.depth_codec.flush(fps=fps):
                    payload_depth = out_d["payload"]
                    out_fid_d     = out_d["frame_id"]
                    is_key        = out_d["is_key"]
                    qp_depth      = out_d["qp"]
                    hdr_d = struct.pack(FMT_DESC, MSG_DESC, FRAME_TYPE_DEPTH,
                                        int(out_fid), int(gop_id), int(qp_depth), 0, 1, 1, int(len(payload_depth)))
                    self.data_channel_depth.send(hdr_d + payload_depth)
                    self.i_bytes_depth_sent       += len(hdr_d) + len(payload_depth)
                    self.total_bytes_depth_sent   += len(hdr_d) + len(payload_depth)
                    self.reliable_bytes_meta_depth += len(hdr_d)
                    self.sent_frames_depth += 1
                    logging.info("Sent frame %04d (%s, %d B depth) [flush]",
                                 out_fid_d, "I" if is_key else "P", len(payload_depth))

            logging.info("[Sender] Completed streaming all frames.")

        except Exception as e:
            logging.exception(f"[Sender] Error in stream_video: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # Main async entry point
    # ────────────────────────────────────────────────────────────────────────

    def _handle_rgb_message(self, msg):
        try:
            if isinstance(msg, bytes):
                msg = msg.decode("utf-8")
            if isinstance(msg, str):
                if msg.startswith("FB:"):
                    self.recv_miss_rate = float(msg[3:])
                    self.feedback_seen = True
                    logging.info(f"[Feedback] miss_rate={self.recv_miss_rate:.1f}%")
                elif msg.startswith("KEYFRAME_REQUEST:"):
                    requested_fid = int(msg[len("KEYFRAME_REQUEST:"):])
                    if requested_fid >= 0:
                        # Cooldown must be measured on the loop index, the same
                        # clock last_forced_keyframe_fid is written from.
                        # sent_frames is a count of frames actually put on the
                        # wire, so back-pressure drops make it drift below
                        # frame_id and the cooldown silently grows longer than
                        # intended -- suppressing exactly the recoveries wanted
                        # during the congestion that caused the drops.
                        current_fid = self.current_frame_id
                        if current_fid - self.last_forced_keyframe_fid >= SALSIFY_KF_COOLDOWN:
                            self.force_keyframe_pending = True
                            logging.info(f"[Feedback] Received KEYFRAME_REQUEST for frame {requested_fid}, current frame {current_fid}. Setting pending force keyframe.")
                        else:
                            logging.info(f"[Feedback] Ignored KEYFRAME_REQUEST for frame {requested_fid} due to cooldown (current frame {current_fid}, last forced {self.last_forced_keyframe_fid}).")
        except Exception as exc:
            logging.warning(f"[Feedback] could not parse '{msg}': {exc}")

    async def run(self):
        """
        Connect to the signaling server, negotiate WebRTC, wait for both
        DataChannels to open, stream video, then teardown.
        """
        self.cfg = RTCConfiguration([RTCIceServer(urls=[self.stun_url])])
        self.pc  = RTCPeerConnection(configuration=self.cfg)

        # Two unreliable unordered DataChannels: late packets are useless for
        # real-time video, so we skip retransmission at the SCTP layer entirely.
        self.data_channel_rgb   = self.pc.createDataChannel("rgb_payload",   ordered=False, maxRetransmits=0)
        self.data_channel_depth = self.pc.createDataChannel("depth_payload", ordered=False, maxRetransmits=0)

        self.i_open = asyncio.Event()   # set when rgb_payload channel is open
        self.p_open = asyncio.Event()   # set when depth_payload channel is open

        @self.pc.on("iceconnectionstatechange")
        async def on_state_change():
            logging.warning(f"[Sender] ICE state: {self.pc.iceConnectionState}")

        @self.data_channel_rgb.on("open")
        def on_rgb_open():
            logging.info("rgb_payload channel open")
            self.i_open.set()
            # Warm up the path before real video data arrives
            asyncio.create_task(self.send_garbage(self.data_channel_rgb, duration_s=1.0, pps=200, size=1200))

        @self.data_channel_depth.on("open")
        def on_depth_open():
            logging.info("depth_payload channel open")
            self.p_open.set()

        @self.data_channel_rgb.on("message")
        def on_rgb_message(msg):
            self._handle_rgb_message(msg)

        @self.data_channel_rgb.on("close")
        def on_rgb_close():
            logging.info("rgb_payload channel closed")

        @self.data_channel_depth.on("close")
        def on_depth_close():
            logging.info("depth_payload channel closed")

        try:
            async with ClientSession() as session:
                async with session.ws_connect(self.signalling_server) as ws:
                    await ws.send_json({"type": "join", "role": "offer"})
                    logging.info("[Sender] Connected to signaling server")

                    # Create and send the WebRTC offer
                    offer = await self.pc.createOffer()
                    await self.pc.setLocalDescription(offer)
                    await asyncio.sleep(1)  # allow ICE candidates to gather
                    await ws.send_json({
                        "type": "offer",
                        "sdp":  self.pc.localDescription.sdp,
                        "role": "offer"
                    })
                    logging.info("[Sender] Offer sent, waiting for answer...")

                    # Wait for the receiver's SDP answer
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = msg.json()
                            if data["type"] == "answer":
                                logging.info("[Sender] Answer received")
                                await self.pc.setRemoteDescription(
                                    RTCSessionDescription(sdp=data["sdp"], type="answer")
                                )
                                break
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            break

                    # Block until both DataChannels are open
                    await asyncio.gather(self.i_open.wait(), self.p_open.wait())
                    logging.info("[Sender] Both channels open — starting stream in 2 s")
                    await asyncio.sleep(2)

                    await self.stream_video()
                    await asyncio.sleep(0.5)  # let last packets drain before closing

                    # Notify the receiver that streaming is complete
                    try:
                        await ws.send_json({"type": "bye"})
                        logging.info("[Sender] bye sent")
                    except Exception as e:
                        logging.warning(f"[Sender] Failed to send bye: {e}")

                    # ── Session summary ──────────────────────────────────────
                    logging.info(
                        f"\n[Sender Summary]\n"
                        f"Frames sent:  {self.sent_frames} RGB, {self.sent_frames_depth} depth\n"
                        f"Total bytes:  {self.total_bytes_sent/1e6:.2f} MB RGB, "
                        f"{self.total_bytes_depth_sent/1e6:.2f} MB depth\n"
                        f"P-bytes RGB:  {self.p_bytes_sent/1e6:.2f} MB\n"
                        f"P-bytes depth:{self.p_bytes_depth_sent/1e6:.2f} MB\n"
                        f"I-bytes RGB (total):   {self.i_bytes_sent/1e6:.2f} MB  "
                        f"(data {self.i_bytes_payload/1e6:.2f} MB + parity {self.i_bytes_parity/1e6:.2f} MB)\n"
                        f"I-bytes depth (total): {self.i_bytes_depth_sent/1e6:.2f} MB  "
                        f"(data {self.i_bytes_depth_payload/1e6:.2f} MB + parity {self.i_bytes_depth_parity/1e6:.2f} MB)\n"
                        f"Headers RGB:   {self.reliable_bytes_meta/1e6:.2f} MB\n"
                        f"Headers depth: {self.reliable_bytes_meta_depth/1e6:.2f} MB"
                    )
                    if SALSIFY_MODE:
                        # Everything needed to diagnose a run from the log alone.
                        # keyframes_sent above the schedule means IDRs are being
                        # injected somewhere they were not accounted for.
                        # Forced keyframes reset x265's keyint, so they SHIFT the
                        # schedule rather than adding to it -- measured: 306 sent,
                        # 28 forced, 278 non-forced against a naive 288. Subtracting
                        # forced from a fixed schedule therefore over-counts. Report
                        # the naive schedule for reference and do not derive an
                        # "unscheduled" figure from it.
                        _sched = self.sent_frames // max(1, SALSIFY_GOP)
                        logging.info(
                            f"[Adaptive Summary] qp_switches={self.qp_switches} "
                            f"forced_keyframes={self.kf_forced_count} "
                            f"keyframes_sent={self.kf_sent_count} scheduled~={_sched} "
                            f"non_forced={self.kf_sent_count - self.kf_forced_count} "
                            f"final_levels={self.quality_levels}"
                        )

                    await self.pc.close()
                    logging.info("[Sender] PeerConnection closed")
                    await ws.close()
                    logging.info("[Sender] WebSocket closed")

        except Exception as e:
            logging.error(f"[Sender] Error during execution: {e}")
        finally:
            # Always clean up TC rules even on crash / KeyboardInterrupt
            self.stop_trace()


# ────────────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="WebRTC RGB+depth video sender")
    p.add_argument("--file",       required=True, help="Path to the RGB video file")
    p.add_argument("--depth_file", required=True, help="Path to the depth video file")
    p.add_argument("--server_ip",  required=True, help="Signaling server IP address")
    p.add_argument("--stun_url",   default="stun:stun.l.google.com:19302", help="STUN server URL")
    p.add_argument("--codec",      required=True, default="h265",
                   choices=["dcvcrt", "h265", "h264"],
                   help="Video codec for both RGB and depth streams")

    # Optional: network trace for loss / bandwidth emulation
    p.add_argument("--trace_path", default=None,          help="Path to the network trace CSV file")
    p.add_argument("--interface",  default="enp130s0",    help="Network interface for tc rules (e.g. enp130s0)")
    p.add_argument("--tc_script",  default="run_loss_trace.py", help="Path to the TC control Python script")

    args = p.parse_args()
    s    = Sender(args)
    asyncio.run(s.run())
