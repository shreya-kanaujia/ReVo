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
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capacity_feedback import (
    MSG_CAPACITY_FEEDBACK,
    SIZE as SZ_CAPACITY_FEEDBACK,
    decode_capacity_feedback,
)
from webrtc_diagnostics import (
    WebRTCDiagnostics,
    apply_ice_consent_timeout,
    apply_sctp_gap_rtt_fix,
    enable_ice_debug_logging,
    sctp_diagnostic_snapshot,
    sender_grace_for_backpressure_pause,
)

# ANSI color codes for log readability
RED   = "\033[31m"
GREEN = '\033[32m'
BLUE  = '\033[34m'
RESET = '\033[0m'

logging.basicConfig(level=logging.INFO)

# Pause sending when the DataChannel's internal send buffer exceeds this limit.
# Prevents memory bloat if the network is slower than the encode rate.
BUFFERED_WATERMARK_HARD = 128 * 1024  # 128 KB

FPS_FALLBACK = 30  # used when the video file has no metadata fps

# ---------------------------------------------------------------------------
# Control-plane message types (shared with receiver)
# ---------------------------------------------------------------------------
MSG_INIT         = 1   # one-time stream parameters
MSG_DESC         = 2   # per-chunk descriptor (precedes every data shard)
FRAME_TYPE_RGB   = 3
FRAME_TYPE_DEPTH = 4
MSG_PROBE        = 5   # validation-only packet train probe

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
#         (raw shard bytes follow immediately after)
# ---------------------------------------------------------------------------
FMT_INIT = "<BHHHHH"
FMT_DESC = "<BBIIBHHHIQdd"
FMT_PROBE = "<BQIIHd"
SZ_INIT  = struct.calcsize(FMT_INIT)
SZ_DESC  = struct.calcsize(FMT_DESC)
SZ_PROBE = struct.calcsize(FMT_PROBE)


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
        self.ice_consent_timeout_s = args.ice_consent_timeout_s
        self._ice_consent_timeout_applied = False
        self.validation_sctp_gap_rtt_fix = args.validation_sctp_gap_rtt_fix

        # ── Codec selection ──────────────────────────────────────────────────
        # RGB codec
        self.codec = h265.H265VideoCodec(intra_period=30)
        if args.codec == "dcvcrt":
            self.codec = dcvc.DCVCVideoCodec(intra_period=30)
        if args.codec == "h264":
            self.codec = h264.H264VideoCodec(intra_period=30)

        # Depth codec (mirrors RGB codec choice)
        self.depth_codec = h265.H265VideoCodec(intra_period=30)
        if args.codec == "dcvcrt":
            self.depth_codec = dcvc.DCVCVideoCodec(intra_period=30)
        if args.codec == "h264":
            self.depth_codec = h264.H264VideoCodec(intra_period=30)

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

        # ── Passive capacity-estimator state ────────────────────────────────
        self.next_packet_seq = 1
        self.last_sent_sequence = 0
        self.last_acknowledged_sequence = 0
        self.last_packet_send_ts = None
        self.outstanding_packets = {}  # seq_id -> packet byte length
        self.outstanding_packet_send_times = {}
        self.latest_capacity_state = {
            "raw_estimated_capacity_mbps": None,
            "filtered_estimated_capacity_mbps": None,
            "published_estimated_capacity_mbps": None,
            "estimate_age_s": None,
            "estimate_fresh": False,
            "freshness_threshold_s": None,
            "last_valid_update_ts": None,
            "feedback_received_ts": None,
        }
        if args.diagnostic_ice:
            enable_ice_debug_logging()
        self.diagnostics = WebRTCDiagnostics(args.diagnostic_csv, "sender")
        self.capacity_delay_target_s = float(args.capacity_delay_target)
        self.capacity_csv_path = args.capacity_csv
        self.capacity_file = None
        self.capacity_writer = None
        self._open_capacity_log()

        # Validation-only probe logging. This is separate from normal ReVo frames.
        self.probe_sender_csv_path = args.probe_sender_csv
        self.probe_sender_file = None
        self.probe_sender_writer = None
        self._open_probe_sender_log()

        # Passive Week 1A measurements; this does not feed back into send logic.
        self.measurement_csv_path = args.measurement_csv
        self.measurement_file = None
        self.measurement_writer = None
        self._open_measurement_log()

    # ────────────────────────────────────────────────────────────────────────
    # Network trace (tc qdisc) control
    # ────────────────────────────────────────────────────────────────────────

    def _open_measurement_log(self):
        if not self.measurement_csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.measurement_csv_path))
        os.makedirs(parent, exist_ok=True)
        self.measurement_file = open(self.measurement_csv_path, "w", newline="")
        self.measurement_writer = csv.DictWriter(
            self.measurement_file,
            fieldnames=[
                "timestamp_monotonic",
                "frame_id",
                "stream",
                "frame_type",
                "encoded_frame_size_bytes",
                "num_chunks",
                "chunk_size",
                "buffered_amount_before_send",
                "buffered_watermark_hard",
                "sent",
                "drop_reason",
                "send_start_monotonic",
                "send_end_monotonic",
            ],
        )
        self.measurement_writer.writeheader()

    def _log_frame_measurement(self, *, frame_id, stream, is_key, encoded_size,
                               num_chunks, chunk_size, buffered_before, sent,
                               drop_reason="", send_start=None, send_end=None):
        if self.measurement_writer is None:
            return
        self.measurement_writer.writerow({
            "timestamp_monotonic": f"{time.perf_counter():.9f}",
            "frame_id": int(frame_id),
            "stream": stream,
            "frame_type": "I" if is_key else "P",
            "encoded_frame_size_bytes": int(encoded_size),
            "num_chunks": int(num_chunks),
            "chunk_size": int(chunk_size),
            "buffered_amount_before_send": int(buffered_before),
            "buffered_watermark_hard": int(BUFFERED_WATERMARK_HARD),
            "sent": "1" if sent else "0",
            "drop_reason": drop_reason,
            "send_start_monotonic": "" if send_start is None else f"{send_start:.9f}",
            "send_end_monotonic": "" if send_end is None else f"{send_end:.9f}",
        })
        self.measurement_file.flush()

    def _close_measurement_log(self):
        if self.measurement_file is not None:
            self.measurement_file.close()
            self.measurement_file = None
            self.measurement_writer = None

    def _open_capacity_log(self):
        if not self.capacity_csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.capacity_csv_path))
        os.makedirs(parent, exist_ok=True)
        self.capacity_file = open(self.capacity_csv_path, "w", newline="")
        self.capacity_writer = csv.DictWriter(
            self.capacity_file,
            fieldnames=[
                "timestamp",
                "receiver_timestamp",
                "acknowledged_sequence_number",
                "last_sent_sequence_number",
                "packets_outstanding",
                "outstanding_packets",
                "raw_interarrival_time",
                "sender_grace_period",
                "corrected_interarrival_time",
                "smoothed_tau",
                "packet_size_bytes",
                "estimated_packet_service_rate",
                "estimated_capacity_bytes_per_sec",
                "estimated_capacity_mbps",
                "filtered_smoothed_tau",
                "filtered_estimated_capacity_mbps",
                "published_estimated_capacity_mbps",
                "estimate_age_s",
                "estimate_fresh",
                "freshness_threshold_s",
                "last_valid_update_timestamp",
                "skip_reason",
                "filter_reason",
                "filter_applied",
                "max_frame_size_bytes",
                "rgb_bufferedAmount",
                "depth_bufferedAmount",
            ],
        )
        self.capacity_writer.writeheader()

    def _close_capacity_log(self):
        if self.capacity_file is not None:
            self.capacity_file.close()
            self.capacity_file = None
            self.capacity_writer = None

    def _open_probe_sender_log(self):
        if not self.probe_sender_csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.probe_sender_csv_path))
        os.makedirs(parent, exist_ok=True)
        self.probe_sender_file = open(self.probe_sender_csv_path, "w", newline="")
        self.probe_sender_writer = csv.DictWriter(
            self.probe_sender_file,
            fieldnames=[
                "timestamp_monotonic",
                "probe_sequence_id",
                "payload_bytes",
                "target_offered_bitrate_mbps",
                "target_packet_interval_seconds",
                "actual_send_interval_seconds",
                "scheduling_lateness_seconds",
                "datachannel_bufferedAmount",
                "sent",
                "paused_due_to_backpressure",
                "backpressure_pause_duration_seconds",
                "sender_grace_period",
                "schedule_reset_due_to_lateness",
                "cumulative_probe_bytes_sent",
            ],
        )
        self.probe_sender_writer.writeheader()

    def _close_probe_sender_log(self):
        if self.probe_sender_file is not None:
            self.probe_sender_file.close()
            self.probe_sender_file = None
            self.probe_sender_writer = None

    def _log_probe_sender(self, *, timestamp, seq_id, payload_bytes, bitrate_mbps,
                          packet_interval, actual_interval, lateness,
                          buffered_amount, sent, paused, cumulative_bytes,
                          backpressure_pause_duration=0.0,
                          sender_grace_period=0.0,
                          schedule_reset_due_to_lateness=False):
        if self.probe_sender_writer is None:
            return
        self.probe_sender_writer.writerow({
            "timestamp_monotonic": f"{float(timestamp):.9f}",
            "probe_sequence_id": "" if seq_id is None else int(seq_id),
            "payload_bytes": int(payload_bytes),
            "target_offered_bitrate_mbps": f"{float(bitrate_mbps):.9f}",
            "target_packet_interval_seconds": f"{float(packet_interval):.9f}",
            "actual_send_interval_seconds": "" if actual_interval is None else f"{float(actual_interval):.9f}",
            "scheduling_lateness_seconds": f"{float(lateness):.9f}",
            "datachannel_bufferedAmount": int(buffered_amount),
            "sent": "1" if sent else "0",
            "paused_due_to_backpressure": "1" if paused else "0",
            "backpressure_pause_duration_seconds": (
                f"{float(backpressure_pause_duration):.9f}"
            ),
            "sender_grace_period": f"{float(sender_grace_period):.9f}",
            "schedule_reset_due_to_lateness": (
                "1" if schedule_reset_due_to_lateness else "0"
            ),
            "cumulative_probe_bytes_sent": int(cumulative_bytes),
        })
        self.probe_sender_file.flush()

    @staticmethod
    def _fmt_float(value):
        return "" if value is None else f"{float(value):.9f}"

    def _handle_feedback_message(self, msg):
        if isinstance(msg, bytes):
            if len(msg) != SZ_CAPACITY_FEEDBACK or msg[0] != MSG_CAPACITY_FEEDBACK:
                return
            feedback = decode_capacity_feedback(msg)
            if feedback is None:
                return
        elif isinstance(msg, str):
            try:
                feedback = json.loads(msg)
            except json.JSONDecodeError:
                return
            if feedback.get("type") != "capacity_feedback":
                return
        else:
            return

        seq_id = int(feedback.get("seq_id", 0))
        feedback_received_ts = time.perf_counter()
        self.diagnostics.feedback_event(seq_id)
        packet_size = int(feedback.get("packet_size", 0))
        if seq_id > self.last_acknowledged_sequence:
            self.last_acknowledged_sequence = seq_id
        acknowledged_bytes = self.outstanding_packets.pop(seq_id, packet_size)
        packet_send_ts = self.outstanding_packet_send_times.pop(seq_id, None)
        feedback_transport_delay = (
            0.0
            if packet_send_ts is None
            else max(0.0, feedback_received_ts - packet_send_ts)
        )
        ewma = feedback.get("ewma_interarrival")
        filtered_ewma = feedback.get("filtered_ewma_interarrival")
        packets_outstanding = max(0, self.last_sent_sequence - self.last_acknowledged_sequence)
        capacity_bps = None
        service_rate_pps = None
        max_frame_size = None
        if ewma is not None and float(ewma) > 0 and acknowledged_bytes > 0:
            service_rate_pps = 1.0 / float(ewma)
            capacity_bps = float(packet_size or acknowledged_bytes) * service_rate_pps
            max_frame_size = float(packet_size or acknowledged_bytes) * (
                self.capacity_delay_target_s * service_rate_pps - packets_outstanding
            )
            max_frame_size = max(0.0, max_frame_size)
        capacity_mbps = None if capacity_bps is None else capacity_bps * 8.0 / 1_000_000.0
        filtered_capacity_mbps = (
            None
            if filtered_ewma is None or float(filtered_ewma) <= 0.0
            else float(packet_size or acknowledged_bytes) * 8.0
            / float(filtered_ewma) / 1_000_000.0
        )
        receiver_age = feedback.get("estimate_age_s")
        estimate_age = (
            None
            if receiver_age is None
            else max(0.0, float(receiver_age)) + feedback_transport_delay
        )
        freshness_threshold = feedback.get("freshness_threshold_s")
        estimate_fresh = (
            bool(feedback.get("estimate_fresh"))
            and estimate_age is not None
            and freshness_threshold is not None
            and estimate_age <= float(freshness_threshold)
        )
        published_capacity_mbps = (
            filtered_capacity_mbps if estimate_fresh else None
        )
        last_valid_update_ts = feedback.get("last_valid_update_ts")
        if last_valid_update_ts is None and receiver_age is not None:
            last_valid_update_ts = (
                float(feedback.get("receiver_ts", 0.0)) - float(receiver_age)
            )
        skip_reason = feedback.get("skip_reason", "")
        filter_applied = (
            ewma is not None
            and filtered_ewma is not None
            and not math.isclose(
                float(ewma), float(filtered_ewma), rel_tol=1e-12, abs_tol=1e-15
            )
        )
        self.latest_capacity_state = {
            "raw_estimated_capacity_mbps": capacity_mbps,
            "filtered_estimated_capacity_mbps": filtered_capacity_mbps,
            "published_estimated_capacity_mbps": published_capacity_mbps,
            "estimate_age_s": estimate_age,
            "estimate_fresh": estimate_fresh,
            "freshness_threshold_s": freshness_threshold,
            "last_valid_update_ts": last_valid_update_ts,
            "feedback_received_ts": feedback_received_ts,
        }

        logging.debug(
            "[Estimator] ack seq=%d last_sent=%d outstanding=%d tau=%s max_frame=%s",
            seq_id, self.last_sent_sequence, packets_outstanding,
            self._fmt_float(ewma), self._fmt_float(max_frame_size)
        )

        if self.capacity_writer is not None:
            self.capacity_writer.writerow({
                "timestamp": f"{feedback_received_ts:.9f}",
                "receiver_timestamp": self._fmt_float(feedback.get("receiver_ts")),
                "acknowledged_sequence_number": seq_id,
                "last_sent_sequence_number": self.last_sent_sequence,
                "packets_outstanding": packets_outstanding,
                "outstanding_packets": packets_outstanding,
                "raw_interarrival_time": self._fmt_float(feedback.get("raw_interarrival")),
                "sender_grace_period": self._fmt_float(feedback.get("sender_grace_period")),
                "corrected_interarrival_time": self._fmt_float(feedback.get("corrected_interarrival")),
                "smoothed_tau": self._fmt_float(ewma),
                "packet_size_bytes": packet_size,
                "estimated_packet_service_rate": self._fmt_float(service_rate_pps),
                "estimated_capacity_bytes_per_sec": self._fmt_float(capacity_bps),
                "estimated_capacity_mbps": self._fmt_float(capacity_mbps),
                "filtered_smoothed_tau": self._fmt_float(filtered_ewma),
                "filtered_estimated_capacity_mbps": self._fmt_float(
                    filtered_capacity_mbps
                ),
                "published_estimated_capacity_mbps": self._fmt_float(
                    published_capacity_mbps
                ),
                "estimate_age_s": self._fmt_float(estimate_age),
                "estimate_fresh": "1" if estimate_fresh else "0",
                "freshness_threshold_s": self._fmt_float(freshness_threshold),
                "last_valid_update_timestamp": self._fmt_float(
                    last_valid_update_ts
                ),
                "skip_reason": skip_reason,
                "filter_reason": feedback.get("filter_reason", ""),
                "filter_applied": "1" if filter_applied else "0",
                "max_frame_size_bytes": self._fmt_float(max_frame_size),
                "rgb_bufferedAmount": int(self.data_channel_rgb.bufferedAmount) if self.data_channel_rgb else "",
                "depth_bufferedAmount": int(self.data_channel_depth.bufferedAmount) if self.data_channel_depth else "",
            })
            self.capacity_file.flush()

    def _current_capacity_state(self):
        """Return the control-safe state, aging it while feedback is absent."""
        state = dict(self.latest_capacity_state)
        received_ts = state.pop("feedback_received_ts", None)
        age = state.get("estimate_age_s")
        threshold = state.get("freshness_threshold_s")
        if received_ts is not None and age is not None:
            age = max(0.0, float(age) + time.perf_counter() - received_ts)
            state["estimate_age_s"] = age
        fresh = (
            bool(state.get("estimate_fresh"))
            and age is not None
            and threshold is not None
            and age <= float(threshold)
        )
        state["estimate_fresh"] = fresh
        if not fresh:
            state["published_estimated_capacity_mbps"] = None
        return state

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

    def _send_data_packet(self, dc, *, stream_name, frame_type, fid, gop_id, qp,
                          chunk_idx, num_chunks, k_data, total_size, shard,
                          sender_idle_boundary=False):
        send_ts = time.perf_counter()
        sender_grace = 0.0
        if sender_idle_boundary and self.last_packet_send_ts is not None:
            sender_grace = max(0.0, send_ts - self.last_packet_send_ts)

        seq_id = self.next_packet_seq
        self.next_packet_seq += 1
        self.last_sent_sequence = seq_id

        hdr = struct.pack(
            FMT_DESC, MSG_DESC, int(frame_type),
            int(fid), int(gop_id), int(qp),
            int(chunk_idx), int(num_chunks), int(k_data), int(total_size),
            int(seq_id), float(send_ts), float(sender_grace)
        )
        packet = hdr + shard
        dc.send(packet)
        self.last_packet_send_ts = send_ts
        self.outstanding_packets[seq_id] = len(packet)
        self.outstanding_packet_send_times[seq_id] = send_ts
        self.diagnostics.packet_sent(seq_id, send_ts)
        return True, len(packet)

    def _send_probe_packet(self, dc, *, burst_id, packet_idx, packets_in_burst,
                           packet_size, sender_grace=0.0):
        seq_id = self.next_packet_seq
        self.next_packet_seq += 1
        self.last_sent_sequence = seq_id

        payload_len = max(0, int(packet_size) - SZ_PROBE)
        hdr = struct.pack(
            FMT_PROBE, MSG_PROBE, int(seq_id), int(burst_id), int(packet_idx),
            int(packets_in_burst), float(max(0.0, sender_grace))
        )
        packet = hdr + (b"\x00" * payload_len)
        send_ts = time.perf_counter()
        dc.send(packet)
        self.outstanding_packets[seq_id] = len(packet)
        self.outstanding_packet_send_times[seq_id] = send_ts
        self.diagnostics.packet_sent(seq_id, send_ts)
        return seq_id, len(packet)

    def _send_paced_probe_packet(self, dc, *, payload_size, packet_index,
                                 sender_grace=0.0):
        seq_id = self.next_packet_seq
        self.next_packet_seq += 1
        self.last_sent_sequence = seq_id

        hdr = struct.pack(
            FMT_PROBE, MSG_PROBE, int(seq_id), 0, int(packet_index),
            0, float(max(0.0, sender_grace))
        )
        packet = hdr + (b"\x00" * max(0, int(payload_size)))
        send_ts = time.perf_counter()
        dc.send(packet)
        self.outstanding_packets[seq_id] = len(packet)
        self.outstanding_packet_send_times[seq_id] = send_ts
        self.diagnostics.packet_sent(seq_id, send_ts)
        return seq_id, len(packet)

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

    async def send_validation_packet_trains(self):
        """
        Validation-only packet train. This probes the passive estimator through
        the same WebRTC DataChannel without changing normal ReVo video behavior.
        """
        packet_size = int(self.args.validation_train_packet_size)
        packets_per_burst = int(self.args.validation_train_packets)
        interval_s = float(self.args.validation_train_interval)
        duration_s = float(self.args.validation_train_duration)
        hard_cap = max(BUFFERED_WATERMARK_HARD, packet_size * packets_per_burst * 2)

        logging.info(
            "[ValidationTrain] duration=%.1fs packet_size=%d packets_per_burst=%d interval=%.3fs",
            duration_s, packet_size, packets_per_burst, interval_s
        )
        start = time.perf_counter()
        next_burst_t = start
        burst_id = 0
        while time.perf_counter() - start < duration_s:
            now = time.perf_counter()
            if now < next_burst_t:
                await asyncio.sleep(min(0.01, next_burst_t - now))
                continue

            sender_grace = 0.0
            if self.last_packet_send_ts is not None:
                sender_grace = max(0.0, now - self.last_packet_send_ts)
            burst_id += 1
            for packet_idx in range(packets_per_burst):
                while self.data_channel_rgb.bufferedAmount > hard_cap:
                    await asyncio.sleep(0.001)
                grace = sender_grace if packet_idx == 0 else 0.0
                self._send_probe_packet(
                    self.data_channel_rgb,
                    burst_id=burst_id,
                    packet_idx=packet_idx,
                    packets_in_burst=packets_per_burst,
                    packet_size=packet_size,
                    sender_grace=grace,
                )
                self.last_packet_send_ts = time.perf_counter()

            next_burst_t += interval_s

        logging.info("[ValidationTrain] completed %d bursts", burst_id)

    async def send_validation_paced_probe(self):
        """
        Validation-only paced probe. Packets are ACKed by the receiver estimator
        path and never enter RGB/depth frame assembly.
        """
        dc = self.data_channel_rgb
        duration_s = float(self.args.validation_paced_duration)
        payload_size = int(self.args.validation_paced_payload_size)
        bitrate_mbps = float(self.args.validation_paced_bitrate_mbps)
        soft_limit = int(self.args.validation_paced_soft_buffer_limit)
        pause_s = float(self.args.validation_paced_backpressure_sleep)
        bitrate_bps = bitrate_mbps * 1_000_000.0
        packet_interval = (payload_size * 8.0) / bitrate_bps

        start_signal_path = self.args.validation_start_signal_path
        if start_signal_path:
            logging.info("[PacedProbe] waiting for validation start signal: %s",
                         start_signal_path)
            while not os.path.exists(start_signal_path):
                if dc.readyState != "open":
                    logging.warning("[PacedProbe] DataChannel closed while waiting for start signal")
                    return
                await asyncio.sleep(0.01)
            logging.info("[PacedProbe] validation start signal observed")

        logging.info(
            "[PacedProbe] duration=%.3fs offered=%.3f Mbps payload=%dB interval=%.9fs soft_buffer=%dB",
            duration_s, bitrate_mbps, payload_size, packet_interval, soft_limit
        )
        start = time.perf_counter()
        end_time = start + duration_s
        next_deadline = start
        last_send_ts = None
        cumulative_bytes = 0
        sent_packets = 0
        pause_events = 0
        missed_deadlines = 0
        max_buffered = 0
        backpressure_pause_started = None

        while time.perf_counter() < end_time:
            if dc.readyState != "open":
                logging.warning("[PacedProbe] DataChannel closed before duration completed")
                break

            now = time.perf_counter()
            if now < next_deadline:
                await asyncio.sleep(min(0.005, next_deadline - now))
                continue

            buffered = int(dc.bufferedAmount)
            max_buffered = max(max_buffered, buffered)
            lateness = max(0.0, now - next_deadline)
            if lateness > packet_interval:
                missed_deadlines += 1

            if buffered > soft_limit:
                pause_events += 1
                if backpressure_pause_started is None:
                    backpressure_pause_started = now
                self._log_probe_sender(
                    timestamp=now, seq_id=None, payload_bytes=payload_size,
                    bitrate_mbps=bitrate_mbps, packet_interval=packet_interval,
                    actual_interval=None if last_send_ts is None else now - last_send_ts,
                    lateness=lateness, buffered_amount=buffered, sent=False,
                    paused=True, cumulative_bytes=cumulative_bytes,
                    backpressure_pause_duration=(
                        now - backpressure_pause_started
                    ),
                )
                await asyncio.sleep(pause_s)
                # Keep schedule deadline-based, but avoid replaying a huge backlog
                # after an intentional validation-only backpressure pause.
                next_deadline = max(next_deadline + packet_interval, time.perf_counter())
                await asyncio.sleep(0)
                continue

            send_start = time.perf_counter()
            actual_interval = None if last_send_ts is None else send_start - last_send_ts
            backpressure_pause_duration = (
                0.0
                if backpressure_pause_started is None
                else max(0.0, send_start - backpressure_pause_started)
            )
            schedule_reset_due_to_lateness = lateness > packet_interval
            sender_grace = sender_grace_for_backpressure_pause(
                actual_interval, buffered, backpressure_pause_duration
            )
            seq_id, packet_len = self._send_paced_probe_packet(
                dc,
                payload_size=payload_size,
                packet_index=sent_packets,
                # A nonempty transport queue remains continuously offered.
                # Only a pause that fully drains bufferedAmount is a
                # receiver-visible sender-idle boundary.
                sender_grace=sender_grace,
            )
            backpressure_pause_started = None
            last_send_ts = time.perf_counter()
            self.last_packet_send_ts = last_send_ts
            sent_packets += 1
            cumulative_bytes += packet_len
            self._log_probe_sender(
                timestamp=last_send_ts, seq_id=seq_id, payload_bytes=payload_size,
                bitrate_mbps=bitrate_mbps, packet_interval=packet_interval,
                actual_interval=actual_interval, lateness=lateness,
                buffered_amount=buffered, sent=True, paused=False,
                cumulative_bytes=cumulative_bytes,
                backpressure_pause_duration=backpressure_pause_duration,
                sender_grace_period=sender_grace,
                schedule_reset_due_to_lateness=schedule_reset_due_to_lateness,
            )
            # Never replay missed application deadlines as a sub-interval
            # catch-up burst; resume causal pacing from the actual send time.
            if schedule_reset_due_to_lateness:
                next_deadline = send_start + packet_interval
            else:
                next_deadline += packet_interval
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - start
        offered_mbps = (cumulative_bytes * 8.0 / elapsed / 1_000_000.0) if elapsed > 0 else 0.0
        logging.info(
            "[PacedProbe] completed elapsed=%.3fs packets=%d bytes=%d offered=%.3f Mbps pauses=%d missed_deadlines=%d max_buffered=%d channel=%s",
            elapsed, sent_packets, cumulative_bytes, offered_mbps, pause_events,
            missed_deadlines, max_buffered, dc.readyState
        )

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

            # Codec wrappers are not async, so run them in a thread pool
            def encode_rgb(raw_tensor, fid, current_fps):
                return list(self.codec.compress_stream(raw_tensor, fid, fps=current_fps))

            def encode_depth(raw_tensor, fid, current_fps):
                return list(self.depth_codec.compress_stream(raw_tensor, fid, fps=current_fps))

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

                # ── Compress RGB and depth concurrently ──────────────────────
                task_rgb   = asyncio.to_thread(encode_rgb,   raw,       frame_id, fps)
                task_depth = asyncio.to_thread(encode_depth, raw_depth, frame_id, fps)
                packet_list, packet_list_depth = await asyncio.gather(task_rgb, task_depth)

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

                    out_depth     = packet_list_depth[0]
                    payload_depth = out_depth["payload"]
                    qp_depth      = out_depth["qp"]

                    # Update GOP id on I-frame so receiver can drop stale P-frames
                    if is_key:
                        self.gop_id       = out_fid
                        self.gop_id_depth = out_fid
                    gop_id = self.gop_id

                    # ── Back-pressure check ──────────────────────────────────
                    # Drop the frame if either channel's send buffer is full.
                    buffered_rgb_before = int(self.data_channel_rgb.bufferedAmount)
                    buffered_depth_before = int(self.data_channel_depth.bufferedAmount)
                    if is_key:
                        num_chunks_log = max(1, (len(payload) + self.chunk_size - 1) // self.chunk_size)
                        num_chunks_log = num_chunks_log + (num_chunks_log + 1) // 2
                        num_chunks_depth_log = max(1, (len(payload_depth) + self.chunk_size_depth - 1) // self.chunk_size_depth)
                        num_chunks_depth_log = num_chunks_depth_log + (num_chunks_depth_log + 1) // 2
                    else:
                        num_chunks_log = max(1, (len(payload) + self.chunk_size - 1) // self.chunk_size)
                        num_chunks_depth_log = max(1, (len(payload_depth) + self.chunk_size_depth - 1) // self.chunk_size_depth)

                    drop_reason = ""
                    if buffered_rgb_before > BUFFERED_WATERMARK_HARD:
                        logging.warning(f"[Sender] RGB buffer full — dropping frame {out_fid}")
                        drop_reason = "rgb_buffer_full"
                    elif buffered_depth_before > BUFFERED_WATERMARK_HARD:
                        logging.warning(f"[Sender] Depth buffer full — dropping frame {out_fid}")
                        drop_reason = "depth_buffer_full"

                    if drop_reason:
                        self._log_frame_measurement(
                            frame_id=out_fid, stream="rgb", is_key=is_key,
                            encoded_size=len(payload), num_chunks=num_chunks_log,
                            chunk_size=self.chunk_size, buffered_before=buffered_rgb_before,
                            sent=False, drop_reason=drop_reason
                        )
                        self._log_frame_measurement(
                            frame_id=out_fid, stream="depth", is_key=is_key,
                            encoded_size=len(payload_depth), num_chunks=num_chunks_depth_log,
                            chunk_size=self.chunk_size_depth, buffered_before=buffered_depth_before,
                            sent=False, drop_reason=drop_reason
                        )
                        break

                    # ── Chunk / shard preparation ────────────────────────────
                    if is_key:
                        # I-frame: encode with FEC (50% parity overhead)
                        k_data   = max(1, (len(payload)       + self.chunk_size       - 1) // self.chunk_size)
                        n_total  = k_data + (k_data + 1) // 2   # ceil(1.5 * k)
                        chunks, _ = self._make_iframe_chunks(payload, k_data, n_total)
                        num_chunks = n_total

                        k_data_depth  = max(1, (len(payload_depth) + self.chunk_size_depth - 1) // self.chunk_size_depth)
                        n_total_depth = k_data_depth + (k_data_depth + 1) // 2
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
                    first_rgb_shard   = None
                    first_depth_shard = None
                    frame_idle_boundary = (
                        buffered_rgb_before == 0 and buffered_depth_before == 0
                    )

                    async def _pace():
                        """Sleep until the next pacing slot."""
                        nonlocal idx
                        target_t = t_send0 + per_pkt_dt * (idx + 1)
                        idx += 1
                        await asyncio.sleep(max(0.0, target_t - time.perf_counter()))

                    # Phase 1: interleaved RGB + depth (while both streams still have chunks)
                    while chunk_idx < num_chunks and chunk_idx_depth < num_chunks_depth:
                        # RGB chunk
                        shard  = chunks[chunk_idx] if is_key else payload[cursor:cursor + self.chunk_size]
                        if not is_key:
                            cursor += len(shard)
                        if chunk_idx == 0 and not is_key:
                            first_rgb_shard = shard
                        sent, packet_len = self._send_data_packet(
                            self.data_channel_rgb,
                            stream_name="rgb", frame_type=FRAME_TYPE_RGB,
                            fid=out_fid, gop_id=gop_id, qp=qp,
                            chunk_idx=chunk_idx, num_chunks=num_chunks,
                            k_data=k_data, total_size=len(payload), shard=shard,
                            sender_idle_boundary=(frame_idle_boundary and chunk_idx == 0),
                        )
                        if sent:
                            logging.debug(f"rgb  frame {out_fid} chunk {chunk_idx} sent")
                            self.reliable_bytes_meta += SZ_DESC
                            if is_key:
                                self.i_bytes_sent += packet_len
                                if chunk_idx >= k_data:
                                    self.i_bytes_parity  += packet_len
                                else:
                                    self.i_bytes_payload += packet_len
                            else:
                                self.p_bytes_sent += packet_len
                            self.total_bytes_sent += packet_len
                        chunk_idx += 1
                        await _pace()

                        # Depth chunk
                        shard_d = chunks_depth[chunk_idx_depth] if is_key else payload_depth[cursor_depth:cursor_depth + self.chunk_size_depth]
                        if not is_key:
                            cursor_depth += len(shard_d)
                        if chunk_idx_depth == 0 and not is_key:
                            first_depth_shard = shard_d
                        sent_depth, packet_len = self._send_data_packet(
                            self.data_channel_depth,
                            stream_name="depth", frame_type=FRAME_TYPE_DEPTH,
                            fid=out_fid, gop_id=gop_id, qp=qp_depth,
                            chunk_idx=chunk_idx_depth, num_chunks=num_chunks_depth,
                            k_data=k_data_depth, total_size=len(payload_depth), shard=shard_d
                        )
                        if sent_depth:
                            logging.debug(f"depth frame {out_fid} chunk {chunk_idx_depth} sent")
                            self.reliable_bytes_meta_depth += SZ_DESC
                            if is_key:
                                self.i_bytes_depth_sent += packet_len
                                if chunk_idx_depth >= k_data_depth:
                                    self.i_bytes_depth_parity  += packet_len
                                else:
                                    self.i_bytes_depth_payload += packet_len
                            else:
                                self.p_bytes_depth_sent += packet_len
                            self.total_bytes_depth_sent += packet_len
                        chunk_idx_depth += 1
                        await _pace()

                    # Phase 2: drain any remaining RGB chunks (if RGB had more than depth)
                    while chunk_idx < num_chunks:
                        shard = chunks[chunk_idx] if is_key else payload[cursor:cursor + self.chunk_size]
                        if not is_key:
                            cursor += len(shard)
                        sent, packet_len = self._send_data_packet(
                            self.data_channel_rgb,
                            stream_name="rgb", frame_type=FRAME_TYPE_RGB,
                            fid=out_fid, gop_id=gop_id, qp=qp,
                            chunk_idx=chunk_idx, num_chunks=num_chunks,
                            k_data=k_data, total_size=len(payload), shard=shard
                        )
                        if sent:
                            self.reliable_bytes_meta += SZ_DESC
                            if is_key:
                                self.i_bytes_sent += packet_len
                                if chunk_idx >= k_data:
                                    self.i_bytes_parity  += packet_len
                                else:
                                    self.i_bytes_payload += packet_len
                            else:
                                self.p_bytes_sent += packet_len
                            self.total_bytes_sent += packet_len
                        chunk_idx += 1
                        await _pace()

                    # Phase 3: drain any remaining depth chunks
                    while chunk_idx_depth < num_chunks_depth:
                        shard_d = chunks_depth[chunk_idx_depth] if is_key else payload_depth[cursor_depth:cursor_depth + self.chunk_size_depth]
                        if not is_key:
                            cursor_depth += len(shard_d)
                        sent_depth, packet_len = self._send_data_packet(
                            self.data_channel_depth,
                            stream_name="depth", frame_type=FRAME_TYPE_DEPTH,
                            fid=out_fid, gop_id=gop_id, qp=qp_depth,
                            chunk_idx=chunk_idx_depth, num_chunks=num_chunks_depth,
                            k_data=k_data_depth, total_size=len(payload_depth), shard=shard_d
                        )
                        if sent_depth:
                            self.reliable_bytes_meta_depth += SZ_DESC
                            if is_key:
                                self.i_bytes_depth_sent += packet_len
                                if chunk_idx_depth >= k_data_depth:
                                    self.i_bytes_depth_parity  += packet_len
                                else:
                                    self.i_bytes_depth_payload += packet_len
                            else:
                                self.p_bytes_depth_sent += packet_len
                            self.total_bytes_depth_sent += packet_len
                        chunk_idx_depth += 1
                        await _pace()

                    # Phase 4 (P-frames only): retransmit chunk 0 of both streams.
                    # The first chunk carries the slice header that the codec needs
                    # to begin decoding, so one extra copy improves delivery odds.
                    if not is_key and first_rgb_shard and first_depth_shard:
                        sent, packet_len = self._send_data_packet(
                            self.data_channel_rgb,
                            stream_name="rgb", frame_type=FRAME_TYPE_RGB,
                            fid=out_fid, gop_id=gop_id, qp=qp,
                            chunk_idx=0, num_chunks=num_chunks,
                            k_data=k_data, total_size=len(payload), shard=first_rgb_shard
                        )
                        if sent:
                            self.p_bytes_sent        += packet_len
                            self.total_bytes_sent     += packet_len
                        sent_depth, packet_len = self._send_data_packet(
                            self.data_channel_depth,
                            stream_name="depth", frame_type=FRAME_TYPE_DEPTH,
                            fid=out_fid, gop_id=gop_id, qp=qp_depth,
                            chunk_idx=0, num_chunks=num_chunks_depth,
                            k_data=k_data_depth, total_size=len(payload_depth), shard=first_depth_shard
                        )
                        if sent_depth:
                            self.p_bytes_depth_sent  += packet_len
                            self.total_bytes_depth_sent += packet_len

                    t_send1 = time.perf_counter()
                    self._log_frame_measurement(
                        frame_id=out_fid, stream="rgb", is_key=is_key,
                        encoded_size=len(payload), num_chunks=num_chunks,
                        chunk_size=self.chunk_size, buffered_before=buffered_rgb_before,
                        sent=True, send_start=t_send0, send_end=t_send1
                    )
                    self._log_frame_measurement(
                        frame_id=out_fid, stream="depth", is_key=is_key,
                        encoded_size=len(payload_depth), num_chunks=num_chunks_depth,
                        chunk_size=self.chunk_size_depth, buffered_before=buffered_depth_before,
                        sent=True, send_start=t_send0, send_end=t_send1
                    )

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
                    t_flush0 = time.perf_counter()
                    buffered_rgb_before = int(self.data_channel_rgb.bufferedAmount)
                    sent, packet_len = self._send_data_packet(
                        self.data_channel_rgb,
                        stream_name="rgb", frame_type=FRAME_TYPE_RGB,
                        fid=out_fid, gop_id=gop_id, qp=qp,
                        chunk_idx=0, num_chunks=1, k_data=1,
                        total_size=len(payload), shard=payload
                    )
                    t_flush1 = time.perf_counter()
                    if sent:
                        self.i_bytes_sent        += packet_len
                        self.total_bytes_sent    += packet_len
                        self.reliable_bytes_meta += SZ_DESC
                        self.sent_frames         += 1
                    self._log_frame_measurement(
                        frame_id=out_fid, stream="rgb", is_key=is_key,
                        encoded_size=len(payload), num_chunks=1,
                        chunk_size=len(payload), buffered_before=buffered_rgb_before,
                        sent=True, send_start=t_flush0, send_end=t_flush1
                    )
                    logging.info("Sent frame %04d (%s, %d B RGB) [flush]",
                                 out_fid, "I" if is_key else "P", len(payload))

                for out_d in self.depth_codec.flush(fps=fps):
                    payload_depth = out_d["payload"]
                    out_fid_d     = out_d["frame_id"]
                    is_key        = out_d["is_key"]
                    qp_depth      = out_d["qp"]
                    t_flush0 = time.perf_counter()
                    buffered_depth_before = int(self.data_channel_depth.bufferedAmount)
                    sent_depth, packet_len = self._send_data_packet(
                        self.data_channel_depth,
                        stream_name="depth", frame_type=FRAME_TYPE_DEPTH,
                        fid=out_fid_d, gop_id=gop_id, qp=qp_depth,
                        chunk_idx=0, num_chunks=1, k_data=1,
                        total_size=len(payload_depth), shard=payload_depth
                    )
                    t_flush1 = time.perf_counter()
                    if sent_depth:
                        self.i_bytes_depth_sent       += packet_len
                        self.total_bytes_depth_sent   += packet_len
                        self.reliable_bytes_meta_depth += SZ_DESC
                        self.sent_frames_depth += 1
                    self._log_frame_measurement(
                        frame_id=out_fid_d, stream="depth", is_key=is_key,
                        encoded_size=len(payload_depth), num_chunks=1,
                        chunk_size=len(payload_depth), buffered_before=buffered_depth_before,
                        sent=True, send_start=t_flush0, send_end=t_flush1
                    )
                    logging.info("Sent frame %04d (%s, %d B depth) [flush]",
                                 out_fid_d, "I" if is_key else "P", len(payload_depth))

            logging.info("[Sender] Completed streaming all frames.")

        except Exception as e:
            logging.exception(f"[Sender] Error in stream_video: {e}")

    # ────────────────────────────────────────────────────────────────────────
    # Main async entry point
    # ────────────────────────────────────────────────────────────────────────

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
        if self.validation_sctp_gap_rtt_fix:
            apply_sctp_gap_rtt_fix(self.pc, "Sender")
        self.diagnostics.start(lambda: {
            "rgb_buffered_amount": int(self.data_channel_rgb.bufferedAmount),
            "depth_buffered_amount": int(self.data_channel_depth.bufferedAmount),
            "outstanding_packets": max(0, self.last_sent_sequence - self.last_acknowledged_sequence),
            "last_sent_sequence": self.last_sent_sequence,
            "connection_state": self.pc.connectionState,
            "ice_state": self.pc.iceConnectionState,
            **self._current_capacity_state(),
            **sctp_diagnostic_snapshot(self.pc),
        })

        self.i_open = asyncio.Event()   # set when rgb_payload channel is open
        self.p_open = asyncio.Event()   # set when depth_payload channel is open

        @self.pc.on("iceconnectionstatechange")
        async def on_state_change():
            logging.warning(f"[Sender] ICE state: {self.pc.iceConnectionState}")
            if (self.pc.iceConnectionState == "completed"
                    and self.ice_consent_timeout_s is not None
                    and not self._ice_consent_timeout_applied):
                apply_ice_consent_timeout(self.ice_consent_timeout_s, "Sender")
                self._ice_consent_timeout_applied = True

        @self.data_channel_rgb.on("open")
        def on_rgb_open():
            logging.info("rgb_payload channel open")
            self.i_open.set()
            # Warm up the path before real video data arrives
            asyncio.create_task(self.send_garbage(self.data_channel_rgb, duration_s=1.0, pps=200, size=1200))

        @self.data_channel_rgb.on("message")
        def on_rgb_message(msg):
            self._handle_feedback_message(msg)

        @self.data_channel_depth.on("open")
        def on_depth_open():
            logging.info("depth_payload channel open")
            self.p_open.set()

        @self.data_channel_depth.on("message")
        def on_depth_message(msg):
            self._handle_feedback_message(msg)

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

                    if self.args.validation_paced_probe_only:
                        await self.send_validation_paced_probe()
                    elif self.args.validation_train_only:
                        await self.send_validation_packet_trains()
                    else:
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

                    await self.pc.close()
                    logging.info("[Sender] PeerConnection closed")
                    await ws.close()
                    logging.info("[Sender] WebSocket closed")

        except Exception as e:
            logging.error(f"[Sender] Error during execution: {e}")
        finally:
            # Always clean up TC rules even on crash / KeyboardInterrupt
            self.stop_trace()
            self._close_measurement_log()
            self._close_capacity_log()
            self._close_probe_sender_log()
            await self.diagnostics.stop()


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
    p.add_argument("--measurement_csv", default="output/sender_frame_measurements.csv",
                   help="CSV path for passive per-frame sender measurements")
    p.add_argument("--capacity_csv", default="output/week1_estimator/capacity_estimator.csv",
                   help="CSV path for passive capacity-estimator feedback logs")
    p.add_argument("--probe_sender_csv", default="",
                   help="CSV path for validation-only probe sender logs")
    p.add_argument("--capacity_delay_target", type=float, default=0.1,
                   help="Salsify-style delay target in seconds for passive max-frame-size logging")
    p.add_argument("--diagnostic_csv", default="",
                   help="Validation only: one-second sender event-loop/feedback diagnostics")
    p.add_argument("--diagnostic_ice", action="store_true",
                   help="Validation only: timestamp aioice STUN consent traffic")
    p.add_argument("--ice_consent_timeout_s", type=float, default=None,
                   help="Validation only: aioice consent timeout applied after ICE completes")
    p.add_argument("--validation_sctp_gap_rtt_fix", action="store_true",
                   help="Validation only: ignore ambiguous RTT samples from already gap-ACKed SCTP data")
    p.add_argument("--validation_train_only", action="store_true",
                   help="Validation only: send synthetic packet trains instead of ReVo video frames")
    p.add_argument("--validation_train_duration", type=float, default=60.0,
                   help="Validation packet-train duration in seconds")
    p.add_argument("--validation_train_packet_size", type=int, default=1200,
                   help="Validation packet-train packet size in bytes, including probe header")
    p.add_argument("--validation_train_packets", type=int, default=320,
                   help="Validation packet-train packets per burst")
    p.add_argument("--validation_train_interval", type=float, default=0.25,
                   help="Validation packet-train burst interval in seconds")
    p.add_argument("--validation_paced_probe_only", action="store_true",
                   help="Validation only: send evenly paced probe packets instead of ReVo video frames")
    p.add_argument("--validation_paced_duration", type=float, default=60.0,
                   help="Validation paced-probe duration in seconds")
    p.add_argument("--validation_paced_bitrate_mbps", type=float, default=9.0,
                   help="Validation paced-probe target payload bitrate in Mbps")
    p.add_argument("--validation_paced_payload_size", type=int, default=1024,
                   help="Validation paced-probe payload bytes, not including probe header")
    p.add_argument("--validation_paced_soft_buffer_limit", type=int, default=32768,
                   help="Validation-only soft DataChannel bufferedAmount limit before pausing probes")
    p.add_argument("--validation_paced_backpressure_sleep", type=float, default=0.002,
                   help="Validation-only sleep when probe bufferedAmount exceeds the soft limit")
    p.add_argument("--validation_start_signal_path", default="",
                   help="Validation only: wait for this file before starting the paced probe")

    args = p.parse_args()
    s    = Sender(args)
    asyncio.run(s.run())
