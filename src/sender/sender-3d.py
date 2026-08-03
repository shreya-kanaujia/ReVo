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
try:
    from torchcodec.decoders import VideoDecoder
except Exception:
    VideoDecoder = None
import torch
import DCVCRT_wrapper as dcvc
import H265_wrapper as h265
import H264_wrapper as h264
import av
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
import queue
import threading
from dataclasses import replace
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from capacity_feedback import (
    MSG_CAPACITY_FEEDBACK,
    SIZE as SZ_CAPACITY_FEEDBACK,
    decode_capacity_feedback,
    validate_capacity_feedback,
)
from receiver_health_feedback import (
    MSG_RECEIVER_HEALTH,
    RECEIVER_HEALTH_SIZES,
    ReceiverHealthInbox,
)
from probe_feedback import (
    MSG_CAPACITY_PROBE_ACK,
    PROBE_FLAG_END,
    decode_probe_ack,
    encode_probe_data,
)
from sender.adaptation_controller import (
    AdaptationController,
    AdaptationMode,
    ControllerConfig,
    ControllerInputs,
    Quality,
)
from sender.capacity_feedback_state import CapacityFeedbackState
from sender.capacity_probe_controller import (
    CapacityProbeController,
    CausalOfferedRate,
    ProbeBufferGrowthBaseline,
    ProbeConfig,
    ProbeInputs,
    ProbeState,
    frames_until_keyframe,
)
from sender.candidate_miss_diagnostics import (
    CandidateDiagnosticLog,
    child_process_snapshot,
)
from sender.quality_encoder_manager import (
    BoundedCandidateQueue,
    CandidateRefillCondition,
    CausalEncodeLead,
    CausalDemandPredictor,
    PreencodedQualityEncoderManager,
    QualityEncoderManager,
    SourceFrameRecord,
    application_bytes_for_payload,
    get_ordered_source_pair,
    packet_pacing_interval,
    prefetch_source_pairs,
    sender_frame_deadline,
    wait_for_candidate_priming,
)
from sender.track_b_logging import (
    CAPACITY_PROBE_FIELDNAMES,
    CONTROLLER_FIELDNAMES,
    neutral_probe_snapshot,
    write_strict_row,
)
from webrtc_diagnostics import (
    SctpEventDiagnostics,
    WebRTCDiagnostics,
    apply_ice_consent_timeout,
    apply_sctp_gap_rtt_fix,
    enable_ice_debug_logging,
    sctp_diagnostic_snapshot,
    sender_grace_for_backpressure_pause,
    process_resource_snapshot,
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


def probe_integer_video_fps(path, fallback=FPS_FALLBACK):
    """Return a positive integer source rate without changing decoder state."""
    try:
        with av.open(path) as container:
            rate = container.streams.video[0].average_rate
            if rate is not None and float(rate) > 0:
                return max(1, int(round(float(rate))))
    except Exception:
        logging.warning("[Sender] Could not read source FPS; using %d", fallback)
    return int(fallback)


class PyAVVideoDecoder:
    """Sequential PyAV fallback for platforms where TorchCodec CPU decode is unavailable."""

    def __init__(self, path):
        self.path = path
        self._count = self._count_frames()
        self._open()

    def _count_frames(self):
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            if stream.frames:
                return stream.frames
            return sum(1 for _ in container.decode(stream))

    def _open(self):
        self.container = av.open(self.path)
        self.stream = self.container.streams.video[0]
        self.frames = self.container.decode(self.stream)
        self.next_index = 0

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if index < self.next_index:
            self.container.close()
            self._open()
        while self.next_index <= index:
            frame = next(self.frames)
            self.next_index += 1
        rgb = frame.to_ndarray(format="rgb24")
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def open_video_decoder(path):
    if VideoDecoder is None:
        logging.warning("[Sender] TorchCodec unavailable; using PyAV fallback")
        return PyAVVideoDecoder(path)
    try:
        decoder = VideoDecoder(path, device="cpu")
        _ = decoder[0]
        return decoder
    except Exception as exc:
        logging.warning("[Sender] TorchCodec decoder failed on this platform; using PyAV fallback (%s)", type(exc).__name__)
        return PyAVVideoDecoder(path)


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
        self.packet_loss_rate  = max(0.0, min(1.0, float(args.packet_loss_rate)))
        self.packet_loss_rng   = random.Random(args.packet_loss_seed)
        self.ice_consent_timeout_s = args.ice_consent_timeout_s
        self._ice_consent_timeout_applied = False
        self.validation_sctp_gap_rtt_fix = args.validation_sctp_gap_rtt_fix
        self.dropped_packets   = 0
        self.dropped_rgb       = 0
        self.dropped_depth     = 0

        # ── Codec selection ──────────────────────────────────────────────────
        self.adaptation_mode = AdaptationMode(args.adaptation_mode)
        if self.adaptation_mode != AdaptationMode.LEGACY and args.codec != "h265":
            raise ValueError("Track B adaptation modes currently require --codec h265")

        if self.adaptation_mode == AdaptationMode.LEGACY:
            self.codec = h265.H265VideoCodec(intra_period=30)
            if args.codec == "dcvcrt":
                self.codec = dcvc.DCVCVideoCodec(intra_period=30)
            if args.codec == "h264":
                self.codec = h264.H264VideoCodec(intra_period=30)
            self.depth_codec = h265.H265VideoCodec(intra_period=30)
            if args.codec == "dcvcrt":
                self.depth_codec = dcvc.DCVCVideoCodec(intra_period=30)
            if args.codec == "h264":
                self.depth_codec = h264.H264VideoCodec(intra_period=30)
        else:
            # Track B parent owns metadata only. The child owns all H.265 state.
            self.codec = SimpleNamespace(
                intra_period=30, qp_p=int(args.track_b_high_qp)
            )
            self.depth_codec = SimpleNamespace(
                intra_period=30, qp_p=int(args.track_b_high_qp)
            )

        self.quality_manager = None
        self.controller = None
        self.demand_predictor = None
        self.encode_lead = None
        if self.adaptation_mode != AdaptationMode.LEGACY:
            pool_threads = int(args.track_b_encoder_pool_threads)
            if pool_threads <= 0:
                pool_threads = None
            manager_class = (
                PreencodedQualityEncoderManager
                if args.diagnostic_preencoded_manifest
                else QualityEncoderManager
            )
            manager_kwargs = {}
            if args.diagnostic_preencoded_manifest:
                manager_kwargs["manifest_path"] = args.diagnostic_preencoded_manifest
            self.quality_manager = manager_class(
                high_qp=args.track_b_high_qp,
                low_qp=args.track_b_low_qp,
                intra_period=30,
                pool_threads=pool_threads,
                max_inflight=args.track_b_encoder_max_inflight,
                completed_lead=args.track_b_encoder_completed_lead,
                sender_reserved_cpus=args.track_b_sender_reserved_cpus,
                enable_affinity=args.track_b_encoder_process,
                diagnostic_live_worker_state=(
                    args.diagnostic_continue_on_candidate_miss
                ),
                **manager_kwargs,
            )
            self.controller = AdaptationController(
                self.adaptation_mode,
                ControllerConfig(
                    buffer_soft_bytes=args.track_b_buffer_soft_bytes,
                    buffer_hard_bytes=args.track_b_buffer_hard_bytes,
                    buffer_growth_bytes=args.track_b_buffer_growth_bytes,
                    impairment_threshold=args.track_b_impairment_threshold,
                    severe_impairment_threshold=args.track_b_severe_impairment_threshold,
                    capacity_safety_margin=args.track_b_capacity_safety_margin,
                    upgrade_headroom_fraction=args.track_b_upgrade_headroom_fraction,
                    downgrade_confirmations=args.track_b_downgrade_confirmations,
                    upgrade_confirmations=args.track_b_upgrade_confirmations,
                    min_dwell_gops=args.track_b_min_dwell_gops,
                    passive_service_floor_fraction=(
                        args.track_b_passive_service_floor_fraction
                    ),
                ),
                initial_quality=(
                    Quality.LOW
                    if self.adaptation_mode == AdaptationMode.COMBINED
                    else None
                ),
            )
            self.demand_predictor = CausalDemandPredictor(
                alpha=args.track_b_demand_alpha
            )
            self.encode_lead = CausalEncodeLead(
                alpha=args.track_b_encode_lead_alpha
            )
        self.offered_rate = CausalOfferedRate(alpha=args.track_b_demand_alpha)
        self.capacity_probe = None
        if self.adaptation_mode == AdaptationMode.COMBINED:
            self.capacity_probe = CapacityProbeController(ProbeConfig(
                payload_bytes=args.track_b_probe_payload_bytes,
                min_duration_s=args.track_b_probe_min_duration_s,
                max_duration_s=args.track_b_probe_max_duration_s,
                max_total_bytes=args.track_b_probe_max_total_bytes,
                max_additional_rate_mbps=(
                    args.track_b_probe_max_additional_rate_mbps
                ),
                max_rate_multiplier=args.track_b_probe_max_rate_multiplier,
                min_confirmed_fraction=(
                    args.track_b_probe_min_confirmed_fraction
                ),
                ack_timeout_s=args.track_b_probe_ack_timeout_s,
                cooldown_s=args.track_b_probe_cooldown_s,
                authorization_ttl_s=args.track_b_probe_authorization_ttl_s,
                buffer_abort_bytes=args.track_b_probe_buffer_abort_bytes,
                buffer_growth_abort_bytes=(
                    args.track_b_probe_buffer_growth_abort_bytes
                ),
                impairment_abort_threshold=(
                    args.track_b_probe_impairment_abort_threshold
                ),
                capacity_safety_margin=args.track_b_capacity_safety_margin,
                boundary_guard_s=args.track_b_probe_boundary_guard_s,
            ))
        self._capacity_probe_task = None
        self._probe_buffer_baseline = None
        self._latest_probe_inputs = None
        self.capacity_probe_csv_path = args.capacity_probe_csv
        self.capacity_probe_file = None
        self.capacity_probe_writer = None
        self._fatal_error = None
        self._active_ws = None

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
        self.capacity_feedback_state = CapacityFeedbackState(
            shared_monotonic_clock=args.track_b_shared_monotonic_clock
        )
        self.receiver_health_freshness_s = float(
            args.receiver_health_freshness_s
        )
        self.health_inbox = ReceiverHealthInbox(
            self.receiver_health_freshness_s,
            shared_monotonic_clock=args.track_b_shared_monotonic_clock,
        )
        self._previous_max_buffered = 0
        self.controller_csv_path = args.controller_csv
        self.controller_file = None
        self.controller_writer = None
        self._open_controller_log()
        self._open_capacity_probe_log()
        if args.diagnostic_ice:
            enable_ice_debug_logging()
        self.diagnostics = WebRTCDiagnostics(args.diagnostic_csv, "sender")
        self.sctp_events = SctpEventDiagnostics(args.sctp_event_csv, "sender")
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
        self.candidate_diagnostic_log = CandidateDiagnosticLog(
            args.candidate_diagnostic_csv
        )

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
                "frame_deadline_monotonic",
                "production_timestamp_monotonic",
                "causal_encode_lead_s",
                "completed_encode_s",
                "send_budget_s",
                "producer_queue_depth",
                "producer_lead_frames",
                "active_encoder_workers",
                "configured_encoder_workers",
                "late_candidate_jobs",
                "cancelled_candidate_jobs",
                "request_submit_timestamp", "worker_start_timestamp",
                "worker_completion_timestamp", "result_received_timestamp",
                "source_load_start_timestamp", "source_load_completion_timestamp",
                "prefetched_pair_wait_s", "request_submission_interval_s",
                "worker_idle_s",
                "shared_memory_slot", "input_queue_depth", "output_queue_depth",
                "encoder_child_pid", "sender_cpu_affinity", "worker_cpu_affinity",
                "high_rgb_worker_encode_s", "low_rgb_worker_encode_s",
                "high_depth_worker_encode_s", "low_depth_worker_encode_s",
                "worker_total_production_s", "ipc_handoff_s",
                "worker_cpu_s", "worker_voluntary_context_switches",
                "worker_involuntary_context_switches",
                "candidate_not_ready_count", "encoder_worker_failures",
                "encoder_worker_exit_code", "collector_buffer_depth",
            ],
        )
        self.measurement_writer.writeheader()

    def _log_frame_measurement(self, *, frame_id, stream, is_key, encoded_size,
                               num_chunks, chunk_size, buffered_before, sent,
                               drop_reason="", send_start=None, send_end=None,
                               frame_deadline=None, causal_encode_lead=None,
                               completed_encode=None, send_budget=None,
                               production_timestamp=None,
                               producer_diagnostics=None):
        if self.measurement_writer is None:
            return
        producer_diagnostics = producer_diagnostics or {}
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
            "frame_deadline_monotonic": (
                "" if frame_deadline is None else f"{frame_deadline:.9f}"
            ),
            "production_timestamp_monotonic": (
                ""
                if production_timestamp is None
                else f"{production_timestamp:.9f}"
            ),
            "causal_encode_lead_s": (
                "" if causal_encode_lead is None else f"{causal_encode_lead:.9f}"
            ),
            "completed_encode_s": (
                "" if completed_encode is None else f"{completed_encode:.9f}"
            ),
            "send_budget_s": (
                "" if send_budget is None else f"{send_budget:.9f}"
            ),
            "producer_queue_depth": producer_diagnostics.get(
                "producer_queue_depth", ""
            ),
            "producer_lead_frames": producer_diagnostics.get(
                "producer_lead_frames", ""
            ),
            "active_encoder_workers": producer_diagnostics.get(
                "active_encoder_workers", ""
            ),
            "configured_encoder_workers": producer_diagnostics.get(
                "configured_encoder_workers", ""
            ),
            "late_candidate_jobs": producer_diagnostics.get(
                "late_candidate_jobs", ""
            ),
            "cancelled_candidate_jobs": producer_diagnostics.get(
                "cancelled_candidate_jobs", ""
            ),
            "request_submit_timestamp": producer_diagnostics.get("request_submitted", ""),
            "worker_start_timestamp": producer_diagnostics.get("worker_started", ""),
            "worker_completion_timestamp": producer_diagnostics.get("worker_completed", ""),
            "result_received_timestamp": producer_diagnostics.get("result_received", ""),
            "source_load_start_timestamp": producer_diagnostics.get("source_load_started", ""),
            "source_load_completion_timestamp": producer_diagnostics.get("source_load_completed", ""),
            "prefetched_pair_wait_s": producer_diagnostics.get("prefetched_pair_wait_s", ""),
            "request_submission_interval_s": producer_diagnostics.get("request_submission_interval_s", ""),
            "worker_idle_s": producer_diagnostics.get("worker_idle_s", ""),
            "shared_memory_slot": producer_diagnostics.get("slot", ""),
            "input_queue_depth": producer_diagnostics.get("input_queue_depth", ""),
            "output_queue_depth": producer_diagnostics.get("output_queue_depth", ""),
            "encoder_child_pid": producer_diagnostics.get("child_pid", ""),
            "sender_cpu_affinity": producer_diagnostics.get("sender_cpu_affinity", ""),
            "worker_cpu_affinity": producer_diagnostics.get("worker_cpu_affinity", ""),
            "high_rgb_worker_encode_s": producer_diagnostics.get("high_rgb_encode_s", ""),
            "low_rgb_worker_encode_s": producer_diagnostics.get("low_rgb_encode_s", ""),
            "high_depth_worker_encode_s": producer_diagnostics.get("high_depth_encode_s", ""),
            "low_depth_worker_encode_s": producer_diagnostics.get("low_depth_encode_s", ""),
            "worker_total_production_s": producer_diagnostics.get("total_production_s", ""),
            "ipc_handoff_s": producer_diagnostics.get("ipc_handoff_s", ""),
            "worker_cpu_s": producer_diagnostics.get("worker_cpu_s", ""),
            "worker_voluntary_context_switches": producer_diagnostics.get(
                "worker_voluntary_context_switches", ""
            ),
            "worker_involuntary_context_switches": producer_diagnostics.get(
                "worker_involuntary_context_switches", ""
            ),
            "candidate_not_ready_count": producer_diagnostics.get("candidate_not_ready_count", ""),
            "encoder_worker_failures": producer_diagnostics.get("worker_failures", ""),
            "encoder_worker_exit_code": producer_diagnostics.get("worker_exit_code", ""),
            "collector_buffer_depth": producer_diagnostics.get(
                "collector_buffer_depth", ""
            ),
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
                "feedback_sequence_id",
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
                "last_published_capacity_mbps",
                "raw_update_age_s",
                "published_estimate_age_s",
                "estimate_age_s",
                "estimate_fresh",
                "estimate_stale",
                "estimate_recovering",
                "estimate_unavailable",
                "estimate_state_reason",
                "freshness_threshold_s",
                "last_valid_update_timestamp",
                "last_published_estimate_timestamp",
                "packet_rtt_s",
                "feedback_delivery_delay_s",
                "feedback_clock_status",
                "capacity_state_update_status",
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

    def _open_controller_log(self):
        if not self.controller_csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.controller_csv_path))
        os.makedirs(parent, exist_ok=True)
        self.controller_file = open(self.controller_csv_path, "w", newline="")
        self.controller_writer = csv.DictWriter(
            self.controller_file,
            fieldnames=CONTROLLER_FIELDNAMES,
        )
        self.controller_writer.writeheader()

    def _close_controller_log(self):
        if self.controller_file is not None:
            self.controller_file.close()
            self.controller_file = None
            self.controller_writer = None

    def _open_capacity_probe_log(self):
        if not self.capacity_probe_csv_path:
            return
        parent = os.path.dirname(os.path.abspath(self.capacity_probe_csv_path))
        os.makedirs(parent, exist_ok=True)
        self.capacity_probe_file = open(
            self.capacity_probe_csv_path, "w", newline=""
        )
        self.capacity_probe_writer = csv.DictWriter(
            self.capacity_probe_file,
            fieldnames=CAPACITY_PROBE_FIELDNAMES,
        )
        self.capacity_probe_writer.writeheader()
        self.capacity_probe_file.flush()

    def _close_capacity_probe_log(self):
        if self.capacity_probe_file is not None:
            self.capacity_probe_file.close()
            self.capacity_probe_file = None
            self.capacity_probe_writer = None

    def _log_capacity_probe(
        self, now, event, snapshot, *, inputs=None, decision=None
    ):
        if self.capacity_probe_writer is None:
            return
        inputs = inputs or self._latest_probe_inputs
        rgb = "" if inputs is None else inputs.rgb_buffer
        depth = "" if inputs is None else inputs.depth_buffer
        requested = "" if decision is None else decision.requested_quality.value
        applied = "" if decision is None else decision.applied_quality.value
        before = "" if decision is None else decision.current_quality.value
        frame_id = "" if inputs is None else getattr(inputs, "frame_id", "")
        gop_id = "" if inputs is None else getattr(inputs, "gop_id", "")
        keyframe = "" if inputs is None else getattr(inputs, "keyframe", "")
        row = {
            "monotonic_timestamp": f"{float(now):.9f}",
            "event": event,
            "probe_id": snapshot["probe_id"],
            "probe_sequence": snapshot["probe_sequence"],
            "probe_state": snapshot["state"],
            "probe_reason": snapshot["reason"],
            "target_additional_rate_mbps": self._fmt_float(
                snapshot["target_additional_mbps"]
            ),
            "target_total_high_quality_rate_mbps": self._fmt_float(
                snapshot["target_high_mbps"]
            ),
            "offered_probe_bytes": snapshot["offered_bytes"],
            "confirmed_probe_bytes": snapshot["confirmed_bytes"],
            "offered_probe_application_bytes": snapshot[
                "offered_application_bytes"
            ],
            "confirmed_probe_application_bytes": snapshot[
                "confirmed_application_bytes"
            ],
            "elapsed_s": f"{snapshot['elapsed_s']:.9f}",
            "measured_probe_delivery_rate_mbps": self._fmt_float(
                snapshot["measured_probe_mbps"]
            ),
            "rgb_buffer": rgb,
            "depth_buffer": depth,
            "maximum_buffer": (
                "" if inputs is None else max(inputs.rgb_buffer, inputs.depth_buffer)
            ),
            "buffer_trend": "" if inputs is None else inputs.buffer_trend,
            "probe_buffer_credit_bytes": (
                "" if inputs is None else inputs.probe_buffer_credit_bytes
            ),
            "media_buffer_baseline_bytes": (
                ""
                if inputs is None or inputs.media_buffer_baseline_bytes is None
                else inputs.media_buffer_baseline_bytes
            ),
            "media_buffer_bytes": (
                ""
                if inputs is None or inputs.media_buffer_bytes is None
                else inputs.media_buffer_bytes
            ),
            "receiver_impairment": self._fmt_float(
                None if inputs is None else inputs.receiver_impairment
            ),
            "receiver_health_fresh": (
                "" if inputs is None else int(inputs.receiver_health_fresh)
            ),
            "acknowledgement_age_s": self._fmt_float(snapshot["ack_age_s"]),
            "success_safety_margin": self.capacity_probe.config.capacity_safety_margin,
            "abort_buffer_threshold": self.capacity_probe.config.buffer_abort_bytes,
            "authorization_created": int(
                snapshot["state"] == ProbeState.SUCCEEDED.value
            ),
            "authorization_expiry": self._fmt_float(
                snapshot["authorization_expiry"]
            ),
            "quality_before": before,
            "requested_quality": requested,
            "applied_quality": applied,
            "frame_id": frame_id,
            "gop_id": gop_id,
            "keyframe": keyframe,
            "current_offered_video_mbps": self._fmt_float(
                None if inputs is None else inputs.current_offered_video_mbps
            ),
            "passive_deterioration_ignored": (
                "1" if snapshot["passive_deterioration_ignored"] else "0"
            ),
        }
        write_strict_row(
            self.capacity_probe_writer,
            row,
            CAPACITY_PROBE_FIELDNAMES,
            "capacity probe CSV",
        )
        self.capacity_probe_file.flush()

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
            if msg and msg[0] == MSG_CAPACITY_PROBE_ACK:
                ack = decode_probe_ack(msg)
                if ack is not None and self.capacity_probe is not None:
                    now = time.perf_counter()
                    status = self.capacity_probe.accept_ack(ack, now)
                    snapshot = self.capacity_probe.snapshot(now)
                    self._log_capacity_probe(
                        now, f"ack_{status}", snapshot
                    )
                return
            if (
                len(msg) in RECEIVER_HEALTH_SIZES
                and msg[0] == MSG_RECEIVER_HEALTH
            ):
                self.health_inbox.accept(msg, time.perf_counter())
                return
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
        if not validate_capacity_feedback(feedback):
            return

        seq_id = int(feedback.get("seq_id", 0))
        feedback_sequence_id = int(
            feedback.get("feedback_sequence_id", 0)
        )
        feedback_received_ts = time.perf_counter()
        self.diagnostics.feedback_event(seq_id)
        packet_size = int(feedback.get("packet_size", 0))
        if seq_id > self.last_acknowledged_sequence:
            self.last_acknowledged_sequence = seq_id
        acknowledged_bytes = self.outstanding_packets.pop(seq_id, packet_size)
        packet_send_ts = self.outstanding_packet_send_times.pop(seq_id, None)
        packet_rtt_s = (
            None
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
        receiver_raw_age = feedback.get("raw_update_age_s")
        receiver_publication_age = feedback.get(
            "published_estimate_age_s"
        )
        freshness_threshold = feedback.get("freshness_threshold_s")
        last_valid_update_ts = feedback.get("last_valid_update_ts")
        last_published_estimate_ts = feedback.get(
            "last_published_estimate_ts"
        )
        if last_valid_update_ts is None and receiver_raw_age is not None:
            last_valid_update_ts = (
                float(feedback.get("receiver_ts", 0.0))
                - float(receiver_raw_age)
            )
        capacity_state_update_status = self.capacity_feedback_state.accept(
            feedback_sequence_id=feedback_sequence_id,
            application_sequence_id=seq_id,
            receiver_timestamp=feedback.get("receiver_ts"),
            receiver_raw_update_age_s=receiver_raw_age,
            receiver_published_estimate_age_s=receiver_publication_age,
            receiver_estimate_fresh=feedback.get("estimate_fresh", False),
            receiver_estimate_stale=feedback.get("estimate_stale", False),
            receiver_estimate_recovering=feedback.get(
                "estimate_recovering", False
            ),
            receiver_estimate_unavailable=feedback.get(
                "estimate_unavailable", False
            ),
            estimate_state_reason=feedback.get(
                "estimate_state_reason", "recovery"
            ),
            freshness_threshold_s=freshness_threshold,
            raw_capacity_mbps=capacity_mbps,
            filtered_capacity_mbps=filtered_capacity_mbps,
            last_valid_update_ts=last_valid_update_ts,
            last_published_estimate_ts=last_published_estimate_ts,
            feedback_received_ts=feedback_received_ts,
            packet_rtt_s=packet_rtt_s,
        )
        capacity_state = self.capacity_feedback_state.snapshot(
            feedback_received_ts
        )
        estimate_age = capacity_state["estimate_age_s"]
        estimate_fresh = capacity_state["estimate_fresh"]
        published_capacity_mbps = capacity_state[
            "published_estimated_capacity_mbps"
        ]
        skip_reason = feedback.get("skip_reason", "")
        filter_applied = (
            ewma is not None
            and filtered_ewma is not None
            and not math.isclose(
                float(ewma), float(filtered_ewma), rel_tol=1e-12, abs_tol=1e-15
            )
        )
        logging.debug(
            "[Estimator] ack seq=%d last_sent=%d outstanding=%d tau=%s max_frame=%s",
            seq_id, self.last_sent_sequence, packets_outstanding,
            self._fmt_float(ewma), self._fmt_float(max_frame_size)
        )

        if self.capacity_writer is not None:
            self.capacity_writer.writerow({
                "timestamp": f"{feedback_received_ts:.9f}",
                "receiver_timestamp": self._fmt_float(feedback.get("receiver_ts")),
                "feedback_sequence_id": feedback_sequence_id,
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
                "last_published_capacity_mbps": self._fmt_float(
                    capacity_state["last_published_capacity_mbps"]
                ),
                "raw_update_age_s": self._fmt_float(
                    capacity_state["raw_update_age_s"]
                ),
                "published_estimate_age_s": self._fmt_float(
                    capacity_state["published_estimate_age_s"]
                ),
                "estimate_age_s": self._fmt_float(estimate_age),
                "estimate_fresh": "1" if estimate_fresh else "0",
                "estimate_stale": (
                    "1" if capacity_state["estimate_stale"] else "0"
                ),
                "estimate_recovering": (
                    "1" if capacity_state["estimate_recovering"] else "0"
                ),
                "estimate_unavailable": (
                    "1" if capacity_state["estimate_unavailable"] else "0"
                ),
                "estimate_state_reason": capacity_state[
                    "estimate_state_reason"
                ],
                "freshness_threshold_s": self._fmt_float(freshness_threshold),
                "last_valid_update_timestamp": self._fmt_float(
                    last_valid_update_ts
                ),
                "last_published_estimate_timestamp": self._fmt_float(
                    last_published_estimate_ts
                ),
                "packet_rtt_s": self._fmt_float(packet_rtt_s),
                "feedback_delivery_delay_s": self._fmt_float(
                    capacity_state["feedback_delivery_delay_s"]
                ),
                "feedback_clock_status": capacity_state[
                    "feedback_clock_status"
                ],
                "capacity_state_update_status": capacity_state_update_status,
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
        return self.capacity_feedback_state.snapshot(time.perf_counter())

    def _ensure_capacity_probe_task(self):
        if (
            self._capacity_probe_task is not None
            and self._capacity_probe_task.done()
        ):
            # Retrieve and propagate task failures before starting another
            # probe worker. A completed successful task returns None.
            self._capacity_probe_task.result()
        if (
            self.capacity_probe is not None
            and self.capacity_probe.snapshot(time.perf_counter())["state"]
            == ProbeState.PROBING.value
            and (
                self._capacity_probe_task is None
                or self._capacity_probe_task.done()
            )
        ):
            self._capacity_probe_task = asyncio.create_task(
                self._run_capacity_probe()
            )

    async def _run_capacity_probe(self):
        """Pace bounded probe-only payloads while continuously checking safety."""
        rgb_at_activation = int(self.data_channel_rgb.bufferedAmount)
        depth_at_activation = int(self.data_channel_depth.bufferedAmount)
        tracker = ProbeBufferGrowthBaseline(
            rgb_at_activation, depth_at_activation
        )
        self._probe_buffer_baseline = tracker
        try:
            while self.capacity_probe is not None:
                now = time.perf_counter()
                latest = self._latest_probe_inputs
                if latest is None:
                    await asyncio.sleep(0.001)
                    continue
                health = self._current_health_state()
                rgb_buffer = int(self.data_channel_rgb.bufferedAmount)
                depth_buffer = int(self.data_channel_depth.bufferedAmount)
                buffer_state = tracker.observe(rgb_buffer, depth_buffer)
                refreshed = replace(
                    latest,
                    now=now,
                    rgb_buffer=rgb_buffer,
                    depth_buffer=depth_buffer,
                    buffer_trend=buffer_state["media_buffer_growth_bytes"],
                    probe_buffer_credit_bytes=buffer_state[
                        "probe_buffer_credit_bytes"
                    ],
                    media_buffer_baseline_bytes=buffer_state[
                        "media_buffer_baseline_bytes"
                    ],
                    media_buffer_bytes=buffer_state["media_buffer_bytes"],
                    receiver_impairment=health["impairment_rate"],
                    receiver_health_fresh=health["health_fresh"],
                    transport_connected=(
                        self.pc is not None
                        and self.pc.connectionState == "connected"
                        and self.data_channel_rgb.readyState == "open"
                    ),
                )
                self._latest_probe_inputs = refreshed
                snapshot = self.capacity_probe.evaluate(refreshed)
                if snapshot["state"] != ProbeState.PROBING.value:
                    self._log_capacity_probe(
                        now, "probe_terminal", snapshot, inputs=refreshed
                    )
                    return
                packet = self.capacity_probe.next_packet(now)
                if packet is None:
                    await asyncio.sleep(0.001)
                    continue
                message = encode_probe_data(
                    packet.probe_id,
                    packet.sequence,
                    packet.flags,
                    now,
                    b"\0" * packet.payload_bytes,
                )
                before_probe_enqueue = int(
                    self.data_channel_rgb.bufferedAmount
                )
                try:
                    self.data_channel_rgb.send(message)
                except Exception:
                    self.capacity_probe.cancel_authorization(
                        "probe_send_failed", now
                    )
                    self._log_capacity_probe(
                        now,
                        "send_failed",
                        self.capacity_probe.snapshot(now),
                        inputs=refreshed,
                    )
                    return
                after_probe_enqueue = int(
                    self.data_channel_rgb.bufferedAmount
                )
                buffer_state = tracker.record_probe_enqueue(
                    before_probe_enqueue, after_probe_enqueue
                )
                refreshed = replace(
                    refreshed,
                    probe_buffer_credit_bytes=buffer_state[
                        "probe_buffer_credit_bytes"
                    ],
                    media_buffer_baseline_bytes=buffer_state[
                        "media_buffer_baseline_bytes"
                    ],
                    media_buffer_bytes=buffer_state["media_buffer_bytes"],
                )
                self._latest_probe_inputs = refreshed
                self._log_capacity_probe(
                    now,
                    "end_sent" if packet.flags & PROBE_FLAG_END else "data_sent",
                    self.capacity_probe.snapshot(now),
                    inputs=refreshed,
                )
                await asyncio.sleep(0)
        finally:
            if self._probe_buffer_baseline is tracker:
                self._probe_buffer_baseline = None

    async def _stop_capacity_probe_for_stream_shutdown(self):
        """Stop probe traffic without recording normal stream completion as failure."""
        if self.capacity_probe is None:
            return
        now = time.perf_counter()
        snapshot = self.capacity_probe.stop_for_stream_shutdown(now)
        task = self._capacity_probe_task
        self._capacity_probe_task = None
        if task is not None:
            if not task.done():
                task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._log_capacity_probe(
            now,
            "normal_stream_shutdown",
            snapshot,
            inputs=self._latest_probe_inputs,
        )

    def _current_health_state(self):
        return self.health_inbox.snapshot(time.perf_counter())

    async def _abort_after_feedback_failure(self):
        if self.pc is not None and self.pc.connectionState != "closed":
            await self.pc.close()
        if self._active_ws is not None:
            await self._active_ws.close()

    def _handle_feedback_callback(self, msg):
        try:
            self._handle_feedback_message(msg)
        except Exception as error:
            logging.exception("[Sender] feedback handler failed")
            if self._fatal_error is None:
                self._fatal_error = RuntimeError(
                    f"feedback handler failed: {error}"
                )
            asyncio.create_task(self._abort_after_feedback_failure())

    @staticmethod
    def _candidate_application_bytes(payload_size, chunk_size, is_key):
        """Exact application bytes offered by the existing DESC/shard policy."""
        return application_bytes_for_payload(
            payload_size, chunk_size, is_key, SZ_DESC
        )

    def _candidate_pair_application_bytes(self, candidate):
        return (
            self._candidate_application_bytes(
                candidate.rgb_bytes, self.chunk_size, candidate.is_keyframe
            )
            + self._candidate_application_bytes(
                candidate.depth_bytes,
                self.chunk_size_depth,
                candidate.is_keyframe,
            )
        )

    def _log_controller_decision(
        self,
        *,
        now,
        candidate_set,
        decision,
        actual_quality,
        capacity,
        health,
        rgb_buffer,
        depth_buffer,
        buffer_trend,
        fallback_reason,
        probe,
    ):
        if self.controller_writer is None:
            return
        high = candidate_set.high
        low = candidate_set.low
        row = {
            "timestamp_monotonic": f"{now:.9f}",
            "decision_sequence": decision.decision_seq,
            "frame_id": candidate_set.frame_id,
            "gop_id": candidate_set.frame_id // 30,
            "keyframe": "1" if high.is_keyframe else "0",
            "adaptation_mode": self.adaptation_mode.value,
            "controller_state": decision.state.value,
            "current_quality": decision.current_quality.value,
            "requested_quality": decision.requested_quality.value,
            "applied_quality": actual_quality.value,
            "switch_requested": "1" if decision.switch_requested else "0",
            "switch_applied": "1" if decision.switch_applied and not fallback_reason else "0",
            "primary_decision_reason": (
                fallback_reason or decision.primary_reason
            ),
            "active_reason_flags": "|".join(decision.reason_flags),
            "raw_capacity_mbps": self._fmt_float(
                capacity["raw_estimated_capacity_mbps"]
            ),
            "filtered_capacity_mbps": self._fmt_float(
                capacity["filtered_estimated_capacity_mbps"]
            ),
            "published_capacity_mbps": self._fmt_float(
                capacity["published_estimated_capacity_mbps"]
            ),
            "capacity_age_s": self._fmt_float(capacity["estimate_age_s"]),
            "capacity_fresh": "1" if capacity["estimate_fresh"] else "0",
            "capacity_raw_update_age_s": self._fmt_float(
                capacity["raw_update_age_s"]
            ),
            "capacity_published_estimate_age_s": self._fmt_float(
                capacity["published_estimate_age_s"]
            ),
            "capacity_stale": "1" if capacity["estimate_stale"] else "0",
            "capacity_recovering": (
                "1" if capacity["estimate_recovering"] else "0"
            ),
            "capacity_unavailable": (
                "1" if capacity["estimate_unavailable"] else "0"
            ),
            "capacity_state_reason": capacity["estimate_state_reason"],
            "capacity_feedback_sequence": capacity["feedback_sequence_id"],
            "safe_capacity_mbps": self._fmt_float(decision.capacity_safe_mbps),
            "predicted_high_demand_mbps": self._fmt_float(
                self.demand_predictor.high_mbps
            ),
            "predicted_low_demand_mbps": self._fmt_float(
                self.demand_predictor.low_mbps
            ),
            "passive_service_estimate_mbps": self._fmt_float(
                capacity["published_estimated_capacity_mbps"]
            ),
            "current_offered_video_mbps": self._fmt_float(
                self.offered_rate.mbps
            ),
            "passive_deterioration_ignored": (
                "1" if probe["passive_deterioration_ignored"] else "0"
            ),
            "current_quality_demand_mbps": self._fmt_float(
                self.demand_predictor.low_mbps
                if decision.current_quality == Quality.LOW
                else self.demand_predictor.high_mbps
            ),
            "target_high_quality_demand_mbps": self._fmt_float(
                self.demand_predictor.high_mbps
            ),
            "probe_state": probe["state"],
            "probe_authorization_valid": (
                "1" if probe["authorization_valid"] else "0"
            ),
            "probe_result_age_s": self._fmt_float(probe["result_age_s"]),
            "probe_measured_rate_mbps": self._fmt_float(
                probe["measured_probe_mbps"]
            ),
            "probe_failure_reason": (
                probe["reason"]
                if probe["state"] in (
                    ProbeState.FAILED.value,
                    ProbeState.ABORTED.value,
                )
                else ""
            ),
            "rgb_bufferedAmount": rgb_buffer,
            "depth_bufferedAmount": depth_buffer,
            "max_bufferedAmount": max(rgb_buffer, depth_buffer),
            "buffer_trend_bytes": buffer_trend,
            "receiver_impairment_rate": self._fmt_float(
                health["impairment_rate"]
            ),
            "receiver_health_age_s": self._fmt_float(health["health_age_s"]),
            "receiver_health_fresh": "1" if health["health_fresh"] else "0",
            "health_feedback_sequence": health["feedback_seq"],
            "health_window_start_frame": (
                "" if health["window_start_frame"] is None
                else health["window_start_frame"]
            ),
            "health_window_end_frame": (
                "" if health["window_end_frame"] is None
                else health["window_end_frame"]
            ),
            "health_total_frames": (
                "" if health["total_frames"] is None else health["total_frames"]
            ),
            "health_full_misses": (
                "" if health["full_misses"] is None else health["full_misses"]
            ),
            "health_partial_frames": (
                "" if health["partial_frames"] is None else health["partial_frames"]
            ),
            "health_decode_failures": (
                "" if health["decode_failures"] is None
                else health["decode_failures"]
            ),
            "health_reference_unavailable": (
                "" if health["reference_unavailable"] is None
                else health["reference_unavailable"]
            ),
            "health_frozen_frames": (
                "" if health["frozen_frames"] is None else health["frozen_frames"]
            ),
            "dwell_gops": decision.dwell_gops,
            "cooldown_active": "1" if decision.cooldown_active else "0",
            "high_rgb_bytes": high.rgb_bytes,
            "high_depth_bytes": high.depth_bytes,
            "low_rgb_bytes": "" if low is None else low.rgb_bytes,
            "low_depth_bytes": "" if low is None else low.depth_bytes,
            "high_rgb_encode_s": f"{high.rgb_encode_s:.9f}",
            "high_depth_encode_s": f"{high.depth_encode_s:.9f}",
            "low_rgb_encode_s": "" if low is None else f"{low.rgb_encode_s:.9f}",
            "low_depth_encode_s": "" if low is None else f"{low.depth_encode_s:.9f}",
            "outstanding_messages": max(
                0, self.last_sent_sequence - self.last_acknowledged_sequence
            ),
            "feedback_delay_s": self._fmt_float(health["feedback_delay_s"]),
            "capacity_packet_rtt_s": self._fmt_float(
                capacity["packet_rtt_s"]
            ),
            "capacity_feedback_delivery_delay_s": self._fmt_float(
                capacity["feedback_delivery_delay_s"]
            ),
            "capacity_feedback_clock_status": capacity[
                "feedback_clock_status"
            ],
            "fallback_reason": fallback_reason,
        }
        write_strict_row(
            self.controller_writer,
            row,
            CONTROLLER_FIELDNAMES,
            "controller CSV",
        )
        self.controller_file.flush()

    def start_trace(self):
        """
        Launch the TC (traffic control) script as a subprocess to emulate
        real-world network loss / bandwidth conditions.  Does nothing if
        --trace_path was not provided.
        """
        if self.trace_process is None and self.args.trace_path:
            logging.info(f"Starting network trace: {self.args.trace_path}")
            cmd = [
                "sudo", "-n",
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
            ["sudo", "-n", "tc", "qdisc", "del", "dev", self.args.interface, "root"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
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
        if self.packet_loss_rate > 0 and self.packet_loss_rng.random() < self.packet_loss_rate:
            self.dropped_packets += 1
            if stream_name == "rgb":
                self.dropped_rgb += 1
            else:
                self.dropped_depth += 1
            logging.debug("[Sender] Dropped %s frame %s chunk %s", stream_name, fid, chunk_idx)
            return False, 0

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

    async def _wait_for_validation_start(self):
        """Optional experiment barrier; absent during every normal ReVo run."""
        start_signal_path = self.args.validation_start_signal_path
        if not start_signal_path:
            return True
        logging.info(
            "[Sender] waiting for validation start signal: %s",
            start_signal_path,
        )
        while not os.path.exists(start_signal_path):
            if (
                self.data_channel_rgb.readyState != "open"
                or self.data_channel_depth.readyState != "open"
            ):
                logging.warning(
                    "[Sender] DataChannel closed while waiting for start signal"
                )
                return False
            await asyncio.sleep(0.01)
        logging.info("[Sender] validation start signal observed")
        return True

    # ────────────────────────────────────────────────────────────────────────
    # Main streaming loop
    # ────────────────────────────────────────────────────────────────────────

    def _stream_fps(self) -> int:
        if self.quality_manager is None:
            return FPS_FALLBACK
        if int(self.args.track_b_fps) > 0:
            return int(self.args.track_b_fps)
        return probe_integer_video_fps(self.media_file)

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
        candidate_producer = None
        try:
            decoder       = open_video_decoder(self.media_file)
            decoder_depth = open_video_decoder(self.media_file_depth)
            fps = self._stream_fps()
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

            t0             = None          # synchronized frame-0 pacing origin
            frame_interval = 1.0 / fps

            candidate_queue = None
            candidate_prefilled = None
            candidate_refill = None
            source_prefetch_task = None
            candidate_submission_thread = None
            candidate_submission_stop = None
            producer_records = {}
            producer_diagnostics = {}
            late_candidate_jobs = 0
            validation_stopped_early = False
            producer_state = {"submitted": 0, "collected": 0}
            diagnostic_continue = bool(
                self.args.diagnostic_continue_on_candidate_miss
            )
            first_candidate_miss = False

            def record_candidate_diagnostic(
                fid, *, missed, first_miss_before, deadline, check_time,
                producer_record=None, send_start=None, send_end=None,
                reason="",
            ):
                if not diagnostic_continue:
                    return
                producer_record = producer_record or producer_records.get(fid, {})
                runtime = self.quality_manager.runtime_snapshot()
                sender_usage = process_resource_snapshot()
                worker_usage = child_process_snapshot(runtime.get("child_pid", ""))
                sctp = sctp_diagnostic_snapshot(self.pc)
                durations = producer_record.get("durations", {})
                submitted = int(producer_state["submitted"])
                collected = int(producer_state["collected"])
                consumed = int(candidate_queue.consumed_count)
                self.candidate_diagnostic_log.record(
                    fid,
                    missed_candidate=int(bool(missed)),
                    first_miss_already_occurred=int(bool(first_miss_before)),
                    diagnostic_only_after_first_miss=int(
                        bool(first_miss_before or missed)
                    ),
                    scheduled_deadline=deadline if deadline is not None else "",
                    actual_check_time=check_time,
                    deadline_lateness_s=(
                        max(0.0, check_time - deadline)
                        if deadline is not None else ""
                    ),
                    candidate_ready_timestamp=producer_record.get(
                        "worker_completed", ""
                    ),
                    source_load_start=producer_record.get("source_load_started", ""),
                    source_load_end=producer_record.get("source_load_completed", ""),
                    prefetch_wait_s=producer_record.get("prefetched_pair_wait_s", ""),
                    request_submit=producer_record.get("request_submitted", ""),
                    worker_start=producer_record.get("worker_started", ""),
                    worker_end=producer_record.get("worker_completed", ""),
                    high_rgb_encode_s=durations.get("high_rgb", ""),
                    low_rgb_encode_s=durations.get("low_rgb", ""),
                    high_depth_encode_s=durations.get("high_depth", ""),
                    low_depth_encode_s=durations.get("low_depth", ""),
                    worker_idle_s=producer_record.get("worker_idle_s", ""),
                    completed_queue_depth=candidate_queue.qsize(),
                    completed_lead=candidate_queue.qsize(),
                    source_prefetch_depth=raw_prefetch_queue.qsize(),
                    inflight_count=runtime.get("unfinished_ipc_requests", ""),
                    submitted_count=submitted,
                    collected_count=collected,
                    consumed_count=consumed,
                    total_submitted_minus_consumed=max(0, submitted - consumed),
                    input_ipc_queue_depth=runtime.get("input_queue_depth", ""),
                    output_ipc_queue_depth=runtime.get("output_queue_depth", ""),
                    active_native_calls=runtime.get("active_encoder_workers", ""),
                    worker_alive=int(bool(worker_usage["worker_alive"])),
                    worker_exit_code=runtime.get("worker_exit_code", ""),
                    sender_cpu_s=sender_usage.get("process_cpu_time_s", ""),
                    worker_cpu_s=producer_record.get("worker_cpu_s", ""),
                    worker_process_cpu_s=worker_usage.get("worker_cpu_s", ""),
                    sender_voluntary_context_switches=sender_usage.get(
                        "process_voluntary_context_switches", ""
                    ),
                    sender_involuntary_context_switches=sender_usage.get(
                        "process_involuntary_context_switches", ""
                    ),
                    worker_voluntary_context_switches=producer_record.get(
                        "worker_voluntary_context_switches", ""
                    ),
                    worker_involuntary_context_switches=producer_record.get(
                        "worker_involuntary_context_switches", ""
                    ),
                    worker_process_voluntary_context_switches=worker_usage.get(
                        "worker_voluntary_context_switches", ""
                    ),
                    worker_process_involuntary_context_switches=worker_usage.get(
                        "worker_involuntary_context_switches", ""
                    ),
                    event_loop_delay_s=self.diagnostics.last_event_loop_delay_s,
                    send_start=send_start if send_start is not None else "",
                    send_end=send_end if send_end is not None else "",
                    rgb_buffered_amount=int(self.data_channel_rgb.bufferedAmount),
                    depth_buffered_amount=int(self.data_channel_depth.bufferedAmount),
                    sctp_outbound_queue=sctp.get("sctp_outbound_queue", ""),
                    sctp_sent_queue=sctp.get("sctp_sent_queue", ""),
                    sctp_outstanding=sctp.get("sctp_sent_outstanding", ""),
                    sctp_flight_size_bytes=sctp.get("sctp_flight_size_bytes", ""),
                    drop_skip_reason=reason,
                )
            if self.quality_manager is not None:
                queue_frames = int(self.quality_manager.completed_lead)
                candidate_queue = BoundedCandidateQueue(queue_frames)
                candidate_refill = CandidateRefillCondition()
                raw_prefetch_queue = queue.Queue(maxsize=3)

                def load_source_pair(fid):
                    rgb = decoder[fid]
                    depth = decoder_depth[fid]
                    if rgb.dtype != torch.uint8:
                        rgb = rgb.clamp(0, 255).to(torch.uint8)
                    if depth.dtype != torch.uint8:
                        depth = depth.clamp(0, 255).to(torch.uint8)
                    rgb = rgb.permute(1, 2, 0).contiguous().cpu().numpy()
                    depth = depth.permute(1, 2, 0).contiguous().cpu().numpy()
                    return rgb, depth

                self.quality_manager.bind_completed_handoff(
                    candidate_queue, producer_records, producer_state
                )

                # Linux multiprocessing startup must happen before any source
                # prefetch or candidate-submission thread exists. Decode only
                # frame 0 on the sender main thread to discover the actual
                # shared-memory layout, then require an explicit child-ready
                # acknowledgement before launching background work.
                first_load_started = time.perf_counter()
                first_rgb, first_depth = load_source_pair(0)
                first_record = SourceFrameRecord(
                    0, first_rgb, first_depth, first_load_started,
                    time.perf_counter(),
                )
                await self.quality_manager.start(
                    first_record.rgb.shape, first_record.depth.shape,
                    first_record.rgb.dtype, first_record.depth.dtype,
                )
                snapshot = self.quality_manager.runtime_snapshot()
                logging.info(
                    "[EncoderWorker] ready sender_pid=%d child_pid=%s "
                    "sender_cpus=%s worker_cpus=%s",
                    os.getpid(), snapshot["child_pid"],
                    snapshot["sender_cpu_affinity"],
                    snapshot["worker_cpu_affinity"],
                )
                source_prefetch_task = asyncio.create_task(
                    prefetch_source_pairs(
                        len(decoder), load_source_pair, raw_prefetch_queue,
                        start_frame_id=1,
                    )
                )

                async def produce_candidates():
                    nonlocal late_candidate_jobs, candidate_submission_thread
                    nonlocal candidate_submission_stop
                    try:
                        total_limit = (
                            self.quality_manager.completed_lead
                            + self.quality_manager.max_inflight
                        )
                        candidate_submission_stop = threading.Event()

                        def submit_prefetched_candidates():
                            submitted = 0
                            first_pending = (first_record, 0.0)
                            previous_request_ts = None
                            try:
                                while (
                                    submitted < len(decoder)
                                    and not candidate_submission_stop.is_set()
                                ):
                                    if not candidate_refill.wait_for_room_blocking(
                                        submitted, candidate_queue, total_limit,
                                        candidate_submission_stop,
                                    ):
                                        break
                                    raw_pair = first_pending
                                    first_pending = None
                                    if raw_pair is None:
                                        wait_started = time.perf_counter()
                                        while not candidate_submission_stop.is_set():
                                            try:
                                                source_record = raw_prefetch_queue.get(
                                                    timeout=0.1
                                                )
                                                break
                                            except queue.Empty:
                                                continue
                                        else:
                                            break
                                        prefetch_wait_s = (
                                            time.perf_counter() - wait_started
                                        )
                                        if source_record.error is not None:
                                            raise source_record.error
                                        if source_record.frame_id != submitted:
                                            raise RuntimeError(
                                                "source prefetch order mismatch: "
                                                f"expected {submitted}, got "
                                                f"{source_record.frame_id}"
                                            )
                                    else:
                                        source_record, prefetch_wait_s = raw_pair
                                    causal_lead = self.encode_lead.predict(
                                        frame_interval
                                    )
                                    request_ts = (
                                        self.quality_manager.submit_frame_blocking(
                                            submitted, fps,
                                            source_record.rgb, source_record.depth,
                                        )
                                    )
                                    producer_records[submitted] = {
                                        "causal_lead": causal_lead,
                                        "request_submitted": request_ts,
                                        "source_load_started": source_record.load_started,
                                        "source_load_completed": source_record.load_completed,
                                        "prefetched_pair_wait_s": prefetch_wait_s,
                                        "request_submission_interval_s": (
                                            0.0
                                            if previous_request_ts is None
                                            else request_ts - previous_request_ts
                                        ),
                                    }
                                    previous_request_ts = request_ts
                                    submitted += 1
                                    producer_state["submitted"] = submitted
                            except BaseException as exc:
                                if not candidate_submission_stop.is_set():
                                    candidate_queue.put_error_from_collector(exc)

                        candidate_submission_thread = threading.Thread(
                            target=submit_prefetched_candidates,
                            name="revo-candidate-submitter",
                            daemon=True,
                        )
                        candidate_submission_thread.start()
                        while candidate_submission_thread.is_alive():
                            await asyncio.to_thread(
                                candidate_submission_thread.join, 0.1
                            )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as exc:
                        await candidate_queue.put_error(exc)
                    finally:
                        if candidate_submission_stop is not None:
                            candidate_submission_stop.set()
                            candidate_refill.notify_consumed_sync()
                        if (
                            candidate_submission_thread is not None
                            and candidate_submission_thread.is_alive()
                        ):
                            await asyncio.to_thread(
                                candidate_submission_thread.join, 2.0
                            )
                        # Release a prefetch put already running in to_thread;
                        # cancellation alone cannot interrupt queue.Queue.put.
                        while True:
                            try:
                                raw_prefetch_queue.get_nowait()
                            except queue.Empty:
                                break
                        if (
                            source_prefetch_task is not None
                            and not source_prefetch_task.done()
                        ):
                            source_prefetch_task.cancel()
                        if source_prefetch_task is not None:
                            await asyncio.gather(
                                source_prefetch_task, return_exceptions=True
                            )

                candidate_producer = asyncio.create_task(produce_candidates())
                t0 = await wait_for_candidate_priming(
                    candidate_queue,
                    None,
                    required=queue_frames,
                )

            for frame_id in range(len(decoder)):
                stop_signal = self.args.validation_stop_signal_path
                if stop_signal and os.path.exists(stop_signal):
                    validation_stopped_early = True
                    logging.info(
                        "[Sender] Event validation stop observed before frame %d",
                        frame_id,
                    )
                    break
                if self.quality_manager is None and frame_id == 1:
                    # Preserve the legacy encoder's established pacing origin.
                    t0 = time.perf_counter()
                # Legacy uses the original two-encoder path unchanged. Track B
                # continuously advances four independent H.265 reference chains.
                if self.quality_manager is None:
                    # ── Decode raw tensors ───────────────────────────────────
                    raw = decoder[frame_id].unsqueeze(0).unsqueeze(0).float().mul_(1.0 / 255.0)
                    raw_depth = decoder_depth[frame_id].unsqueeze(0).unsqueeze(0).float().mul_(1.0 / 255.0)
                    causal_encode_lead = 0.0
                    encode_started = time.perf_counter()
                    task_rgb = asyncio.to_thread(
                        encode_rgb, raw, frame_id, fps
                    )
                    task_depth = asyncio.to_thread(
                        encode_depth, raw_depth, frame_id, fps
                    )
                    packet_list, packet_list_depth = await asyncio.gather(
                        task_rgb, task_depth
                    )
                    packet_pairs = [
                        (out, packet_list_depth[0])
                        for out in packet_list
                        if packet_list_depth
                    ]
                else:
                    candidate_wait_deadline = None
                    if frame_id > 0:
                        pending_lead = producer_records.get(frame_id, {}).get(
                            "causal_lead",
                            self.encode_lead.predict(frame_interval),
                        )
                        candidate_wait_deadline = sender_frame_deadline(
                            t0,
                            frame_id,
                            frame_interval,
                            pending_lead,
                        )
                    try:
                        if candidate_wait_deadline is None:
                            candidate_record = candidate_queue.get_nowait()
                            discarded_late_records = []
                        elif diagnostic_continue:
                            (
                                candidate_record,
                                discarded_late_records,
                            ) = await candidate_queue.get_frame_before(
                                frame_id, candidate_wait_deadline
                            )
                        else:
                            candidate_record = await candidate_queue.get_before(
                                candidate_wait_deadline
                            )
                            discarded_late_records = []
                        for late_record in discarded_late_records:
                            late_fid = int(late_record[0])
                            late_details = producer_records.pop(late_fid, {})
                            durations = late_details.get("durations", {})
                            self.candidate_diagnostic_log.record(
                                late_fid,
                                candidate_ready_timestamp=late_details.get(
                                    "worker_completed", ""
                                ),
                                worker_start=late_details.get("worker_started", ""),
                                worker_end=late_details.get("worker_completed", ""),
                                worker_idle_s=late_details.get("worker_idle_s", ""),
                                high_rgb_encode_s=durations.get("high_rgb", ""),
                                low_rgb_encode_s=durations.get("low_rgb", ""),
                                high_depth_encode_s=durations.get("high_depth", ""),
                                low_depth_encode_s=durations.get("low_depth", ""),
                                worker_cpu_s=late_details.get("worker_cpu_s", ""),
                                worker_voluntary_context_switches=late_details.get(
                                    "worker_voluntary_context_switches", ""
                                ),
                                worker_involuntary_context_switches=late_details.get(
                                    "worker_involuntary_context_switches", ""
                                ),
                                drop_skip_reason="candidate_not_ready_late_result_discarded",
                            )
                            await candidate_refill.notify_consumed()
                        if candidate_record is None:
                            raise asyncio.QueueEmpty
                        (
                            produced_fid,
                            candidate_sets,
                            encode_started,
                            production_timestamp,
                            causal_encode_lead,
                            producer_error,
                        ) = candidate_record
                        completed_details = producer_records.get(produced_fid, {})
                        if completed_details:
                            production_s = float(
                                completed_details.get("total_production_s", 0.0)
                            )
                            if production_s > frame_interval:
                                late_candidate_jobs += 1
                            self.encode_lead.observe(production_s)
                            durations = completed_details.get("durations", {})
                            completed_details.update({
                                **self.quality_manager.runtime_snapshot(),
                                "high_rgb_encode_s": durations.get("high_rgb", 0.0),
                                "low_rgb_encode_s": durations.get("low_rgb", 0.0),
                                "high_depth_encode_s": durations.get("high_depth", 0.0),
                                "low_depth_encode_s": durations.get("low_depth", 0.0),
                                "ipc_handoff_s": float(
                                    completed_details.get("result_received", 0.0)
                                ) - float(
                                    completed_details.get("worker_completed", 0.0)
                                ),
                            })
                    except (asyncio.QueueEmpty, asyncio.TimeoutError) as exc:
                        snapshot = self.quality_manager.runtime_snapshot()
                        if diagnostic_continue:
                            checked = time.perf_counter()
                            producer_snapshot = {
                                **producer_records.get(frame_id, {}),
                                **snapshot,
                                "producer_queue_depth": candidate_queue.qsize(),
                                "producer_lead_frames": candidate_queue.qsize(),
                                "late_candidate_jobs": late_candidate_jobs,
                            }
                            record_candidate_diagnostic(
                                frame_id,
                                missed=True,
                                first_miss_before=first_candidate_miss,
                                deadline=candidate_wait_deadline,
                                check_time=checked,
                                producer_record=producer_snapshot,
                                reason="candidate_not_ready",
                            )
                            first_candidate_miss = True
                            for stream_name in ("rgb", "depth"):
                                self._log_frame_measurement(
                                    frame_id=frame_id,
                                    stream=stream_name,
                                    is_key=(frame_id % 30 == 0),
                                    encoded_size=0,
                                    num_chunks=0,
                                    chunk_size=(
                                        self.chunk_size
                                        if stream_name == "rgb"
                                        else self.chunk_size_depth
                                    ),
                                    buffered_before=(
                                        self.data_channel_rgb.bufferedAmount
                                        if stream_name == "rgb"
                                        else self.data_channel_depth.bufferedAmount
                                    ),
                                    sent=False,
                                    drop_reason="diagnostic_candidate_not_ready",
                                    frame_deadline=candidate_wait_deadline,
                                    producer_diagnostics=producer_snapshot,
                                )
                            logging.error(
                                "[DIAGNOSTIC_ONLY] candidate_not_ready frame=%d "
                                "deadline=%s check=%.9f worker=%s; continuing",
                                frame_id, candidate_wait_deadline, checked, snapshot,
                            )
                            continue
                        raise RuntimeError(
                            "candidate_not_ready "
                            f"frame={frame_id} queue_depth={candidate_queue.qsize()} "
                            f"candidate_ready_timestamp=unavailable "
                            f"scheduled_deadline={candidate_wait_deadline} "
                            f"worker={snapshot}"
                        ) from exc
                    if producer_error is not None:
                        raise producer_error
                    await candidate_refill.notify_consumed()
                    if self.quality_manager is not None:
                        await asyncio.sleep(0)
                    if produced_fid != frame_id:
                        raise RuntimeError(
                            "candidate pipeline frame order mismatch: "
                            f"expected {frame_id}, got {produced_fid}"
                        )
                    packet_pairs = []
                    for candidates in candidate_sets:
                        high = candidates.high
                        if high is None:
                            continue
                        now = time.perf_counter()
                        capacity = self._current_capacity_state()
                        health = self._current_health_state()
                        rgb_buffer = int(self.data_channel_rgb.bufferedAmount)
                        depth_buffer = int(self.data_channel_depth.bufferedAmount)
                        max_buffer = max(rgb_buffer, depth_buffer)
                        buffer_trend = max_buffer - self._previous_max_buffered
                        self._previous_max_buffered = max_buffer
                        probe_buffer_credit = 0
                        media_buffer_baseline = None
                        media_buffer = None
                        if (
                            self._probe_buffer_baseline is not None
                            and self.capacity_probe is not None
                            and self.capacity_probe.snapshot(now)["state"]
                            == ProbeState.PROBING.value
                        ):
                            buffer_state = self._probe_buffer_baseline.observe(
                                rgb_buffer, depth_buffer
                            )
                            buffer_trend = buffer_state[
                                "media_buffer_growth_bytes"
                            ]
                            probe_buffer_credit = buffer_state[
                                "probe_buffer_credit_bytes"
                            ]
                            media_buffer_baseline = buffer_state[
                                "media_buffer_baseline_bytes"
                            ]
                            media_buffer = buffer_state["media_buffer_bytes"]
                        probe_snapshot = neutral_probe_snapshot()
                        if self.capacity_probe is not None:
                            gop_size = int(self.codec.intra_period)
                            frames_to_keyframe = frames_until_keyframe(
                                candidates.frame_id, gop_size
                            )
                            offered = self.offered_rate.mbps
                            passive_low = (
                                capacity["estimate_fresh"]
                                and capacity[
                                    "published_estimated_capacity_mbps"
                                ] is not None
                                and offered is not None
                                and capacity[
                                    "published_estimated_capacity_mbps"
                                ]
                                < offered
                                * self.controller.config
                                .passive_service_floor_fraction
                            )
                            probe_inputs = ProbeInputs(
                                now=now,
                                current_quality=(
                                    self.controller.current_quality.value
                                ),
                                rgb_buffer=rgb_buffer,
                                depth_buffer=depth_buffer,
                                buffer_trend=buffer_trend,
                                receiver_impairment=health["impairment_rate"],
                                receiver_health_fresh=health["health_fresh"],
                                dwell_complete=self.controller.dwell_complete(
                                    candidates.frame_id // 30
                                ),
                                transport_connected=(
                                    self.pc.connectionState == "connected"
                                ),
                                seconds_to_keyframe=(
                                    frames_to_keyframe * frame_interval
                                ),
                                current_offered_video_mbps=offered,
                                predicted_low_demand_mbps=(
                                    self.demand_predictor.low_mbps
                                ),
                                predicted_high_demand_mbps=(
                                    self.demand_predictor.high_mbps
                                ),
                                severe_downgrade=(
                                    max_buffer
                                    >= self.controller.config.buffer_hard_bytes
                                    or (
                                        health["health_fresh"]
                                        and health["impairment_rate"] is not None
                                        and health["impairment_rate"]
                                        >= self.controller.config
                                        .severe_impairment_threshold
                                    )
                                ),
                                passive_deterioration_unaligned=passive_low,
                                frame_id=candidates.frame_id,
                                gop_id=candidates.frame_id // 30,
                                keyframe=high.is_keyframe,
                                probe_buffer_credit_bytes=probe_buffer_credit,
                                media_buffer_baseline_bytes=(
                                    media_buffer_baseline
                                ),
                                media_buffer_bytes=media_buffer,
                            )
                            self._latest_probe_inputs = probe_inputs
                            previous_probe_state = self.capacity_probe.snapshot(
                                now
                            )["state"]
                            probe_snapshot = self.capacity_probe.evaluate(
                                probe_inputs
                            )
                            if (
                                probe_snapshot["state"]
                                == ProbeState.PROBING.value
                                and probe_snapshot[
                                    "passive_deterioration_ignored"
                                ]
                            ):
                                self._log_capacity_probe(
                                    now,
                                    "passive_deterioration_ignored_unaligned",
                                    probe_snapshot,
                                    inputs=probe_inputs,
                                )
                            if probe_snapshot["state"] != previous_probe_state:
                                self._log_capacity_probe(
                                    now,
                                    "state_transition",
                                    probe_snapshot,
                                    inputs=probe_inputs,
                                )
                            self._ensure_capacity_probe_task()
                        controller_inputs = ControllerInputs(
                            timestamp=now,
                            frame_id=candidates.frame_id,
                            gop_id=candidates.frame_id // 30,
                            is_keyframe=high.is_keyframe,
                            capacity_raw_mbps=capacity[
                                "raw_estimated_capacity_mbps"
                            ],
                            capacity_filtered_mbps=capacity[
                                "filtered_estimated_capacity_mbps"
                            ],
                            capacity_published_mbps=capacity[
                                "published_estimated_capacity_mbps"
                            ],
                            capacity_age_s=capacity["estimate_age_s"],
                            capacity_fresh=capacity["estimate_fresh"],
                            rgb_buffered_amount=rgb_buffer,
                            depth_buffered_amount=depth_buffer,
                            buffer_trend_bytes=buffer_trend,
                            receiver_impairment_rate=health["impairment_rate"],
                            receiver_health_age_s=health["health_age_s"],
                            receiver_health_fresh=health["health_fresh"],
                            predicted_high_demand_mbps=self.demand_predictor.high_mbps,
                            predicted_low_demand_mbps=self.demand_predictor.low_mbps,
                            current_offered_video_mbps=self.offered_rate.mbps,
                            probe_state=probe_snapshot["state"],
                            probe_authorization_valid=probe_snapshot[
                                "authorization_valid"
                            ],
                            probe_measured_mbps=probe_snapshot[
                                "measured_probe_mbps"
                            ],
                            probe_failure_reason=probe_snapshot["reason"],
                            low_candidate_available=candidates.low_usable,
                            outstanding_messages=max(
                                0,
                                self.last_sent_sequence
                                - self.last_acknowledged_sequence,
                            ),
                        )
                        decision = self.controller.decide(controller_inputs)
                        selected, fallback_reason = self.quality_manager.select(
                            candidates, decision.applied_quality
                        )
                        actual_quality = (
                            Quality.LOW
                            if candidates.low is not None
                            and selected is candidates.low
                            else Quality.HIGH
                        )
                        self._log_controller_decision(
                            now=now,
                            candidate_set=candidates,
                            decision=decision,
                            actual_quality=actual_quality,
                            capacity=capacity,
                            health=health,
                            rgb_buffer=rgb_buffer,
                            depth_buffer=depth_buffer,
                            buffer_trend=buffer_trend,
                            fallback_reason=fallback_reason,
                            probe=probe_snapshot,
                        )
                        high_bytes = self._candidate_pair_application_bytes(high)
                        low_bytes = (
                            high_bytes
                            if candidates.low is None
                            else self._candidate_pair_application_bytes(
                                candidates.low
                            )
                        )
                        self.demand_predictor.observe_completed_frame(
                            high_application_bytes=high_bytes,
                            low_application_bytes=low_bytes,
                            frame_interval_s=frame_interval,
                        )
                        selected_bytes = self._candidate_pair_application_bytes(
                            selected
                        )
                        self.offered_rate.observe(
                            selected_bytes, frame_interval
                        )
                        if self.capacity_probe is not None:
                            self._log_capacity_probe(
                                now,
                                "controller_decision",
                                probe_snapshot,
                                inputs=self._latest_probe_inputs,
                                decision=decision,
                            )
                        packet_pairs.append((selected.rgb, selected.depth))
                    producer_diagnostics = {
                        **self.quality_manager.runtime_snapshot(),
                        **producer_records.pop(frame_id, {}),
                        "producer_queue_depth": candidate_queue.qsize(),
                        "producer_lead_frames": candidate_queue.qsize(),
                        "late_candidate_jobs": late_candidate_jobs,
                    }
                    record_candidate_diagnostic(
                        frame_id,
                        missed=False,
                        first_miss_before=first_candidate_miss,
                        deadline=candidate_wait_deadline,
                        check_time=time.perf_counter(),
                        producer_record=producer_diagnostics,
                    )
                if self.quality_manager is None:
                    production_timestamp = time.perf_counter()
                completed_encode_s = production_timestamp - encode_started
                if self.encode_lead is not None and self.quality_manager is None:
                    self.encode_lead.observe(completed_encode_s)

                t_send0 = time.perf_counter()
                if self.trace_t0_sender is None:
                    self.trace_t0_sender = t_send0

                # ── Compute send budget (time remaining in this frame slot) ──
                send_budget   = frame_interval
                frame_deadline = None
                if frame_id > 0:
                    frame_deadline = sender_frame_deadline(
                        t0,
                        frame_id,
                        frame_interval,
                        causal_encode_lead,
                    )
                    send_budget    = max(0.0, frame_deadline - t_send0)

                # ── Extract compressed outputs ───────────────────────────────
                for out, out_depth in packet_pairs:
                    payload       = out["payload"]
                    out_fid       = out["frame_id"]
                    is_key        = out["is_key"]
                    qp            = out["qp"]

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
                            sent=False, drop_reason=drop_reason,
                            frame_deadline=frame_deadline,
                            causal_encode_lead=causal_encode_lead,
                            completed_encode=completed_encode_s,
                            send_budget=send_budget,
                            production_timestamp=production_timestamp,
                            producer_diagnostics=producer_diagnostics,
                        )
                        self._log_frame_measurement(
                            frame_id=out_fid, stream="depth", is_key=is_key,
                            encoded_size=len(payload_depth), num_chunks=num_chunks_depth_log,
                            chunk_size=self.chunk_size_depth, buffered_before=buffered_depth_before,
                            sent=False, drop_reason=drop_reason,
                            frame_deadline=frame_deadline,
                            causal_encode_lead=causal_encode_lead,
                            completed_encode=completed_encode_s,
                            send_budget=send_budget,
                            production_timestamp=production_timestamp,
                            producer_diagnostics=producer_diagnostics,
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
                    per_pkt_dt = packet_pacing_interval(
                        send_budget,
                        num_chunks + num_chunks_depth,
                        is_keyframe=is_key,
                    )

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
                        if per_pkt_dt == 0.0:
                            idx += 1
                            return
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
                    if diagnostic_continue:
                        sctp_after_send = sctp_diagnostic_snapshot(self.pc)
                        self.candidate_diagnostic_log.record(
                            out_fid,
                            send_start=t_send0,
                            send_end=t_send1,
                            rgb_buffered_amount=int(
                                self.data_channel_rgb.bufferedAmount
                            ),
                            depth_buffered_amount=int(
                                self.data_channel_depth.bufferedAmount
                            ),
                            sctp_outbound_queue=sctp_after_send.get(
                                "sctp_outbound_queue", ""
                            ),
                            sctp_sent_queue=sctp_after_send.get(
                                "sctp_sent_queue", ""
                            ),
                            sctp_outstanding=sctp_after_send.get(
                                "sctp_sent_outstanding", ""
                            ),
                            sctp_flight_size_bytes=sctp_after_send.get(
                                "sctp_flight_size_bytes", ""
                            ),
                        )
                    self._log_frame_measurement(
                        frame_id=out_fid, stream="rgb", is_key=is_key,
                        encoded_size=len(payload), num_chunks=num_chunks,
                        chunk_size=self.chunk_size, buffered_before=buffered_rgb_before,
                        sent=True, send_start=t_send0, send_end=t_send1,
                        frame_deadline=frame_deadline,
                        causal_encode_lead=causal_encode_lead,
                        completed_encode=completed_encode_s,
                        send_budget=send_budget,
                        production_timestamp=production_timestamp,
                        producer_diagnostics=producer_diagnostics,
                    )
                    self._log_frame_measurement(
                        frame_id=out_fid, stream="depth", is_key=is_key,
                        encoded_size=len(payload_depth), num_chunks=num_chunks_depth,
                        chunk_size=self.chunk_size_depth, buffered_before=buffered_depth_before,
                        sent=True, send_start=t_send0, send_end=t_send1,
                        frame_deadline=frame_deadline,
                        causal_encode_lead=causal_encode_lead,
                        completed_encode=completed_encode_s,
                        send_budget=send_budget,
                        production_timestamp=production_timestamp,
                        producer_diagnostics=producer_diagnostics,
                    )

                    self.sent_frames       += 1
                    self.sent_frames_depth += 1
                    if is_key or (out_fid % 15 == 0):
                        logging.info(
                            "Sent frame %04d (%s) RGB=%d B depth=%d B",
                            out_fid, "I" if is_key else "P", len(payload), len(payload_depth)
                        )
                # Release exactly one unit of producer lead after this source
                # frame is classified, including an explicitly logged media
                # drop.  Otherwise a drop can strand the bounded producer.
            if candidate_producer is not None:
                if validation_stopped_early and not candidate_producer.done():
                    candidate_producer.cancel()
                    await asyncio.gather(
                        candidate_producer, return_exceptions=True
                    )
                else:
                    await candidate_producer

            # ── Encoder flush ────────────────────────────────────────────────
            # H.265 and H264 codecs may buffer a few frames internally; flush them.
            if (
                not validation_stopped_early
                and isinstance(self.codec, (h265.H265VideoCodec, h264.H264VideoCodec))
            ):
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
            raise
        finally:
            if candidate_producer is not None and not candidate_producer.done():
                candidate_producer.cancel()
                await asyncio.gather(candidate_producer, return_exceptions=True)
            if source_prefetch_task is not None and not source_prefetch_task.done():
                source_prefetch_task.cancel()
                await asyncio.gather(
                    source_prefetch_task, return_exceptions=True
                )

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
        self.capacity_feedback_state.reset()

        # Two unreliable unordered DataChannels: late packets are useless for
        # real-time video, so we skip retransmission at the SCTP layer entirely.
        self.data_channel_rgb   = self.pc.createDataChannel("rgb_payload",   ordered=False, maxRetransmits=0)
        self.data_channel_depth = self.pc.createDataChannel("depth_payload", ordered=False, maxRetransmits=0)
        self.sctp_events.instrument(self.pc)
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
            **process_resource_snapshot(),
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
            self._handle_feedback_callback(msg)

        @self.data_channel_depth.on("open")
        def on_depth_open():
            logging.info("depth_payload channel open")
            self.p_open.set()

        @self.data_channel_depth.on("message")
        def on_depth_message(msg):
            self._handle_feedback_callback(msg)

        @self.data_channel_rgb.on("close")
        def on_rgb_close():
            logging.info("rgb_payload channel closed")

        @self.data_channel_depth.on("close")
        def on_depth_close():
            logging.info("depth_payload channel closed")

        try:
            async with ClientSession() as session:
                async with session.ws_connect(self.signalling_server) as ws:
                    self._active_ws = ws
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
                    # Validation trace barriers release only after the receiver
                    # has observed INIT. This prevents the trace's first loss
                    # burst from destroying required application metadata.
                    if (
                        self.args.validation_start_signal_path
                        and not self.args.validation_paced_probe_only
                        and not self.args.validation_train_only
                    ):
                        self._send_init(512, 512, self._stream_fps())
                    if not await self._wait_for_validation_start():
                        raise RuntimeError("validation start barrier failed")

                    if self.args.validation_paced_probe_only:
                        await self.send_validation_paced_probe()
                    elif self.args.validation_train_only:
                        await self.send_validation_packet_trains()
                    else:
                        await self.stream_video()
                    await self._stop_capacity_probe_for_stream_shutdown()
                    if self._fatal_error is not None:
                        raise self._fatal_error
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
                        f"Headers depth: {self.reliable_bytes_meta_depth/1e6:.2f} MB\n"
                        f"App packet loss: {self.packet_loss_rate:.3f}; dropped "
                        f"{self.dropped_packets} packets ({self.dropped_rgb} RGB, {self.dropped_depth} depth)"
                    )

                    await self.pc.close()
                    logging.info("[Sender] PeerConnection closed")
                    await ws.close()
                    logging.info("[Sender] WebSocket closed")

        except Exception as e:
            logging.exception(f"[Sender] Error during execution: {e}")
            raise
        finally:
            # Always clean up TC rules even on crash / KeyboardInterrupt
            if self.pc is not None and self.pc.connectionState != "closed":
                await self.pc.close()
            if self.quality_manager is not None:
                await self.quality_manager.aclose()
            self.stop_trace()
            self._close_measurement_log()
            self._close_capacity_log()
            self._close_controller_log()
            self._close_capacity_probe_log()
            self._close_probe_sender_log()
            await self.diagnostics.stop()
            self.candidate_diagnostic_log.close()
            self.sctp_events.close()
            self._active_ws = None


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
    p.add_argument("--packet_loss_rate", type=float, default=0.0,
                   help="Mac-friendly app-level random data packet drop rate, e.g. 0.05 for 5%%")
    p.add_argument("--packet_loss_seed", type=int, default=7,
                   help="Random seed for --packet_loss_rate")
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
    p.add_argument("--sctp_event_csv", default="",
                   help="Validation only: event-level SCTP send/SACK/T3 diagnostics")
    p.add_argument("--diagnostic_ice", action="store_true",
                   help="Validation only: timestamp aioice STUN consent traffic")
    p.add_argument("--ice_consent_timeout_s", type=float, default=None,
                   help="Validation only: aioice consent timeout applied after ICE completes")
    p.add_argument("--validation_sctp_gap_rtt_fix", action="store_true",
                   help="Validation only: ignore ambiguous RTT samples from already gap-ACKed SCTP data")
    p.add_argument(
        "--adaptation_mode",
        choices=[mode.value for mode in AdaptationMode],
        default=AdaptationMode.LEGACY.value,
        help="Quality policy; legacy preserves the existing ReVo send path",
    )
    p.add_argument("--controller_csv", default="",
                   help="Track B controller-decision CSV")
    p.add_argument("--capacity_probe_csv", default="",
                   help="Track B bounded active-probe event CSV")
    p.add_argument("--track_b_high_qp", type=int, default=20)
    p.add_argument("--track_b_low_qp", type=int, default=30)
    p.add_argument("--track_b_buffer_soft_bytes", type=int, default=64 * 1024)
    p.add_argument("--track_b_buffer_hard_bytes", type=int, default=128 * 1024)
    p.add_argument("--track_b_buffer_growth_bytes", type=int, default=8 * 1024)
    p.add_argument("--track_b_impairment_threshold", type=float, default=0.05)
    p.add_argument("--track_b_severe_impairment_threshold", type=float, default=0.20)
    p.add_argument("--track_b_capacity_safety_margin", type=float, default=0.85)
    p.add_argument("--track_b_upgrade_headroom_fraction", type=float, default=0.15)
    p.add_argument("--track_b_downgrade_confirmations", type=int, default=1)
    p.add_argument("--track_b_upgrade_confirmations", type=int, default=3)
    p.add_argument("--track_b_min_dwell_gops", type=int, default=2)
    p.add_argument("--track_b_demand_alpha", type=float, default=0.2)
    p.add_argument(
        "--track_b_encode_lead_alpha",
        type=float,
        default=0.2,
        help="Causal EWMA weight for completed dual-candidate encode duration",
    )
    p.add_argument(
        "--track_b_passive_service_floor_fraction",
        type=float,
        default=0.80,
        help="Fresh passive service below this fraction of measured offered load triggers downgrade",
    )
    p.add_argument("--track_b_probe_payload_bytes", type=int, default=1024)
    p.add_argument("--track_b_probe_min_duration_s", type=float, default=0.40)
    p.add_argument("--track_b_probe_max_duration_s", type=float, default=0.60)
    p.add_argument("--track_b_probe_max_total_bytes", type=int, default=512 * 1024)
    p.add_argument(
        "--track_b_probe_max_additional_rate_mbps", type=float, default=20.0
    )
    p.add_argument("--track_b_probe_max_rate_multiplier", type=float, default=8.0)
    p.add_argument(
        "--track_b_probe_min_confirmed_fraction", type=float, default=0.90
    )
    p.add_argument("--track_b_probe_ack_timeout_s", type=float, default=0.25)
    p.add_argument("--track_b_probe_cooldown_s", type=float, default=2.0)
    p.add_argument(
        "--track_b_probe_authorization_ttl_s", type=float, default=1.5
    )
    p.add_argument(
        "--track_b_probe_buffer_abort_bytes", type=int, default=128 * 1024
    )
    p.add_argument(
        "--track_b_probe_buffer_growth_abort_bytes",
        type=int,
        default=8 * 1024,
    )
    p.add_argument(
        "--track_b_probe_impairment_abort_threshold",
        type=float,
        default=0.05,
    )
    p.add_argument(
        "--track_b_probe_boundary_guard_s", type=float, default=0.08
    )
    p.add_argument(
        "--track_b_encoder_pool_threads",
        type=int,
        default=0,
        help="x265 worker pool per adaptive encoder; 0 preserves x265 default",
    )
    p.add_argument("--track_b_encoder_process", action="store_true")
    p.add_argument("--track_b_encoder_max_inflight", type=int, default=2)
    p.add_argument("--track_b_encoder_completed_lead", type=int, default=5)
    p.add_argument("--track_b_sender_reserved_cpus", type=int, default=2)
    p.add_argument(
        "--diagnostic_preencoded_manifest",
        default="",
        help="Diagnostic only: replay retained Track B candidates without an encoder",
    )
    p.add_argument(
        "--diagnostic_continue_on_candidate_miss",
        action="store_true",
        help="Diagnostic only: record and skip late candidates instead of failing",
    )
    p.add_argument(
        "--candidate_diagnostic_csv",
        default="",
        help="Diagnostic-only per-frame candidate/send/transport CSV",
    )
    p.add_argument(
        "--track_b_fps",
        type=int,
        default=0,
        help="adaptive-mode stream FPS; 0 reads the source rate",
    )
    p.add_argument(
        "--receiver_health_freshness_s",
        type=float,
        default=2.5,
        help="Maximum age of receiver-health feedback used by Track B",
    )
    p.add_argument(
        "--track_b_shared_monotonic_clock",
        action="store_true",
        help="Validation only: log one-way health delay when peers share CLOCK_MONOTONIC",
    )
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
    p.add_argument("--validation_stop_signal_path", default="",
                   help="Validation only: stop between frames after an event gate completes")

    args = p.parse_args()
    s    = Sender(args)
    asyncio.run(s.run())
