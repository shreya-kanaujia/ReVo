"""Opt-in WebRTC validation diagnostics; inactive during normal ReVo runs."""

import asyncio
import csv
import logging
import math
import os
import resource
import struct
import sys
import time
from collections import Counter


def classify_revo_message(data):
    """Best-effort wire classification for diagnostics; never affects delivery."""
    result = {
        "message_type": "unknown",
        "frame_id": "",
        "media_chunk_index": "",
        "media_sequence": "",
        "probe_id": "",
        "probe_sequence": "",
    }
    if not isinstance(data, bytes) or not data:
        return result
    result["message_type"] = f"type_{data[0]}"
    try:
        if data[0] == 1:
            result["message_type"] = "init"
        elif data[0] == 2 and len(data) >= struct.calcsize("<BBIIBHHHIQdd"):
            values = struct.unpack("<BBIIBHHHIQdd", data[:struct.calcsize("<BBIIBHHHIQdd")])
            result.update(
                message_type="rgb_media" if values[1] == 3 else "depth_media",
                frame_id=values[2],
                media_chunk_index=values[5],
                media_sequence=values[9],
            )
        elif data[0] == 7 and len(data) >= struct.calcsize("<BBQQBdI"):
            values = struct.unpack("<BBQQBdI", data[:struct.calcsize("<BBQQBdI")])
            result.update(
                message_type="capacity_probe",
                probe_id=values[2],
                probe_sequence=values[3],
            )
        elif data[0] == 8 and len(data) == struct.calcsize("<BBQQBdI"):
            values = struct.unpack("<BBQQBdI", data)
            result.update(
                message_type="capacity_probe_ack",
                probe_id=values[2],
                probe_sequence=values[3],
            )
        elif data[0] == 5:
            result["message_type"] = "validation_probe"
        elif data[0] == 6:
            result["message_type"] = "capacity_feedback"
        elif data[0] == 9:
            result["message_type"] = "receiver_health"
    except (struct.error, ValueError, IndexError):
        result["message_type"] += "_malformed"
    return result


class SctpEventDiagnostics:
    """Event-level, validation-only SCTP observability with no policy changes."""

    FIELDNAMES = (
        "timestamp_monotonic", "role", "event", "direction",
        "stream_id", "channel_label", "message_type", "message_bytes",
        "frame_id", "media_chunk_index", "media_sequence", "probe_id",
        "probe_sequence", "tsn", "tsn_start", "tsn_end", "fragment_flags",
        "fragment_bytes", "sent_count", "cumulative_tsn", "sack_gaps",
        "sent_queue", "outbound_queue", "data_channel_queue", "flight_size",
        "cwnd", "rto", "t3_active", "advanced_peer_ack_tsn",
    )

    def __init__(self, path, role):
        self.enabled = bool(path)
        self.role = role
        self._file = None
        self._writer = None
        self.last_event_loop_delay_s = 0.0
        self._metadata = {}
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._file = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()
            self._file.flush()

    def _state(self, sctp):
        return {
            "sent_queue": len(getattr(sctp, "_sent_queue", ())),
            "outbound_queue": len(getattr(sctp, "_outbound_queue", ())),
            "data_channel_queue": len(getattr(sctp, "_data_channel_queue", ())),
            "flight_size": getattr(sctp, "_flight_size", ""),
            "cwnd": getattr(sctp, "_cwnd", ""),
            "rto": getattr(sctp, "_rto", ""),
            "t3_active": int(bool(getattr(sctp, "_t3_handle", None))),
            "advanced_peer_ack_tsn": getattr(
                sctp, "_advanced_peer_ack_tsn", ""
            ),
        }

    def log(self, sctp, event, *, direction="", metadata=None, **values):
        if not self.enabled:
            return
        row = {name: "" for name in self.FIELDNAMES}
        row.update(self._state(sctp))
        row.update(metadata or {})
        row.update(values)
        row.update(
            timestamp_monotonic=f"{time.perf_counter():.9f}",
            role=self.role,
            event=event,
            direction=direction,
        )
        self._writer.writerow({name: row.get(name, "") for name in self.FIELDNAMES})
        self._file.flush()

    def instrument(self, peer_connection):
        if not self.enabled:
            return
        sctp = getattr(peer_connection, "sctp", None)
        if sctp is None:
            raise RuntimeError("SCTP transport does not exist yet")
        if getattr(sctp, "_revo_event_diagnostics_applied", False):
            return
        sctp._revo_event_diagnostics_applied = True

        original_send = sctp._send
        original_send_chunk = sctp._send_chunk
        original_sack = sctp._receive_sack_chunk
        original_t3 = sctp._t3_expired
        original_receive_data = sctp._receive_data_chunk
        original_receive = sctp._receive

        async def send(stream_id, pp_id, user_data, **kwargs):
            start = int(sctp._local_tsn)
            fragments = max(1, math.ceil(len(user_data) / 1200))
            metadata = classify_revo_message(user_data)
            metadata.update(
                stream_id=stream_id,
                channel_label=getattr(
                    getattr(sctp, "_data_channels", {}).get(stream_id),
                    "label", "",
                ),
                message_bytes=len(user_data),
                tsn_start=start,
                tsn_end=(start + fragments - 1) & 0xFFFFFFFF,
            )
            for offset in range(fragments):
                self._metadata[(start + offset) & 0xFFFFFFFF] = metadata.copy()
            self.log(sctp, "message_submit", direction="out", metadata=metadata)
            return await original_send(stream_id, pp_id, user_data, **kwargs)

        async def send_chunk(chunk):
            tsn = getattr(chunk, "tsn", None)
            metadata = self._metadata.get(tsn, {})
            self.log(
                sctp,
                "forward_tsn_send" if chunk.__class__.__name__ == "ForwardTsnChunk" else "data_send",
                direction="out",
                metadata=metadata,
                tsn=tsn if tsn is not None else "",
                cumulative_tsn=getattr(chunk, "cumulative_tsn", ""),
                fragment_flags=getattr(chunk, "flags", ""),
                fragment_bytes=len(getattr(chunk, "user_data", b"")),
                sent_count=getattr(chunk, "_sent_count", ""),
            )
            return await original_send_chunk(chunk)

        async def receive_sack(chunk):
            self.log(
                sctp, "sack_receive", direction="in",
                cumulative_tsn=getattr(chunk, "cumulative_tsn", ""),
                sack_gaps=repr(getattr(chunk, "gaps", ())),
            )
            return await original_sack(chunk)

        def t3_expired():
            self.log(sctp, "t3_expire", direction="local")
            return original_t3()

        async def receive_data(chunk):
            metadata = classify_revo_message(getattr(chunk, "user_data", b""))
            self.log(
                sctp, "data_receive", direction="in", metadata=metadata,
                stream_id=getattr(chunk, "stream_id", ""),
                tsn=getattr(chunk, "tsn", ""),
                fragment_flags=getattr(chunk, "flags", ""),
                fragment_bytes=len(getattr(chunk, "user_data", b"")),
            )
            return await original_receive_data(chunk)

        async def receive(stream_id, pp_id, data):
            metadata = classify_revo_message(data)
            self.log(
                sctp, "message_deliver", direction="in", metadata=metadata,
                stream_id=stream_id, message_bytes=len(data),
            )
            return await original_receive(stream_id, pp_id, data)

        sctp._send = send
        sctp._send_chunk = send_chunk
        sctp._receive_sack_chunk = receive_sack
        sctp._t3_expired = t3_expired
        sctp._receive_data_chunk = receive_data
        sctp._receive = receive
        self.log(sctp, "instrumented", direction="local")

    def close(self):
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None


class _MonotonicLogFilter(logging.Filter):
    def filter(self, record):
        record.monotonic = time.perf_counter()
        return True


def enable_ice_debug_logging():
    """Timestamp aioice STUN requests/responses without enabling app debug spam."""
    formatter = logging.Formatter(
        "%(monotonic).9f %(levelname)s:%(name)s:%(message)s"
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_MonotonicLogFilter())
        handler.setFormatter(formatter)
    logging.getLogger("aioice.ice").setLevel(logging.DEBUG)


def apply_ice_consent_timeout(timeout_s, role):
    """Apply an explicit validation-only consent timeout after ICE completes."""
    if timeout_s is None:
        return
    import aioice.stun

    aioice.stun.RETRY_RTO = float(timeout_s)
    logging.warning(
        "[%s] Applied validation-only ICE consent timeout: %.3f s",
        role,
        timeout_s,
    )


def sender_grace_for_backpressure_pause(
    inter_send_interval_s, buffered_amount, pause_duration_s
):
    """Mark an idle boundary only if the transport queue actually drained."""
    if (
        float(pause_duration_s) > 0.0
        and int(buffered_amount) == 0
        and inter_send_interval_s is not None
    ):
        return max(0.0, float(inter_send_interval_s))
    return 0.0


def apply_sctp_gap_rtt_fix(peer_connection, role):
    """
    Avoid treating an already gap-ACKed DATA chunk as a new RTT sample.

    aiortc 1.14 updates its RTO when that chunk is later removed by a
    cumulative SACK. Under partial reliability, the delay can include one or
    more FORWARD-TSN timeout cycles and is therefore not an RTT measurement.
    The resulting inflated RTO can leave the DataChannel queue blocked for
    tens of seconds. This opt-in validation workaround suppresses only that
    ambiguous sample; unambiguous cumulative ACK samples are unchanged.
    """
    sctp = getattr(peer_connection, "sctp", None)
    if sctp is None:
        raise RuntimeError("SCTP transport does not exist yet")
    if getattr(sctp, "_revo_gap_rtt_fix_applied", False):
        return

    from aiortc.rtcsctptransport import uint32_gte

    original_receive_sack = sctp._receive_sack_chunk
    original_update_rto = sctp._update_rto
    sctp._revo_gap_rtt_fix_applied = True
    sctp._revo_suppress_rtt_sample = False
    sctp._revo_suppressed_rtt_samples = 0
    sctp._revo_last_suppressed_rtt_s = None

    async def receive_sack(chunk):
        sent_queue = getattr(sctp, "_sent_queue", ())
        first = sent_queue[0] if sent_queue else None
        sctp._revo_suppress_rtt_sample = bool(
            first is not None
            and getattr(first, "_acked", False)
            and getattr(first, "_sent_count", 0) == 1
            and uint32_gte(chunk.cumulative_tsn, first.tsn)
        )
        try:
            await original_receive_sack(chunk)
        finally:
            sctp._revo_suppress_rtt_sample = False

    def update_rto(sample_s):
        if sctp._revo_suppress_rtt_sample:
            sctp._revo_suppressed_rtt_samples += 1
            sctp._revo_last_suppressed_rtt_s = float(sample_s)
            return
        original_update_rto(sample_s)

    sctp._receive_sack_chunk = receive_sack
    sctp._update_rto = update_rto
    logging.warning(
        "[%s] Applied validation-only aiortc gap-ACK RTT workaround", role
    )


def sctp_diagnostic_snapshot(peer_connection):
    """Return a read-only snapshot of aiortc SCTP queues and timers."""
    sctp = getattr(peer_connection, "sctp", None)
    if sctp is None:
        return {}

    sent_queue = list(getattr(sctp, "_sent_queue", ()))
    outbound_queue = list(getattr(sctp, "_outbound_queue", ()))
    data_channel_queue = list(getattr(sctp, "_data_channel_queue", ()))
    oldest = sent_queue[0] if sent_queue else None
    t3_handle = getattr(sctp, "_t3_handle", None)
    try:
        t3_due_s = max(0.0, t3_handle.when() - asyncio.get_running_loop().time())
    except (AttributeError, RuntimeError):
        t3_due_s = None

    sent_states = Counter(
        "abandoned" if getattr(chunk, "_abandoned", False)
        else "gap_acked" if getattr(chunk, "_acked", False)
        else "retransmit" if getattr(chunk, "_retransmit", False)
        else "outstanding"
        for chunk in sent_queue
    )
    return {
        "sctp_state": str(getattr(sctp, "state", "")),
        "sctp_rto_s": getattr(sctp, "_rto", None),
        "sctp_srtt_s": getattr(sctp, "_srtt", None),
        "sctp_rttvar_s": getattr(sctp, "_rttvar", None),
        "sctp_cwnd_bytes": getattr(sctp, "_cwnd", None),
        "sctp_flight_size_bytes": getattr(sctp, "_flight_size", None),
        "sctp_data_channel_queue": len(data_channel_queue),
        "sctp_outbound_queue": len(outbound_queue),
        "sctp_sent_queue": len(sent_queue),
        "sctp_sent_outstanding": sent_states["outstanding"],
        "sctp_sent_gap_acked": sent_states["gap_acked"],
        "sctp_sent_abandoned": sent_states["abandoned"],
        "sctp_sent_retransmit": sent_states["retransmit"],
        "sctp_t3_active": bool(t3_handle),
        "sctp_t3_due_s": t3_due_s,
        "sctp_forward_tsn_pending": bool(
            getattr(sctp, "_forward_tsn_chunk", None)
        ),
        "sctp_last_sacked_tsn": getattr(sctp, "_last_sacked_tsn", None),
        "sctp_advanced_peer_ack_tsn": getattr(
            sctp, "_advanced_peer_ack_tsn", None
        ),
        "sctp_oldest_tsn": getattr(oldest, "tsn", None),
        "sctp_oldest_sent_count": getattr(oldest, "_sent_count", None),
        "sctp_oldest_misses": getattr(oldest, "_misses", None),
        "sctp_oldest_gap_acked": (
            bool(getattr(oldest, "_acked", False)) if oldest else ""
        ),
        "sctp_oldest_abandoned": (
            bool(getattr(oldest, "_abandoned", False)) if oldest else ""
        ),
        "sctp_gap_rtt_fix_applied": bool(
            getattr(sctp, "_revo_gap_rtt_fix_applied", False)
        ),
        "sctp_suppressed_rtt_samples": getattr(
            sctp, "_revo_suppressed_rtt_samples", 0
        ),
        "sctp_last_suppressed_rtt_s": getattr(
            sctp, "_revo_last_suppressed_rtt_s", None
        ),
    }


def process_resource_snapshot():
    """Portable process CPU time and peak RSS for experiment accounting."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss = int(usage.ru_maxrss)
    if sys.platform != "darwin":
        rss *= 1024
    return {
        "process_cpu_time_s": time.process_time(),
        "process_max_rss_bytes": rss,
        "process_voluntary_context_switches": int(usage.ru_nvcsw),
        "process_involuntary_context_switches": int(usage.ru_nivcsw),
    }


class WebRTCDiagnostics:
    FIELDNAMES = [
        "timestamp_monotonic",
        "role",
        "event_loop_delay_s",
        "feedback_events_total",
        "feedback_events_per_s",
        "feedback_delay_mean_s",
        "feedback_delay_max_s",
        "rgb_buffered_amount",
        "depth_buffered_amount",
        "outstanding_packets",
        "last_sent_sequence",
        "last_feedback_sequence",
        "connection_state",
        "ice_state",
        "raw_estimated_capacity_mbps",
        "filtered_estimated_capacity_mbps",
        "published_estimated_capacity_mbps",
        "estimate_age_s",
        "estimate_fresh",
        "freshness_threshold_s",
        "last_valid_update_ts",
        "process_cpu_time_s",
        "process_max_rss_bytes",
        "process_voluntary_context_switches",
        "process_involuntary_context_switches",
        "sctp_state",
        "sctp_rto_s",
        "sctp_srtt_s",
        "sctp_rttvar_s",
        "sctp_cwnd_bytes",
        "sctp_flight_size_bytes",
        "sctp_data_channel_queue",
        "sctp_outbound_queue",
        "sctp_sent_queue",
        "sctp_sent_outstanding",
        "sctp_sent_gap_acked",
        "sctp_sent_abandoned",
        "sctp_sent_retransmit",
        "sctp_t3_active",
        "sctp_t3_due_s",
        "sctp_forward_tsn_pending",
        "sctp_last_sacked_tsn",
        "sctp_advanced_peer_ack_tsn",
        "sctp_oldest_tsn",
        "sctp_oldest_sent_count",
        "sctp_oldest_misses",
        "sctp_oldest_gap_acked",
        "sctp_oldest_abandoned",
        "sctp_gap_rtt_fix_applied",
        "sctp_suppressed_rtt_samples",
        "sctp_last_suppressed_rtt_s",
    ]

    def __init__(self, path, role):
        self.enabled = bool(path)
        self.role = role
        self.feedback_total = 0
        self.feedback_interval = 0
        self.feedback_delays = []
        self.send_times = {}
        self.last_feedback_sequence = 0
        self._task = None
        self._stop = False
        self._file = None
        self._writer = None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self._file = open(path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDNAMES)
            self._writer.writeheader()
            self._file.flush()

    def packet_sent(self, seq_id, timestamp):
        if self.enabled:
            self.send_times[int(seq_id)] = float(timestamp)

    def feedback_event(self, seq_id, timestamp=None):
        if not self.enabled:
            return
        now = time.perf_counter() if timestamp is None else float(timestamp)
        seq_id = int(seq_id)
        self.feedback_total += 1
        self.feedback_interval += 1
        self.last_feedback_sequence = max(self.last_feedback_sequence, seq_id)
        sent = self.send_times.pop(seq_id, None)
        if sent is not None:
            self.feedback_delays.append(max(0.0, now - sent))

    def start(self, snapshot):
        if self.enabled and self._task is None:
            self._task = asyncio.create_task(self._monitor(snapshot))

    async def _monitor(self, snapshot):
        loop = asyncio.get_running_loop()
        previous = time.perf_counter()
        while not self._stop:
            deadline = loop.time() + 1.0
            await asyncio.sleep(1.0)
            now = time.perf_counter()
            elapsed = max(1e-9, now - previous)
            previous = now
            state = snapshot()
            delays = self.feedback_delays
            mean_delay = sum(delays) / len(delays) if delays else None
            max_delay = max(delays) if delays else None
            row = {
                "timestamp_monotonic": f"{now:.9f}",
                "role": self.role,
                "event_loop_delay_s": f"{max(0.0, loop.time() - deadline):.9f}",
                "feedback_events_total": self.feedback_total,
                "feedback_events_per_s": f"{self.feedback_interval / elapsed:.6f}",
                "feedback_delay_mean_s": "" if mean_delay is None else f"{mean_delay:.9f}",
                "feedback_delay_max_s": "" if max_delay is None else f"{max_delay:.9f}",
                "last_feedback_sequence": self.last_feedback_sequence,
            }
            self.last_event_loop_delay_s = max(0.0, loop.time() - deadline)
            row["event_loop_delay_s"] = f"{self.last_event_loop_delay_s:.9f}"
            for field in self.FIELDNAMES:
                if field not in row:
                    value = state.get(field, "")
                    if isinstance(value, float) and not math.isfinite(value):
                        value = ""
                    row[field] = value
            self._writer.writerow(row)
            self._file.flush()
            self.feedback_interval = 0
            self.feedback_delays = []

    async def stop(self):
        if not self.enabled:
            return
        self._stop = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        if self._file is not None:
            self._file.close()
            self._file = None
