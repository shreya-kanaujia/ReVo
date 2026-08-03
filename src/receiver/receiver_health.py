"""Thread-safe, display-clock-driven receiver impairment accounting."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

from receiver_health_feedback import ReceiverHealthFeedback


def lacks_required_reference(
    *, is_keyframe: bool, gop_id: int, last_decoded_keyframe_id
) -> bool:
    """True only when a P-frame has no matching successfully decoded keyframe."""
    return (
        not bool(is_keyframe)
        and (
            last_decoded_keyframe_id is None
            or int(gop_id) != int(last_decoded_keyframe_id)
        )
    )


@dataclass
class _FrameHealth:
    full_miss: bool = False
    partial: bool = False
    decode_failure: bool = False
    reference_unavailable: bool = False
    frozen: bool = False
    finalized: bool = False


class ReceiverHealthTracker:
    """Produces non-overlapping health windows independent of decode success."""

    def __init__(self, window_frames: int = 30):
        if window_frames <= 0:
            raise ValueError("window_frames must be positive")
        self.window_frames = int(window_frames)
        self._lock = threading.Lock()
        self._frames: dict[int, _FrameHealth] = {}
        self._next_window_start = 0
        self._feedback_seq = 0

    def record(
        self,
        frame_id: int,
        *,
        full_miss: bool = False,
        partial: bool = False,
        decode_failure: bool = False,
        reference_unavailable: bool = False,
        frozen: bool = False,
        finalized: bool = False,
    ) -> None:
        frame_id = int(frame_id)
        if frame_id < 0:
            return
        with self._lock:
            item = self._frames.setdefault(frame_id, _FrameHealth())
            item.full_miss |= bool(full_miss)
            item.partial |= bool(partial)
            item.decode_failure |= bool(decode_failure)
            item.reference_unavailable |= bool(reference_unavailable)
            item.frozen |= bool(frozen)
            item.finalized |= bool(finalized)

    def complete_display_frame(
        self, frame_id: int, receiver_ts: float, *, frozen: bool
    ) -> Optional[ReceiverHealthFeedback]:
        """Record display outcome and emit when a full display-clock window ends."""
        self.record(frame_id, frozen=frozen)
        with self._lock:
            window_end = self._next_window_start + self.window_frames - 1
            if frame_id < window_end:
                return None
            if not all(
                self._frames.get(fid, _FrameHealth()).finalized
                for fid in range(self._next_window_start, window_end + 1)
            ):
                return None
            rows = [
                self._frames.pop(fid, _FrameHealth())
                for fid in range(self._next_window_start, window_end + 1)
            ]
            full = sum(row.full_miss for row in rows)
            partial = sum(row.partial for row in rows)
            decode = sum(row.decode_failure for row in rows)
            reference = sum(row.reference_unavailable for row in rows)
            frozen_count = sum(row.frozen for row in rows)
            total = len(rows)
            # Component counts intentionally overlap; the aggregate is the
            # fraction of frames with at least one observed impairment.
            impaired = sum(
                row.full_miss
                or row.partial
                or row.decode_failure
                or row.reference_unavailable
                or row.frozen
                for row in rows
            )
            rate = impaired / total
            self._feedback_seq += 1
            result = ReceiverHealthFeedback(
                feedback_seq=self._feedback_seq,
                receiver_ts=float(receiver_ts),
                window_start_frame=self._next_window_start,
                window_end_frame=window_end,
                total_frames=total,
                full_misses=full,
                partial_frames=partial,
                decode_failures=decode,
                reference_unavailable=reference,
                frozen_frames=frozen_count,
                impairment_rate=rate,
            )
            self._next_window_start = window_end + 1
            return result
