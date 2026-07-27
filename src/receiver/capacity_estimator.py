"""Receiver-side estimator for delivered application-message throughput.

The size-normalized EWMA remains the raw, auditable service-rate estimate.  A
separate causal publication path adds a short time-window median, freshness,
and recovery state for consumers that must not treat an old estimate as live.
Neither path uses configured link rates or trace data.
"""

from collections import deque
from statistics import median


class ArrivalCapacityEstimator:
    def __init__(
        self,
        alpha=0.1,
        min_corrected_interval=1e-6,
        robust_window_s=0.1,
        freshness_multiplier=10.0,
        freshness_min_s=0.05,
        freshness_max_s=1.0,
        recovery_samples=5,
        recovery_window_s=None,
    ):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        if float(min_corrected_interval) <= 0.0:
            raise ValueError("min_corrected_interval must be positive")
        if float(robust_window_s) <= 0.0:
            raise ValueError("robust_window_s must be positive")
        if float(freshness_multiplier) <= 0.0:
            raise ValueError("freshness_multiplier must be positive")
        if float(freshness_min_s) <= 0.0:
            raise ValueError("freshness_min_s must be positive")
        if float(freshness_max_s) < float(freshness_min_s):
            raise ValueError("freshness_max_s must be >= freshness_min_s")
        if int(recovery_samples) < 1:
            raise ValueError("recovery_samples must be positive")
        if recovery_window_s is not None and float(recovery_window_s) < 0.0:
            raise ValueError("recovery_window_s must be nonnegative")

        self.alpha = float(alpha)
        self.min_corrected_interval = float(min_corrected_interval)
        self.robust_window_s = float(robust_window_s)
        self.freshness_multiplier = float(freshness_multiplier)
        self.freshness_min_s = float(freshness_min_s)
        self.freshness_max_s = float(freshness_max_s)
        self.recovery_samples = int(recovery_samples)
        self.recovery_window_s = (
            self.robust_window_s
            if recovery_window_s is None
            else float(recovery_window_s)
        )

        self.last_receiver_ts = None
        self.last_sequence_id = None
        self.ewma_service_time_per_byte = None

        self.filtered_service_time_per_byte = None
        self._service_samples = deque()
        self.last_valid_update_ts = None
        self._valid_update_spacing_ewma = None
        self._previous_observation_valid = False
        self._recovering = False
        self._recovery_count = 0
        self._recovery_start_ts = None

    def _effective_interval(self, service_time_per_byte, packet_size_bytes):
        if service_time_per_byte is None:
            return None
        return service_time_per_byte * packet_size_bytes

    def _freshness_threshold(self):
        cadence = self._valid_update_spacing_ewma
        if cadence is None:
            return self.freshness_min_s
        return min(
            self.freshness_max_s,
            max(self.freshness_min_s, self.freshness_multiplier * cadence),
        )

    def snapshot(self, receiver_ts, packet_size_bytes=1):
        """Return current publication state without changing the estimator."""
        receiver_ts = float(receiver_ts)
        packet_size_bytes = int(packet_size_bytes)
        if packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")

        age = (
            None
            if self.last_valid_update_ts is None
            else max(0.0, receiver_ts - self.last_valid_update_ts)
        )
        threshold = self._freshness_threshold()
        fresh = (
            age is not None
            and age <= threshold
            and not self._recovering
            and self.filtered_service_time_per_byte is not None
        )
        raw_interval = self._effective_interval(
            self.ewma_service_time_per_byte, packet_size_bytes
        )
        filtered_interval = self._effective_interval(
            self.filtered_service_time_per_byte, packet_size_bytes
        )
        return {
            "ewma_interarrival": raw_interval,
            "filtered_ewma_interarrival": filtered_interval,
            "last_valid_update_ts": self.last_valid_update_ts,
            "estimate_age_s": age,
            "freshness_threshold_s": threshold,
            "estimate_fresh": fresh,
            "published_ewma_interarrival": filtered_interval if fresh else None,
        }

    def _update_raw_ewma(self, service_time_per_byte):
        if self.ewma_service_time_per_byte is None:
            self.ewma_service_time_per_byte = service_time_per_byte
        else:
            self.ewma_service_time_per_byte = (
                self.alpha * service_time_per_byte
                + (1.0 - self.alpha) * self.ewma_service_time_per_byte
            )

    def _update_filtered_ewma(self, receiver_ts, service_time_per_byte):
        self._service_samples.append((receiver_ts, service_time_per_byte))
        cutoff = receiver_ts - self.robust_window_s
        while self._service_samples and self._service_samples[0][0] < cutoff:
            self._service_samples.popleft()
        robust_sample = median(value for _timestamp, value in self._service_samples)
        # This is intentionally the causal rolling median itself, rather than
        # a second EWMA.  It therefore cannot leave the range of valid samples
        # in the active window or retain samples which have aged out.
        self.filtered_service_time_per_byte = robust_sample
        return robust_sample

    def _begin_recovery(self):
        self._recovering = True
        self._recovery_count = 0
        self._recovery_start_ts = None
        self._service_samples.clear()
        self.filtered_service_time_per_byte = None
        self._previous_observation_valid = False

    def observe(
        self,
        receiver_ts,
        sender_grace_period=0.0,
        sequence_id=None,
        packet_size_bytes=1,
    ):
        receiver_ts = float(receiver_ts)
        sender_grace_period = max(0.0, float(sender_grace_period))
        sequence_id = None if sequence_id is None else int(sequence_id)
        packet_size_bytes = int(packet_size_bytes)
        if packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")

        if self.last_receiver_ts is None:
            self.last_receiver_ts = receiver_ts
            self.last_sequence_id = sequence_id
            state = self.snapshot(receiver_ts, packet_size_bytes)
            return {
                "raw_interarrival": None,
                "sender_grace_period": sender_grace_period,
                "corrected_interarrival": None,
                "skip_reason": "initial_sample",
                "filter_reason": "no_raw_estimate",
                **state,
            }

        raw = max(0.0, receiver_ts - self.last_receiver_ts)
        contiguous = (
            sequence_id is None
            or self.last_sequence_id is None
            or sequence_id == self.last_sequence_id + 1
        )
        state_before = self.snapshot(receiver_ts, packet_size_bytes)
        timed_out = (
            raw > state_before["freshness_threshold_s"]
        )
        self.last_receiver_ts = receiver_ts
        self.last_sequence_id = sequence_id

        # A sender-created idle boundary and a sequence gap both span an
        # unknown service interval. Exclude that boundary and require a new
        # adjacent observation window before publishing again.
        if sender_grace_period > 0.0 or not contiguous:
            self._begin_recovery()
            state = self.snapshot(receiver_ts, packet_size_bytes)
            return {
                "raw_interarrival": raw,
                "sender_grace_period": sender_grace_period,
                "corrected_interarrival": None,
                "skip_reason": (
                    "sender_idle_boundary"
                    if sender_grace_period > 0.0
                    else "sequence_gap"
                ),
                "filter_reason": "no_valid_spacing_sample",
                **state,
            }

        if timed_out:
            # This interval spans a period in which no receiver service was
            # observed.  It is neither a one-message service sample nor a
            # valid seed for the recovered publication path.
            self._begin_recovery()
            state = self.snapshot(receiver_ts, packet_size_bytes)
            return {
                "raw_interarrival": raw,
                "sender_grace_period": sender_grace_period,
                "corrected_interarrival": None,
                "skip_reason": "stale_boundary",
                "filter_reason": "stale_boundary_excluded",
                **state,
            }

        corrected = max(self.min_corrected_interval, raw)
        service_time_per_byte = corrected / packet_size_bytes

        # Preserve the original EWMA update rule for every valid service sample.
        self._update_raw_ewma(service_time_per_byte)

        if self._previous_observation_valid:
            if self._valid_update_spacing_ewma is None:
                self._valid_update_spacing_ewma = raw
            else:
                self._valid_update_spacing_ewma = (
                    self.alpha * raw
                    + (1.0 - self.alpha) * self._valid_update_spacing_ewma
                )

        if self._recovering:
            self._service_samples.append((receiver_ts, service_time_per_byte))
            cutoff = receiver_ts - self.robust_window_s
            while self._service_samples and self._service_samples[0][0] < cutoff:
                self._service_samples.popleft()
            if self._recovery_start_ts is None:
                self._recovery_start_ts = receiver_ts
            self._recovery_count += 1
            recovery_duration = receiver_ts - self._recovery_start_ts
            if (
                self._recovery_count >= self.recovery_samples
                and recovery_duration + 1e-12 >= self.recovery_window_s
            ):
                self.filtered_service_time_per_byte = median(
                    value for _timestamp, value in self._service_samples
                )
                self._recovering = False
                self._recovery_start_ts = None
                filter_reason = "recovery_seeded"
            else:
                filter_reason = "recovery_buffering"
        else:
            robust_sample = self._update_filtered_ewma(
                receiver_ts, service_time_per_byte
            )
            filter_reason = "rolling_time_median"
            if robust_sample == service_time_per_byte:
                filter_reason = "sample_accepted"
        self.last_valid_update_ts = receiver_ts
        self._previous_observation_valid = True
        state = self.snapshot(receiver_ts, packet_size_bytes)
        return {
            "raw_interarrival": raw,
            "sender_grace_period": sender_grace_period,
            "corrected_interarrival": corrected,
            "skip_reason": "" if state["estimate_fresh"] else "recovering",
            "filter_reason": filter_reason,
            **state,
        }
