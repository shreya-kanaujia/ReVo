"""Receiver-side EWMA estimator for application-message arrival spacing.

This passive Week 1 helper observes application-message arrival times and
sender idle boundaries, but it does not make send/drop/quality decisions.
"""


class ArrivalCapacityEstimator:
    def __init__(self, alpha=0.1, min_corrected_interval=1e-6):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.min_corrected_interval = float(min_corrected_interval)
        self.last_receiver_ts = None
        self.last_sequence_id = None
        self.ewma_service_time_per_byte = None

    def observe(self, receiver_ts, sender_grace_period=0.0, sequence_id=None,
                packet_size_bytes=1):
        receiver_ts = float(receiver_ts)
        sender_grace_period = max(0.0, float(sender_grace_period))
        sequence_id = None if sequence_id is None else int(sequence_id)
        packet_size_bytes = int(packet_size_bytes)
        if packet_size_bytes <= 0:
            raise ValueError("packet_size_bytes must be positive")

        def effective_interarrival():
            if self.ewma_service_time_per_byte is None:
                return None
            return self.ewma_service_time_per_byte * packet_size_bytes

        if self.last_receiver_ts is None:
            self.last_receiver_ts = receiver_ts
            self.last_sequence_id = sequence_id
            return {
                "raw_interarrival": None,
                "sender_grace_period": sender_grace_period,
                "corrected_interarrival": None,
                "ewma_interarrival": effective_interarrival(),
            }

        raw = max(0.0, receiver_ts - self.last_receiver_ts)
        contiguous = (
            sequence_id is None
            or self.last_sequence_id is None
            or sequence_id == self.last_sequence_id + 1
        )
        self.last_receiver_ts = receiver_ts
        self.last_sequence_id = sequence_id

        # Receiver dispersion measures packet service time only for adjacent
        # packets in a continuously offered train.  A sender-created idle gap
        # is not service time, but subtracting the whole sender inter-send
        # interval also subtracts real serialization time.  Exclude that
        # boundary sample instead.  Likewise, a sequence gap spans an unknown
        # number of packets and cannot provide a one-packet service interval.
        if sender_grace_period > 0.0 or not contiguous:
            return {
                "raw_interarrival": raw,
                "sender_grace_period": sender_grace_period,
                "corrected_interarrival": None,
                "ewma_interarrival": effective_interarrival(),
            }

        corrected = max(self.min_corrected_interval, raw)
        service_time_per_byte = corrected / packet_size_bytes
        if self.ewma_service_time_per_byte is None:
            self.ewma_service_time_per_byte = service_time_per_byte
        else:
            self.ewma_service_time_per_byte = (
                self.alpha * service_time_per_byte
                + (1.0 - self.alpha) * self.ewma_service_time_per_byte
            )
        return {
            "raw_interarrival": raw,
            "sender_grace_period": sender_grace_period,
            "corrected_interarrival": corrected,
            # Express the size-normalized EWMA as the effective service time
            # for this message so the existing feedback wire field remains
            # backward-compatible.  packet_size / value is the EWMA byte rate.
            "ewma_interarrival": effective_interarrival(),
        }
