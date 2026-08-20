"""Mentor-provided GCC control logic, isolated from ReVo/ABR policy.

The receiver estimator works on differences of sender departure timestamps and
receiver arrival timestamps.  The fixed sender/receiver clock offset therefore
cancels; clocks need only be monotonic on their respective hosts.
"""

from collections import deque
import math
import time


class ReceiverGccEstimator:
    def __init__(self):
        self.seen = set()
        self.seen_order = deque()
        self.seen_max = 20000
        self.loss_seqs = []
        self.last_seq = -1
        self.frames = {}
        self.max_fid = -1
        self.last_processed_fid = -1
        self.prev_group = None
        self.rate_window = deque()
        self.m = 0.0
        self.gamma = 12.5
        self.kf_p = 1.0
        self.kf_q = 1e-3
        self.kf_sigma = 1.0
        self.ku = 0.01
        self.kd = 0.00018
        self.overuse_since = None
        self.state = "HOLD"
        self.rate_bps = 800_000.0
        self.min_rate = 150_000.0
        self.max_rate = 1_200_000.0
        self.eta = 1.05
        self.alpha = 0.85
        self.last_rate_update = time.perf_counter()

    def observe(self, *, seq, fid, sender_time, receiver_time, size):
        """Record one unique media packet; repeated proactive copies share seq."""
        if seq in self.seen:
            return False
        self.seen.add(seq)
        self.seen_order.append(seq)
        if len(self.seen_order) > self.seen_max:
            self.seen.discard(self.seen_order.popleft())
        self.loss_seqs.append(seq)
        self.max_fid = max(self.max_fid, fid)
        row = self.frames.setdefault(fid, {
            "t_send_first": sender_time,
            "t_recv_last": receiver_time,
        })
        row["t_send_first"] = min(row["t_send_first"], sender_time)
        row["t_recv_last"] = max(row["t_recv_last"], receiver_time)
        self.rate_window.append((receiver_time, size))
        return True

    def _loss(self, consume=True):
        if not self.loss_seqs:
            return 0.0
        seqs = sorted(set(self.loss_seqs))
        lo, hi = seqs[0], seqs[-1]
        expected = hi - lo + 1
        if self.last_seq >= 0 and lo > self.last_seq + 1:
            expected += lo - self.last_seq - 1
        loss = max(0.0, min(1.0,
                            (expected - len(seqs)) / max(1, expected)))
        # The 50 ms estimator tick peeks at loss for diagnostics, while mentor
        # GCC advances the sequence-window boundary only when REMB is emitted.
        # Mutating last_seq during a peek forgets a leading gap before the 1 Hz
        # report consumes it.
        if consume:
            self.loss_seqs.clear()
            self.last_seq = max(self.last_seq, hi)
        return loss

    def _signal(self, now):
        if self.m > self.gamma:
            if self.overuse_since is None:
                self.overuse_since = now
            return "overuse" if now - self.overuse_since > 0.1 else "normal"
        self.overuse_since = None
        return "underuse" if self.m < -self.gamma else "normal"

    def _transition(self, signal):
        if signal == "overuse":
            self.state = "DECREASE"
        elif signal == "underuse":
            self.state = "HOLD"
        elif self.state == "DECREASE":
            self.state = "HOLD"
        elif self.state == "HOLD":
            self.state = "INCREASE"

    def feedback(self, now=None, consume_loss=True):
        """Advance closed packet groups and return the current REMB payload."""
        now = time.perf_counter() if now is None else now
        while self.rate_window and now - self.rate_window[0][0] > 0.5:
            self.rate_window.popleft()
        span = max(0.1, now - self.rate_window[0][0]) if self.rate_window else 0.1
        receive_rate = sum(size for _, size in self.rate_window) * 8.0 / span

        closed = [fid for fid in sorted(self.frames)
                  if self.last_processed_fid < fid < self.max_fid]
        for fid in closed:
            timing = self.frames[fid]
            self.last_processed_fid = fid
            if self.prev_group is None:
                self.prev_group = (fid, timing["t_send_first"], timing["t_recv_last"])
                continue
            _, prev_send, prev_recv = self.prev_group
            arrival_delta = (timing["t_recv_last"] - prev_recv) * 1000.0
            send_delta = (timing["t_send_first"] - prev_send) * 1000.0
            raw_gradient = arrival_delta - send_delta
            self.prev_group = (fid, timing["t_send_first"], timing["t_recv_last"])
            innovation = raw_gradient - self.m
            self.kf_sigma = 0.95 * self.kf_sigma + 0.05 * innovation ** 2
            gain = ((self.kf_p + self.kf_q) /
                    (self.kf_p + self.kf_q + self.kf_sigma))
            self.m = (1.0 - gain) * self.m + gain * raw_gradient
            self.kf_p = (1.0 - gain) * (self.kf_p + self.kf_q)
            if abs(self.m) - self.gamma <= 15.0:
                k = self.kd if abs(self.m) < self.gamma else self.ku
                self.gamma += max(0.0, arrival_delta) * k * (abs(self.m) - self.gamma)
                self.gamma = max(6.0, min(600.0, self.gamma))
            self._transition(self._signal(timing["t_recv_last"]))

        dt = min(1.0, max(0.0, now - self.last_rate_update))
        if self.state == "INCREASE":
            grown = self.rate_bps * self.eta ** dt
            self.rate_bps = max(self.rate_bps,
                                min(grown, receive_rate * 1.5) if receive_rate else grown)
        elif self.state == "DECREASE":
            candidate = receive_rate * self.alpha if receive_rate else self.rate_bps * self.alpha
            self.rate_bps = max(self.min_rate, min(self.rate_bps, candidate))
        self.rate_bps = min(self.max_rate, self.rate_bps)
        self.last_rate_update = now
        for fid in list(self.frames):
            if fid < self.last_processed_fid - 50:
                del self.frames[fid]
        return {
            "type": "remb_feedback",
            "A_r": self.rate_bps,
            "loss_rate": self._loss(consume=consume_loss),
            "state": self.state,
            "delay_gradient_ms": self.m,
            "threshold_ms": self.gamma,
            "receive_rate_bps": receive_rate,
        }


class SenderGccController:
    def __init__(self):
        self.min_rate = 150_000.0
        self.max_rate = 1_200_000.0
        self.loss_rate_bps = 800_000.0
        self.remote_rate_bps = 800_000.0
        self.target_rate_bps = 800_000.0
        self.smoothed_rate_bps = None
        self.target_qp = 30
        self.alpha = 0.3
        self.last_qp_change = 0.0
        self.dwell_s = 2.0
        self.qp_hysteresis = 3
        self.rate_at_last_qp = None

    def qp_for_rate(self, rate_bps):
        clamped = max(self.min_rate, min(self.max_rate, rate_bps))
        ratio = math.log(clamped / self.min_rate) / math.log(self.max_rate / self.min_rate)
        return int(40 - ratio * 15)

    def update(self, feedback, now=None):
        now = time.perf_counter() if now is None else now
        self.remote_rate_bps = float(feedback["A_r"])
        loss = float(feedback.get("loss_rate", 0.0))
        if loss < 0.02:
            self.loss_rate_bps = min(self.max_rate, self.loss_rate_bps * 1.05)
        elif loss > 0.10:
            self.loss_rate_bps = max(self.min_rate,
                                     self.loss_rate_bps * (1.0 - 0.5 * loss))
        self.target_rate_bps = min(self.remote_rate_bps, self.loss_rate_bps)
        if self.smoothed_rate_bps is None:
            self.smoothed_rate_bps = self.target_rate_bps
        else:
            self.smoothed_rate_bps = ((1.0 - self.alpha) * self.smoothed_rate_bps
                                      + self.alpha * self.target_rate_bps)
        candidate = self.qp_for_rate(self.smoothed_rate_bps)
        emergency = (self.rate_at_last_qp is not None
                     and self.smoothed_rate_bps < 0.7 * self.rate_at_last_qp
                     and candidate > self.target_qp)
        changed = ((abs(candidate - self.target_qp) >= self.qp_hysteresis
                    and now - self.last_qp_change >= self.dwell_s) or emergency)
        if changed:
            self.target_qp = candidate
            self.last_qp_change = now
            self.rate_at_last_qp = self.smoothed_rate_bps
        return changed
