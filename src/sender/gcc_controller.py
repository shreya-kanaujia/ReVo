import time
from collections import deque


class GCCController:
    """Sender-side GCC controller from the 2016 GCC paper.

    The sender keeps a loss-based estimate As. The receiver supplies Ar, the
    delay-based estimate. The effective target is min(Ar, As).
    """
    def __init__(self, initial_rate_bps, min_rate_bps=100_000, max_rate_bps=50_000_000):
        self.initial_rate_bps = float(initial_rate_bps)
        self.as_bps = float(initial_rate_bps)
        self.ar_bps = float(initial_rate_bps)
        self.target_rate_bps = float(initial_rate_bps)
        self.min_rate_bps = float(min_rate_bps)
        self.max_rate_bps = float(max_rate_bps)
        self._last_as_update = None
        # Loss is averaged over this window before driving As. Individual 50 ms
        # reports are far too noisy on a bursty link: most windows read exactly
        # 0.0 even when the mean loss is 7.6%.
        self._loss_hist = deque()
        self._loss_window_s = 1.0

    def update(self, ar_bps=None, loss_fraction=0.0, now=None):
        now = time.monotonic() if now is None else float(now)
        if ar_bps is not None and ar_bps > 0:
            self.ar_bps = self._clamp(float(ar_bps))

        f = max(0.0, min(1.0, float(loss_fraction)))
        self._loss_hist.append((now, f))
        while self._loss_hist and now - self._loss_hist[0][0] > self._loss_window_s:
            self._loss_hist.popleft()
        f_avg = (sum(x for _, x in self._loss_hist) / len(self._loss_hist)
                 if self._loss_hist else f)

        # Time-scale the adaptation. The paper's +5% / -0.5f steps are per
        # UPDATE INTERVAL (~1 s), but feedback arrives every 50 ms here, so
        # applying them per message compounds to 1.05^20 ~ 2.65x per second and
        # As pegs at the ceiling within seconds -- measured at 36-45 Mbps on a
        # 2.5 Mbps link, which makes the loss-based half of GCC inert.
        if self._last_as_update is None:
            dt = 0.0
        else:
            dt = max(0.0, min(1.0, now - self._last_as_update))
        self._last_as_update = now

        if dt > 0.0:
            if f_avg > 0.10:
                self.as_bps *= (1.0 - 0.5 * f_avg) ** dt
            elif f_avg < 0.02:
                self.as_bps *= 1.05 ** dt

        self.as_bps = self._clamp(self.as_bps)
        self.target_rate_bps = self._clamp(min(self.ar_bps, self.as_bps))
        return self.target_rate_bps

    def get_target_rate_bps(self):
        return self.target_rate_bps

    def get_target_rate_mbps(self):
        return self.target_rate_bps / 1e6

    @property
    def As(self):
        return self.as_bps

    @property
    def Ar(self):
        return self.ar_bps

    def _clamp(self, value):
        return max(self.min_rate_bps, min(self.max_rate_bps, value))


class GCCReceiverController:
    """Receiver-side delay-based GCC controller described in the paper.

    Measurement:
        d_m = (arrival_i-arrival_prev) - (send_i-send_prev)

    A scalar Kalman filter estimates the delay gradient. An adaptive threshold
    drives the overuse detector. Ar is updated from the 500-ms receive rate.
    """
    def __init__(self, initial_rate_bps=8_000_000, min_rate_bps=100_000, max_rate_bps=50_000_000):
        self.ar_bps = float(initial_rate_bps)
        self.min_rate_bps = float(min_rate_bps)
        self.max_rate_bps = float(max_rate_bps)

        # Paper parameters
        self.Q = 1e-3
        self.P = 1e-1
        self.m = 0.0
        self.sigma2 = 1e-3
        self.beta = 0.95
        self.gamma = 0.0125       # 12.5 ms initial threshold
        self.ku = 0.01
        self.kd = 0.00018
        self.overuse_time = 0.0
        self.state = 'NORMAL'

        self.prev_send_ts = None
        self.prev_arrival_ts = None
        self.prev_frame_id = None

        self.rx_samples = deque()
        self.last_rate_update = None

    def note_received_bytes(self, nbytes, now=None):
        now = time.monotonic() if now is None else float(now)
        self.rx_samples.append((now, int(nbytes)))
        cutoff = now - 0.5
        while self.rx_samples and self.rx_samples[0][0] < cutoff:
            self.rx_samples.popleft()

    def receive_rate_bps(self, now=None):
        now = time.monotonic() if now is None else float(now)
        cutoff = now - 0.5
        while self.rx_samples and self.rx_samples[0][0] < cutoff:
            self.rx_samples.popleft()
        if not self.rx_samples:
            return 0.0
        return sum(n for _, n in self.rx_samples) * 8.0 / 0.5

    def note_frame(self, frame_id, send_ts, arrival_ts):
        """Feed one frame's first-send/last-arrival timestamps to GCC."""
        if send_ts is None or arrival_ts is None:
            return None
        if self.prev_send_ts is None:
            self.prev_send_ts = float(send_ts)
            self.prev_arrival_ts = float(arrival_ts)
            self.prev_frame_id = int(frame_id)
            return None

        # Ignore late/out-of-order frames for the gradient; use monotonically
        # increasing frame IDs so retransmission/order artifacts cannot invert dt.
        if int(frame_id) <= int(self.prev_frame_id):
            return None

        dm = (float(arrival_ts) - self.prev_arrival_ts) - (float(send_ts) - self.prev_send_ts)
        dt = max(1e-3, float(arrival_ts) - self.prev_arrival_ts)

        # Scalar Kalman filter. dm is represented in seconds internally.
        dm_s = dm
        residual = dm_s - self.m
        self.sigma2 = self.beta * self.sigma2 + (1.0 - self.beta) * residual * residual
        K = (self.P + self.Q) / (self.P + self.Q + max(self.sigma2, 1e-9))
        self.m = (1.0 - K) * self.m + K * dm_s
        self.P = (1.0 - K) * (self.P + self.Q)

        # Adaptive threshold from the paper.
        k = self.kd if abs(self.m) < self.gamma else self.ku
        self.gamma += dt * k * (abs(self.m) - self.gamma)
        self.gamma = max(0.006, min(0.600, self.gamma))

        if self.m > self.gamma:
            self.overuse_time += dt
            if self.overuse_time >= 0.100:
                self.state = 'OVERUSE'
        elif self.m < -self.gamma:
            self.overuse_time = 0.0
            self.state = 'UNDERUSE'
        else:
            self.overuse_time = max(0.0, self.overuse_time - dt)
            if self.overuse_time == 0.0:
                self.state = 'NORMAL'

        self.prev_send_ts = float(send_ts)
        self.prev_arrival_ts = float(arrival_ts)
        self.prev_frame_id = int(frame_id)
        return dm * 1000.0

    def update_rate(self, now=None):
        now = time.monotonic() if now is None else float(now)
        rr = self.receive_rate_bps(now)

        if rr >= 50_000:
            if self.state == 'OVERUSE':
                # Decrease rate upon delay overuse
                self.ar_bps = 0.85 * rr
            elif self.state == 'UNDERUSE':
                # Multiplicative increase when queue is draining
                self.ar_bps = max(self.ar_bps * 1.05, rr * 1.05)
            elif self.state == 'NORMAL':
                # Probing/Additive increase when channel is stable
                # Increase by 5% or minimum step of 100 kbps
                # The cap must be relative to the MEASURED receive rate. Capping
                # at 1.5 * max(rr, ar_bps) becomes 1.5 * ar_bps as soon as ar_bps
                # exceeds rr, which is always above ar_bps + 5% -- so the cap
                # never binds and Ar compounds without limit. Observed on a
                # 2.5 Mbps link: Ar climbed 8.0 -> 9.7 Mbps while only 0.2 Mbps
                # was actually arriving. Capacity cannot be inferred above what
                # has actually been pushed through the path.
                increase = max(self.ar_bps * 0.05, 100_000.0)
                self.ar_bps = min(self.ar_bps + increase, 1.5 * rr)

        self.ar_bps = max(self.min_rate_bps, min(self.max_rate_bps, self.ar_bps))
        self.last_rate_update = now
        return self.ar_bps, rr

    def feedback(self):
        ar, rr = self.update_rate()
        return ar, rr, self.m * 1000.0, self.gamma * 1000.0, self.state
