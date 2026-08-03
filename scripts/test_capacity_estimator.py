#!/usr/bin/env python3
"""Deterministic checks for the passive Week 1B capacity estimator."""

import math
import inspect
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "receiver"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from capacity_estimator import ArrivalCapacityEstimator  # noqa: E402
from webrtc_diagnostics import (  # noqa: E402
    sender_grace_for_backpressure_pause,
)


def assert_close(actual, expected, label, tol=1e-9):
    if not math.isclose(actual, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def main():
    est = ArrivalCapacityEstimator(alpha=0.1)

    first = est.observe(receiver_ts=10.000, sender_grace_period=0.0)
    if first["raw_interarrival"] is not None:
        raise AssertionError("first sample should not have an inter-arrival time")

    second = est.observe(receiver_ts=10.020, sender_grace_period=0.0)
    assert_close(second["raw_interarrival"], 0.020, "raw sample 2")
    assert_close(second["corrected_interarrival"], 0.020, "corrected sample 2")
    assert_close(second["ewma_interarrival"], 0.020, "ewma sample 2")

    third = est.observe(receiver_ts=10.140, sender_grace_period=0.100)
    assert_close(third["raw_interarrival"], 0.120, "raw sample 3")
    assert_close(third["sender_grace_period"], 0.100, "grace sample 3")
    if third["corrected_interarrival"] is not None:
        raise AssertionError("sender-idle boundary must not update service time")
    assert_close(third["ewma_interarrival"], 0.020, "idle preserves EWMA")

    fourth = est.observe(receiver_ts=10.180, sender_grace_period=0.0)
    assert_close(fourth["raw_interarrival"], 0.040, "raw sample 4")
    assert_close(fourth["corrected_interarrival"], 0.040, "corrected sample 4")
    assert_close(fourth["ewma_interarrival"], 0.022, "ewma sample 4")

    clamped = est.observe(receiver_ts=10.180, sender_grace_period=0.0)
    assert_close(clamped["corrected_interarrival"], 1e-6, "clamped corrected sample")

    sequenced = ArrivalCapacityEstimator(alpha=0.1)
    sequenced.observe(20.000, sequence_id=10)
    normal = sequenced.observe(20.010, sequence_id=11)
    assert_close(normal["ewma_interarrival"], 0.010, "contiguous sequence sample")
    gap = sequenced.observe(20.040, sequence_id=13)
    if gap["corrected_interarrival"] is not None:
        raise AssertionError("non-contiguous sequence must not update service time")
    assert_close(gap["ewma_interarrival"], 0.010, "sequence gap preserves EWMA")
    resumed = sequenced.observe(20.052, sequence_id=14)
    assert_close(resumed["corrected_interarrival"], 0.012, "post-gap adjacent sample")

    variable = ArrivalCapacityEstimator(alpha=0.1)
    variable.observe(30.000, sequence_id=1, packet_size_bytes=1000)
    small = variable.observe(30.010, sequence_id=2, packet_size_bytes=1000)
    assert_close(1000 / small["ewma_interarrival"], 100_000.0,
                 "small-message byte rate", tol=1e-7)
    large = variable.observe(30.030, sequence_id=3, packet_size_bytes=2000)
    assert_close(2000 / large["ewma_interarrival"], 100_000.0,
                 "variable-size byte rate", tol=1e-7)
    idle_large = variable.observe(
        30.130, sender_grace_period=0.080, sequence_id=4,
        packet_size_bytes=2000,
    )
    if idle_large["corrected_interarrival"] is not None:
        raise AssertionError("variable-size idle boundary must be excluded")
    assert_close(2000 / idle_large["ewma_interarrival"], 100_000.0,
                 "idle preserves variable-size byte rate", tol=1e-7)

    freshness = ArrivalCapacityEstimator(
        alpha=1.0,
        freshness_multiplier=5.0,
        freshness_min_s=0.02,
        freshness_max_s=0.10,
        recovery_samples=3,
        recovery_window_s=0.02,
    )
    freshness.observe(40.000, sequence_id=1, packet_size_bytes=1000)
    startup_one = freshness.observe(
        40.010, sequence_id=2, packet_size_bytes=1000
    )
    if startup_one["estimate_fresh"]:
        raise AssertionError("startup must not publish from one spacing sample")
    freshness.observe(40.020, sequence_id=3, packet_size_bytes=1000)
    live = freshness.observe(40.030, sequence_id=4, packet_size_bytes=1000)
    if not live["estimate_fresh"]:
        raise AssertionError("startup should publish after configured evidence")
    raw_before_stale = live["ewma_interarrival"]
    stale = freshness.snapshot(40.200, packet_size_bytes=1000)
    if stale["estimate_fresh"] or stale["published_ewma_interarrival"] is not None:
        raise AssertionError("prolonged absence must invalidate publication")
    assert_close(
        stale["ewma_interarrival"], raw_before_stale,
        "stale state retains last raw value",
    )

    stale_boundary = freshness.observe(
        40.200, sequence_id=5, packet_size_bytes=1000
    )
    if stale_boundary["filter_reason"] != "stale_boundary_excluded":
        raise AssertionError("first post-stale spacing must remain raw-only")
    if stale_boundary["estimate_fresh"]:
        raise AssertionError("one post-stale sample must not republish")
    for seq, timestamp in ((6, 40.210), (7, 40.220)):
        recovering = freshness.observe(
            timestamp, sequence_id=seq, packet_size_bytes=1000
        )
        if recovering["estimate_fresh"]:
            raise AssertionError("recovery must require configured adjacency")
    recovered = freshness.observe(
        40.230, sequence_id=8, packet_size_bytes=1000
    )
    if not recovered["estimate_fresh"]:
        raise AssertionError("estimate should resume after adjacent recovery samples")

    recovery_gap = freshness.observe(
        40.250, sequence_id=10, packet_size_bytes=1000
    )
    if recovery_gap["corrected_interarrival"] is not None:
        raise AssertionError("recovery sequence gap must remain excluded")
    resumed_after_gap = freshness.observe(
        40.260, sequence_id=11, packet_size_bytes=1000
    )
    if resumed_after_gap["corrected_interarrival"] is None:
        raise AssertionError("adjacent sample after a gap must update normally")

    if sender_grace_for_backpressure_pause(0.25, 32768, 0.20) != 0.0:
        raise AssertionError("queued backpressure is not a sender-idle boundary")
    drained_grace = sender_grace_for_backpressure_pause(0.25, 0, 0.20)
    assert_close(drained_grace, 0.25, "drained backpressure idle boundary")
    pacing = ArrivalCapacityEstimator(alpha=1.0)
    pacing.observe(50.000, sequence_id=1, packet_size_bytes=1000)
    paced = pacing.observe(
        50.010,
        sender_grace_period=sender_grace_for_backpressure_pause(
            0.010, 4096, 0.002
        ),
        sequence_id=2,
        packet_size_bytes=1000,
    )
    if paced["corrected_interarrival"] is None:
        raise AssertionError("ordinary queued pacing must not be removed")
    drained = pacing.observe(
        50.260,
        sender_grace_period=drained_grace,
        sequence_id=3,
        packet_size_bytes=1000,
    )
    if drained["corrected_interarrival"] is not None:
        raise AssertionError("a genuinely drained sender pause must be excluded")
    if drained["estimate_recovering"]:
        raise AssertionError("normal drained frame boundary must not enter recovery")

    frame_rate = ArrivalCapacityEstimator(
        alpha=0.2,
        robust_window_s=0.1,
        freshness_multiplier=3.0,
        freshness_min_s=0.01,
        freshness_max_s=0.5,
        recovery_samples=3,
        recovery_window_s=0.1,
    )
    seq = 1
    frame_rate.observe(70.0, sequence_id=seq, packet_size_bytes=1000)
    fresh_frames = []
    for frame in range(8):
        base = 70.0 + frame * 0.04
        if frame:
            seq += 1
            boundary = frame_rate.observe(
                base, sender_grace_period=0.03,
                sequence_id=seq, packet_size_bytes=1000,
            )
            if boundary["estimate_recovering"]:
                raise AssertionError("25 fps idle boundary reset recovery")
        for offset in (0.001, 0.002, 0.003):
            seq += 1
            sample = frame_rate.observe(
                base + offset, sequence_id=seq, packet_size_bytes=1000
            )
        fresh_frames.append(sample["estimate_fresh"])
    if not all(fresh_frames):
        raise AssertionError("repeated 25 fps bursts must remain publishable")
    between_frames = frame_rate.snapshot(70.0 + 7 * 0.04 + 0.035)
    if not between_frames["estimate_fresh"]:
        raise AssertionError("freshness must span the 25 fps controller cadence")
    expired = frame_rate.snapshot(71.0)
    if not expired["estimate_stale"] or expired["estimate_fresh"]:
        raise AssertionError("genuine long confirmation gap must become stale")

    reordered = ArrivalCapacityEstimator(
        alpha=1.0, recovery_samples=2, recovery_window_s=0.1
    )
    reordered.observe(80.000, sequence_id=1, packet_size_bytes=1000)
    reordered.observe(80.001, sequence_id=2, packet_size_bytes=1000)
    reordered.observe(80.002, sequence_id=3, packet_size_bytes=1000)
    jump = reordered.observe(80.003, sequence_id=5, packet_size_bytes=1000)
    if jump["estimate_state_reason"] != "sequence_gap":
        raise AssertionError("real forward gap must enter recovery")
    late = reordered.observe(80.004, sequence_id=4, packet_size_bytes=1000)
    if late["skip_reason"] != "reordered_message":
        raise AssertionError("late cross-channel message must be classified")
    boundary = reordered.observe(80.005, sequence_id=6, packet_size_bytes=1000)
    if boundary["skip_reason"] != "reorder_boundary":
        raise AssertionError("post-reorder physical boundary must be excluded")
    first_recovery = reordered.observe(
        80.006, sequence_id=7, packet_size_bytes=1000
    )
    second_recovery = reordered.observe(
        80.007, sequence_id=8, packet_size_bytes=1000
    )
    if first_recovery["estimate_fresh"] or not second_recovery["estimate_fresh"]:
        raise AssertionError("reorder must not repeatedly reset recovery")
    if reordered._recovery_count != len(reordered._service_samples):
        raise AssertionError("recovery count must match active-window samples")

    robust = ArrivalCapacityEstimator(
        alpha=1.0,
        robust_window_s=0.1,
        freshness_min_s=0.02,
        freshness_max_s=0.2,
    )
    robust.observe(60.000, sequence_id=1, packet_size_bytes=1000)
    sample = None
    for seq in range(2, 9):
        sample = robust.observe(
            60.000 + (seq - 1) * 0.010,
            sequence_id=seq,
            packet_size_bytes=1000,
        )
    filtered_before = sample["filtered_ewma_interarrival"]
    spike = robust.observe(
        60.070001, sequence_id=9, packet_size_bytes=1000
    )
    if spike["ewma_interarrival"] >= filtered_before:
        raise AssertionError("short interval should remain visible in raw EWMA")
    assert_close(
        spike["filtered_ewma_interarrival"], filtered_before,
        "causal median rejects isolated short interval",
    )
    snapshot_before_future = robust.snapshot(60.070001, packet_size_bytes=1000)
    robust.observe(60.080, sequence_id=10, packet_size_bytes=1000)
    assert_close(
        snapshot_before_future["filtered_ewma_interarrival"],
        filtered_before,
        "filter output cannot depend on a future sample",
    )

    signature = inspect.signature(ArrivalCapacityEstimator)
    forbidden_inputs = {"bandwidth", "loss", "trace", "ground_truth", "expected_rate"}
    if any(any(word in name for word in forbidden_inputs) for name in signature.parameters):
        raise AssertionError("estimator constructor must not accept trace ground truth")
    print("capacity estimator unit test passed")


if __name__ == "__main__":
    main()
