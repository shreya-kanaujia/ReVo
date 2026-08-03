"""Pure, causal Track B adaptation policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class AdaptationMode(str, Enum):
    LEGACY = "legacy"
    FIXED_HIGH = "fixed_high"
    FIXED_LOW = "fixed_low"
    BUFFER_HEALTH = "buffer_health"
    ESTIMATOR_ONLY = "estimator_only"
    COMBINED = "combined"


class Quality(str, Enum):
    HIGH = "high"
    LOW = "low"


class ControllerState(str, Enum):
    HIGH_STABLE = "HIGH_STABLE"
    LOW_CONGESTED = "LOW_CONGESTED"
    RECOVERY_HOLD = "RECOVERY_HOLD"
    OBSERVABILITY_DEGRADED = "OBSERVABILITY_DEGRADED"


@dataclass(frozen=True)
class ControllerConfig:
    buffer_soft_bytes: int = 64 * 1024
    buffer_hard_bytes: int = 128 * 1024
    buffer_growth_bytes: int = 8 * 1024
    impairment_threshold: float = 0.05
    severe_impairment_threshold: float = 0.20
    capacity_safety_margin: float = 0.85
    upgrade_headroom_fraction: float = 0.15
    downgrade_confirmations: int = 1
    upgrade_confirmations: int = 3
    min_dwell_gops: int = 2
    passive_service_floor_fraction: float = 0.80

    def __post_init__(self):
        if not 0 < self.buffer_soft_bytes < self.buffer_hard_bytes:
            raise ValueError("buffer thresholds must be positive and ordered")
        if not 0.0 < self.capacity_safety_margin <= 1.0:
            raise ValueError("capacity_safety_margin must be in (0, 1]")
        if self.upgrade_confirmations <= self.downgrade_confirmations:
            raise ValueError("upgrade evidence must exceed downgrade evidence")
        if not 0.0 < self.passive_service_floor_fraction <= 1.0:
            raise ValueError("passive service floor must be in (0, 1]")


@dataclass(frozen=True)
class ControllerInputs:
    timestamp: float
    frame_id: int
    gop_id: int
    is_keyframe: bool
    capacity_raw_mbps: Optional[float]
    capacity_filtered_mbps: Optional[float]
    capacity_published_mbps: Optional[float]
    capacity_age_s: Optional[float]
    capacity_fresh: bool
    rgb_buffered_amount: int
    depth_buffered_amount: int
    buffer_trend_bytes: int
    receiver_impairment_rate: Optional[float]
    receiver_health_age_s: Optional[float]
    receiver_health_fresh: bool
    predicted_high_demand_mbps: Optional[float]
    predicted_low_demand_mbps: Optional[float]
    low_candidate_available: bool = True
    outstanding_messages: int = 0
    feedback_delay_s: Optional[float] = None
    current_offered_video_mbps: Optional[float] = None
    probe_state: str = "IDLE"
    probe_authorization_valid: bool = False
    probe_measured_mbps: Optional[float] = None
    probe_failure_reason: str = ""


@dataclass(frozen=True)
class ControllerDecision:
    decision_seq: int
    state: ControllerState
    current_quality: Quality
    requested_quality: Quality
    applied_quality: Quality
    switch_requested: bool
    switch_applied: bool
    primary_reason: str
    reason_flags: tuple[str, ...]
    capacity_safe_mbps: Optional[float]
    dwell_gops: int
    cooldown_active: bool


class AdaptationController:
    def __init__(
        self,
        mode: AdaptationMode | str = AdaptationMode.LEGACY,
        config: ControllerConfig | None = None,
        initial_quality: Quality | str | None = None,
    ):
        self.mode = AdaptationMode(mode)
        self.config = config or ControllerConfig()
        if initial_quality is None:
            self.current_quality = (
                Quality.LOW
                if self.mode == AdaptationMode.FIXED_LOW
                else Quality.HIGH
            )
        else:
            self.current_quality = Quality(initial_quality)
        self.requested_quality = self.current_quality
        self.state = (
            ControllerState.LOW_CONGESTED
            if self.current_quality == Quality.LOW
            else ControllerState.HIGH_STABLE
        )
        self._decision_seq = 0
        self._down_votes = 0
        self._up_votes = 0
        self._last_applied_gop: Optional[int] = None
        self._quality_since_gop: Optional[int] = (
            0 if initial_quality is not None else None
        )
        self._last_upgrade_evidence_gop: Optional[int] = None
        self._downgrade_latched = False
        self._latched_down_reason = ""
        self._latched_down_flags: tuple[str, ...] = ()
        self._post_upgrade_until_gop: Optional[int] = None

    def _dwell_gops(self, gop_id: int) -> int:
        if self._quality_since_gop is None:
            return self.config.min_dwell_gops
        return max(0, int(gop_id) - self._quality_since_gop)

    def dwell_complete(self, gop_id: int) -> bool:
        return self._dwell_gops(gop_id) >= self.config.min_dwell_gops

    def decide(self, inputs: ControllerInputs) -> ControllerDecision:
        self._decision_seq += 1
        previous_quality = self.current_quality
        cfg = self.config
        max_buffer = max(inputs.rgb_buffered_amount, inputs.depth_buffered_amount)
        capacity_safe = (
            None
            if inputs.capacity_published_mbps is None
            else inputs.capacity_published_mbps * cfg.capacity_safety_margin
        )
        flags: list[str] = []
        severe_buffer = max_buffer >= cfg.buffer_hard_bytes
        buffer_above_soft = max_buffer >= cfg.buffer_soft_bytes
        buffer_growing = (
            max_buffer > 0
            and inputs.buffer_trend_bytes >= cfg.buffer_growth_bytes
        )
        sustained_buffer_growth = (
            buffer_above_soft and buffer_growing
        )
        health_bad = (
            inputs.receiver_health_fresh
            and inputs.receiver_impairment_rate is not None
            and inputs.receiver_impairment_rate >= cfg.impairment_threshold
        )
        severe_health = (
            inputs.receiver_health_fresh
            and inputs.receiver_impairment_rate is not None
            and inputs.receiver_impairment_rate >= cfg.severe_impairment_threshold
        )
        # Passive service is conditional on currently offered traffic. It can
        # detect deterioration below measured offered load, but cannot prove
        # unused headroom or authorize a low-to-high upgrade.
        passive_service_low = (
            inputs.capacity_fresh
            and inputs.capacity_published_mbps is not None
            and inputs.current_offered_video_mbps is not None
            and inputs.capacity_published_mbps
            < inputs.current_offered_video_mbps
            * cfg.passive_service_floor_fraction
        )
        # A single keyframe enqueue can create a short buffer delta while the
        # queue remains far below the soft threshold.  It is not independent
        # congestion evidence.  Corroboration requires sustained queue
        # pressure or receiver health, while hard/severe signals remain
        # immediate safety overrides.
        direct_congestion_evidence = (
            severe_buffer
            or sustained_buffer_growth
            or health_bad
            or severe_health
        )
        passive_service_corroborated = (
            passive_service_low and direct_congestion_evidence
        )
        passive_only_suppressed = (
            self.mode == AdaptationMode.COMBINED
            and passive_service_low
            and not direct_congestion_evidence
        )
        post_upgrade_validation = (
            self.mode == AdaptationMode.COMBINED
            and self.current_quality == Quality.HIGH
            and self._post_upgrade_until_gop is not None
            and inputs.gop_id < self._post_upgrade_until_gop
        )
        for active, name in (
            (severe_buffer, "buffer_hard"),
            (buffer_above_soft, "buffer_soft"),
            (buffer_growing, "buffer_growing"),
            (severe_health, "health_severe"),
            (health_bad, "health_bad"),
            (passive_service_low, "passive_service_deterioration"),
            (
                passive_service_corroborated,
                "passive_service_corroborated",
            ),
            (
                passive_only_suppressed,
                "passive_only_downgrade_suppressed",
            ),
            (post_upgrade_validation, "post_upgrade_validation"),
            (inputs.probe_authorization_valid, "probe_upgrade_authorized"),
            (not inputs.capacity_fresh, "capacity_stale"),
            (not inputs.receiver_health_fresh, "health_stale"),
        ):
            if active:
                flags.append(name)

        target = self.current_quality
        reason = "hold"
        forced_down = severe_buffer or severe_health
        policy_buffer = sustained_buffer_growth or health_bad

        if self.mode == AdaptationMode.FIXED_HIGH:
            target, reason = Quality.HIGH, "fixed_high"
        elif self.mode == AdaptationMode.FIXED_LOW:
            target, reason = Quality.LOW, "fixed_low"
        elif self.mode == AdaptationMode.LEGACY:
            target, reason = self.current_quality, "legacy"
        else:
            use_buffer = self.mode in (
                AdaptationMode.BUFFER_HEALTH,
                AdaptationMode.COMBINED,
            )
            use_capacity = self.mode in (
                AdaptationMode.ESTIMATOR_ONLY,
                AdaptationMode.COMBINED,
            )
            passive_wants_down = (
                use_capacity
                and passive_service_low
                and (
                    self.mode != AdaptationMode.COMBINED
                    or direct_congestion_evidence
                )
            )
            wants_down = (use_buffer and policy_buffer) or passive_wants_down
            post_upgrade_hold = post_upgrade_validation and not forced_down
            if post_upgrade_hold:
                # The bounded validation dwell prevents the keyframe and rate
                # transition caused by the authorized upgrade from voting
                # against itself. Emergency signals still bypass the hold.
                wants_down = False
            if use_buffer and forced_down:
                self._down_votes = cfg.downgrade_confirmations
                reason = "safety_override"
            elif wants_down:
                self._down_votes += 1
                reason = (
                    "fallback_congestion"
                    if use_buffer and policy_buffer
                    else "passive_service_deterioration_corroborated"
                )
            else:
                self._down_votes = 0
                if post_upgrade_hold and (
                    policy_buffer or passive_service_low
                ):
                    reason = "post_upgrade_validation_hold"
                elif passive_only_suppressed:
                    reason = "passive_service_uncorroborated_hold"

            health_ok = (
                not use_buffer
                or (
                    inputs.receiver_health_fresh
                    and not health_bad
                    and max_buffer < cfg.buffer_soft_bytes
                    and inputs.buffer_trend_bytes <= 0
                )
            )
            if self.mode == AdaptationMode.COMBINED:
                upgrade_authorized = inputs.probe_authorization_valid
                capacity_observable = True
            elif self.mode == AdaptationMode.ESTIMATOR_ONLY:
                # Passive conditional service can force a downgrade but never
                # prove spare headroom for an upgrade.
                upgrade_authorized = False
                capacity_observable = inputs.capacity_fresh
            else:
                upgrade_authorized = True
                capacity_observable = True
            capacity_ok = not use_capacity or upgrade_authorized
            observability_ok = (
                capacity_observable
                and (not use_buffer or inputs.receiver_health_fresh)
            )
            new_upgrade_observation = (
                inputs.is_keyframe
                and inputs.gop_id != self._last_upgrade_evidence_gop
            )
            if (
                health_ok
                and capacity_ok
                and observability_ok
                and new_upgrade_observation
            ):
                self._up_votes += 1
                self._last_upgrade_evidence_gop = inputs.gop_id
            else:
                if not (health_ok and capacity_ok and observability_ok):
                    self._up_votes = 0
            if not observability_ok:
                self.state = ControllerState.OBSERVABILITY_DEGRADED
                reason = "observability_degraded" if reason == "hold" else reason
                # Evidence collected before loss of observability cannot authorize upgrade.
                self._up_votes = 0

            if self._down_votes >= cfg.downgrade_confirmations:
                target = Quality.LOW
            elif (
                self.current_quality == Quality.LOW
                and self._up_votes >= (
                    1
                    if (
                        self.mode == AdaptationMode.COMBINED
                        and inputs.probe_authorization_valid
                    )
                    else cfg.upgrade_confirmations
                )
                and self._dwell_gops(inputs.gop_id) >= cfg.min_dwell_gops
            ):
                target = Quality.HIGH
                reason = "sustained_recovery"

            if self.current_quality == Quality.HIGH and target == Quality.LOW:
                self._downgrade_latched = True
                self._latched_down_reason = reason
                self._latched_down_flags = tuple(flags)

            if self.current_quality == Quality.HIGH and self._downgrade_latched:
                # A valid downgrade survives intervening non-keyframes. It may
                # be cancelled only at the next safe boundary, and only when
                # every signal used by the active mode is currently observable
                # and positively supports high quality.
                cancellation_safe = (
                    inputs.is_keyframe
                    and not forced_down
                    and not wants_down
                    and observability_ok
                    and health_ok
                    and capacity_ok
                )
                if cancellation_safe:
                    self._downgrade_latched = False
                    self._latched_down_reason = ""
                    self._latched_down_flags = ()
                    target = Quality.HIGH
                    reason = "latched_downgrade_cancelled_at_keyframe"
                else:
                    target = Quality.LOW
                    reason = self._latched_down_reason or "latched_downgrade"
                    for flag in self._latched_down_flags:
                        if flag not in flags:
                            flags.append(flag)
                    if "downgrade_latched" not in flags:
                        flags.append("downgrade_latched")

        pending_switch = target != self.current_quality
        switch_requested = pending_switch and target != self.requested_quality
        self.requested_quality = target
        switch_applied = False
        if (
            pending_switch
            and inputs.is_keyframe
            and self._last_applied_gop != inputs.gop_id
        ):
            if target != Quality.LOW or inputs.low_candidate_available:
                quality_before_switch = self.current_quality
                self.current_quality = target
                self._last_applied_gop = inputs.gop_id
                self._quality_since_gop = inputs.gop_id
                self._down_votes = 0
                self._up_votes = 0
                self._downgrade_latched = False
                self._latched_down_reason = ""
                self._latched_down_flags = ()
                if (
                    self.mode == AdaptationMode.COMBINED
                    and quality_before_switch == Quality.LOW
                    and target == Quality.HIGH
                ):
                    self._post_upgrade_until_gop = (
                        inputs.gop_id + cfg.min_dwell_gops
                    )
                    if "post_upgrade_validation" not in flags:
                        flags.append("post_upgrade_validation")
                elif target == Quality.LOW:
                    self._post_upgrade_until_gop = None
                switch_applied = True
            else:
                reason = "low_candidate_missing_or_mismatched"
                flags.append("low_candidate_unavailable")

        dwell = self._dwell_gops(inputs.gop_id)
        if self.state != ControllerState.OBSERVABILITY_DEGRADED:
            if self.current_quality == Quality.HIGH:
                self.state = ControllerState.HIGH_STABLE
            elif self._up_votes:
                self.state = ControllerState.RECOVERY_HOLD
            else:
                self.state = ControllerState.LOW_CONGESTED
        elif inputs.capacity_fresh and (
            self.mode == AdaptationMode.ESTIMATOR_ONLY
            or inputs.receiver_health_fresh
        ):
            self.state = (
                ControllerState.HIGH_STABLE
                if self.current_quality == Quality.HIGH
                else ControllerState.LOW_CONGESTED
            )

        return ControllerDecision(
            decision_seq=self._decision_seq,
            state=self.state,
            current_quality=previous_quality,
            requested_quality=target,
            applied_quality=self.current_quality,
            switch_requested=switch_requested,
            switch_applied=switch_applied,
            primary_reason=reason,
            reason_flags=tuple(flags),
            capacity_safe_mbps=capacity_safe,
            dwell_gops=dwell,
            cooldown_active=dwell < cfg.min_dwell_gops,
        )
