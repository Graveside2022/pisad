"""
SDR Priority Manager

Implements priority decision making, conflict resolution, and safety override
for dual SDR coordination system.

PRD References:
- FR11: Operator override capability
- FR15: Immediate command cessation on mode change
- NFR2: <100ms latency requirement
- NFR12: Deterministic timing for safety-critical functions
"""

import time
from dataclasses import dataclass
from typing import Any

from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SignalQuality:
    """Signal quality metrics for priority decisions."""

    score: float  # 0-100 quality score
    confidence: float  # 0-1 confidence level
    rssi: float  # RSSI in dBm
    snr: float | None = None  # SNR in dB
    stability: float | None = None  # Signal stability 0-1


@dataclass
class PriorityDecision:
    """Priority decision result."""

    selected_source: str  # "ground" or "drone"
    reason: str  # Decision rationale
    confidence: float  # Decision confidence 0-1
    latency_ms: float  # Decision latency
    switch_recommended: bool = False
    safety_critical: bool = False


@dataclass
class ConflictResolution:
    """Frequency command conflict resolution."""

    selected_command: dict[str, Any]
    conflict_type: str
    resolution_time_ms: float
    rejected_command: dict[str, Any] | None = None


class SDRPriorityMatrix:
    """
    Signal quality scoring and priority decision matrix.

    Implements intelligent source selection with hysteresis
    and safety override capabilities.
    """

    def __init__(self, hysteresis_threshold: float = 5.0):
        """
        Initialize priority matrix.

        Args:
            hysteresis_threshold: Minimum score difference to switch sources
        """
        self.hysteresis_threshold = hysteresis_threshold

        # Scoring weights
        self.rssi_weight = 0.5
        self.snr_weight = 0.3
        self.stability_weight = 0.2

    def calculate_signal_quality(
        self, rssi: float, snr: float | None = None, stability: float | None = None
    ) -> SignalQuality:
        """
        Calculate signal quality score from metrics.

        Args:
            rssi: RSSI in dBm
            snr: SNR in dB (optional)
            stability: Signal stability 0-1 (optional)

        Returns:
            SignalQuality with calculated score and confidence
        """
        # RSSI scoring: -30dBm = 100, -80dBm = 0
        rssi_score = max(0, min(100, (rssi + 80) * 2))

        # SNR scoring: 20dB = 100, 0dB = 0
        snr_score = 50.0  # Default if not provided
        if snr is not None:
            snr_score = max(0, min(100, snr * 5))

        # Stability scoring: direct mapping
        stability_score = 50.0  # Default if not provided
        if stability is not None:
            stability_score = stability * 100

        # Weighted total score
        total_score = (
            rssi_score * self.rssi_weight
            + snr_score * self.snr_weight
            + stability_score * self.stability_weight
        )

        # Confidence based on available metrics
        confidence = 0.6  # Base confidence
        if snr is not None:
            confidence += 0.2
        if stability is not None:
            confidence += 0.2

        return SignalQuality(
            score=total_score, confidence=confidence, rssi=rssi, snr=snr, stability=stability
        )

    def make_priority_decision(
        self,
        ground_quality: SignalQuality,
        drone_quality: SignalQuality,
        current_source: str,
        emergency_override: bool = False,
    ) -> PriorityDecision:
        """
        Make priority decision between ground and drone sources.

        Args:
            ground_quality: Ground SDR signal quality
            drone_quality: Drone SDR signal quality
            current_source: Currently active source
            emergency_override: Force drone selection for safety

        Returns:
            PriorityDecision with selected source and rationale
        """
        start_time = time.perf_counter()

        # Emergency override always selects drone for safety
        if emergency_override:
            latency_ms = (time.perf_counter() - start_time) * 1000
            return PriorityDecision(
                selected_source="drone",
                reason="emergency_override",
                confidence=1.0,
                latency_ms=latency_ms,
                safety_critical=True,
            )

        # Calculate score difference
        score_diff = ground_quality.score - drone_quality.score

        # Apply hysteresis to prevent oscillation
        if current_source == "ground":
            # Need significant drone advantage to switch
            if score_diff < -self.hysteresis_threshold:
                selected = "drone"
                reason = "drone_signal_superior"
                switch = True
            else:
                selected = "ground"
                reason = (
                    "hysteresis_maintained"
                    if abs(score_diff) < self.hysteresis_threshold
                    else "ground_signal_superior"
                )
                switch = False
        else:  # current_source == "drone"
            # Need significant ground advantage to switch
            if score_diff > self.hysteresis_threshold:
                selected = "ground"
                reason = "ground_signal_superior"
                switch = True
            else:
                selected = "drone"
                reason = (
                    "hysteresis_maintained"
                    if abs(score_diff) < self.hysteresis_threshold
                    else "drone_signal_superior"
                )
                switch = False

        # Calculate decision confidence
        confidence = min(ground_quality.confidence, drone_quality.confidence)

        latency_ms = (time.perf_counter() - start_time) * 1000

        return PriorityDecision(
            selected_source=selected,
            reason=reason,
            confidence=confidence,
            latency_ms=latency_ms,
            switch_recommended=switch,
        )


class SDRPriorityManager:
    """
    Manages SDR priority decisions, conflict resolution, and safety overrides.

    Coordinates between ground SDR++ and drone PISAD systems while maintaining
    safety authority and performance requirements.
    """

    def __init__(self, coordinator: Any = None, safety_manager: Any = None):
        """
        Initialize priority manager.

        Args:
            coordinator: DualSDRCoordinator instance
            safety_manager: SafetyManager instance
        """
        self._coordinator = coordinator
        self._safety_manager = safety_manager
        self._matrix = SDRPriorityMatrix()

        # Priority state
        self._emergency_override_active = False
        self._conflict_history: list[ConflictResolution] = []
        self._last_decision_timestamp = 0.0

        # Performance tracking
        self._decision_latencies: list[float] = []

        logger.info("SDRPriorityManager initialized")

    async def evaluate_source_switch(self) -> PriorityDecision:
        """
        Evaluate whether to switch active SDR source.

        Returns:
            PriorityDecision with switching recommendation
        """
        start_time = time.perf_counter()

        # Get signal qualities from both sources
        ground_rssi = None
        drone_rssi = None

        if self._coordinator and hasattr(self._coordinator, "get_ground_rssi"):
            try:
                ground_rssi = self._coordinator.get_ground_rssi()
            except Exception as e:
                logger.warning(f"Failed to get ground RSSI: {e}")

        if self._coordinator and hasattr(self._coordinator, "get_drone_rssi"):
            try:
                drone_rssi = self._coordinator.get_drone_rssi()
            except Exception as e:
                logger.warning(f"Failed to get drone RSSI: {e}")

        # Calculate signal qualities
        if ground_rssi is not None and drone_rssi is not None:
            ground_quality = self._matrix.calculate_signal_quality(ground_rssi)
            drone_quality = self._matrix.calculate_signal_quality(drone_rssi)

            current_source = getattr(self._coordinator, "active_source", "drone")

            decision = self._matrix.make_priority_decision(
                ground_quality, drone_quality, current_source, self._emergency_override_active
            )
        else:
            # Fallback to drone if unable to get signals
            latency_ms = (time.perf_counter() - start_time) * 1000
            decision = PriorityDecision(
                selected_source="drone",
                reason="fallback_no_signals",
                confidence=0.5,
                latency_ms=latency_ms,
                switch_recommended=True,
                safety_critical=True,
            )

        # Track performance
        self._decision_latencies.append(decision.latency_ms)
        if len(self._decision_latencies) > 100:
            self._decision_latencies = self._decision_latencies[-50:]

        self._last_decision_timestamp = time.time()

        return decision

    async def resolve_frequency_conflict(
        self, ground_command: dict[str, Any], drone_command: dict[str, Any]
    ) -> ConflictResolution:
        """
        Resolve conflict between ground and drone frequency commands.

        Args:
            ground_command: Frequency command from ground SDR++
            drone_command: Frequency command from drone PISAD

        Returns:
            ConflictResolution with selected command
        """
        start_time = time.perf_counter()

        # Safety-first priority: drone commands win for emergency scenarios
        if self._emergency_override_active:
            selected = drone_command
            conflict_type = "emergency_override_active"
        else:
            # Compare timestamps - newer command wins
            ground_time = ground_command.get("timestamp", 0)
            drone_time = drone_command.get("timestamp", 0)

            if drone_time > ground_time:
                selected = drone_command
                conflict_type = "drone_command_newer"
            elif ground_time > drone_time:
                selected = ground_command
                conflict_type = "ground_command_newer"
            else:
                # Same timestamp - prefer drone for safety
                selected = drone_command
                conflict_type = "safety_priority_drone"

        resolution_time_ms = (time.perf_counter() - start_time) * 1000

        resolution = ConflictResolution(
            selected_command=selected,
            conflict_type=conflict_type,
            resolution_time_ms=resolution_time_ms,
            rejected_command=ground_command if selected == drone_command else drone_command,
        )

        # Log conflict for history
        self._conflict_history.append(resolution)
        if len(self._conflict_history) > 50:
            self._conflict_history = self._conflict_history[-25:]

        logger.info(
            f"Resolved frequency conflict: {conflict_type}, latency: {resolution_time_ms:.1f}ms"
        )

        return resolution

    async def trigger_emergency_override(self) -> dict[str, Any]:
        """
        Trigger emergency override ensuring drone maintains control.

        Returns:
            Emergency override result with timing
        """
        start_time = time.perf_counter()

        self._emergency_override_active = True

        # Force switch to drone source
        if self._coordinator and hasattr(self._coordinator, "active_source"):
            self._coordinator.active_source = "drone"

        # Trigger safety manager if available
        safety_result = {}
        if self._safety_manager and hasattr(self._safety_manager, "trigger_emergency_stop"):
            try:
                safety_result = self._safety_manager.trigger_emergency_stop()
            except Exception as e:
                logger.error(f"Emergency stop failed: {e}")
                safety_result = {"success": False, "error": str(e)}

        response_time_ms = (time.perf_counter() - start_time) * 1000

        result = {
            "source_switched_to": "drone",
            "safety_activated": safety_result.get("success", False),
            "response_time_ms": response_time_ms,
            "emergency_override_active": True,
        }

        logger.warning(f"Emergency override activated, response time: {response_time_ms:.1f}ms")

        return result

    async def handle_communication_loss(self) -> dict[str, Any]:
        """
        Handle graceful degradation on ground communication loss.

        Returns:
            Degradation handling result
        """
        start_time = time.perf_counter()

        # Switch to drone-only operation
        if self._coordinator:
            if hasattr(self._coordinator, "active_source"):
                self._coordinator.active_source = "drone"
            if hasattr(self._coordinator, "fallback_active"):
                self._coordinator.fallback_active = True

        degradation_time_s = time.perf_counter() - start_time

        result = {
            "fallback_source": "drone",
            "degradation_time_s": degradation_time_s,
            "safety_maintained": True,
            "communication_lost": True,
        }

        logger.warning(f"Communication loss handled, degradation time: {degradation_time_s:.3f}s")

        return result

    async def get_priority_status(self) -> dict[str, Any]:
        """
        Get current priority status for operator awareness.

        Returns:
            Priority status dictionary
        """
        # Get current active source
        active_source = "unknown"
        if self._coordinator and hasattr(self._coordinator, "active_source"):
            active_source = self._coordinator.active_source

        # Calculate average decision latency
        avg_latency = 0.0
        if self._decision_latencies:
            avg_latency = sum(self._decision_latencies) / len(self._decision_latencies)

        return {
            "active_source": active_source,
            "priority_score": 0.0,  # TODO: Calculate from recent decisions
            "conflict_history": len(self._conflict_history),
            "emergency_override_active": self._emergency_override_active,
            "last_decision_timestamp": self._last_decision_timestamp,
            "average_decision_latency_ms": avg_latency,
            "recent_conflicts": len(
                [
                    c
                    for c in self._conflict_history
                    if time.time() - c.resolution_time_ms / 1000 < 60
                ]
            ),
        }
