"""ASV Degradation Recovery Strategies Implementation.

SUBTASK-6.1.2.3 [16b] - Implement RSSI degradation recovery strategies using ASV signal confidence metrics

This module implements intelligent recovery strategies when signal quality degrades, leveraging
ASV's professional signal confidence metrics and integrating with existing safety authority hierarchy.

Recovery strategies include:
- Degradation detection using ASV signal quality confidence thresholds
- Return-to-last-good-position algorithm
- Spiral search expansion patterns
- Safety authority integration and validation
- Recovery event logging and operator notifications
"""

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol

from src.backend.services.homing_algorithm import VelocityCommand
from src.backend.services.safety_authority_manager import (
    SafetyAuthorityLevel,
    SafetyAuthorityManager,
)

logger = logging.getLogger(__name__)


class ASVMetricsProtocol(Protocol):
    """Protocol for ASV signal metrics compatibility."""
    
    confidence: float
    signal_strength_dbm: float
    interference_detected: bool
    processing_time_ms: float
    bearing_precision_deg: float


class SafetyManagerProtocol(Protocol):
    """Protocol for safety manager compatibility."""
    
    def get_current_authority_level(self) -> SafetyAuthorityLevel:
        ...
    
    def validate_command(self, command: Any) -> bool:
        ...


class DegradationSeverity(Enum):
    """Severity levels for signal degradation events."""

    MINOR = "MINOR"
    MODERATE = "MODERATE"
    SIGNIFICANT = "SIGNIFICANT"
    CRITICAL = "CRITICAL"


class RecoveryStrategy(Enum):
    """Available recovery strategies for signal degradation."""

    CONTINUE_GRADIENT_CLIMB = "continue_gradient_climb"
    RETURN_TO_LAST_GOOD = "return_to_last_good"
    SPIRAL_SEARCH = "spiral_search"
    S_TURN_SAMPLING = "s_turn_sampling"


class NotificationPriority(Enum):
    """Priority levels for operator notifications."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DegradationEvent:
    """Event representing detected signal degradation."""

    timestamp: float
    is_degrading: bool
    confidence_trend: float  # Rate of confidence change
    severity: DegradationSeverity
    trigger_recovery: bool
    effective_confidence: float | None = None
    interference_penalty_applied: bool = False


@dataclass
class LastGoodPosition:
    """Position record with good signal quality."""

    x: float
    y: float
    confidence: float
    timestamp: float


@dataclass
class SpiralSearchPattern:
    """Spiral search pattern parameters."""

    center_x: float
    center_y: float
    initial_radius: float
    radius_increment: float
    waypoints: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class RecoveryAction:
    """Recovery action with strategy and parameters."""

    strategy: RecoveryStrategy
    target_position: LastGoodPosition | None = None
    velocity_command: VelocityCommand | None = None
    spiral_pattern: SpiralSearchPattern | None = None
    estimated_time_seconds: float = 0.0
    safety_validated: bool = False
    authority_level: SafetyAuthorityLevel | None = None


class RecoveryBlockedException(Exception):
    """Exception raised when recovery is blocked by safety authority."""

    pass


class ASVDegradationDetector:
    """Detects signal degradation using ASV confidence metrics."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        trend_window_size: int = 5,
        degradation_rate_threshold: float = 0.15,
        interference_penalty: float = 0.2,
    ):
        """Initialize degradation detector.

        Args:
            confidence_threshold: Minimum acceptable confidence level
            trend_window_size: Number of samples for trend analysis
            degradation_rate_threshold: Rate of degradation to trigger detection
            interference_penalty: Confidence penalty for interference detection
        """
        self.confidence_threshold = confidence_threshold
        self.trend_window_size = trend_window_size
        self.degradation_rate_threshold = degradation_rate_threshold
        self.interference_penalty = interference_penalty

        self._confidence_history: deque[float] = deque(maxlen=trend_window_size)

    def analyze_degradation(
        self, asv_metrics: ASVMetricsProtocol
    ) -> DegradationEvent | None:
        """Analyze ASV metrics for signal degradation.

        Args:
            asv_metrics: Current ASV signal processing metrics

        Returns:
            DegradationEvent if degradation detected, None otherwise
        """
        # Apply interference penalty if detected
        effective_confidence = asv_metrics.confidence
        interference_penalty_applied = False

        if asv_metrics.interference_detected:
            effective_confidence -= self.interference_penalty
            interference_penalty_applied = True

        # Add to history
        self._confidence_history.append(effective_confidence)

        # Check immediate threshold breach even with single sample
        immediate_breach = effective_confidence < self.confidence_threshold

        # Need enough samples for trend analysis (but not for immediate threshold breach)
        if len(self._confidence_history) < 2 and not immediate_breach:
            return None

        # Calculate confidence trend
        confidence_values = list(self._confidence_history)
        if len(confidence_values) >= 2:
            confidence_trend = confidence_values[-1] - confidence_values[0]
        else:
            confidence_trend = 0.0

        # Determine if degrading
        is_degrading = effective_confidence < self.confidence_threshold or (
            len(self._confidence_history) >= 2
            and confidence_trend < -self.degradation_rate_threshold
        )

        if not is_degrading:
            return None

        # Determine severity
        if effective_confidence < 0.1 or confidence_trend < -0.4:
            severity = DegradationSeverity.CRITICAL
        elif effective_confidence < 0.2 or confidence_trend < -0.15:
            severity = DegradationSeverity.SIGNIFICANT
        elif effective_confidence < 0.3 or confidence_trend < -0.1:
            severity = DegradationSeverity.MODERATE
        else:
            severity = DegradationSeverity.MINOR

        return DegradationEvent(
            timestamp=time.time(),
            is_degrading=True,
            confidence_trend=confidence_trend,
            severity=severity,
            trigger_recovery=severity
            in [DegradationSeverity.SIGNIFICANT, DegradationSeverity.CRITICAL],
            effective_confidence=effective_confidence,
            interference_penalty_applied=interference_penalty_applied,
        )


class ASVRecoveryManager:
    """Manages recovery strategies for signal degradation."""

    def __init__(
        self,
        safety_manager: SafetyManagerProtocol | None = None,
        event_logger: Optional["RecoveryEventLogger"] = None,
        operator_notifier: Optional["OperatorNotifier"] = None,
    ):
        """Initialize recovery manager.

        Args:
            safety_manager: Safety authority manager for validation
            event_logger: Logger for recovery events
            operator_notifier: Notifier for operator alerts
        """
        self.safety_manager = safety_manager
        self.event_logger = event_logger
        self.operator_notifier = operator_notifier

        self._good_positions: deque[LastGoodPosition] = deque(maxlen=10)

    def record_good_position(self, position: LastGoodPosition) -> None:
        """Record a position with good signal quality.

        Args:
            position: Position with good signal metrics
        """
        self._good_positions.append(position)

    def generate_recovery_action(
        self,
        strategy: RecoveryStrategy,
        current_position: tuple[float, float],
        signal_loss_severity: DegradationSeverity,
        notify_operator: bool = False,
    ) -> RecoveryAction:
        """Generate recovery action for given strategy.

        Args:
            strategy: Recovery strategy to use
            current_position: Current drone position (x, y)
            signal_loss_severity: Severity of signal loss
            notify_operator: Whether to notify operator

        Returns:
            RecoveryAction with strategy parameters
        """
        recovery_action = None
        if strategy == RecoveryStrategy.RETURN_TO_LAST_GOOD:
            recovery_action = self._generate_return_to_last_good(
                current_position, signal_loss_severity
            )
        elif strategy == RecoveryStrategy.SPIRAL_SEARCH:
            recovery_action = self._generate_spiral_search(current_position, signal_loss_severity)
        else:
            # Default fallback
            recovery_action = RecoveryAction(strategy=strategy, estimated_time_seconds=10.0)

        # Send operator notification if requested
        if notify_operator and self.operator_notifier:
            priority = (
                NotificationPriority.HIGH
                if signal_loss_severity
                in [DegradationSeverity.SIGNIFICANT, DegradationSeverity.CRITICAL]
                else NotificationPriority.MEDIUM
            )

            message = f"Signal degradation detected - initiating {strategy.value} recovery strategy"
            self.operator_notifier.send_notification(message=message, priority=priority)

        return recovery_action

    async def generate_safe_recovery_action(
        self,
        strategy: RecoveryStrategy,
        current_position: tuple[float, float],
        degradation_severity: DegradationSeverity,
    ) -> RecoveryAction:
        """Generate recovery action with safety validation.

        Args:
            strategy: Recovery strategy to use
            current_position: Current position
            degradation_severity: Severity of degradation

        Returns:
            Safety-validated recovery action

        Raises:
            RecoveryBlockedException: If recovery blocked by safety authority
        """
        if not self.safety_manager:
            # No safety manager - generate basic action
            return self.generate_recovery_action(strategy, current_position, degradation_severity)

        # Check safety authority level
        authority_level = self.safety_manager.get_current_authority_level()

        # Emergency stop blocks all recovery
        if authority_level == SafetyAuthorityLevel.EMERGENCY_STOP:
            raise RecoveryBlockedException("Recovery blocked by EMERGENCY_STOP authority level")

        # Generate recovery action
        recovery_action = self.generate_recovery_action(
            strategy, current_position, degradation_severity
        )

        # Validate with safety manager
        is_valid = self.safety_manager.validate_command(recovery_action.velocity_command)

        if not is_valid:
            raise RecoveryBlockedException("Recovery action failed safety validation")

        # Mark as safety validated
        recovery_action.safety_validated = True
        recovery_action.authority_level = authority_level

        return recovery_action

    def _generate_return_to_last_good(
        self, current_position: tuple[float, float], severity: DegradationSeverity
    ) -> RecoveryAction:
        """Generate return-to-last-good-position recovery action."""
        # Find most recent good position
        if not self._good_positions:
            # No good positions recorded - use current position
            target = LastGoodPosition(
                x=current_position[0], y=current_position[1], confidence=0.5, timestamp=time.time()
            )
        else:
            target = self._good_positions[-1]  # Most recent

        # Calculate velocity command to target
        dx = target.x - current_position[0]
        dy = target.y - current_position[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 0.1:
            heading = math.atan2(dy, dx)
            velocity = VelocityCommand(
                forward_velocity=min(3.0, distance * 0.5),  # Scale with distance
                yaw_rate=heading * 0.3,  # Proportional yaw control
            )
        else:
            velocity = VelocityCommand(forward_velocity=0.0, yaw_rate=0.0)

        estimated_time = (
            distance / max(velocity.forward_velocity, 0.1) if velocity.forward_velocity > 0 else 5.0
        )

        return RecoveryAction(
            strategy=RecoveryStrategy.RETURN_TO_LAST_GOOD,
            target_position=target,
            velocity_command=velocity,
            estimated_time_seconds=estimated_time,
        )

    def _generate_spiral_search(
        self, current_position: tuple[float, float], severity: DegradationSeverity
    ) -> RecoveryAction:
        """Generate spiral search recovery action."""
        # Determine search parameters based on severity
        if severity == DegradationSeverity.CRITICAL:
            initial_radius = 20.0
            radius_increment = 10.0
        elif severity == DegradationSeverity.SIGNIFICANT:
            initial_radius = 15.0
            radius_increment = 7.5
        else:
            initial_radius = 10.0
            radius_increment = 5.0

        # Generate spiral waypoints
        waypoints = []
        for i in range(8):  # 8 points around circle
            angle = i * (2 * math.pi / 8)
            x = current_position[0] + initial_radius * math.cos(angle)
            y = current_position[1] + initial_radius * math.sin(angle)
            waypoints.append((x, y))

        pattern = SpiralSearchPattern(
            center_x=current_position[0],
            center_y=current_position[1],
            initial_radius=initial_radius,
            radius_increment=radius_increment,
            waypoints=waypoints,
        )

        # Initial velocity command toward first waypoint
        if waypoints:
            dx = waypoints[0][0] - current_position[0]
            dy = waypoints[0][1] - current_position[1]
            heading = math.atan2(dy, dx)
            velocity = VelocityCommand(forward_velocity=2.0, yaw_rate=heading * 0.4)
        else:
            velocity = VelocityCommand(forward_velocity=1.0, yaw_rate=0.2)

        return RecoveryAction(
            strategy=RecoveryStrategy.SPIRAL_SEARCH,
            spiral_pattern=pattern,
            velocity_command=velocity,
            estimated_time_seconds=len(waypoints) * 10.0,  # Estimate 10s per waypoint
        )


@dataclass
class RecoveryEvent:
    """Recovery event for logging."""

    timestamp: float
    event_type: str
    strategy: RecoveryStrategy | None = None
    severity: DegradationSeverity | None = None
    estimated_time_seconds: float = 0.0
    trigger_recovery: bool | None = None


class RecoveryEventLogger:
    """Logs degradation and recovery events."""

    def __init__(self) -> None:
        """Initialize event logger."""
        self._events: list[RecoveryEvent] = []

    def log_degradation_event(self, event: DegradationEvent) -> None:
        """Log a degradation event.

        Args:
            event: Degradation event to log
        """
        recovery_event = RecoveryEvent(
            timestamp=event.timestamp,
            event_type="degradation",
            severity=event.severity,
            trigger_recovery=event.trigger_recovery,
        )
        self._events.append(recovery_event)

        logger.info(
            f"Degradation detected: severity={event.severity}, "
            f"confidence_trend={event.confidence_trend:.3f}, "
            f"trigger_recovery={event.trigger_recovery}"
        )

    def log_recovery_action(self, action: RecoveryAction) -> None:
        """Log a recovery action.

        Args:
            action: Recovery action to log
        """
        recovery_event = RecoveryEvent(
            timestamp=time.time(),
            event_type="recovery",
            strategy=action.strategy,
            estimated_time_seconds=action.estimated_time_seconds,
        )
        self._events.append(recovery_event)

        logger.info(
            f"Recovery action initiated: strategy={action.strategy}, "
            f"estimated_time={action.estimated_time_seconds:.1f}s"
        )

    def get_recent_events(self, limit: int = 50) -> list[RecoveryEvent]:
        """Get recent recovery events.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of recent recovery events
        """
        return self._events[-limit:] if self._events else []


class OperatorNotifier:
    """Handles operator notifications for recovery events."""

    def __init__(self) -> None:
        """Initialize operator notifier."""
        self._notifications: list[dict[str, Any]] = []

    def send_notification(
        self, message: str, priority: NotificationPriority, **kwargs: Any
    ) -> None:
        """Send notification to operator.

        Args:
            message: Notification message
            priority: Priority level
            **kwargs: Additional notification parameters
        """
        notification = {
            "timestamp": time.time(),
            "message": message,
            "priority": priority,
            **kwargs,
        }
        self._notifications.append(notification)

        logger.warning(f"Operator notification [{priority}]: {message}")


# SafetyIntegratedRecovery is an alias for ASVRecoveryManager for test compatibility
SafetyIntegratedRecovery = ASVRecoveryManager
