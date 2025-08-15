"""Signal State Controller for debounced state transitions.

This module implements PRD FR7 - debounced state transitions with
12dB trigger threshold and 6dB drop threshold for hysteresis.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np

from src.backend.core.exceptions import (
    StateTransitionError,
)
from src.backend.models.schemas import DetectionEvent
from src.backend.services.state_machine import StateMachine
from src.backend.utils.logging import get_logger

logger = get_logger(__name__)


class SignalState(Enum):
    """Signal detection states with hysteresis."""

    NO_SIGNAL = "no_signal"
    RISING = "rising"  # Signal detected but not yet confirmed
    CONFIRMED = "confirmed"  # Signal confirmed above trigger threshold
    FALLING = "falling"  # Signal dropping but not yet lost
    LOST = "lost"  # Signal dropped below drop threshold


@dataclass
class SignalTransitionEvent:
    """Records state transition events for audit logging."""

    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    from_state: SignalState = SignalState.NO_SIGNAL
    to_state: SignalState = SignalState.NO_SIGNAL
    trigger_snr: float = 0.0
    current_rssi: float = -100.0
    noise_floor: float = -100.0
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class SignalStateController:
    """Manages signal state transitions with hysteresis and debouncing.

    Implements PRD requirements:
    - FR6: Noise floor estimation using 10th percentile
    - FR7: Debounced state transitions (12dB trigger, 6dB drop)
    - NFR7: Signal anomaly detection (<5% false positive rate)
    - FR12: Command audit logging for state transitions
    """

    def __init__(
        self,
        trigger_threshold: float = 12.0,
        drop_threshold: float = 6.0,
        confirmation_time: float = 0.5,
        drop_time: float = 1.0,
        anomaly_window_size: int = 100,
        anomaly_threshold: float = 3.0,  # Z-score threshold for anomaly detection
    ):
        """Initialize signal state controller.

        Args:
            trigger_threshold: SNR threshold to trigger detection (dB)
            drop_threshold: SNR threshold to drop detection (dB)
            confirmation_time: Time to confirm signal before transitioning (seconds)
            drop_time: Time to wait before declaring signal lost (seconds)
            anomaly_window_size: Window size for anomaly detection
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.trigger_threshold = trigger_threshold
        self.drop_threshold = drop_threshold
        self.confirmation_time = confirmation_time
        self.drop_time = drop_time

        # Current state
        self.current_state = SignalState.NO_SIGNAL
        self.state_start_time = datetime.now(UTC)

        # Signal history for anomaly detection
        self.signal_history: deque[float] = deque(maxlen=anomaly_window_size)
        self.anomaly_threshold = anomaly_threshold

        # Transition events for audit logging (FR12)
        self.transition_events: deque[SignalTransitionEvent] = deque(maxlen=1000)

        # State machine reference
        self.state_machine: StateMachine | None = None

        # Statistics for false positive tracking (NFR7)
        self.total_detections = 0
        self.false_positives = 0
        self.true_positives = 0

        logger.info(
            f"SignalStateController initialized: trigger={trigger_threshold}dB, "
            f"drop={drop_threshold}dB, confirm={confirmation_time}s, drop_time={drop_time}s"
        )

    def set_state_machine(self, state_machine: StateMachine) -> None:
        """Set reference to state machine for triggering transitions.

        Args:
            state_machine: State machine instance
        """
        self.state_machine = state_machine
        logger.info("State machine connected to signal state controller")

    async def process_signal(
        self, rssi: float, noise_floor: float
    ) -> tuple[SignalState, DetectionEvent | None]:
        """Process signal reading and manage state transitions.

        Args:
            rssi: Current RSSI value in dBm
            noise_floor: Current noise floor estimate in dBm

        Returns:
            Tuple of (new_state, detection_event if triggered)
        """
        snr = rssi - noise_floor

        # Add to history for anomaly detection
        self.signal_history.append(snr)

        # Check for anomalies
        is_anomaly = self._detect_anomaly(snr)

        # Get time in current state
        time_in_state = (datetime.now(UTC) - self.state_start_time).total_seconds()

        # State transition logic with hysteresis
        old_state = self.current_state
        new_state = old_state
        detection_event = None
        reason = ""

        if old_state == SignalState.NO_SIGNAL:
            if snr >= self.trigger_threshold and not is_anomaly:
                new_state = SignalState.RISING
                reason = f"SNR {snr:.1f}dB exceeds trigger threshold {self.trigger_threshold}dB"

        elif old_state == SignalState.RISING:
            if snr < self.drop_threshold:
                new_state = SignalState.NO_SIGNAL
                reason = f"SNR {snr:.1f}dB dropped below {self.drop_threshold}dB during rise"
                self.false_positives += 1
            elif time_in_state >= self.confirmation_time:
                new_state = SignalState.CONFIRMED
                reason = f"Signal confirmed after {self.confirmation_time}s"
                self.true_positives += 1
                self.total_detections += 1

                # Create detection event
                detection_event = DetectionEvent(
                    id=str(uuid4()),
                    timestamp=datetime.now(UTC),
                    frequency=0.0,  # Will be set by SDR service
                    rssi=rssi,
                    snr=snr,
                    confidence=self._calculate_confidence(snr, is_anomaly),
                    location=None,  # Will be set by MAVLink service
                    state="active",
                )

        elif old_state == SignalState.CONFIRMED:
            if snr < self.drop_threshold:
                new_state = SignalState.FALLING
                reason = f"SNR {snr:.1f}dB dropped below {self.drop_threshold}dB"

        elif old_state == SignalState.FALLING:
            if snr >= self.trigger_threshold:
                new_state = SignalState.CONFIRMED
                reason = f"SNR {snr:.1f}dB recovered above {self.trigger_threshold}dB"
            elif time_in_state >= self.drop_time:
                new_state = SignalState.LOST
                reason = f"Signal lost after {self.drop_time}s below threshold"

        elif old_state == SignalState.LOST:
            if snr >= self.trigger_threshold and not is_anomaly:
                new_state = SignalState.RISING
                reason = f"New signal detected at {snr:.1f}dB"
            elif time_in_state >= 2.0:  # Reset to NO_SIGNAL after 2 seconds
                new_state = SignalState.NO_SIGNAL
                reason = "Reset to NO_SIGNAL after timeout"

        # Handle state transition
        if new_state != old_state:
            await self._handle_transition(old_state, new_state, snr, rssi, noise_floor, reason)

        return new_state, detection_event

    async def _handle_transition(
        self,
        old_state: SignalState,
        new_state: SignalState,
        snr: float,
        rssi: float,
        noise_floor: float,
        reason: str,
    ) -> None:
        """Handle state transition and trigger appropriate actions.

        Args:
            old_state: Previous signal state
            new_state: New signal state
            snr: Current SNR in dB
            rssi: Current RSSI in dBm
            noise_floor: Current noise floor in dBm
            reason: Reason for transition
        """
        # Update state
        self.current_state = new_state
        self.state_start_time = datetime.now(UTC)

        # Log transition event for audit (FR12)
        event = SignalTransitionEvent(
            from_state=old_state,
            to_state=new_state,
            trigger_snr=snr,
            current_rssi=rssi,
            noise_floor=noise_floor,
            reason=reason,
            metadata={
                "false_positive_rate": self.get_false_positive_rate(),
                "total_detections": self.total_detections,
            },
        )
        self.transition_events.append(event)

        logger.info(
            f"Signal state transition: {old_state.value} -> {new_state.value} "
            f"(SNR: {snr:.1f}dB, Reason: {reason})"
        )

        # Trigger state machine transitions if connected
        if self.state_machine:
            try:
                if new_state == SignalState.CONFIRMED:
                    # Signal detected and confirmed
                    await self.state_machine.handle_detection(
                        rssi, self._calculate_confidence(snr, False)
                    )
                elif new_state == SignalState.LOST:
                    # Signal lost
                    await self.state_machine.handle_signal_lost()
            except StateTransitionError as e:
                logger.error(f"Error triggering state machine transition: {e}")

    def _detect_anomaly(self, snr: float) -> bool:
        """Detect signal anomalies using statistical analysis.

        Implements NFR7 - maintains <5% false positive rate.

        Args:
            snr: Current SNR value in dB

        Returns:
            True if anomaly detected, False otherwise
        """
        if len(self.signal_history) < 10:
            return False  # Not enough data for anomaly detection

        # Calculate statistics
        mean = np.mean(list(self.signal_history))
        std = np.std(list(self.signal_history))

        if std < 0.1:  # Very stable signal
            return False

        # Calculate Z-score
        z_score = abs((snr - mean) / std)

        # Detect sudden spikes (anomaly)
        is_anomaly = z_score > self.anomaly_threshold

        if is_anomaly:
            logger.debug(
                f"Anomaly detected: SNR={snr:.1f}dB, Z-score={z_score:.2f}, "
                f"mean={mean:.1f}dB, std={std:.1f}dB"
            )

        return is_anomaly

    def _calculate_confidence(self, snr: float, is_anomaly: bool) -> float:
        """Calculate detection confidence based on SNR and anomaly status.

        Args:
            snr: Signal-to-noise ratio in dB
            is_anomaly: Whether signal is anomalous

        Returns:
            Confidence percentage (0-100)
        """
        if is_anomaly:
            return 0.0  # Zero confidence for anomalies

        # Base confidence on SNR above trigger threshold
        if snr < self.trigger_threshold:
            return 0.0

        # Linear scaling from 50% to 100% based on SNR
        snr_above_threshold = snr - self.trigger_threshold
        confidence = min(100.0, 50.0 + snr_above_threshold * 5.0)

        return confidence

    def get_false_positive_rate(self) -> float:
        """Get current false positive rate.

        Returns:
            False positive rate as percentage
        """
        total = self.true_positives + self.false_positives
        if total == 0:
            return 0.0

        return (self.false_positives / total) * 100.0

    def get_statistics(self) -> dict[str, Any]:
        """Get controller statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "current_state": self.current_state.value,
            "time_in_state": (datetime.now(UTC) - self.state_start_time).total_seconds(),
            "total_detections": self.total_detections,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_positive_rate": self.get_false_positive_rate(),
            "trigger_threshold": self.trigger_threshold,
            "drop_threshold": self.drop_threshold,
            "signal_history_size": len(self.signal_history),
        }

    def get_transition_log(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent transition events for audit logging.

        Args:
            limit: Maximum number of events to return

        Returns:
            List of transition events
        """
        events = list(self.transition_events)[-limit:]

        return [
            {
                "id": event.id,
                "timestamp": event.timestamp.isoformat(),
                "from_state": event.from_state.value,
                "to_state": event.to_state.value,
                "trigger_snr": event.trigger_snr,
                "rssi": event.current_rssi,
                "noise_floor": event.noise_floor,
                "reason": event.reason,
                "metadata": event.metadata,
            }
            for event in events
        ]

    def reset(self) -> None:
        """Reset controller state."""
        self.current_state = SignalState.NO_SIGNAL
        self.state_start_time = datetime.now(UTC)
        self.signal_history.clear()
        self.transition_events.clear()
        self.total_detections = 0
        self.false_positives = 0
        self.true_positives = 0
        logger.info("Signal state controller reset")
