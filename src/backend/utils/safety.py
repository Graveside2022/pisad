"""Safety Interlock System for PiSAD.

Provides multiple independent safety mechanisms to ensure flight safety
and maintain operator control at all times.
"""

import asyncio
import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

logger = logging.getLogger(__name__)


class SafetyEventType(Enum):
    """Types of safety events."""

    INTERLOCK_TRIGGERED = "interlock_triggered"
    EMERGENCY_STOP = "emergency_stop"
    SAFETY_OVERRIDE = "safety_override"
    SAFETY_ENABLED = "safety_enabled"
    SAFETY_DISABLED = "safety_disabled"
    SAFETY_WARNING = "safety_warning"


class SafetyTrigger(Enum):
    """Safety event triggers."""

    MODE_CHANGE = "mode_change"
    LOW_BATTERY = "low_battery"
    SIGNAL_LOSS = "signal_loss"
    GEOFENCE_VIOLATION = "geofence_violation"
    OPERATOR_DISABLE = "operator_disable"
    EMERGENCY_STOP = "emergency_stop"
    TIMEOUT = "timeout"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SafetyEvent:
    """Represents a safety event."""

    id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_type: SafetyEventType = SafetyEventType.INTERLOCK_TRIGGERED
    trigger: SafetyTrigger = SafetyTrigger.MODE_CHANGE
    details: dict[str, Any] = field(default_factory=dict)
    resolved: bool = False


class SafetyCheck(ABC):
    """Abstract base class for all safety checks."""

    def __init__(self, name: str):
        """Initialize safety check.

        Args:
            name: Name of the safety check
        """
        self.name = name
        self.is_safe = False
        self.last_check = datetime.now(UTC)
        self.failure_reason: str | None = None

    @abstractmethod
    async def check(self) -> bool:
        """Perform the safety check.

        Returns:
            True if safe, False otherwise
        """
        pass

    def get_status(self) -> dict[str, Any]:
        """Get current status of the safety check.

        Returns:
            Dictionary with check status
        """
        return {
            "name": self.name,
            "is_safe": self.is_safe,
            "last_check": self.last_check.isoformat(),
            "failure_reason": self.failure_reason,
        }


class ModeCheck(SafetyCheck):
    """Validates flight mode is GUIDED."""

    def __init__(self) -> None:
        """Initialize mode check."""
        super().__init__("mode_check")
        self.current_mode = "UNKNOWN"
        self.required_mode = "GUIDED"

    async def check(self) -> bool:
        """Check if flight mode is GUIDED.

        Returns:
            True if in GUIDED mode, False otherwise
        """
        self.last_check = datetime.now(UTC)
        self.is_safe = self.current_mode == self.required_mode

        if not self.is_safe:
            self.failure_reason = f"Mode is {self.current_mode}, requires {self.required_mode}"
        else:
            self.failure_reason = None

        return self.is_safe

    def update_mode(self, mode: str) -> None:
        """Update current flight mode.

        Args:
            mode: Current flight mode
        """
        self.current_mode = mode
        logger.debug(f"Mode updated to: {mode}")


class OperatorActivationCheck(SafetyCheck):
    """Verifies homing is enabled by operator."""

    def __init__(self) -> None:
        """Initialize operator activation check."""
        super().__init__("operator_check")
        self.homing_enabled = False
        self.activation_time: datetime | None = None

    async def check(self) -> bool:
        """Check if operator has enabled homing.

        Returns:
            True if homing enabled, False otherwise
        """
        self.last_check = datetime.now(UTC)
        self.is_safe = self.homing_enabled

        if not self.is_safe:
            self.failure_reason = "Operator has not enabled homing"
        else:
            self.failure_reason = None

        return self.is_safe

    def enable_homing(self) -> None:
        """Enable homing."""
        self.homing_enabled = True
        self.activation_time = datetime.now(UTC)
        logger.info("Homing enabled by operator")

    def disable_homing(self, reason: str = "Manual disable") -> None:
        """Disable homing.

        Args:
            reason: Reason for disabling
        """
        self.homing_enabled = False
        self.activation_time = None
        logger.info(f"Homing disabled: {reason}")


class SignalLossCheck(SafetyCheck):
    """Monitors signal quality with 10-second timeout."""

    def __init__(self, snr_threshold: float = 6.0, timeout_seconds: int = 10):
        """Initialize signal loss check.

        Args:
            snr_threshold: Minimum SNR in dB
            timeout_seconds: Seconds of low signal before triggering
        """
        super().__init__("signal_check")
        self.snr_threshold = snr_threshold
        self.timeout_seconds = timeout_seconds
        self.current_snr = 0.0
        self.signal_lost_time: datetime | None = None
        self.snr_history: list[tuple[datetime, float]] = []
        self.history_window = timedelta(seconds=timeout_seconds)

    async def check(self) -> bool:
        """Check if signal quality is acceptable.

        Returns:
            True if signal quality good, False otherwise
        """
        self.last_check = datetime.now(UTC)

        # Clean old history
        cutoff_time = self.last_check - self.history_window
        self.snr_history = [(t, snr) for t, snr in self.snr_history if t > cutoff_time]

        # Check if signal has been lost for too long
        if self.current_snr < self.snr_threshold:
            if self.signal_lost_time is None:
                self.signal_lost_time = datetime.now(UTC)

            time_lost = (datetime.now(UTC) - self.signal_lost_time).total_seconds()

            if time_lost >= self.timeout_seconds:
                self.is_safe = False
                self.failure_reason = (
                    f"Signal lost for {time_lost:.1f} seconds (SNR: {self.current_snr:.1f} dB)"
                )
            else:
                self.is_safe = True
                self.failure_reason = (
                    f"Signal weak for {time_lost:.1f}s (waiting for {self.timeout_seconds}s)"
                )
        else:
            self.signal_lost_time = None
            self.is_safe = True
            self.failure_reason = None

        return self.is_safe

    def update_snr(self, snr: float) -> None:
        """Update current SNR value.

        Args:
            snr: Signal-to-noise ratio in dB
        """
        self.current_snr = snr
        self.snr_history.append((datetime.now(UTC), snr))

        # Reset timer if signal recovered
        if snr >= self.snr_threshold:
            self.signal_lost_time = None


class BatteryCheck(SafetyCheck):
    """Monitors battery level."""

    def __init__(self, threshold_percent: float = 20.0):
        """Initialize battery check.

        Args:
            threshold_percent: Minimum battery percentage
        """
        super().__init__("battery_check")
        self.threshold_percent = threshold_percent
        self.current_battery_percent = 100.0
        self.warning_levels = [30.0, 25.0, 20.0]
        self.last_warning_level: float | None = None

    async def check(self) -> bool:
        """Check if battery level is safe.

        Returns:
            True if battery above threshold, False otherwise
        """
        self.last_check = datetime.now(UTC)
        self.is_safe = self.current_battery_percent > self.threshold_percent

        if not self.is_safe:
            self.failure_reason = f"Battery at {self.current_battery_percent:.1f}%, below {self.threshold_percent}% threshold"
        else:
            self.failure_reason = None

        # Check for warning levels
        for level in self.warning_levels:
            if self.current_battery_percent <= level and (
                self.last_warning_level is None or level < self.last_warning_level
            ):
                self.last_warning_level = level
                logger.warning(f"Battery warning: {self.current_battery_percent:.1f}% remaining")
                break

        return self.is_safe

    def update_battery(self, percent: float) -> None:
        """Update battery percentage.

        Args:
            percent: Battery percentage (0-100)
        """
        self.current_battery_percent = max(0.0, min(100.0, percent))


class GeofenceCheck(SafetyCheck):
    """Validates position against geofence."""

    def __init__(self) -> None:
        """Initialize geofence check."""
        super().__init__("geofence_check")
        self.center_lat: float | None = None
        self.center_lon: float | None = None
        self.radius_meters: float | None = None
        self.current_lat: float | None = None
        self.current_lon: float | None = None
        self.fence_enabled = False

    async def check(self) -> bool:
        """Check if position is within geofence.

        Returns:
            True if within geofence or fence disabled, False otherwise
        """
        self.last_check = datetime.now(UTC)

        if not self.fence_enabled:
            self.is_safe = True
            self.failure_reason = None
            return True

        if any(
            x is None
            for x in [
                self.center_lat,
                self.center_lon,
                self.radius_meters,
                self.current_lat,
                self.current_lon,
            ]
        ):
            self.is_safe = False
            self.failure_reason = "Geofence or position not configured"
            return False

        # Calculate distance using Haversine formula
        # Type assertions since we checked for None above
        assert self.current_lat is not None
        assert self.current_lon is not None
        assert self.center_lat is not None
        assert self.center_lon is not None
        assert self.radius_meters is not None

        distance = self._calculate_distance(
            self.current_lat, self.current_lon, self.center_lat, self.center_lon
        )

        self.is_safe = distance <= self.radius_meters

        if not self.is_safe:
            self.failure_reason = (
                f"Position {distance:.1f}m from center, exceeds {self.radius_meters}m radius"
            )
        else:
            self.failure_reason = None

        return self.is_safe

    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in meters.

        Args:
            lat1: Latitude of first point
            lon1: Longitude of first point
            lat2: Latitude of second point
            lon2: Longitude of second point

        Returns:
            Distance in meters
        """
        from math import atan2, cos, radians, sin, sqrt

        R = 6371000  # Earth radius in meters

        lat1_rad = radians(lat1)
        lat2_rad = radians(lat2)
        delta_lat = radians(lat2 - lat1)
        delta_lon = radians(lon2 - lon1)

        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return R * c

    def set_geofence(self, center_lat: float, center_lon: float, radius_meters: float) -> None:
        """Set geofence parameters.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_meters: Radius in meters
        """
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.radius_meters = radius_meters
        self.fence_enabled = True
        logger.info(f"Geofence set: center=({center_lat}, {center_lon}), radius={radius_meters}m")

    def update_position(self, lat: float, lon: float) -> None:
        """Update current position.

        Args:
            lat: Current latitude
            lon: Current longitude
        """
        self.current_lat = lat
        self.current_lon = lon


class SafetyInterlockSystem:
    """Main safety interlock system coordinator."""

    def __init__(self) -> None:
        """Initialize safety interlock system."""
        self.checks: dict[str, SafetyCheck] = {
            "mode": ModeCheck(),
            "operator": OperatorActivationCheck(),
            "signal": SignalLossCheck(),
            "battery": BatteryCheck(),
            "geofence": GeofenceCheck(),
        }

        self.emergency_stopped = False
        self.safety_events: list[SafetyEvent] = []
        self.max_events = 1000  # Keep last 1000 events

        self._check_task: asyncio.Task | None = None
        self._check_interval = 0.1  # 100ms for mode detection requirement

        logger.info("Safety interlock system initialized")

    async def start_monitoring(self) -> None:
        """Start continuous safety monitoring."""
        if self._check_task is None or self._check_task.done():
            self._check_task = asyncio.create_task(self._monitor_loop())
            logger.info("Safety monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop safety monitoring."""
        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._check_task
            logger.info("Safety monitoring stopped")

    async def _monitor_loop(self) -> None:
        """Continuous monitoring loop."""
        while True:
            try:
                await self.check_all_safety()
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in safety monitor loop: {e}")
                await asyncio.sleep(self._check_interval)

    async def check_all_safety(self) -> dict[str, bool]:
        """Check all safety interlocks.

        Returns:
            Dictionary of check results
        """
        if self.emergency_stopped:
            return dict.fromkeys(self.checks, False)

        results = {}
        for name, check in self.checks.items():
            try:
                is_safe = await check.check()
                results[name] = is_safe

                # Log safety failures
                if not is_safe and check.failure_reason:
                    await self._log_safety_event(
                        SafetyEventType.INTERLOCK_TRIGGERED,
                        self._get_trigger_for_check(name),
                        {"check": name, "reason": check.failure_reason},
                    )
            except Exception as e:
                logger.error(f"Error in {name} check: {e}")
                results[name] = False

        return results

    async def is_safe_to_proceed(self) -> bool:
        """Check if all safety conditions are met.

        Returns:
            True if safe to proceed, False otherwise
        """
        if self.emergency_stopped:
            return False

        results = await self.check_all_safety()
        return all(results.values())

    async def emergency_stop(self, reason: str = "Manual emergency stop") -> None:
        """Activate emergency stop.

        Args:
            reason: Reason for emergency stop
        """
        self.emergency_stopped = True

        # Disable all systems
        if "operator" in self.checks:
            operator_check = self.checks["operator"]
            if isinstance(operator_check, OperatorActivationCheck):
                operator_check.disable_homing("Emergency stop")

        await self._log_safety_event(
            SafetyEventType.EMERGENCY_STOP, SafetyTrigger.EMERGENCY_STOP, {"reason": reason}
        )

        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

    async def reset_emergency_stop(self) -> None:
        """Reset emergency stop (requires manual intervention)."""
        self.emergency_stopped = False

        await self._log_safety_event(
            SafetyEventType.SAFETY_OVERRIDE,
            SafetyTrigger.MANUAL_OVERRIDE,
            {"action": "Emergency stop reset"},
        )

        logger.info("Emergency stop reset")

    async def enable_homing(self, confirmation_token: str | None = None) -> bool:
        """Enable homing if safe.

        Args:
            confirmation_token: Optional confirmation token

        Returns:
            True if enabled, False if blocked by safety
        """
        if self.emergency_stopped:
            logger.warning("Cannot enable homing: Emergency stop active")
            return False

        # Check all safety conditions except operator activation
        safety_ok = True
        for name, check in self.checks.items():
            if name != "operator":
                is_safe = await check.check()
                if not is_safe:
                    safety_ok = False
                    logger.warning(
                        f"Cannot enable homing: {name} check failed - {check.failure_reason}"
                    )

        if safety_ok:
            operator_check = self.checks["operator"]
            if isinstance(operator_check, OperatorActivationCheck):
                operator_check.enable_homing()

                await self._log_safety_event(
                    SafetyEventType.SAFETY_ENABLED,
                    SafetyTrigger.MANUAL_OVERRIDE,
                    {"system": "homing", "token": confirmation_token},
                )

                return True

        return False

    async def disable_homing(self, reason: str) -> None:
        """Disable homing.

        Args:
            reason: Reason for disabling
        """
        operator_check = self.checks["operator"]
        if isinstance(operator_check, OperatorActivationCheck):
            operator_check.disable_homing(reason)

            await self._log_safety_event(
                SafetyEventType.SAFETY_DISABLED,
                SafetyTrigger.OPERATOR_DISABLE,
                {"system": "homing", "reason": reason},
            )

    def update_flight_mode(self, mode: str) -> None:
        """Update current flight mode.

        Args:
            mode: Flight mode string
        """
        mode_check = self.checks.get("mode")
        if isinstance(mode_check, ModeCheck):
            mode_check.update_mode(mode)

    def update_battery(self, percent: float) -> None:
        """Update battery percentage.

        Args:
            percent: Battery percentage
        """
        battery_check = self.checks.get("battery")
        if isinstance(battery_check, BatteryCheck):
            battery_check.update_battery(percent)

    def update_signal_snr(self, snr: float) -> None:
        """Update signal SNR.

        Args:
            snr: Signal-to-noise ratio in dB
        """
        signal_check = self.checks.get("signal")
        if isinstance(signal_check, SignalLossCheck):
            signal_check.update_snr(snr)

    def update_position(self, lat: float, lon: float) -> None:
        """Update current position.

        Args:
            lat: Latitude
            lon: Longitude
        """
        geofence_check = self.checks.get("geofence")
        if isinstance(geofence_check, GeofenceCheck):
            geofence_check.update_position(lat, lon)

    def get_safety_status(self) -> dict[str, Any]:
        """Get comprehensive safety status.

        Returns:
            Dictionary with all safety information
        """
        return {
            "emergency_stopped": self.emergency_stopped,
            "checks": {name: check.get_status() for name, check in self.checks.items()},
            "timestamp": datetime.now(UTC).isoformat(),
        }

    def get_safety_events(
        self, since: datetime | None = None, limit: int = 100
    ) -> list[SafetyEvent]:
        """Get safety events.

        Args:
            since: Get events after this time
            limit: Maximum number of events

        Returns:
            List of safety events
        """
        events = self.safety_events

        if since:
            events = [e for e in events if e.timestamp > since]

        return events[-limit:]

    async def _log_safety_event(
        self, event_type: SafetyEventType, trigger: SafetyTrigger, details: dict[str, Any]
    ) -> None:
        """Log a safety event.

        Args:
            event_type: Type of event
            trigger: What triggered the event
            details: Event details
        """
        event = SafetyEvent(event_type=event_type, trigger=trigger, details=details)

        self.safety_events.append(event)

        # Trim events list if too long
        if len(self.safety_events) > self.max_events:
            self.safety_events = self.safety_events[-self.max_events :]

        logger.info(f"Safety event: {event_type.value} triggered by {trigger.value}")

    def _get_trigger_for_check(self, check_name: str) -> SafetyTrigger:
        """Get appropriate trigger for a check name.

        Args:
            check_name: Name of the check

        Returns:
            Corresponding trigger
        """
        trigger_map = {
            "mode": SafetyTrigger.MODE_CHANGE,
            "operator": SafetyTrigger.OPERATOR_DISABLE,
            "signal": SafetyTrigger.SIGNAL_LOSS,
            "battery": SafetyTrigger.LOW_BATTERY,
            "geofence": SafetyTrigger.GEOFENCE_VIOLATION,
        }
        return trigger_map.get(check_name, SafetyTrigger.MANUAL_OVERRIDE)
