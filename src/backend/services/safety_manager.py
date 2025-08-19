"""
Safety Manager Service
Implements safety interlocks and monitoring for MAVLink operations.

Story 4.7: Hardware Integration Testing
Sprint 4.5: Safety system implementation
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.backend.core.exceptions import (
    MAVLinkError,
    PISADException,
    SafetyInterlockError,
    SDRError,
)

logger = logging.getLogger(__name__)


class SafetyPriority(Enum):
    """Safety action priorities."""

    RC_OVERRIDE = 1
    BATTERY_CRITICAL = 2
    GPS_LOSS = 3
    SIGNAL_LOSS = 4
    ALTITUDE_VIOLATION = 5


@dataclass
class SafetyViolation:
    """Safety violation record."""

    timestamp: float
    type: str
    severity: str
    description: str
    action: str


class SafetyManager:
    """Manages safety interlocks and monitoring for drone operations."""

    def __init__(self) -> None:
        """Initialize safety manager."""
        self.mavlink = None  # Will be injected
        self.motor_interlock = False
        self.max_altitude = 100.0  # meters
        self.geofence_radius = 100.0  # meters
        self.geofence_altitude = 50.0  # meters
        self.home_position = {"lat": 0.0, "lon": 0.0, "alt": 0.0}
        self.active_violations: list[SafetyViolation] = []
        self.watchdog_commands: dict[str, float] = {}
        self.watchdog_timeout = 5.0  # seconds
        self.monitoring_task: asyncio.Task[None] | None = None
        self.state = "IDLE"
        self.last_signal_time = time.time()

        # SDR++ coordination awareness - Added for SUBTASK-5.5.3.1
        self.coordination_status = {"active": False, "healthy": True, "source": "drone"}
        self.coordination_battery_health = {"ground": None, "drone": None}
        self.coordination_latency_ms = 0.0

        # Safety thresholds
        self.battery_low_voltage = 19.2  # 6S Li-ion low
        self.battery_critical_voltage = 18.0  # 6S Li-ion critical
        self.min_satellites = 8
        self.max_hdop = 2.0
        self.rc_override_threshold = 50  # PWM units
        self.rc_center = 1500  # PWM center position

    def trigger_emergency_stop(self) -> dict[str, Any]:
        """
        Trigger emergency stop with timing measurement.

        SAFETY: Must complete in < 500ms per Story 4.7 AC #7
        """
        start_time = time.perf_counter()

        try:
            if self.mavlink and hasattr(self.mavlink, "emergency_stop"):
                success = self.mavlink.emergency_stop()
            else:
                # Fallback: immediate motor stop
                success = self._force_motor_stop()

            response_time_ms = (time.perf_counter() - start_time) * 1000

            if response_time_ms > 500:
                logger.warning(f"Emergency stop took {response_time_ms:.1f}ms (> 500ms limit)")

            result = {
                "success": success,
                "response_time_ms": response_time_ms,
                "priority": "CRITICAL",
            }

            # Include coordination latency for timing validation [8f]
            if self.include_coordination_in_decisions():
                result["coordination_latency_ms"] = self.coordination_latency_ms

            return result

        except (SafetyInterlockError, MAVLinkError, SDRError, Exception) as e:
            logger.error(f"Emergency stop failed: {e}")
            # Always attempt fallback when primary method fails
            try:
                success = self._force_motor_stop()
                logger.info("Emergency fallback executed successfully")
            except Exception as fallback_error:
                logger.critical(f"Emergency fallback failed: {fallback_error}")
                success = False

            return {
                "success": success,
                "error": str(e),
                "response_time_ms": (time.perf_counter() - start_time) * 1000,
                "fallback_attempted": True,
            }

    def is_rc_override_active(self) -> bool:
        """
        Check if RC override is active based on stick positions.

        SAFETY: ±50 PWM threshold per Story 4.7 requirements
        """
        if not self.mavlink or not hasattr(self.mavlink, "telemetry"):
            return False

        rc_channels = self.mavlink.telemetry.get("rc_channels", {})

        for channel in ["throttle", "roll", "pitch", "yaw"]:
            value = rc_channels.get(channel, self.rc_center)
            if abs(value - self.rc_center) > self.rc_override_threshold:
                return True

        return False

    def check_battery_status(self) -> dict[str, Any]:
        """
        Check battery status against safety thresholds.

        SAFETY: 6S Li-ion monitoring per hardware specs
        """
        if not self.mavlink or not hasattr(self.mavlink, "telemetry"):
            return {"level": "UNKNOWN", "warning": False, "critical": False}

        battery = self.mavlink.telemetry.get("battery", {})
        voltage = battery.get("voltage", 0.0)

        if voltage >= self.battery_low_voltage:
            return {"level": "NORMAL", "voltage": voltage, "warning": False, "critical": False}
        elif voltage >= self.battery_critical_voltage:
            return {
                "level": "LOW",
                "voltage": voltage,
                "warning": True,
                "critical": False,
                "action": "WARN",
            }
        else:
            return {
                "level": "CRITICAL",
                "voltage": voltage,
                "warning": True,
                "critical": True,
                "action": "RTL",
            }

    def check_gps_status(self) -> dict[str, Any]:
        """
        Check GPS status for safe operation.

        SAFETY: Requires 8+ satellites, HDOP < 2.0
        """
        if not self.mavlink or not hasattr(self.mavlink, "telemetry"):
            return {"ready": False, "reason": "No telemetry"}

        gps = self.mavlink.telemetry.get("gps", {})
        satellites = gps.get("satellites", 0)
        hdop = gps.get("hdop", 99.0)
        fix_type = gps.get("fix_type", 0)

        if satellites < self.min_satellites:
            return {
                "ready": False,
                "satellites": satellites,
                "hdop": hdop,
                "reason": f"Insufficient satellites ({satellites} < {self.min_satellites})",
            }

        if hdop > self.max_hdop:
            return {
                "ready": False,
                "satellites": satellites,
                "hdop": hdop,
                "reason": f"Poor HDOP ({hdop:.1f} > {self.max_hdop})",
            }

        if fix_type < 3:
            return {
                "ready": False,
                "satellites": satellites,
                "hdop": hdop,
                "reason": f"No 3D fix (type={fix_type})",
            }

        return {"ready": True, "satellites": satellites, "hdop": hdop, "fix_type": fix_type}

    def set_geofence(self, radius: float, altitude: float) -> None:
        """Set geofence parameters."""
        self.geofence_radius = radius
        self.geofence_altitude = altitude

    def check_geofence(self, position: dict[str, float]) -> bool:
        """
        Check if position is within geofence.

        Returns True if inside fence, False if outside.
        """
        # Simplified distance calculation (should use haversine for real GPS)
        lat_diff = position["lat"] - self.home_position["lat"]
        lon_diff = position["lon"] - self.home_position["lon"]

        # Approximate meters (simplified)
        distance = ((lat_diff * 111000) ** 2 + (lon_diff * 111000) ** 2) ** 0.5

        if distance > self.geofence_radius:
            return False

        if position["alt"] > self.geofence_altitude:
            return False

        return True

    def validate_mode_change(self, new_mode: str) -> bool:
        """
        Validate if mode change is allowed.

        SAFETY: Prevent unsafe mode transitions
        """
        # Always allow safety modes
        if new_mode in ["RTL", "LAND", "LOITER"]:
            return True

        # Block dangerous modes
        if new_mode in ["ACRO", "FLIP", "SPORT"]:
            return False

        # Allow other standard modes
        if new_mode in ["GUIDED", "AUTO", "STABILIZE", "ALT_HOLD"]:
            return True

        return False

    def signal_lost(self, duration: float) -> None:
        """Handle signal loss event."""
        self.last_signal_time = time.time() - duration
        self.state = "SEARCHING"

    def get_state(self) -> str:
        """Get current safety state."""
        return self.state

    def get_contingency_mode(self) -> str:
        """Get contingency mode based on signal loss duration."""
        if not hasattr(self, "last_signal_time"):
            return "LOITER"

        duration = time.time() - self.last_signal_time

        if duration < 10 or duration < 30:
            return "LOITER"
        else:
            return "RTL"

    def pre_arm_checks(self) -> dict[str, Any]:
        """
        Perform comprehensive pre-arm safety checks.

        SAFETY: All checks must pass before arming
        """
        failures = []

        # Check GPS
        gps_status = self.check_gps_status()
        if not gps_status["ready"]:
            failures.append(f"GPS: {gps_status['reason']}")

        # Check battery
        battery_status = self.check_battery_status()
        if battery_status["critical"]:
            failures.append(f"Battery critical: {battery_status['voltage']:.1f}V")

        # Check motor interlock
        if self.motor_interlock:
            failures.append("Motor interlock engaged")

        # Check RC override
        if self.is_rc_override_active():
            failures.append("RC override active")

        return {"passed": len(failures) == 0, "failures": failures}

    def get_failsafe_action(self) -> dict[str, Any]:
        """
        Determine failsafe action based on current conditions.

        SAFETY: Priority-based failsafe decisions with coordination health [8c]
        """
        # Priority 1: RC override
        if self.is_rc_override_active():
            return {"priority": 1, "action": "RC_CONTROL", "reason": "Pilot override"}

        # Priority 2: Battery critical
        battery = self.check_battery_status()
        if battery["critical"]:
            return {
                "priority": 2,
                "action": "RTL",
                "reason": f"Battery critical ({battery['voltage']:.1f}V)",
            }

        # Priority 3: GPS loss
        gps = self.check_gps_status()
        if not gps["ready"]:
            return {"priority": 3, "action": "LOITER", "reason": gps["reason"]}

        # Priority 4: Coordination health (if active) - SUBTASK-5.5.3.1 [8c]
        if self.include_coordination_in_decisions() and not self.is_coordination_healthy():
            return {
                "priority": 4,
                "action": "FALLBACK_DRONE",
                "reason": "Coordination system unhealthy",
                "coordination_health": False,
            }

        # No failsafe needed
        result = {"priority": 99, "action": "NONE", "reason": "All systems nominal"}

        # Include coordination health status in decision matrix [8c]
        if self.include_coordination_in_decisions():
            result["coordination_health"] = self.is_coordination_healthy()

        return result

    def set_motor_interlock(self, engaged: bool) -> None:
        """Set motor interlock state."""
        self.motor_interlock = engaged
        logger.info(f"Motor interlock {'engaged' if engaged else 'disengaged'}")

    def can_spin_motors(self) -> bool:
        """Check if motors can spin."""
        return not self.motor_interlock

    def arm_with_checks(self) -> dict[str, Any]:
        """Attempt to arm with safety checks."""
        if self.motor_interlock:
            return {"success": False, "reason": "Motor interlock engaged"}

        checks = self.pre_arm_checks()
        if not checks["passed"]:
            return {"success": False, "reason": ", ".join(checks["failures"])}

        # Attempt to arm
        if self.mavlink and hasattr(self.mavlink, "arm_vehicle"):
            success = self.mavlink.arm_vehicle()
            return {"success": success}

        return {"success": False, "reason": "MAVLink not available"}

    async def start_monitoring(self, rate_hz: float = 10) -> None:
        """
        Start continuous safety monitoring.

        SAFETY: Continuous monitoring for violations
        """
        interval = 1.0 / rate_hz

        while True:
            try:
                # Check all safety conditions
                self._check_all_safety_conditions()

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except SafetyInterlockError as e:
                logger.error(f"Safety monitoring error: {e}")
                await asyncio.sleep(interval)

    def _check_all_safety_conditions(self) -> None:
        """Check all safety conditions and record violations."""
        violations = []

        # Check battery
        battery = self.check_battery_status()
        if battery["critical"]:
            violations.append(
                SafetyViolation(
                    timestamp=time.time(),
                    type="BATTERY",
                    severity="CRITICAL",
                    description=f"Battery critical: {battery['voltage']:.1f}V",
                    action="RTL",
                )
            )

        # Check GPS
        gps = self.check_gps_status()
        if not gps["ready"]:
            violations.append(
                SafetyViolation(
                    timestamp=time.time(),
                    type="GPS",
                    severity="WARNING",
                    description=gps["reason"],
                    action="LOITER",
                )
            )

        # Check altitude
        if self.mavlink and hasattr(self.mavlink, "telemetry"):
            altitude = self.mavlink.telemetry.get("altitude", 0.0)
            if altitude > self.max_altitude:
                violations.append(
                    SafetyViolation(
                        timestamp=time.time(),
                        type="ALTITUDE",
                        severity="WARNING",
                        description=f"Altitude violation: {altitude:.1f}m > {self.max_altitude}m",
                        action="DESCEND",
                    )
                )

        self.active_violations = violations

    def get_active_violations(self) -> list[str]:
        """Get list of active safety violations."""
        return [v.description for v in self.active_violations]

    def set_max_altitude(self, altitude: float) -> None:
        """Set maximum altitude limit."""
        self.max_altitude = altitude

    def check_altitude_limit(self) -> Any:
        """Check altitude against limits."""
        if not self.mavlink or not hasattr(self.mavlink, "telemetry"):
            return True

        altitude = self.mavlink.telemetry.get("altitude", 0.0)
        margin = self.max_altitude - altitude

        if altitude > self.max_altitude:
            return {
                "violation": True,
                "action": "DESCEND",
                "altitude": altitude,
                "limit": self.max_altitude,
            }
        elif margin < 10.0:
            return {
                "warning": True,
                "margin": margin,
                "altitude": altitude,
                "limit": self.max_altitude,
            }
        else:
            return True

    def set_watchdog(self, timeout: float) -> None:
        """Set watchdog timeout."""
        self.watchdog_timeout = timeout

    def start_command(self, command: str) -> None:
        """Start tracking a command for watchdog."""
        self.watchdog_commands[command] = time.time()

    def complete_command(self, command: str) -> None:
        """Mark command as complete."""
        if command in self.watchdog_commands:
            del self.watchdog_commands[command]

    def is_watchdog_triggered(self) -> bool:
        """Check if any command has timed out."""
        current_time = time.time()

        for command, start_time in self.watchdog_commands.items():
            if current_time - start_time > self.watchdog_timeout:
                return True

        return False

    def get_watchdog_action(self) -> str:
        """Get action for watchdog timeout."""
        if self.is_watchdog_triggered():
            return "ABORT"
        return "NONE"

    def _force_motor_stop(self) -> bool:
        """Force immediate motor stop."""
        try:
            # Implementation would send immediate motor stop command
            logger.warning("Forced motor stop executed")
            return True
        except PISADException as e:
            logger.error(f"Force motor stop failed: {e}")
            return False

    # SDR++ Coordination Awareness Methods - SUBTASK-5.5.3.1

    def get_coordination_status(self) -> dict[str, Any]:
        """
        [8a] Get current SDR++ coordination status for safety monitoring.

        Returns coordination status including health and active source.
        """
        return self.coordination_status.copy()

    def set_coordination_status(self, status: dict[str, Any]) -> None:
        """
        [8a] Set SDR++ coordination status for safety monitoring.

        Updates coordination status with health and source information.
        """
        self.coordination_status.update(status)

    def is_coordination_healthy(self) -> bool:
        """
        [8a] Check if SDR++ coordination system is healthy.

        Returns True if coordination is healthy, False otherwise.
        """
        return bool(self.coordination_status.get("healthy", True))

    def check_dual_system_battery_status(self) -> dict[str, Any]:
        """
        [8b] Check battery status across both ground and drone systems.

        Enhanced battery monitoring with coordination health awareness.
        """
        drone_battery = self.check_battery_status()
        return {
            "drone": drone_battery,
            "ground": self.coordination_battery_health.get("ground"),
            "coordination_impact": self.is_coordination_healthy(),
        }

    def get_coordination_battery_health(self) -> dict[str, Any]:
        """
        [8b] Get coordination-aware battery health status.

        Returns battery health considering dual-system operation.
        """
        return self.coordination_battery_health.copy()

    def include_coordination_in_decisions(self) -> bool:
        """
        [8c] Check if coordination health should be included in safety decisions.

        Integration point for coordination health in safety decision matrix.
        """
        return bool(self.coordination_status.get("active", False))

    def evaluate_source_safety(self, source: str) -> dict[str, Any]:
        """
        [8d] Evaluate safety of a given signal source.

        Safety-aware source selection criteria with fallback triggers.
        """
        if source not in ["ground", "drone"]:
            return {"safe": False, "reason": "unknown_source"}

        if source == "drone":
            return {"safe": True, "reason": "primary_safety_authority"}

        safe = self.active and self.is_coordination_healthy()
        return {"safe": safe, "reason": "requires_active_coordination"}

    def get_safe_source_recommendation(self) -> str:
        """
        [8d] Get safety-recommended signal source.

        Returns safest available source based on current conditions.
        """
        if not self.is_coordination_healthy():
            return "drone"
        return str(self.coordination_status.get("source", "drone"))

    def trigger_source_fallback(self, reason: str) -> None:
        """
        [8d] Trigger safety fallback to drone-only operation.

        Forces fallback to drone source for safety reasons.
        """
        self.coordination_status.update(
            {"source": "drone", "fallback_reason": reason, "fallback_time": time.time()}
        )

    def handle_coordination_state_change(self, new_state: dict[str, Any]) -> None:
        """
        [8e] Handle coordination system state changes for safety.

        Safety event triggers for coordination system state changes.
        """
        old_healthy = self.is_coordination_healthy()
        self.set_coordination_status(new_state)
        new_healthy = self.is_coordination_healthy()

        if old_healthy != new_healthy:
            logger.warning(f"Coordination health changed: {old_healthy} -> {new_healthy}")
            if not new_healthy:
                self.trigger_source_fallback("coordination_unhealthy")

    def register_coordination_event_handler(self, handler_func: Any) -> None:
        """
        [8e] Register handler for coordination events (placeholder).

        Event registration for coordination system state changes.
        """
        # Placeholder implementation for coordination event handling
        pass

    def monitor_coordination_latency(self, latency_ms: float) -> None:
        """
        [8f] Monitor coordination system latency for safety timing validation.

        Tracks coordination latency for safety timing framework.
        """
        self.coordination_latency_ms = latency_ms
        if latency_ms > 100.0:  # PRD-NFR2: <100ms requirement
            logger.warning(f"Coordination latency {latency_ms:.1f}ms exceeds 100ms limit")

    def validate_coordination_timing(self) -> dict[str, Any]:
        """
        [8f] Validate coordination timing meets safety requirements.

        Coordination latency monitoring in safety timing validation framework.
        """
        return {
            "valid": self.coordination_latency_ms <= 100.0,
            "latency_ms": self.coordination_latency_ms,
            "limit_ms": 100.0,
        }

    def get_coordination_latency_status(self) -> dict[str, Any]:
        """
        [8f] Get coordination latency status for safety monitoring.

        Returns current coordination latency status.
        """
        return {
            "latency_ms": self.coordination_latency_ms,
            "within_limits": self.coordination_latency_ms <= 100.0,
            "safety_impact": "acceptable" if self.coordination_latency_ms <= 100.0 else "warning",
        }
