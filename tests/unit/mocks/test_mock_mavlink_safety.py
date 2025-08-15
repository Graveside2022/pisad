"""
Mock MAVLink Safety Tests - Sprint 4.5
Tests for safety interlocks and overrides without physical hardware.

Story 4.7 AC #5: All hardware-dependent code paths tested
Story 4.7 AC #7: Safety system validation
"""

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.serial
# Mark all tests as mock hardware tests
pytestmark = pytest.mark.mock_hardware


class TestMockMAVLinkSafety:
    """Test MAVLink safety interlocks with mock hardware."""

    @pytest.fixture
    def mock_mavlink(self):
        """Create mock MAVLink interface with safety features."""
        mock = MagicMock()
        mock.connection = MagicMock()
        mock.connected = True
        mock.telemetry = {
            "armed": False,
            "flight_mode": "GUIDED",
            "altitude": 0.0,
            "battery": {
                "voltage": 22.2,  # 6S nominal
                "current": 5.0,
                "percentage": 75.0,
            },
            "gps": {"fix_type": 3, "satellites": 12, "hdop": 1.2},
            "rc_channels": {
                "throttle": 1500,  # Center position
                "roll": 1500,
                "pitch": 1500,
                "yaw": 1500,
            },
        }
        return mock

    @pytest.fixture
    def safety_manager(self, mock_mavlink):
        """Create safety manager with mock MAVLink."""
        with patch("src.backend.services.safety_manager.MAVLinkService") as mock_class:
            mock_class.return_value = mock_mavlink
            from src.backend.services.safety_manager import SafetyManager

            manager = SafetyManager()
            manager.mavlink = mock_mavlink
            return manager

    def test_emergency_stop_response_time(self, safety_manager, mock_mavlink):
        """Test emergency stop response time < 500ms."""
        # SAFETY: Emergency stop < 500ms per Story 4.7 AC #7
        mock_mavlink.emergency_stop = MagicMock(return_value=True)

        # Measure response time
        start_time = time.perf_counter()
        result = safety_manager.trigger_emergency_stop()
        response_time = (time.perf_counter() - start_time) * 1000  # ms

        # Verify timing requirement
        assert response_time < 500, f"Emergency stop took {response_time}ms"
        mock_mavlink.emergency_stop.assert_called_once()
        assert result["success"] is True
        assert result["response_time_ms"] < 500

    def test_rc_override_detection(self, safety_manager, mock_mavlink):
        """Test RC override detection with ±50 PWM threshold."""
        # SAFETY: RC override per Story 4.7 Sprint 5 requirements

        # Normal state - no override
        assert not safety_manager.is_rc_override_active()

        # Simulate RC stick movement > 50 PWM
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1600  # +100 PWM
        assert safety_manager.is_rc_override_active()

        # Test threshold boundary
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1545  # +45 PWM
        assert not safety_manager.is_rc_override_active()

        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1551  # +51 PWM
        assert safety_manager.is_rc_override_active()

    def test_battery_monitoring_thresholds(self, safety_manager, mock_mavlink):
        """Test battery monitoring with voltage thresholds."""
        # SAFETY: Battery monitoring per Story 4.7 hardware specs
        # 6S Li-ion: 19.2V low, 18.0V critical

        # Normal voltage
        mock_mavlink.telemetry["battery"]["voltage"] = 22.2
        status = safety_manager.check_battery_status()
        assert status["level"] == "NORMAL"
        assert not status["warning"]
        assert not status["critical"]

        # Low voltage warning
        mock_mavlink.telemetry["battery"]["voltage"] = 19.2
        status = safety_manager.check_battery_status()
        assert status["level"] == "LOW"
        assert status["warning"] is True
        assert not status["critical"]

        # Critical voltage
        mock_mavlink.telemetry["battery"]["voltage"] = 18.0
        status = safety_manager.check_battery_status()
        assert status["level"] == "CRITICAL"
        assert status["warning"] is True
        assert status["critical"] is True

        # Should trigger RTL
        assert status["action"] == "RTL"

    def test_gps_requirements_check(self, safety_manager, mock_mavlink):
        """Test GPS requirements for safe operation."""
        # SAFETY: GPS requirements per Story 4.7 hardware specs
        # Requires: 8+ satellites, HDOP < 2.0

        # Good GPS
        status = safety_manager.check_gps_status()
        assert status["ready"] is True
        assert status["satellites"] >= 8
        assert status["hdop"] < 2.0

        # Insufficient satellites
        mock_mavlink.telemetry["gps"]["satellites"] = 6
        status = safety_manager.check_gps_status()
        assert status["ready"] is False
        assert "satellites" in status["reason"]

        # Poor HDOP
        mock_mavlink.telemetry["gps"]["satellites"] = 10
        mock_mavlink.telemetry["gps"]["hdop"] = 2.5
        status = safety_manager.check_gps_status()
        assert status["ready"] is False
        assert "hdop" in status["reason"].lower()

    def test_geofence_enforcement(self, safety_manager, mock_mavlink):
        """Test geofence boundary enforcement."""
        # SAFETY: Geofence per Story 4.7 field testing requirements

        # Set geofence (100m radius from home)
        safety_manager.set_geofence(radius=100.0, altitude=50.0)

        # Inside fence
        position = {"lat": 0.0, "lon": 0.0, "alt": 25.0}
        assert safety_manager.check_geofence(position) is True

        # Outside horizontal fence
        position = {"lat": 0.001, "lon": 0.001, "alt": 25.0}  # ~150m away
        assert safety_manager.check_geofence(position) is False

        # Outside vertical fence
        position = {"lat": 0.0, "lon": 0.0, "alt": 60.0}
        assert safety_manager.check_geofence(position) is False

    def test_mode_change_validation(self, safety_manager, mock_mavlink):
        """Test flight mode change validations."""
        # SAFETY: Mode changes per Story 4.7 requirements

        # GUIDED to LOITER - allowed
        mock_mavlink.telemetry["flight_mode"] = "GUIDED"
        assert safety_manager.validate_mode_change("LOITER") is True

        # GUIDED to ACRO - not allowed (unsafe)
        assert safety_manager.validate_mode_change("ACRO") is False

        # Any mode to RTL - always allowed (safety)
        mock_mavlink.telemetry["flight_mode"] = "ACRO"
        assert safety_manager.validate_mode_change("RTL") is True

        # Any mode to LAND - always allowed (safety)
        assert safety_manager.validate_mode_change("LAND") is True

    def test_signal_loss_transition(self, safety_manager, mock_mavlink):
        """Test state transition on signal loss."""
        # SAFETY: Signal loss handling per Story 4.7

        # Simulate signal loss
        safety_manager.signal_lost(duration=5.0)

        # Should transition to SEARCHING state
        state = safety_manager.get_state()
        assert state == "SEARCHING"

        # Should trigger loiter mode
        mode = safety_manager.get_contingency_mode()
        assert mode == "LOITER"

        # Extended signal loss should trigger RTL
        safety_manager.signal_lost(duration=30.0)
        mode = safety_manager.get_contingency_mode()
        assert mode == "RTL"

    def test_arming_safety_checks(self, safety_manager, mock_mavlink):
        """Test comprehensive arming safety checks."""
        # SAFETY: Pre-arm checks per Story 4.7

        # All checks pass
        checks = safety_manager.pre_arm_checks()
        assert checks["passed"] is True
        assert len(checks["failures"]) == 0

        # GPS check fail
        mock_mavlink.telemetry["gps"]["fix_type"] = 0
        checks = safety_manager.pre_arm_checks()
        assert checks["passed"] is False
        assert "GPS" in str(checks["failures"])

        # Battery check fail
        mock_mavlink.telemetry["gps"]["fix_type"] = 3
        mock_mavlink.telemetry["battery"]["voltage"] = 17.5
        checks = safety_manager.pre_arm_checks()
        assert checks["passed"] is False
        assert "battery" in str(checks["failures"]).lower()

    def test_failsafe_priorities(self, safety_manager, mock_mavlink):
        """Test failsafe action priorities."""
        # SAFETY: Failsafe priorities per safety requirements

        # Priority 1: RC override
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1600
        action = safety_manager.get_failsafe_action()
        assert action["priority"] == 1
        assert action["action"] == "RC_CONTROL"

        # Priority 2: Battery critical (no RC override)
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1500
        mock_mavlink.telemetry["battery"]["voltage"] = 17.5
        action = safety_manager.get_failsafe_action()
        assert action["priority"] == 2
        assert action["action"] == "RTL"

        # Priority 3: GPS loss
        mock_mavlink.telemetry["battery"]["voltage"] = 22.0
        mock_mavlink.telemetry["gps"]["fix_type"] = 0
        action = safety_manager.get_failsafe_action()
        assert action["priority"] == 3
        assert action["action"] == "LOITER"

    def test_motor_interlock(self, safety_manager, mock_mavlink):
        """Test motor interlock safety feature."""
        # SAFETY: Motor interlock prevents accidental starts

        # Interlock engaged - motors disabled
        safety_manager.set_motor_interlock(True)
        assert not safety_manager.can_spin_motors()

        # Try to arm with interlock
        mock_mavlink.arm_vehicle = MagicMock(return_value=False)
        result = safety_manager.arm_with_checks()
        assert result["success"] is False
        assert "interlock" in result["reason"].lower()

        # Disengage interlock
        safety_manager.set_motor_interlock(False)
        assert safety_manager.can_spin_motors()

    @pytest.mark.asyncio
    async def test_safety_loop_monitoring(self, safety_manager, mock_mavlink):
        """Test continuous safety monitoring loop."""
        # SAFETY: Continuous monitoring per safety requirements

        # Start safety monitoring
        monitoring_task = asyncio.create_task(safety_manager.start_monitoring(rate_hz=10))

        # Let it run for a bit
        await asyncio.sleep(0.5)

        # Inject a safety violation
        mock_mavlink.telemetry["battery"]["voltage"] = 17.0

        # Wait for detection
        await asyncio.sleep(0.2)

        # Check if violation detected
        violations = safety_manager.get_active_violations()
        assert len(violations) > 0
        assert any("battery" in v.lower() for v in violations)

        # Stop monitoring
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    def test_altitude_limit_enforcement(self, safety_manager, mock_mavlink):
        """Test altitude limit enforcement."""
        # SAFETY: Altitude limits per Story 4.7 command_pipeline.py:332

        # Set max altitude
        safety_manager.set_max_altitude(100.0)

        # Below limit - OK
        mock_mavlink.telemetry["altitude"] = 50.0
        assert safety_manager.check_altitude_limit() is True

        # At limit - warning
        mock_mavlink.telemetry["altitude"] = 95.0
        status = safety_manager.check_altitude_limit()
        assert status["warning"] is True
        assert status["margin"] < 10.0

        # Above limit - violation
        mock_mavlink.telemetry["altitude"] = 105.0
        status = safety_manager.check_altitude_limit()
        assert status["violation"] is True
        assert status["action"] == "DESCEND"

    def test_watchdog_timer(self, safety_manager, mock_mavlink):
        """Test watchdog timer for command execution."""
        # SAFETY: Watchdog prevents hung commands

        # Set watchdog for 2 seconds
        safety_manager.set_watchdog(timeout=2.0)

        # Start a command
        safety_manager.start_command("TAKEOFF")

        # Command completes in time
        time.sleep(1.0)
        safety_manager.complete_command("TAKEOFF")
        assert not safety_manager.is_watchdog_triggered()

        # Start another command
        safety_manager.start_command("LAND")

        # Command times out
        time.sleep(2.5)
        assert safety_manager.is_watchdog_triggered()
        assert safety_manager.get_watchdog_action() == "ABORT"
