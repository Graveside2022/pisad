"""Unit tests for SafetyManager service.

Tests safety interlocks, emergency procedures, battery monitoring,
RC override detection, and violation handling per PRD requirements.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from src.backend.core.exceptions import SafetyInterlockError
from src.backend.services.safety_manager import SafetyManager, SafetyPriority, SafetyViolation


class TestSafetyManager:
    """Test safety manager service."""

    @pytest.fixture
    def safety_manager(self):
        """Provide SafetyManager instance."""
        return SafetyManager()

    @pytest.fixture
    def mock_mavlink(self):
        """Provide mock MAVLink service."""
        mock = Mock()
        mock.emergency_stop.return_value = True
        mock.telemetry = {
            "battery": {"voltage": 22.0},
            "rc_channels": {"throttle": 1500, "roll": 1500, "pitch": 1500, "yaw": 1500},
            "gps": {"satellites": 10, "hdop": 1.5},
            "position": {"lat": 37.7749, "lon": -122.4194, "alt": 30.0},
        }
        return mock

    def test_safety_manager_initialization(self, safety_manager):
        """Test SafetyManager initializes with correct defaults."""
        assert safety_manager.motor_interlock is False
        assert safety_manager.max_altitude == 100.0
        assert safety_manager.geofence_radius == 100.0
        assert safety_manager.battery_low_voltage == 19.2
        assert safety_manager.battery_critical_voltage == 18.0
        assert safety_manager.min_satellites == 8
        assert safety_manager.max_hdop == 2.0
        assert safety_manager.rc_override_threshold == 50
        assert safety_manager.rc_center == 1500
        assert isinstance(safety_manager.active_violations, list)
        assert safety_manager.state == "IDLE"

    def test_emergency_stop_timing_requirement(self, safety_manager, mock_mavlink):
        """Test emergency stop meets <500ms timing requirement per Story 4.7."""
        safety_manager.mavlink = mock_mavlink

        start_time = time.perf_counter()
        result = safety_manager.trigger_emergency_stop()
        end_time = time.perf_counter()

        # Should complete quickly
        actual_time_ms = (end_time - start_time) * 1000
        assert actual_time_ms < 100  # Much faster than 500ms requirement

        # Result should indicate success
        assert result["success"] is True
        assert result["response_time_ms"] < 500
        assert result["priority"] == "CRITICAL"

    def test_emergency_stop_without_mavlink(self, safety_manager):
        """Test emergency stop fallback without MAVLink connection."""
        # No MAVLink service connected
        safety_manager.mavlink = None

        result = safety_manager.trigger_emergency_stop()

        # Should still execute fallback procedure
        assert isinstance(result, dict)
        assert "success" in result
        assert "response_time_ms" in result

    def test_emergency_stop_mavlink_failure(self, safety_manager):
        """Test emergency stop handles MAVLink failures gracefully."""
        mock_mavlink = Mock()
        mock_mavlink.emergency_stop.side_effect = SafetyInterlockError("MAVLink failure")
        safety_manager.mavlink = mock_mavlink

        result = safety_manager.trigger_emergency_stop()

        # Should handle error gracefully and succeed via fallback
        assert result["success"] is True  # Fallback should succeed
        assert "error" in result
        assert "response_time_ms" in result
        assert result["fallback_attempted"] is True

    def test_rc_override_detection_normal_sticks(self, safety_manager, mock_mavlink):
        """Test RC override detection with normal stick positions."""
        safety_manager.mavlink = mock_mavlink

        # All sticks at center (1500 PWM)
        is_override = safety_manager.is_rc_override_active()

        assert is_override is False

    def test_rc_override_detection_stick_movement(self, safety_manager, mock_mavlink):
        """Test RC override detection with stick movement per PRD requirements."""
        safety_manager.mavlink = mock_mavlink

        # Move throttle stick beyond threshold (±50 PWM)
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1600  # +100 from center

        is_override = safety_manager.is_rc_override_active()

        assert is_override is True

    def test_rc_override_threshold_boundary(self, safety_manager, mock_mavlink):
        """Test RC override threshold boundary conditions."""
        safety_manager.mavlink = mock_mavlink

        # Just at threshold (50 PWM)
        mock_mavlink.telemetry["rc_channels"]["roll"] = 1550  # +50 from center

        is_override = safety_manager.is_rc_override_active()

        # Implementation uses abs(value - center) > threshold, so exactly at threshold should be False
        assert is_override is False

    def test_rc_override_without_mavlink(self, safety_manager):
        """Test RC override detection without MAVLink connection."""
        safety_manager.mavlink = None

        is_override = safety_manager.is_rc_override_active()

        assert is_override is False

    def test_battery_status_normal(self, safety_manager, mock_mavlink):
        """Test battery status monitoring with normal voltage."""
        safety_manager.mavlink = mock_mavlink

        # Normal battery voltage (22.0V)
        status = safety_manager.check_battery_status()

        assert status["level"] == "NORMAL"
        assert status["voltage"] == 22.0
        assert status["warning"] is False
        assert status["critical"] is False

    def test_battery_status_low_warning(self, safety_manager, mock_mavlink):
        """Test battery status with low voltage warning."""
        safety_manager.mavlink = mock_mavlink

        # Low battery voltage (19.0V - between low and critical)
        mock_mavlink.telemetry["battery"]["voltage"] = 19.0

        status = safety_manager.check_battery_status()

        assert status["level"] == "LOW"
        assert status["voltage"] == 19.0
        assert status["warning"] is True
        assert status["critical"] is False
        assert status["action"] == "WARN"

    def test_battery_status_critical(self, safety_manager, mock_mavlink):
        """Test battery status with critical voltage."""
        safety_manager.mavlink = mock_mavlink

        # Critical battery voltage (17.0V - below critical threshold)
        mock_mavlink.telemetry["battery"]["voltage"] = 17.0

        status = safety_manager.check_battery_status()

        assert status["level"] == "CRITICAL"
        assert status["voltage"] == 17.0
        assert status["warning"] is True
        assert status["critical"] is True

    def test_battery_status_without_mavlink(self, safety_manager):
        """Test battery status without MAVLink connection."""
        safety_manager.mavlink = None

        status = safety_manager.check_battery_status()

        assert status["level"] == "UNKNOWN"
        assert status["warning"] is False
        assert status["critical"] is False

    def test_safety_priority_enum(self):
        """Test SafetyPriority enum values."""
        assert SafetyPriority.RC_OVERRIDE.value == 1
        assert SafetyPriority.BATTERY_CRITICAL.value == 2
        assert SafetyPriority.GPS_LOSS.value == 3
        assert SafetyPriority.SIGNAL_LOSS.value == 4
        assert SafetyPriority.ALTITUDE_VIOLATION.value == 5

    def test_safety_violation_creation(self):
        """Test SafetyViolation dataclass creation."""
        violation = SafetyViolation(
            timestamp=time.time(),
            type="BATTERY_LOW",
            severity="WARNING",
            description="Battery voltage below threshold",
            action="RETURN_TO_LAUNCH",
        )

        assert isinstance(violation.timestamp, float)
        assert violation.type == "BATTERY_LOW"
        assert violation.severity == "WARNING"
        assert violation.description == "Battery voltage below threshold"
        assert violation.action == "RETURN_TO_LAUNCH"

    def test_violation_tracking(self, safety_manager):
        """Test active violation tracking."""
        initial_count = len(safety_manager.active_violations)

        # Add violation manually for testing
        violation = SafetyViolation(
            timestamp=time.time(),
            type="TEST_VIOLATION",
            severity="HIGH",
            description="Test violation",
            action="STOP",
        )
        safety_manager.active_violations.append(violation)

        assert len(safety_manager.active_violations) == initial_count + 1
        assert safety_manager.active_violations[-1].type == "TEST_VIOLATION"

    def test_force_motor_stop_method(self, safety_manager):
        """Test force motor stop fallback method."""
        # Test internal motor stop method
        result = safety_manager._force_motor_stop()

        # Should return success status
        assert isinstance(result, bool)

    def test_watchdog_timeout_configuration(self, safety_manager):
        """Test watchdog timeout configuration."""
        assert safety_manager.watchdog_timeout == 5.0
        assert isinstance(safety_manager.watchdog_commands, dict)

    def test_geofence_parameters(self, safety_manager):
        """Test geofence safety parameters."""
        assert safety_manager.geofence_radius == 100.0
        assert safety_manager.geofence_altitude == 50.0
        assert isinstance(safety_manager.home_position, dict)
        assert "lat" in safety_manager.home_position
        assert "lon" in safety_manager.home_position
        assert "alt" in safety_manager.home_position

    def test_signal_loss_tracking(self, safety_manager):
        """Test signal loss time tracking."""
        initial_time = safety_manager.last_signal_time

        # Update signal time
        safety_manager.last_signal_time = time.time()

        assert safety_manager.last_signal_time > initial_time

    def test_motor_interlock_state(self, safety_manager):
        """Test motor interlock state management."""
        # Initially should be disabled
        assert safety_manager.motor_interlock is False

        # Enable interlock
        safety_manager.motor_interlock = True
        assert safety_manager.motor_interlock is True

        # Disable interlock
        safety_manager.motor_interlock = False
        assert safety_manager.motor_interlock is False

    def test_battery_voltage_thresholds(self, safety_manager):
        """Test battery voltage threshold configuration."""
        # 6S Li-ion thresholds per hardware specs
        assert safety_manager.battery_low_voltage == 19.2
        assert safety_manager.battery_critical_voltage == 18.0

        # Thresholds should be logical
        assert safety_manager.battery_low_voltage > safety_manager.battery_critical_voltage

    def test_gps_safety_parameters(self, safety_manager):
        """Test GPS safety monitoring parameters."""
        assert safety_manager.min_satellites == 8
        assert safety_manager.max_hdop == 2.0

        # Should be reasonable GPS requirements
        assert safety_manager.min_satellites >= 4  # Minimum for 3D fix
        assert safety_manager.max_hdop <= 5.0  # Reasonable precision requirement

    def test_altitude_safety_limits(self, safety_manager):
        """Test altitude safety limit configuration."""
        assert safety_manager.max_altitude == 100.0

        # Should be reasonable altitude limit
        assert safety_manager.max_altitude > 0
        assert safety_manager.max_altitude <= 400  # FAA Part 107 limit

    def test_emergency_stop_performance_logging(self, safety_manager, mock_mavlink):
        """Test emergency stop performance is logged when slow."""
        safety_manager.mavlink = mock_mavlink

        # Mock slow emergency stop
        def slow_emergency_stop():
            time.sleep(0.6)  # Simulate slow response (>500ms)
            return True

        mock_mavlink.emergency_stop = slow_emergency_stop

        with patch("src.backend.services.safety_manager.logger") as mock_logger:
            result = safety_manager.trigger_emergency_stop()

            # Should log warning for slow response
            mock_logger.warning.assert_called()
            assert result["response_time_ms"] > 500

    def test_rc_channel_validation(self, safety_manager, mock_mavlink):
        """Test RC channel value validation."""
        safety_manager.mavlink = mock_mavlink

        # Test each RC channel for override detection
        channels = ["throttle", "roll", "pitch", "yaw"]

        for channel in channels:
            # Reset all channels to center
            for ch in channels:
                mock_mavlink.telemetry["rc_channels"][ch] = 1500

            # Move specific channel beyond threshold
            mock_mavlink.telemetry["rc_channels"][channel] = 1600

            is_override = safety_manager.is_rc_override_active()
            assert is_override is True, f"Failed to detect override on {channel}"

    def test_monitoring_task_management(self, safety_manager):
        """Test safety monitoring task management."""
        assert safety_manager.monitoring_task is None  # Initially not running

        # Task management interface should exist
        assert hasattr(safety_manager, "monitoring_task")

    def test_safety_state_management(self, safety_manager):
        """Test safety manager state tracking."""
        assert safety_manager.state == "IDLE"

        # State should be manageable
        safety_manager.state = "MONITORING"
        assert safety_manager.state == "MONITORING"

    def test_watchdog_command_tracking(self, safety_manager):
        """Test watchdog command timestamp tracking."""
        initial_commands = len(safety_manager.watchdog_commands)

        # Add watchdog command
        safety_manager.watchdog_commands["test_command"] = time.time()

        assert len(safety_manager.watchdog_commands) == initial_commands + 1
        assert "test_command" in safety_manager.watchdog_commands

    def test_home_position_setting(self, safety_manager):
        """Test home position configuration for geofence."""
        test_position = {"lat": 37.7749, "lon": -122.4194, "alt": 50.0}

        safety_manager.home_position = test_position

        assert safety_manager.home_position["lat"] == 37.7749
        assert safety_manager.home_position["lon"] == -122.4194
        assert safety_manager.home_position["alt"] == 50.0

    def test_concurrent_safety_operations(self, safety_manager, mock_mavlink):
        """Test concurrent safety operation handling."""
        safety_manager.mavlink = mock_mavlink

        # Simulate concurrent safety checks
        results = []

        for _ in range(3):
            result = safety_manager.trigger_emergency_stop()
            results.append(result)

        # All operations should complete successfully
        for result in results:
            assert result["success"] is True
            assert "response_time_ms" in result

    # NEW TDD TESTS FOR 90% COVERAGE

    def test_gps_status_normal(self, safety_manager, mock_mavlink):
        """Test GPS status monitoring with normal GPS fix."""
        safety_manager.mavlink = mock_mavlink

        # Normal GPS with good satellites and HDOP
        mock_mavlink.telemetry["gps"] = {"satellites": 12, "hdop": 1.2, "fix_type": 3}

        status = safety_manager.check_gps_status()

        assert status["ready"] is True
        assert status["satellites"] == 12
        assert status["hdop"] == 1.2
        assert status["fix_type"] == 3

    def test_gps_status_insufficient_satellites(self, safety_manager, mock_mavlink):
        """Test GPS status with insufficient satellites."""
        safety_manager.mavlink = mock_mavlink

        # Low satellite count
        mock_mavlink.telemetry["gps"] = {"satellites": 6, "hdop": 1.2, "fix_type": 3}

        status = safety_manager.check_gps_status()

        assert status["ready"] is False
        assert "Insufficient satellites" in status["reason"]
        assert status["satellites"] == 6

    def test_gps_status_poor_hdop(self, safety_manager, mock_mavlink):
        """Test GPS status with poor HDOP."""
        safety_manager.mavlink = mock_mavlink

        # Poor HDOP
        mock_mavlink.telemetry["gps"] = {"satellites": 10, "hdop": 3.5, "fix_type": 3}

        status = safety_manager.check_gps_status()

        assert status["ready"] is False
        assert "Poor HDOP" in status["reason"]
        assert status["hdop"] == 3.5

    def test_gps_status_no_3d_fix(self, safety_manager, mock_mavlink):
        """Test GPS status without 3D fix."""
        safety_manager.mavlink = mock_mavlink

        # No 3D fix
        mock_mavlink.telemetry["gps"] = {
            "satellites": 10,
            "hdop": 1.5,
            "fix_type": 2,  # 2D fix only
        }

        status = safety_manager.check_gps_status()

        assert status["ready"] is False
        assert "No 3D fix" in status["reason"]
        assert status["satellites"] == 10  # Should still return satellites count

    def test_gps_status_without_mavlink(self, safety_manager):
        """Test GPS status without MAVLink connection."""
        safety_manager.mavlink = None

        status = safety_manager.check_gps_status()

        assert status["ready"] is False
        assert status["reason"] == "No telemetry"

    def test_geofence_configuration(self, safety_manager):
        """Test geofence parameter setting."""
        radius = 150.0
        altitude = 75.0

        safety_manager.set_geofence(radius, altitude)

        assert safety_manager.geofence_radius == radius
        assert safety_manager.geofence_altitude == altitude

    def test_geofence_check_inside_fence(self, safety_manager):
        """Test geofence check with position inside fence."""
        # Set home position
        safety_manager.home_position = {"lat": 37.7749, "lon": -122.4194, "alt": 0.0}

        # Position close to home (within fence)
        test_position = {"lat": 37.7750, "lon": -122.4195, "alt": 30.0}

        is_inside = safety_manager.check_geofence(test_position)

        assert is_inside is True

    def test_geofence_check_outside_fence(self, safety_manager):
        """Test geofence check with position outside fence."""
        # Set home position
        safety_manager.home_position = {"lat": 37.7749, "lon": -122.4194, "alt": 0.0}

        # Position far from home (outside 100m fence)
        test_position = {"lat": 37.8000, "lon": -122.5000, "alt": 30.0}

        is_inside = safety_manager.check_geofence(test_position)

        assert is_inside is False

    def test_geofence_check_altitude_violation(self, safety_manager):
        """Test geofence check with altitude violation."""
        # Set home position
        safety_manager.home_position = {"lat": 37.7749, "lon": -122.4194, "alt": 0.0}

        # Position close horizontally but high altitude
        test_position = {"lat": 37.7750, "lon": -122.4195, "alt": 80.0}  # Above 50m limit

        is_inside = safety_manager.check_geofence(test_position)

        assert is_inside is False

    def test_mode_validation_safety_modes(self, safety_manager):
        """Test mode change validation for safety modes."""
        # Safety modes should always be allowed
        assert safety_manager.validate_mode_change("RTL") is True
        assert safety_manager.validate_mode_change("LAND") is True
        assert safety_manager.validate_mode_change("LOITER") is True

    def test_mode_validation_dangerous_modes(self, safety_manager):
        """Test mode change validation blocks dangerous modes."""
        # Dangerous modes should be blocked
        assert safety_manager.validate_mode_change("ACRO") is False
        assert safety_manager.validate_mode_change("FLIP") is False
        assert safety_manager.validate_mode_change("SPORT") is False

    def test_mode_validation_standard_modes(self, safety_manager):
        """Test mode change validation for standard modes."""
        # Standard modes should be allowed
        assert safety_manager.validate_mode_change("GUIDED") is True
        assert safety_manager.validate_mode_change("AUTO") is True
        assert safety_manager.validate_mode_change("STABILIZE") is True
        assert safety_manager.validate_mode_change("ALT_HOLD") is True

    def test_mode_validation_unknown_mode(self, safety_manager):
        """Test mode change validation for unknown mode."""
        # Unknown modes should be blocked
        assert safety_manager.validate_mode_change("UNKNOWN_MODE") is False

    def test_signal_loss_handling(self, safety_manager):
        """Test signal loss event handling."""
        initial_state = safety_manager.state

        # Simulate signal loss
        safety_manager.signal_lost(5.0)  # 5 seconds of signal loss

        assert safety_manager.state == "SEARCHING"
        # Verify signal time was updated
        current_time = time.time()
        assert safety_manager.last_signal_time < current_time

    def test_get_state_method(self, safety_manager):
        """Test get_state method returns current state."""
        # Test initial state
        assert safety_manager.get_state() == "IDLE"

        # Change state and test
        safety_manager.state = "MONITORING"
        assert safety_manager.get_state() == "MONITORING"

    def test_get_contingency_mode_short_signal_loss(self, safety_manager):
        """Test contingency mode for short signal loss."""
        safety_manager.last_signal_time = time.time() - 5.0  # 5 seconds ago

        mode = safety_manager.get_contingency_mode()

        assert mode == "LOITER"

    def test_get_contingency_mode_long_signal_loss(self, safety_manager):
        """Test contingency mode for long signal loss."""
        safety_manager.last_signal_time = time.time() - 45.0  # 45 seconds ago

        mode = safety_manager.get_contingency_mode()

        assert mode == "RTL"

    def test_get_contingency_mode_no_signal_time(self, safety_manager):
        """Test contingency mode when no signal time recorded."""
        # Remove last_signal_time attribute
        if hasattr(safety_manager, "last_signal_time"):
            delattr(safety_manager, "last_signal_time")

        mode = safety_manager.get_contingency_mode()

        assert mode == "LOITER"

    def test_pre_arm_checks_all_pass(self, safety_manager, mock_mavlink):
        """Test pre-arm checks when all conditions are good."""
        safety_manager.mavlink = mock_mavlink

        # Set good conditions
        mock_mavlink.telemetry["gps"] = {"satellites": 12, "hdop": 1.2, "fix_type": 3}
        mock_mavlink.telemetry["battery"] = {"voltage": 22.0}
        safety_manager.motor_interlock = False
        mock_mavlink.telemetry["rc_channels"] = {
            "throttle": 1500,
            "roll": 1500,
            "pitch": 1500,
            "yaw": 1500,
        }

        checks = safety_manager.pre_arm_checks()

        assert checks["passed"] is True
        assert len(checks["failures"]) == 0

    def test_pre_arm_checks_multiple_failures(self, safety_manager, mock_mavlink):
        """Test pre-arm checks with multiple failure conditions."""
        safety_manager.mavlink = mock_mavlink

        # Set bad conditions
        mock_mavlink.telemetry["gps"] = {"satellites": 4, "hdop": 3.5, "fix_type": 2}  # Bad GPS
        mock_mavlink.telemetry["battery"] = {"voltage": 17.0}  # Critical battery
        safety_manager.motor_interlock = True  # Interlock engaged
        mock_mavlink.telemetry["rc_channels"] = {
            "throttle": 1600,
            "roll": 1500,
            "pitch": 1500,
            "yaw": 1500,
        }  # RC override

        checks = safety_manager.pre_arm_checks()

        assert checks["passed"] is False
        assert len(checks["failures"]) > 0
        assert any("GPS" in failure for failure in checks["failures"])
        assert any("Battery critical" in failure for failure in checks["failures"])
        assert any("Motor interlock" in failure for failure in checks["failures"])
        assert any("RC override" in failure for failure in checks["failures"])

    def test_get_failsafe_action_rc_override(self, safety_manager, mock_mavlink):
        """Test failsafe action priority for RC override."""
        safety_manager.mavlink = mock_mavlink

        # RC override active (highest priority)
        mock_mavlink.telemetry["rc_channels"]["throttle"] = 1600

        action = safety_manager.get_failsafe_action()

        assert action["priority"] == 1
        assert action["action"] == "RC_CONTROL"
        assert "Pilot override" in action["reason"]

    def test_get_failsafe_action_critical_battery(self, safety_manager, mock_mavlink):
        """Test failsafe action for critical battery."""
        safety_manager.mavlink = mock_mavlink

        # Critical battery (priority 2)
        mock_mavlink.telemetry["battery"]["voltage"] = 17.0

        action = safety_manager.get_failsafe_action()

        assert action["priority"] == 2
        assert action["action"] == "RTL"
        assert "Battery critical" in action["reason"]

    def test_get_failsafe_action_gps_loss(self, safety_manager, mock_mavlink):
        """Test failsafe action for GPS loss."""
        safety_manager.mavlink = mock_mavlink

        # GPS loss (priority 3)
        mock_mavlink.telemetry["gps"] = {"satellites": 4, "hdop": 3.5, "fix_type": 2}

        action = safety_manager.get_failsafe_action()

        assert action["priority"] == 3
        assert action["action"] == "LOITER"
        # Should contain GPS reason from check_gps_status

    def test_get_failsafe_action_no_failsafe(self, safety_manager, mock_mavlink):
        """Test failsafe action when no failsafe needed."""
        safety_manager.mavlink = mock_mavlink

        # Set up good conditions for all systems
        mock_mavlink.telemetry["rc_channels"] = {
            "throttle": 1500,
            "roll": 1500,
            "pitch": 1500,
            "yaw": 1500,
        }  # No RC override
        mock_mavlink.telemetry["battery"] = {"voltage": 22.0}  # Good battery
        mock_mavlink.telemetry["gps"] = {"satellites": 12, "hdop": 1.2, "fix_type": 3}  # Good GPS

        action = safety_manager.get_failsafe_action()

        assert action["priority"] == 99
        assert action["action"] == "NONE"
        assert action["reason"] == "All systems nominal"

    def test_motor_interlock_management(self, safety_manager):
        """Test motor interlock set/get operations."""
        # Test setting interlock
        safety_manager.set_motor_interlock(True)
        assert safety_manager.motor_interlock is True

        # Test clearing interlock
        safety_manager.set_motor_interlock(False)
        assert safety_manager.motor_interlock is False

    def test_can_spin_motors(self, safety_manager):
        """Test motor spin permission check."""
        # Should be able to spin when interlock disengaged
        safety_manager.motor_interlock = False
        assert safety_manager.can_spin_motors() is True

        # Should not be able to spin when interlock engaged
        safety_manager.motor_interlock = True
        assert safety_manager.can_spin_motors() is False

    def test_arm_with_checks_interlock_engaged(self, safety_manager):
        """Test arming blocked by motor interlock."""
        safety_manager.motor_interlock = True

        result = safety_manager.arm_with_checks()

        assert result["success"] is False
        assert "Motor interlock engaged" in result["reason"]

    def test_arm_with_checks_safety_failures(self, safety_manager, mock_mavlink):
        """Test arming blocked by safety check failures."""
        safety_manager.mavlink = mock_mavlink
        safety_manager.motor_interlock = False

        # Set up failing conditions
        mock_mavlink.telemetry["battery"]["voltage"] = 17.0  # Critical battery

        result = safety_manager.arm_with_checks()

        assert result["success"] is False
        assert "Battery critical" in result["reason"]

    def test_arm_with_checks_success(self, safety_manager, mock_mavlink):
        """Test successful arming with all checks passed."""
        safety_manager.mavlink = mock_mavlink
        safety_manager.motor_interlock = False

        # Set up good conditions
        mock_mavlink.telemetry["gps"] = {"satellites": 12, "hdop": 1.2, "fix_type": 3}
        mock_mavlink.telemetry["battery"] = {"voltage": 22.0}
        mock_mavlink.arm_vehicle = Mock(return_value=True)

        result = safety_manager.arm_with_checks()

        assert result["success"] is True
        mock_mavlink.arm_vehicle.assert_called_once()

    def test_arm_with_checks_no_mavlink(self, safety_manager):
        """Test arming attempt without MAVLink connection."""
        safety_manager.mavlink = None
        safety_manager.motor_interlock = False

        result = safety_manager.arm_with_checks()

        assert result["success"] is False
        # Should fail pre-arm checks due to no telemetry
        expected_reasons = ["MAVLink not available", "GPS: No telemetry"]
        assert any(reason in result["reason"] for reason in expected_reasons)

    # Additional TDD tests for remaining uncovered methods

    @pytest.mark.asyncio
    async def test_start_monitoring_async(self, safety_manager):
        """Test async safety monitoring startup."""
        # Should start monitoring without error
        monitoring_task = asyncio.create_task(safety_manager.start_monitoring(rate_hz=5))

        # Let it run briefly
        await asyncio.sleep(0.01)

        # Cancel and cleanup
        monitoring_task.cancel()
        try:
            await monitoring_task
        except asyncio.CancelledError:
            pass

    def test_check_all_safety_conditions(self, safety_manager, mock_mavlink):
        """Test comprehensive safety condition checking."""
        safety_manager.mavlink = mock_mavlink

        # Set up conditions that trigger violations
        mock_mavlink.telemetry["battery"] = {"voltage": 17.0}  # Critical battery
        mock_mavlink.telemetry["gps"] = {"satellites": 4, "hdop": 3.5, "fix_type": 2}  # Bad GPS
        mock_mavlink.telemetry["altitude"] = 120.0  # Above max altitude

        # Trigger safety condition check
        safety_manager._check_all_safety_conditions()

        # Should have recorded violations
        assert len(safety_manager.active_violations) > 0

        # Check violation types
        violation_types = [v.type for v in safety_manager.active_violations]
        assert "BATTERY" in violation_types
        assert "GPS" in violation_types
        assert "ALTITUDE" in violation_types

    def test_get_active_violations(self, safety_manager):
        """Test active violations retrieval."""
        # Initially no violations
        violations = safety_manager.get_active_violations()
        assert len(violations) == 0

        # Add test violation
        from src.backend.services.safety_manager import SafetyViolation

        test_violation = SafetyViolation(
            timestamp=time.time(),
            type="TEST",
            severity="HIGH",
            description="Test violation",
            action="TEST_ACTION",
        )
        safety_manager.active_violations.append(test_violation)

        violations = safety_manager.get_active_violations()
        assert len(violations) == 1
        assert "Test violation" in violations

    def test_set_max_altitude(self, safety_manager):
        """Test maximum altitude configuration."""
        new_altitude = 150.0

        safety_manager.set_max_altitude(new_altitude)

        assert safety_manager.max_altitude == new_altitude

    def test_check_altitude_limit_normal(self, safety_manager, mock_mavlink):
        """Test altitude limit check with normal altitude."""
        safety_manager.mavlink = mock_mavlink
        mock_mavlink.telemetry["altitude"] = 50.0  # Well below 100m limit

        result = safety_manager.check_altitude_limit()

        assert result is True

    def test_check_altitude_limit_warning(self, safety_manager, mock_mavlink):
        """Test altitude limit check with warning zone."""
        safety_manager.mavlink = mock_mavlink
        mock_mavlink.telemetry["altitude"] = 95.0  # Close to 100m limit

        result = safety_manager.check_altitude_limit()

        assert isinstance(result, dict)
        assert result["warning"] is True
        assert result["margin"] == 5.0
        assert result["altitude"] == 95.0

    def test_check_altitude_limit_violation(self, safety_manager, mock_mavlink):
        """Test altitude limit check with violation."""
        safety_manager.mavlink = mock_mavlink
        mock_mavlink.telemetry["altitude"] = 120.0  # Above 100m limit

        result = safety_manager.check_altitude_limit()

        assert isinstance(result, dict)
        assert result["violation"] is True
        assert result["action"] == "DESCEND"
        assert result["altitude"] == 120.0
        assert result["limit"] == 100.0

    def test_check_altitude_limit_no_telemetry(self, safety_manager):
        """Test altitude limit check without telemetry."""
        safety_manager.mavlink = None

        result = safety_manager.check_altitude_limit()

        assert result is True

    def test_set_watchdog(self, safety_manager):
        """Test watchdog timeout configuration."""
        new_timeout = 10.0

        safety_manager.set_watchdog(new_timeout)

        assert safety_manager.watchdog_timeout == new_timeout

    def test_start_command(self, safety_manager):
        """Test command tracking for watchdog."""
        command = "test_command"

        safety_manager.start_command(command)

        assert command in safety_manager.watchdog_commands
        assert isinstance(safety_manager.watchdog_commands[command], float)

    def test_complete_command(self, safety_manager):
        """Test command completion tracking."""
        command = "test_command"

        # Start command
        safety_manager.start_command(command)
        assert command in safety_manager.watchdog_commands

        # Complete command
        safety_manager.complete_command(command)
        assert command not in safety_manager.watchdog_commands

    def test_is_watchdog_triggered_not_triggered(self, safety_manager):
        """Test watchdog not triggered with recent command."""
        # Start recent command
        safety_manager.start_command("recent_command")

        triggered = safety_manager.is_watchdog_triggered()

        assert triggered is False

    def test_is_watchdog_triggered_timeout(self, safety_manager):
        """Test watchdog triggered by timeout."""
        # Start old command by manipulating timestamp
        command = "old_command"
        safety_manager.watchdog_commands[command] = time.time() - 10.0  # 10 seconds ago
        safety_manager.watchdog_timeout = 5.0  # 5 second timeout

        triggered = safety_manager.is_watchdog_triggered()

        assert triggered is True

    def test_get_watchdog_action_triggered(self, safety_manager):
        """Test watchdog action when triggered."""
        # Set up triggered watchdog
        safety_manager.watchdog_commands["old_command"] = time.time() - 10.0
        safety_manager.watchdog_timeout = 5.0

        action = safety_manager.get_watchdog_action()

        assert action == "ABORT"

    def test_get_watchdog_action_not_triggered(self, safety_manager):
        """Test watchdog action when not triggered."""
        action = safety_manager.get_watchdog_action()

        assert action == "NONE"
