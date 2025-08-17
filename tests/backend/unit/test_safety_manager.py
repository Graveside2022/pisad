"""Unit tests for SafetyManager service.

Tests safety interlocks, emergency procedures, battery monitoring,
RC override detection, and violation handling per PRD requirements.
"""

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

        # Should handle error gracefully
        assert result["success"] is False
        assert "error" in result
        assert "response_time_ms" in result

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
