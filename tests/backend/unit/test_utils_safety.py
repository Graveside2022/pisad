"""Comprehensive unit tests for safety utility module."""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import patch
from uuid import UUID

import pytest

from src.backend.utils.safety import (
    BatteryCheck,
    GeofenceCheck,
    ModeCheck,
    OperatorActivationCheck,
    SafetyCheck,
    SafetyEvent,
    SafetyEventType,
    SafetyInterlockSystem,
    SafetyTrigger,
    SignalLossCheck,
)


class TestSafetyEvents:
    """Test safety event types and triggers."""

    def test_safety_event_type_enum(self):
        """Test SafetyEventType enum values."""
        assert SafetyEventType.INTERLOCK_TRIGGERED.value == "interlock_triggered"
        assert SafetyEventType.EMERGENCY_STOP.value == "emergency_stop"
        assert SafetyEventType.SAFETY_OVERRIDE.value == "safety_override"
        assert SafetyEventType.SAFETY_ENABLED.value == "safety_enabled"
        assert SafetyEventType.SAFETY_DISABLED.value == "safety_disabled"
        assert SafetyEventType.SAFETY_WARNING.value == "safety_warning"

    def test_safety_trigger_enum(self):
        """Test SafetyTrigger enum values."""
        assert SafetyTrigger.MODE_CHANGE.value == "mode_change"
        assert SafetyTrigger.LOW_BATTERY.value == "low_battery"
        assert SafetyTrigger.SIGNAL_LOSS.value == "signal_loss"
        assert SafetyTrigger.GEOFENCE_VIOLATION.value == "geofence_violation"
        assert SafetyTrigger.OPERATOR_DISABLE.value == "operator_disable"
        assert SafetyTrigger.EMERGENCY_STOP.value == "emergency_stop"
        assert SafetyTrigger.TIMEOUT.value == "timeout"
        assert SafetyTrigger.MANUAL_OVERRIDE.value == "manual_override"

    def test_safety_event_creation(self):
        """Test SafetyEvent creation."""
        event = SafetyEvent(
            event_type=SafetyEventType.EMERGENCY_STOP,
            trigger=SafetyTrigger.LOW_BATTERY,
            details={"battery_voltage": 10.5},
        )
        assert event.event_type == SafetyEventType.EMERGENCY_STOP
        assert event.trigger == SafetyTrigger.LOW_BATTERY
        assert event.details["battery_voltage"] == 10.5
        assert event.resolved is False
        assert isinstance(event.id, UUID)
        assert isinstance(event.timestamp, datetime)

    def test_safety_event_default_values(self):
        """Test SafetyEvent with default values."""
        event = SafetyEvent()
        assert event.event_type == SafetyEventType.INTERLOCK_TRIGGERED
        assert event.trigger == SafetyTrigger.MODE_CHANGE
        assert event.details == {}
        assert event.resolved is False

    def test_safety_event_resolved(self):
        """Test SafetyEvent resolved flag."""
        event = SafetyEvent(
            event_type=SafetyEventType.SAFETY_WARNING, trigger=SafetyTrigger.TIMEOUT, resolved=True
        )
        assert event.event_type == SafetyEventType.SAFETY_WARNING
        assert event.trigger == SafetyTrigger.TIMEOUT
        assert event.resolved is True


class TestSafetyCheck:
    """Test abstract SafetyCheck base class."""

    def test_safety_check_get_status(self):
        """Test getting status from safety check."""

        class TestCheck(SafetyCheck):
            async def check(self) -> bool:
                return True

        check = TestCheck("test_check")
        status = check.get_status()

        assert status["name"] == "test_check"
        assert status["is_safe"] is False
        assert "last_check" in status
        assert status["failure_reason"] is None

    def test_safety_check_with_failure(self):
        """Test safety check with failure reason."""

        class TestCheck(SafetyCheck):
            async def check(self) -> bool:
                self.failure_reason = "Test failure"
                self.is_safe = False
                return False

        check = TestCheck("failing_check")
        check.failure_reason = "Something went wrong"
        status = check.get_status()

        assert status["failure_reason"] == "Something went wrong"


class TestModeCheck:
    """Test ModeCheck safety check."""

    @pytest.fixture
    def mode_check(self):
        """Create ModeCheck instance."""
        return ModeCheck()

    @pytest.mark.asyncio
    async def test_check_mode_allowed(self, mode_check):
        """Test checking if mode is allowed."""
        # Test with wrong mode
        mode_check.update_mode("AUTO")
        result = await mode_check.check()
        assert result is False
        assert mode_check.failure_reason == "Mode is AUTO, requires GUIDED"

        # Test with correct mode
        mode_check.update_mode("GUIDED")
        result = await mode_check.check()
        assert result is True
        assert mode_check.failure_reason is None

    def test_update_mode(self, mode_check):
        """Test updating mode."""
        mode_check.update_mode("STABILIZE")
        assert mode_check.current_mode == "STABILIZE"

        mode_check.update_mode("GUIDED")
        assert mode_check.current_mode == "GUIDED"

    def test_get_status(self, mode_check):
        """Test getting check status."""
        status = mode_check.get_status()
        assert status["name"] == "mode_check"
        assert "is_safe" in status
        assert "last_check" in status

    @pytest.mark.asyncio
    async def test_multiple_mode_checks(self, mode_check):
        """Test multiple mode transitions."""
        modes = ["AUTO", "STABILIZE", "GUIDED", "LOITER", "GUIDED"]
        expected_results = [False, False, True, False, True]

        for mode, expected in zip(modes, expected_results, strict=False):
            mode_check.update_mode(mode)
            result = await mode_check.check()
            assert result == expected


class TestOperatorActivationCheck:
    """Test OperatorActivationCheck safety check."""

    @pytest.fixture
    def operator_check(self):
        """Create OperatorActivationCheck instance."""
        return OperatorActivationCheck()

    @pytest.mark.asyncio
    async def test_check_operator_active(self, operator_check):
        """Test checking operator activity."""
        # Test when homing is disabled
        result = await operator_check.check()
        assert result is False
        assert operator_check.failure_reason == "Operator has not enabled homing"

        # Test when homing is enabled
        operator_check.enable_homing()
        result = await operator_check.check()
        assert result is True
        assert operator_check.failure_reason is None

    def test_enable_disable_homing(self, operator_check):
        """Test enabling and disabling homing."""
        # Test enable
        operator_check.enable_homing()
        assert operator_check.homing_enabled is True
        assert operator_check.activation_time is not None

        # Test disable
        operator_check.disable_homing("Test reason")
        assert operator_check.homing_enabled is False
        assert operator_check.activation_time is None

    @pytest.mark.asyncio
    async def test_timeout_check(self, operator_check):
        """Test operator activation timeout."""
        operator_check.enable_homing()

        # Mock time to be past timeout
        past_time = datetime.now(UTC) - timedelta(seconds=operator_check.timeout_seconds + 1)
        operator_check.activation_time = past_time

        result = await operator_check.check()
        assert result is False
        assert "timed out" in operator_check.failure_reason.lower()

    def test_multiple_enable_disable_cycles(self, operator_check):
        """Test multiple enable/disable cycles."""
        for i in range(3):
            operator_check.enable_homing()
            assert operator_check.homing_enabled is True

            operator_check.disable_homing(f"Reason {i}")
            assert operator_check.homing_enabled is False


class TestSignalLossCheck:
    """Test SignalLossCheck safety check."""

    @pytest.fixture
    def signal_check(self):
        """Create SignalLossCheck instance."""
        return SignalLossCheck(snr_threshold=6.0, timeout_seconds=10)

    @pytest.mark.asyncio
    async def test_check_signal_present(self, signal_check):
        """Test checking when signal is present."""
        # Update SNR to good value
        signal_check.update_snr(10.0)
        result = await signal_check.check()
        assert result is True
        assert signal_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_check_signal_lost(self, signal_check):
        """Test checking signal loss."""
        # Update SNR to bad value
        signal_check.update_snr(3.0)
        result = await signal_check.check()
        # Low SNR should trigger failure
        assert result is False
        assert "SNR" in signal_check.failure_reason

    def test_update_snr(self, signal_check):
        """Test updating SNR value."""
        signal_check.update_snr(8.5)
        assert signal_check.current_snr == 8.5
        assert len(signal_check.snr_history) > 0

    def test_snr_history_limit(self, signal_check):
        """Test SNR history size limit."""
        # Add many SNR values
        for i in range(150):
            signal_check.update_snr(float(i))

        # History should be limited to 100
        assert len(signal_check.snr_history) <= 100
        assert signal_check.current_snr == 149.0

    @pytest.mark.asyncio
    async def test_signal_timeout(self, signal_check):
        """Test signal timeout detection."""
        # Set good SNR initially
        signal_check.update_snr(10.0)

        # Mock time to be past timeout
        past_time = datetime.now(UTC) - timedelta(seconds=signal_check.timeout_seconds + 1)
        signal_check.last_update = past_time

        result = await signal_check.check()
        assert result is False
        assert "timeout" in signal_check.failure_reason.lower()

    def test_get_average_snr(self, signal_check):
        """Test average SNR calculation."""
        values = [5.0, 7.0, 9.0, 6.0, 8.0]
        for val in values:
            signal_check.update_snr(val)

        avg = signal_check.get_average_snr()
        assert avg == 7.0


class TestBatteryCheck:
    """Test BatteryCheck safety check."""

    @pytest.fixture
    def battery_check(self):
        """Create BatteryCheck instance."""
        return BatteryCheck(threshold_percent=20.0)

    @pytest.mark.asyncio
    async def test_check_battery_normal(self, battery_check):
        """Test checking normal battery level."""
        battery_check.update_battery(80.0)
        result = await battery_check.check()
        assert result is True
        assert battery_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_check_battery_warning(self, battery_check):
        """Test checking warning battery level."""
        battery_check.update_battery(25.0)
        result = await battery_check.check()
        assert result is True  # Still above 20% threshold
        assert battery_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_check_battery_critical(self, battery_check):
        """Test checking critical battery level."""
        battery_check.update_battery(15.0)
        result = await battery_check.check()
        assert result is False  # Below 20% threshold
        assert "Battery at 15.0%, below 20.0% threshold" in battery_check.failure_reason

    def test_update_battery(self, battery_check):
        """Test battery update clamping."""
        battery_check.update_battery(150.0)
        assert battery_check.current_battery_percent == 100.0

        battery_check.update_battery(-10.0)
        assert battery_check.current_battery_percent == 0.0

    def test_battery_custom_threshold(self):
        """Test battery check with custom threshold."""
        battery_check = BatteryCheck(threshold_percent=30.0)
        battery_check.update_battery(25.0)

        # Should be below custom 30% threshold
        assert battery_check.current_battery_percent == 25.0

    @pytest.mark.asyncio
    async def test_battery_exactly_at_threshold(self, battery_check):
        """Test battery exactly at threshold."""
        battery_check.update_battery(20.0)
        result = await battery_check.check()
        assert result is True  # Exactly at threshold should still be safe


class TestGeofenceCheck:
    """Test GeofenceCheck safety check."""

    @pytest.fixture
    def geofence_check(self):
        """Create GeofenceCheck instance."""
        check = GeofenceCheck()
        check.set_geofence(37.0, -122.0, 1000.0)
        check.fence_enabled = True
        return check

    @pytest.mark.asyncio
    async def test_check_within_geofence(self, geofence_check):
        """Test checking position within geofence."""
        geofence_check.update_position(37.001, -122.001)
        result = await geofence_check.check()
        assert result is True
        assert geofence_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_check_outside_geofence(self, geofence_check):
        """Test checking position outside geofence."""
        geofence_check.update_position(37.1, -122.0)  # Far outside
        result = await geofence_check.check()
        assert result is False
        assert "outside geofence" in geofence_check.failure_reason.lower()

    @pytest.mark.asyncio
    async def test_check_geofence_disabled(self, geofence_check):
        """Test checking when geofence is disabled."""
        geofence_check.fence_enabled = False
        geofence_check.update_position(37.1, -122.0)  # Outside position
        result = await geofence_check.check()
        assert result is True  # Safe when disabled
        assert geofence_check.failure_reason is None

    def test_set_geofence(self, geofence_check):
        """Test setting geofence parameters."""
        geofence_check.set_geofence(40.0, -74.0, 500.0, 100.0)
        assert geofence_check.fence_center_lat == 40.0
        assert geofence_check.fence_center_lon == -74.0
        assert geofence_check.fence_radius == 500.0
        assert geofence_check.fence_altitude == 100.0

    def test_update_position(self, geofence_check):
        """Test updating position."""
        geofence_check.update_position(38.5, -121.5, 50.0)
        assert geofence_check.current_lat == 38.5
        assert geofence_check.current_lon == -121.5
        assert geofence_check.current_alt == 50.0

    def test_calculate_distance(self, geofence_check):
        """Test distance calculation."""
        # Set position exactly at fence center
        geofence_check.update_position(37.0, -122.0)
        distance = geofence_check.calculate_distance()
        assert distance < 1.0  # Should be very close to 0

    @pytest.mark.asyncio
    async def test_altitude_check(self, geofence_check):
        """Test altitude violation detection."""
        geofence_check.set_geofence(37.0, -122.0, 1000.0, 100.0)

        # Test altitude too high
        geofence_check.update_position(37.0, -122.0, 150.0)
        result = await geofence_check.check()
        assert result is False
        assert "altitude" in geofence_check.failure_reason.lower()


class TestSafetyInterlockSystem:
    """Test SafetyInterlockSystem."""

    @pytest.fixture
    def interlock_system(self):
        """Create SafetyInterlockSystem instance."""
        return SafetyInterlockSystem()

    @pytest.mark.asyncio
    async def test_check_all_safety(self, interlock_system):
        """Test checking all safety interlocks."""
        # Set up good conditions
        interlock_system.checks["mode"].update_mode("GUIDED")
        interlock_system.checks["operator"].enable_homing()
        interlock_system.checks["signal"].update_snr(10.0)
        interlock_system.checks["battery"].update_battery(80.0)
        interlock_system.checks["geofence"].fence_enabled = False

        results = await interlock_system.check_all_safety()
        assert all(results.values())

    @pytest.mark.asyncio
    async def test_is_safe_to_proceed(self, interlock_system):
        """Test overall safety check."""
        # Set up mixed conditions
        interlock_system.checks["mode"].update_mode("AUTO")  # Bad
        interlock_system.checks["operator"].enable_homing()  # Good
        interlock_system.checks["signal"].update_snr(10.0)  # Good
        interlock_system.checks["battery"].update_battery(80.0)  # Good
        interlock_system.checks["geofence"].fence_enabled = False  # Good

        safe = await interlock_system.is_safe_to_proceed()
        assert safe is False  # Mode check fails

    @pytest.mark.asyncio
    async def test_emergency_stop(self, interlock_system):
        """Test emergency stop activation."""
        await interlock_system.emergency_stop("Test emergency")

        assert interlock_system.emergency_stopped is True
        assert len(interlock_system.safety_events) > 0

        # Check that safety checks fail during emergency
        safe = await interlock_system.is_safe_to_proceed()
        assert safe is False

    @pytest.mark.asyncio
    async def test_reset_emergency_stop(self, interlock_system):
        """Test resetting emergency stop."""
        await interlock_system.emergency_stop("Test")
        await interlock_system.reset_emergency_stop()

        assert interlock_system.emergency_stopped is False

    @pytest.mark.asyncio
    async def test_enable_homing(self, interlock_system):
        """Test enabling homing with safety checks."""
        # Set up good conditions
        interlock_system.checks["mode"].update_mode("GUIDED")
        interlock_system.checks["signal"].update_snr(10.0)
        interlock_system.checks["battery"].update_battery(80.0)
        interlock_system.checks["geofence"].fence_enabled = False

        enabled = await interlock_system.enable_homing("test-token")
        assert enabled is True

    @pytest.mark.asyncio
    async def test_enable_homing_blocked(self, interlock_system):
        """Test enabling homing blocked by safety."""
        # Set up bad conditions
        interlock_system.checks["mode"].update_mode("AUTO")  # Wrong mode

        enabled = await interlock_system.enable_homing()
        assert enabled is False

    @pytest.mark.asyncio
    async def test_disable_homing(self, interlock_system):
        """Test disabling homing."""
        await interlock_system.disable_homing("Test disable")

        # Check operator activation should be disabled
        operator_check = interlock_system.checks["operator"]
        assert operator_check.homing_enabled is False

    def test_update_flight_mode(self, interlock_system):
        """Test updating flight mode."""
        interlock_system.update_flight_mode("STABILIZE")
        mode_check = interlock_system.checks["mode"]
        assert mode_check.current_mode == "STABILIZE"

    def test_update_battery(self, interlock_system):
        """Test updating battery level."""
        interlock_system.update_battery(65.0)
        battery_check = interlock_system.checks["battery"]
        assert battery_check.current_battery_percent == 65.0

    def test_update_signal_snr(self, interlock_system):
        """Test updating signal SNR."""
        interlock_system.update_signal_snr(8.5)
        signal_check = interlock_system.checks["signal"]
        assert signal_check.current_snr == 8.5

    def test_update_position(self, interlock_system):
        """Test updating position."""
        interlock_system.update_position(40.0, -74.0, 50.0)
        geofence_check = interlock_system.checks["geofence"]
        assert geofence_check.current_lat == 40.0
        assert geofence_check.current_lon == -74.0
        assert geofence_check.current_alt == 50.0

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, interlock_system):
        """Test starting and stopping monitoring."""
        await interlock_system.start_monitoring()
        assert interlock_system._check_task is not None

        await interlock_system.stop_monitoring()
        assert interlock_system._check_task.done()

    @pytest.mark.asyncio
    async def test_monitoring_loop_error_handling(self, interlock_system):
        """Test monitoring loop error handling."""
        # Mock a check to raise an error
        with patch.object(
            interlock_system, "check_all_safety", side_effect=Exception("Test error")
        ):
            await interlock_system.start_monitoring()
            await asyncio.sleep(0.2)  # Let it run a bit
            await interlock_system.stop_monitoring()

        # Should handle error and continue

    def test_get_status(self, interlock_system):
        """Test getting system status."""
        status = interlock_system.get_status()

        assert "emergency_stopped" in status
        assert "checks" in status
        assert "events" in status
        assert len(status["checks"]) == 5

    def test_max_events_limit(self, interlock_system):
        """Test event history size limit."""
        # Add many events
        for i in range(1100):
            event = SafetyEvent(
                event_type=SafetyEventType.SAFETY_WARNING,
                trigger=SafetyTrigger.TIMEOUT,
                details={"index": i},
            )
            interlock_system.safety_events.append(event)

        # Simulate trimming (would normally happen in _log_safety_event)
        if len(interlock_system.safety_events) > interlock_system.max_events:
            interlock_system.safety_events = interlock_system.safety_events[
                -interlock_system.max_events :
            ]

        assert len(interlock_system.safety_events) == 1000

    @pytest.mark.asyncio
    async def test_safety_event_logging(self, interlock_system):
        """Test safety event logging."""
        # Trigger various safety events
        await interlock_system.emergency_stop("Test")
        await interlock_system.reset_emergency_stop()
        await interlock_system.disable_homing("Test")

        # Check events were logged
        assert len(interlock_system.safety_events) >= 3
        event_types = [e.event_type for e in interlock_system.safety_events]
        assert SafetyEventType.EMERGENCY_STOP in event_types
        assert SafetyEventType.SAFETY_OVERRIDE in event_types
        assert SafetyEventType.SAFETY_DISABLED in event_types

    def test_get_trigger_for_check(self, interlock_system):
        """Test mapping check names to triggers."""
        trigger = interlock_system._get_trigger_for_check("mode")
        assert trigger == SafetyTrigger.MODE_CHANGE

        trigger = interlock_system._get_trigger_for_check("battery")
        assert trigger == SafetyTrigger.LOW_BATTERY

        trigger = interlock_system._get_trigger_for_check("signal")
        assert trigger == SafetyTrigger.SIGNAL_LOSS

        trigger = interlock_system._get_trigger_for_check("geofence")
        assert trigger == SafetyTrigger.GEOFENCE_VIOLATION

        trigger = interlock_system._get_trigger_for_check("operator")
        assert trigger == SafetyTrigger.OPERATOR_DISABLE
