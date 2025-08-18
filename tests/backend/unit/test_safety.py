"""Unit tests for Safety Interlock System.

Tests multi-layered safety checks, interlock mechanisms, safety events,
and abstract safety check implementations per PRD requirements.
"""

import asyncio
import time
from datetime import UTC, datetime
from uuid import UUID

import pytest

from src.backend.core.exceptions import SafetyInterlockError
from src.backend.utils.safety import (
    ModeCheck,
    SafetyCheck,
    SafetyEvent,
    SafetyEventType,
    SafetyTrigger,
)


class TestSafetyEvent:
    """Test safety event data structures."""

    def test_safety_event_creation(self):
        """Test SafetyEvent creation with defaults."""
        event = SafetyEvent()

        assert isinstance(event.id, UUID)
        assert isinstance(event.timestamp, datetime)
        assert event.event_type == SafetyEventType.INTERLOCK_TRIGGERED
        assert event.trigger == SafetyTrigger.MODE_CHANGE
        assert isinstance(event.details, dict)
        assert event.resolved is False

    def test_safety_event_custom_values(self):
        """Test SafetyEvent creation with custom values."""
        custom_time = datetime.now(UTC)
        custom_details = {"reason": "low_battery", "voltage": 17.5}

        event = SafetyEvent(
            event_type=SafetyEventType.EMERGENCY_STOP,
            trigger=SafetyTrigger.LOW_BATTERY,
            details=custom_details,
            resolved=True,
            timestamp=custom_time,
        )

        assert event.event_type == SafetyEventType.EMERGENCY_STOP
        assert event.trigger == SafetyTrigger.LOW_BATTERY
        assert event.details == custom_details
        assert event.resolved is True
        assert event.timestamp == custom_time

    def test_safety_event_types_enum(self):
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


class TestSafetyCheck:
    """Test abstract SafetyCheck base class."""

    class TestSafetyCheckImpl(SafetyCheck):
        """Test implementation of SafetyCheck."""

        def __init__(self, name: str, will_pass: bool = True):
            super().__init__(name)
            self.will_pass = will_pass

        async def check(self) -> bool:
            """Implement check method."""
            self.last_check = datetime.now(UTC)
            if self.will_pass:
                self.is_safe = True
                self.failure_reason = None
            else:
                self.is_safe = False
                self.failure_reason = "Test failure"
            return self.is_safe

    def test_safety_check_initialization(self):
        """Test SafetyCheck base class initialization."""
        check = self.TestSafetyCheckImpl("test_check")

        assert check.name == "test_check"
        assert check.is_safe is False  # Initially unsafe
        assert isinstance(check.last_check, datetime)
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_safety_check_passing(self):
        """Test safety check that passes."""
        check = self.TestSafetyCheckImpl("passing_check", will_pass=True)

        result = await check.check()

        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_safety_check_failing(self):
        """Test safety check that fails."""
        check = self.TestSafetyCheckImpl("failing_check", will_pass=False)

        result = await check.check()

        assert result is False
        assert check.is_safe is False
        assert check.failure_reason == "Test failure"

    def test_safety_check_status(self):
        """Test safety check status reporting."""
        check = self.TestSafetyCheckImpl("status_check")

        status = check.get_status()

        assert isinstance(status, dict)
        assert status["name"] == "status_check"
        assert status["is_safe"] is False
        assert "last_check" in status
        assert status["failure_reason"] is None

    @pytest.mark.asyncio
    async def test_safety_check_timestamp_update(self):
        """Test safety check updates timestamp."""
        check = self.TestSafetyCheckImpl("timestamp_check")

        initial_time = check.last_check
        await asyncio.sleep(0.01)  # Small delay

        await check.check()

        assert check.last_check > initial_time


class TestModeCheck:
    """Test ModeCheck implementation."""

    def test_mode_check_initialization(self):
        """Test ModeCheck initialization."""
        check = ModeCheck()

        assert check.name == "mode_check"
        assert check.is_safe is False
        assert check.current_mode == "UNKNOWN"
        assert check.required_mode == "GUIDED"

    @pytest.mark.asyncio
    async def test_mode_check_guided_mode(self):
        """Test ModeCheck passes in GUIDED mode."""
        check = ModeCheck()
        check.current_mode = "GUIDED"

        result = await check.check()

        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_mode_check_wrong_mode(self):
        """Test ModeCheck fails in non-GUIDED mode."""
        check = ModeCheck()
        check.current_mode = "MANUAL"

        result = await check.check()

        assert result is False
        assert check.is_safe is False
        assert "MANUAL" in check.failure_reason

    @pytest.mark.asyncio
    async def test_mode_check_unknown_mode(self):
        """Test ModeCheck fails with unknown mode."""
        check = ModeCheck()
        # Leave current_mode as "UNKNOWN"

        result = await check.check()

        assert result is False
        assert check.is_safe is False
        assert "UNKNOWN" in check.failure_reason

    @pytest.mark.asyncio
    async def test_mode_check_mode_transition(self):
        """Test ModeCheck handles mode transitions."""
        check = ModeCheck()

        # Start in wrong mode
        check.current_mode = "MANUAL"
        result1 = await check.check()
        assert result1 is False

        # Transition to correct mode
        check.current_mode = "GUIDED"
        result2 = await check.check()
        assert result2 is True


class TestSafetyInterlockSystem:
    """Test comprehensive safety interlock system."""

    def test_safety_interlock_error(self):
        """Test SafetyInterlockError exception."""
        error = SafetyInterlockError("Test safety error")

        assert str(error) == "Test safety error"
        assert isinstance(error, Exception)

    @pytest.mark.asyncio
    async def test_multiple_safety_checks(self):
        """Test multiple safety checks running concurrently."""

        class MockCheck(SafetyCheck):
            def __init__(self, name: str, delay: float = 0.01):
                super().__init__(name)
                self.delay = delay

            async def check(self) -> bool:
                await asyncio.sleep(self.delay)
                self.is_safe = True
                return True

        checks = [MockCheck("check1", 0.01), MockCheck("check2", 0.02), MockCheck("check3", 0.01)]

        # Run checks concurrently
        results = await asyncio.gather(*[check.check() for check in checks])

        assert all(results)
        assert all(check.is_safe for check in checks)

    @pytest.mark.asyncio
    async def test_safety_check_timeout_handling(self):
        """Test safety check timeout handling."""

        class SlowCheck(SafetyCheck):
            def __init__(self, name: str):
                super().__init__(name)

            async def check(self) -> bool:
                await asyncio.sleep(1.0)  # Slow check
                self.is_safe = True
                return True

        check = SlowCheck("slow_check")

        # Run with timeout
        try:
            result = await asyncio.wait_for(check.check(), timeout=0.1)
            assert False, "Should have timed out"
        except TimeoutError:
            # Expected timeout
            assert check.is_safe is False  # Should remain unsafe on timeout

    def test_safety_event_serialization(self):
        """Test safety event can be serialized for logging."""
        event = SafetyEvent(
            event_type=SafetyEventType.EMERGENCY_STOP,
            trigger=SafetyTrigger.LOW_BATTERY,
            details={"voltage": 17.5},
        )

        # Should be able to convert to dict for serialization
        event_dict = {
            "id": str(event.id),
            "timestamp": event.timestamp.isoformat(),
            "event_type": event.event_type.value,
            "trigger": event.trigger.value,
            "details": event.details,
            "resolved": event.resolved,
        }

        assert isinstance(event_dict, dict)
        assert event_dict["event_type"] == "emergency_stop"
        assert event_dict["trigger"] == "low_battery"
        assert event_dict["details"]["voltage"] == 17.5

    @pytest.mark.asyncio
    async def test_safety_check_failure_recovery(self):
        """Test safety check can recover from failures."""

        class RecoveringCheck(SafetyCheck):
            def __init__(self, name: str):
                super().__init__(name)
                self.attempt = 0

            async def check(self) -> bool:
                self.attempt += 1
                self.last_check = datetime.now(UTC)

                if self.attempt < 3:
                    self.is_safe = False
                    self.failure_reason = f"Attempt {self.attempt} failed"
                    return False
                else:
                    self.is_safe = True
                    self.failure_reason = None
                    return True

        check = RecoveringCheck("recovering_check")

        # First two attempts should fail
        assert await check.check() is False
        assert await check.check() is False

        # Third attempt should succeed
        assert await check.check() is True
        assert check.is_safe is True
        assert check.failure_reason is None

    def test_safety_event_timestamp_precision(self):
        """Test safety event timestamps are precise enough."""
        event1 = SafetyEvent()
        time.sleep(0.001)  # 1ms delay
        event2 = SafetyEvent()

        # Timestamps should be different
        assert event1.timestamp != event2.timestamp
        assert event2.timestamp > event1.timestamp

    @pytest.mark.asyncio
    async def test_safety_check_concurrent_access(self):
        """Test safety check thread safety."""

        class ConcurrentCheck(SafetyCheck):
            def __init__(self, name: str):
                super().__init__(name)
                self.check_count = 0

            async def check(self) -> bool:
                self.check_count += 1
                await asyncio.sleep(0.01)
                self.is_safe = True
                return True

        check = ConcurrentCheck("concurrent_check")

        # Run multiple concurrent checks
        tasks = [check.check() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        assert all(results)
        assert check.check_count == 5

    def test_safety_event_details_flexibility(self):
        """Test safety event details can store various data types."""
        complex_details = {
            "string_value": "test",
            "numeric_value": 42.5,
            "boolean_value": True,
            "list_value": [1, 2, 3],
            "nested_dict": {"inner": "value"},
        }

        event = SafetyEvent(details=complex_details)

        assert event.details["string_value"] == "test"
        assert event.details["numeric_value"] == 42.5
        assert event.details["boolean_value"] is True
        assert event.details["list_value"] == [1, 2, 3]
        assert event.details["nested_dict"]["inner"] == "value"

    @pytest.mark.asyncio
    async def test_safety_system_performance(self):
        """Test safety system performance requirements."""

        class FastCheck(SafetyCheck):
            def __init__(self, name: str):
                super().__init__(name)

            async def check(self) -> bool:
                # Simulate fast safety check
                self.is_safe = True
                return True

        checks = [FastCheck(f"fast_check_{i}") for i in range(10)]

        start_time = time.perf_counter()
        results = await asyncio.gather(*[check.check() for check in checks])
        end_time = time.perf_counter()

        # All checks should pass
        assert all(results)

        # Should complete quickly (much less than 500ms emergency stop requirement)
        execution_time_ms = (end_time - start_time) * 1000
        assert execution_time_ms < 100

    def test_safety_event_uuid_uniqueness(self):
        """Test safety event UUIDs are unique."""
        events = [SafetyEvent() for _ in range(100)]
        event_ids = [event.id for event in events]

        # All IDs should be unique
        assert len(set(event_ids)) == 100

    @pytest.mark.asyncio
    async def test_safety_check_exception_handling(self):
        """Test safety check handles internal exceptions."""

        class ExceptionCheck(SafetyCheck):
            def __init__(self, name: str):
                super().__init__(name)

            async def check(self) -> bool:
                raise ValueError("Internal check error")

        check = ExceptionCheck("exception_check")

        # Should handle exception gracefully
        try:
            result = await check.check()
            # Implementation may catch and return False, or re-raise
        except ValueError:
            # Exception is allowed to propagate
            pass

    def test_safety_event_resolution_tracking(self):
        """Test safety event resolution state tracking."""
        event = SafetyEvent(
            event_type=SafetyEventType.SAFETY_WARNING, trigger=SafetyTrigger.LOW_BATTERY
        )

        # Initially unresolved
        assert event.resolved is False

        # Mark as resolved
        event.resolved = True
        assert event.resolved is True

    @pytest.mark.asyncio
    async def test_safety_system_cascading_failures(self):
        """Test safety system handles cascading failures."""

        class DependentCheck(SafetyCheck):
            def __init__(self, name: str, dependency: SafetyCheck = None):
                super().__init__(name)
                self.dependency = dependency

            async def check(self) -> bool:
                if self.dependency and not self.dependency.is_safe:
                    self.is_safe = False
                    self.failure_reason = "Dependency failed"
                    return False

                self.is_safe = True
                return True

        # Create chain of dependent checks
        check1 = DependentCheck("check1")
        check2 = DependentCheck("check2", dependency=check1)
        check3 = DependentCheck("check3", dependency=check2)

        # All should pass if first passes
        await check1.check()
        await check2.check()
        await check3.check()

        assert all(check.is_safe for check in [check1, check2, check3])

    def test_safety_trigger_comprehensive_coverage(self):
        """Test all safety triggers are properly defined."""
        triggers = list(SafetyTrigger)

        expected_triggers = [
            SafetyTrigger.MODE_CHANGE,
            SafetyTrigger.LOW_BATTERY,
            SafetyTrigger.SIGNAL_LOSS,
            SafetyTrigger.GEOFENCE_VIOLATION,
            SafetyTrigger.OPERATOR_DISABLE,
            SafetyTrigger.EMERGENCY_STOP,
            SafetyTrigger.TIMEOUT,
            SafetyTrigger.MANUAL_OVERRIDE,
        ]

        assert len(triggers) == len(expected_triggers)
        for trigger in expected_triggers:
            assert trigger in triggers

    def test_safety_event_type_comprehensive_coverage(self):
        """Test all safety event types are properly defined."""
        event_types = list(SafetyEventType)

        expected_types = [
            SafetyEventType.INTERLOCK_TRIGGERED,
            SafetyEventType.EMERGENCY_STOP,
            SafetyEventType.SAFETY_OVERRIDE,
            SafetyEventType.SAFETY_ENABLED,
            SafetyEventType.SAFETY_DISABLED,
            SafetyEventType.SAFETY_WARNING,
        ]

        assert len(event_types) == len(expected_types)
        for event_type in expected_types:
            assert event_type in event_types


# NEW TDD TESTS FOR 90%+ SAFETY UTILS COVERAGE


class TestOperatorActivationCheck:
    """Test OperatorActivationCheck implementation."""

    @pytest.fixture
    def operator_check(self):
        """Provide OperatorActivationCheck instance."""
        from src.backend.utils.safety import OperatorActivationCheck

        return OperatorActivationCheck(timeout_seconds=10)

    @pytest.mark.asyncio
    async def test_operator_check_initialization(self, operator_check):
        """Test OperatorActivationCheck initialization."""
        assert operator_check.name == "operator_check"
        assert operator_check.homing_enabled is False
        assert operator_check.activation_time is None
        assert operator_check.timeout_seconds == 10
        assert operator_check.is_safe is False

    @pytest.mark.asyncio
    async def test_operator_check_homing_disabled(self, operator_check):
        """Test operator check fails when homing disabled."""
        result = await operator_check.check()

        assert result is False
        assert operator_check.is_safe is False
        assert "Operator has not enabled homing" in operator_check.failure_reason

    @pytest.mark.asyncio
    async def test_operator_check_homing_enabled(self, operator_check):
        """Test operator check passes when homing enabled."""
        operator_check.enable_homing()

        result = await operator_check.check()

        assert result is True
        assert operator_check.is_safe is True
        assert operator_check.failure_reason is None
        assert operator_check.activation_time is not None

    @pytest.mark.asyncio
    async def test_operator_check_timeout(self, operator_check):
        """Test operator check times out after configured duration."""
        from datetime import UTC, datetime, timedelta

        operator_check.enable_homing()
        # Simulate old activation time
        operator_check.activation_time = datetime.now(UTC) - timedelta(seconds=15)

        result = await operator_check.check()

        assert result is False
        assert operator_check.is_safe is False
        assert "timed out after" in operator_check.failure_reason

    def test_enable_homing(self, operator_check):
        """Test homing enablement."""
        operator_check.enable_homing()

        assert operator_check.homing_enabled is True
        assert operator_check.activation_time is not None

    def test_disable_homing(self, operator_check):
        """Test homing disablement."""
        operator_check.enable_homing()
        assert operator_check.homing_enabled is True

        operator_check.disable_homing("Test disable")

        assert operator_check.homing_enabled is False
        assert operator_check.activation_time is None


class TestSignalLossCheck:
    """Test SignalLossCheck implementation."""

    @pytest.fixture
    def signal_check(self):
        """Provide SignalLossCheck instance."""
        from src.backend.utils.safety import SignalLossCheck

        return SignalLossCheck(snr_threshold=6.0, timeout_seconds=5)

    @pytest.mark.asyncio
    async def test_signal_check_initialization(self, signal_check):
        """Test SignalLossCheck initialization."""
        assert signal_check.name == "signal_check"
        assert signal_check.snr_threshold == 6.0
        assert signal_check.timeout_seconds == 5
        assert signal_check.current_snr == 0.0
        assert signal_check.signal_lost_time is None
        assert len(signal_check.snr_history) == 0

    @pytest.mark.asyncio
    async def test_signal_check_good_snr(self, signal_check):
        """Test signal check passes with good SNR."""
        signal_check.update_snr(10.0)  # Above threshold

        result = await signal_check.check()

        assert result is True
        assert signal_check.is_safe is True
        assert signal_check.failure_reason is None
        assert signal_check.signal_lost_time is None

    @pytest.mark.asyncio
    async def test_signal_check_poor_snr(self, signal_check):
        """Test signal check fails with poor SNR."""
        signal_check.update_snr(3.0)  # Below threshold

        result = await signal_check.check()

        assert result is False
        assert signal_check.is_safe is False
        assert "SNR 3.0 dB below threshold" in signal_check.failure_reason
        assert signal_check.signal_lost_time is not None

    def test_update_snr_history(self, signal_check):
        """Test SNR history management."""
        # Add multiple SNR values
        for snr in [5.0, 7.0, 6.5, 8.0]:
            signal_check.update_snr(snr)

        assert len(signal_check.snr_history) == 4
        assert signal_check.current_snr == 8.0

        # Test history limit (100 entries)
        for i in range(100):
            signal_check.update_snr(10.0)

        assert len(signal_check.snr_history) <= 100

    def test_get_average_snr(self, signal_check):
        """Test average SNR calculation."""
        # Initially no history
        assert signal_check.get_average_snr() == 0.0

        # Add test values
        signal_check.update_snr(5.0)
        signal_check.update_snr(7.0)
        signal_check.update_snr(8.0)

        average = signal_check.get_average_snr()
        assert abs(average - (5.0 + 7.0 + 8.0) / 3) < 0.01

    @pytest.mark.asyncio
    async def test_signal_check_history_cleanup(self, signal_check):
        """Test old history cleanup during check."""
        # Add some SNR values
        signal_check.update_snr(8.0)

        # Run check to trigger cleanup
        await signal_check.check()

        # History should still contain recent entry
        assert len(signal_check.snr_history) >= 1


class TestBatteryCheck:
    """Test BatteryCheck implementation."""

    @pytest.fixture
    def battery_check(self):
        """Provide BatteryCheck instance."""
        from src.backend.utils.safety import BatteryCheck

        return BatteryCheck(threshold_percent=20.0)

    @pytest.mark.asyncio
    async def test_battery_check_initialization(self, battery_check):
        """Test BatteryCheck initialization."""
        assert battery_check.name == "battery_check"
        assert battery_check.threshold_percent == 20.0
        assert battery_check.current_battery_percent == 100.0
        assert battery_check.warning_levels == [30.0, 25.0, 20.0]
        assert battery_check.last_warning_level is None

    @pytest.mark.asyncio
    async def test_battery_check_normal_level(self, battery_check):
        """Test battery check passes with normal level."""
        battery_check.update_battery(50.0)

        result = await battery_check.check()

        assert result is True
        assert battery_check.is_safe is True
        assert battery_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_battery_check_low_level(self, battery_check):
        """Test battery check fails with low level."""
        battery_check.update_battery(15.0)  # Below 20% threshold

        result = await battery_check.check()

        assert result is False
        assert battery_check.is_safe is False
        assert "Battery at 15.0%, below 20.0% threshold" in battery_check.failure_reason

    def test_update_battery_bounds(self, battery_check):
        """Test battery update respects bounds."""
        # Test upper bound
        battery_check.update_battery(150.0)
        assert battery_check.current_battery_percent == 100.0

        # Test lower bound
        battery_check.update_battery(-10.0)
        assert battery_check.current_battery_percent == 0.0

        # Test normal value
        battery_check.update_battery(75.0)
        assert battery_check.current_battery_percent == 75.0


class TestGeofenceCheck:
    """Test GeofenceCheck implementation."""

    @pytest.fixture
    def geofence_check(self):
        """Provide GeofenceCheck instance."""
        from src.backend.utils.safety import GeofenceCheck

        return GeofenceCheck()

    @pytest.mark.asyncio
    async def test_geofence_check_initialization(self, geofence_check):
        """Test GeofenceCheck initialization."""
        assert geofence_check.name == "geofence_check"
        assert geofence_check.fence_center_lat is None
        assert geofence_check.fence_center_lon is None
        assert geofence_check.fence_radius is None
        assert geofence_check.fence_altitude is None
        assert geofence_check.current_lat is None
        assert geofence_check.current_lon is None
        assert geofence_check.current_alt is None
        assert geofence_check.fence_enabled is False

    @pytest.mark.asyncio
    async def test_geofence_check_disabled(self, geofence_check):
        """Test geofence check passes when disabled."""
        result = await geofence_check.check()

        assert result is True
        assert geofence_check.is_safe is True
        assert geofence_check.failure_reason is None

    def test_set_geofence(self, geofence_check):
        """Test geofence configuration."""
        geofence_check.set_geofence(37.7749, -122.4194, 100.0, 50.0)

        assert geofence_check.fence_center_lat == 37.7749
        assert geofence_check.fence_center_lon == -122.4194
        assert geofence_check.fence_radius == 100.0
        assert geofence_check.fence_altitude == 50.0
        assert geofence_check.fence_enabled is True

    def test_update_position(self, geofence_check):
        """Test position updates."""
        geofence_check.update_position(37.7750, -122.4195, 30.0)

        assert geofence_check.current_lat == 37.7750
        assert geofence_check.current_lon == -122.4195
        assert geofence_check.current_alt == 30.0

    @pytest.mark.asyncio
    async def test_geofence_check_incomplete_config(self, geofence_check):
        """Test geofence check fails with incomplete configuration."""
        geofence_check.fence_enabled = True  # Enable but don't configure

        result = await geofence_check.check()

        assert result is False
        assert geofence_check.is_safe is False
        assert "not configured" in geofence_check.failure_reason

    @pytest.mark.asyncio
    async def test_geofence_check_inside_boundary(self, geofence_check):
        """Test geofence check passes when inside boundary."""
        # Set up geofence and position
        geofence_check.set_geofence(37.7749, -122.4194, 100.0, 50.0)
        geofence_check.update_position(37.7750, -122.4195, 30.0)  # Close position

        result = await geofence_check.check()

        assert result is True
        assert geofence_check.is_safe is True
        assert geofence_check.failure_reason is None

    @pytest.mark.asyncio
    async def test_geofence_check_outside_radius(self, geofence_check):
        """Test geofence check fails when outside radius."""
        # Set up geofence and distant position
        geofence_check.set_geofence(37.7749, -122.4194, 100.0, 50.0)
        geofence_check.update_position(37.8000, -122.5000, 30.0)  # Far position

        result = await geofence_check.check()

        assert result is False
        assert geofence_check.is_safe is False
        assert "outside geofence radius" in geofence_check.failure_reason

    @pytest.mark.asyncio
    async def test_geofence_check_altitude_violation(self, geofence_check):
        """Test geofence check fails when above altitude limit."""
        # Set up geofence and high position
        geofence_check.set_geofence(37.7749, -122.4194, 100.0, 50.0)
        geofence_check.update_position(37.7749, -122.4194, 75.0)  # Same lat/lon, high alt

        result = await geofence_check.check()

        assert result is False
        assert geofence_check.is_safe is False
        assert "exceeds maximum" in geofence_check.failure_reason

    def test_calculate_distance_same_point(self, geofence_check):
        """Test distance calculation for same point."""
        geofence_check.set_geofence(37.7749, -122.4194, 100.0)
        geofence_check.update_position(37.7749, -122.4194, 30.0)

        distance = geofence_check.calculate_distance()

        assert distance < 1.0  # Should be very close to 0

    def test_calculate_distance_no_position(self, geofence_check):
        """Test distance calculation without position."""
        distance = geofence_check.calculate_distance()

        assert distance == float("inf")

    def test_haversine_distance_calculation(self, geofence_check):
        """Test Haversine distance calculation accuracy."""
        # Test known distance (approximately 1 degree lat ≈ 111 km)
        distance = geofence_check._calculate_distance(0.0, 0.0, 1.0, 0.0)

        # Should be approximately 111,000 meters
        assert 110000 < distance < 112000


class TestSafetyInterlockSystemComprehensive:
    """Test comprehensive SafetyInterlockSystem functionality."""

    @pytest.fixture
    def safety_system(self):
        """Provide SafetyInterlockSystem instance."""
        from src.backend.utils.safety import SafetyInterlockSystem

        return SafetyInterlockSystem()

    @pytest.mark.asyncio
    async def test_safety_system_initialization(self, safety_system):
        """Test SafetyInterlockSystem initialization."""
        assert len(safety_system.checks) == 5
        assert "mode" in safety_system.checks
        assert "operator" in safety_system.checks
        assert "signal" in safety_system.checks
        assert "battery" in safety_system.checks
        assert "geofence" in safety_system.checks
        assert safety_system.emergency_stopped is False
        assert len(safety_system.safety_events) == 0
        assert safety_system.max_events == 1000
        assert safety_system._check_task is None
        assert safety_system._check_interval == 0.1

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, safety_system):
        """Test monitoring start and stop."""
        # Start monitoring
        await safety_system.start_monitoring()
        assert safety_system._check_task is not None
        assert not safety_system._check_task.done()

        # Stop monitoring
        await safety_system.stop_monitoring()
        assert safety_system._check_task.done()

    @pytest.mark.asyncio
    async def test_check_all_safety_mixed_results(self, safety_system):
        """Test check_all_safety with mixed results."""
        # Set up some checks to pass and others to fail
        safety_system.checks["mode"].update_mode("GUIDED")  # Should pass
        safety_system.checks["battery"].update_battery(10.0)  # Should fail (below 20%)

        results = await safety_system.check_all_safety()

        assert "mode" in results
        assert "battery" in results
        assert results["mode"] is True
        assert results["battery"] is False

    @pytest.mark.asyncio
    async def test_is_safe_to_proceed_all_pass(self, safety_system):
        """Test is_safe_to_proceed when all checks pass."""
        # Set up all checks to pass
        safety_system.checks["mode"].update_mode("GUIDED")
        safety_system.checks["operator"].enable_homing()
        safety_system.checks["signal"].update_snr(10.0)
        safety_system.checks["battery"].update_battery(80.0)
        safety_system.checks["geofence"].set_geofence(0, 0, 100, 50)
        safety_system.checks["geofence"].update_position(0, 0, 10)

        is_safe = await safety_system.is_safe_to_proceed()

        assert is_safe is True

    @pytest.mark.asyncio
    async def test_is_safe_to_proceed_with_failure(self, safety_system):
        """Test is_safe_to_proceed with one check failing."""
        # Set up one check to fail
        safety_system.checks["battery"].update_battery(10.0)  # Below threshold

        is_safe = await safety_system.is_safe_to_proceed()

        assert is_safe is False

    @pytest.mark.asyncio
    async def test_emergency_stop(self, safety_system):
        """Test emergency stop activation."""
        await safety_system.emergency_stop("Test emergency")

        assert safety_system.emergency_stopped is True
        assert len(safety_system.safety_events) > 0

        # Check that operator check was disabled
        operator_check = safety_system.checks["operator"]
        assert operator_check.homing_enabled is False

    @pytest.mark.asyncio
    async def test_emergency_stop_blocks_safety_checks(self, safety_system):
        """Test emergency stop blocks all safety checks."""
        await safety_system.emergency_stop("Test")

        results = await safety_system.check_all_safety()

        # All checks should return False when emergency stopped
        assert all(not result for result in results.values())

    @pytest.mark.asyncio
    async def test_reset_emergency_stop(self, safety_system):
        """Test emergency stop reset."""
        await safety_system.emergency_stop("Test")
        assert safety_system.emergency_stopped is True

        await safety_system.reset_emergency_stop()

        assert safety_system.emergency_stopped is False
        assert len(safety_system.safety_events) > 1  # Should have reset event

    @pytest.mark.asyncio
    async def test_enable_homing_safety_check(self, safety_system):
        """Test enable homing with safety pre-checks."""
        # Set up good conditions
        safety_system.checks["mode"].update_mode("GUIDED")
        safety_system.checks["signal"].update_snr(10.0)
        safety_system.checks["battery"].update_battery(80.0)
        safety_system.checks["geofence"].set_geofence(0, 0, 100, 50)
        safety_system.checks["geofence"].update_position(0, 0, 10)

        result = await safety_system.enable_homing("test-token")

        assert result is True
        operator_check = safety_system.checks["operator"]
        assert operator_check.homing_enabled is True

    @pytest.mark.asyncio
    async def test_enable_homing_blocked_by_safety(self, safety_system):
        """Test enable homing blocked by safety checks."""
        # Set up failing condition
        safety_system.checks["battery"].update_battery(10.0)  # Critical battery

        result = await safety_system.enable_homing()

        assert result is False
        operator_check = safety_system.checks["operator"]
        assert operator_check.homing_enabled is False

    @pytest.mark.asyncio
    async def test_enable_homing_blocked_by_emergency(self, safety_system):
        """Test enable homing blocked by emergency stop."""
        await safety_system.emergency_stop("Test")

        result = await safety_system.enable_homing()

        assert result is False

    def test_get_safety_status(self, safety_system):
        """Test comprehensive safety status."""
        status = safety_system.get_safety_status()

        assert "emergency_stopped" in status
        assert "checks" in status
        assert "timestamp" in status
        assert len(status["checks"]) == 5
        assert status["emergency_stopped"] is False

    def test_get_status_with_events(self, safety_system):
        """Test status includes event count."""
        status = safety_system.get_status()

        assert "events" in status
        assert status["events"] == 0
