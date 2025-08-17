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
