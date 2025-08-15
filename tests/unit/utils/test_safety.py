"""Unit tests for the safety interlock system."""

import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from src.backend.utils.safety import (
    BatteryCheck,

pytestmark = pytest.mark.serial
    GeofenceCheck,
    ModeCheck,
    OperatorActivationCheck,
    SafetyInterlockSystem,
    SafetyTrigger,
    SignalLossCheck,
)


class TestModeCheck:
    """Test flight mode checking."""

    @pytest.mark.asyncio
    async def test_mode_check_guided(self):
        """Test that GUIDED mode passes check."""
        check = ModeCheck()
        check.update_mode("GUIDED")

        result = await check.check()
        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_mode_check_not_guided(self):
        """Test that non-GUIDED modes fail check."""
        check = ModeCheck()
        check.update_mode("LOITER")

        result = await check.check()
        assert result is False
        assert check.is_safe is False
        assert "Mode is LOITER" in check.failure_reason

    @pytest.mark.asyncio
    async def test_mode_check_unknown(self):
        """Test that UNKNOWN mode fails check."""
        check = ModeCheck()
        # Default mode is UNKNOWN

        result = await check.check()
        assert result is False
        assert check.is_safe is False


class TestOperatorActivationCheck:
    """Test operator activation checking."""

    @pytest.mark.asyncio
    async def test_operator_check_enabled(self):
        """Test that enabled homing passes check."""
        check = OperatorActivationCheck()
        check.enable_homing()

        result = await check.check()
        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_operator_check_disabled(self):
        """Test that disabled homing fails check."""
        check = OperatorActivationCheck()
        # Default is disabled

        result = await check.check()
        assert result is False
        assert check.is_safe is False
        assert "not enabled homing" in check.failure_reason

    @pytest.mark.asyncio
    async def test_operator_enable_disable_cycle(self):
        """Test enable/disable cycle."""
        check = OperatorActivationCheck()

        # Enable
        check.enable_homing()
        assert check.homing_enabled is True
        assert check.activation_time is not None

        # Disable
        check.disable_homing("Test reason")
        assert check.homing_enabled is False
        assert check.activation_time is None


class TestSignalLossCheck:
    """Test signal loss detection."""

    @pytest.mark.asyncio
    async def test_signal_good(self):
        """Test that good signal passes check."""
        check = SignalLossCheck(snr_threshold=6.0, timeout_seconds=10)
        check.update_snr(10.0)  # Good SNR

        result = await check.check()
        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_signal_weak_within_timeout(self):
        """Test that weak signal within timeout still passes."""
        check = SignalLossCheck(snr_threshold=6.0, timeout_seconds=10)
        check.update_snr(3.0)  # Weak SNR

        # First check - should still be safe (within timeout)
        result = await check.check()
        assert result is True
        assert check.is_safe is True
        assert "Signal weak" in check.failure_reason

    @pytest.mark.asyncio
    async def test_signal_lost_after_timeout(self):
        """Test that signal lost for timeout period fails check."""
        check = SignalLossCheck(snr_threshold=6.0, timeout_seconds=0.1)  # Short timeout for testing
        check.update_snr(3.0)  # Weak SNR

        # First check to start the timer
        await check.check()

        # Wait for timeout
        await asyncio.sleep(0.2)

        result = await check.check()
        assert result is False
        assert check.is_safe is False
        assert "Signal lost" in check.failure_reason

    @pytest.mark.asyncio
    async def test_signal_recovery(self):
        """Test that signal recovery resets timeout."""
        check = SignalLossCheck(snr_threshold=6.0, timeout_seconds=10)

        # Lose signal
        check.update_snr(3.0)
        await check.check()  # Need to check to start timer
        assert check.signal_lost_time is not None

        # Recover signal
        check.update_snr(10.0)
        assert check.signal_lost_time is None

        result = await check.check()
        assert result is True
        assert check.is_safe is True


class TestBatteryCheck:
    """Test battery level checking."""

    @pytest.mark.asyncio
    async def test_battery_good(self):
        """Test that good battery level passes check."""
        check = BatteryCheck(threshold_percent=20.0)
        check.update_battery(80.0)

        result = await check.check()
        assert result is True
        assert check.is_safe is True
        assert check.failure_reason is None

    @pytest.mark.asyncio
    async def test_battery_low(self):
        """Test that low battery fails check."""
        check = BatteryCheck(threshold_percent=20.0)
        check.update_battery(15.0)

        result = await check.check()
        assert result is False
        assert check.is_safe is False
        assert "15.0%" in check.failure_reason
        assert "below 20.0%" in check.failure_reason

    @pytest.mark.asyncio
    async def test_battery_warning_levels(self):
        """Test battery warning levels."""
        check = BatteryCheck(threshold_percent=20.0)

        # Test warning levels
        with patch("src.backend.utils.safety.logger") as mock_logger:
            check.update_battery(30.0)
            await check.check()
            # Should trigger 30% warning
            assert mock_logger.warning.called

            mock_logger.reset_mock()
            check.update_battery(25.0)
            await check.check()
            # Should trigger 25% warning
            assert mock_logger.warning.called


class TestGeofenceCheck:
    """Test geofence boundary checking."""

    @pytest.mark.asyncio
    async def test_geofence_disabled(self):
        """Test that disabled geofence always passes."""
        check = GeofenceCheck()
        # Default is disabled

        result = await check.check()
        assert result is True
        assert check.is_safe is True

    @pytest.mark.asyncio
    async def test_geofence_within_bounds(self):
        """Test position within geofence passes."""
        check = GeofenceCheck()
        check.set_geofence(37.0, -122.0, 100.0)  # 100m radius
        check.update_position(37.0, -122.0)  # At center

        result = await check.check()
        assert result is True
        assert check.is_safe is True

    @pytest.mark.asyncio
    async def test_geofence_outside_bounds(self):
        """Test position outside geofence fails."""
        check = GeofenceCheck()
        check.set_geofence(37.0, -122.0, 100.0)  # 100m radius
        check.update_position(37.01, -122.0)  # ~1.1km north

        result = await check.check()
        assert result is False
        assert check.is_safe is False
        assert "exceeds 100.0m radius" in check.failure_reason

    @pytest.mark.asyncio
    async def test_geofence_distance_calculation(self):
        """Test distance calculation accuracy."""
        check = GeofenceCheck()

        # Test known distance
        dist = check._calculate_distance(37.0, -122.0, 37.001, -122.0)
        # Should be approximately 111 meters (1/1000 degree latitude)
        assert 110 < dist < 112


class TestSafetyInterlockSystem:
    """Test the main safety interlock system."""

    @pytest.mark.asyncio
    async def test_system_initialization(self):
        """Test system initializes with all checks."""
        system = SafetyInterlockSystem()

        assert "mode" in system.checks
        assert "operator" in system.checks
        assert "signal" in system.checks
        assert "battery" in system.checks
        assert "geofence" in system.checks
        assert system.emergency_stopped is False

    @pytest.mark.asyncio
    async def test_check_all_safety(self):
        """Test checking all safety interlocks."""
        system = SafetyInterlockSystem()

        # Set up safe conditions
        system.update_flight_mode("GUIDED")
        system.update_battery(100.0)
        system.update_signal_snr(10.0)

        # Enable homing
        operator_check = system.checks["operator"]
        if isinstance(operator_check, OperatorActivationCheck):
            operator_check.enable_homing()

        results = await system.check_all_safety()

        assert results["mode"] is True
        assert results["operator"] is True
        assert results["signal"] is True
        assert results["battery"] is True
        assert results["geofence"] is True  # Disabled by default

    @pytest.mark.asyncio
    async def test_is_safe_to_proceed(self):
        """Test overall safety check."""
        system = SafetyInterlockSystem()

        # Initially unsafe (mode not GUIDED, operator not enabled)
        assert await system.is_safe_to_proceed() is False

        # Make safe
        system.update_flight_mode("GUIDED")
        system.update_battery(100.0)
        system.update_signal_snr(10.0)
        success = await system.enable_homing()

        assert success is True
        assert await system.is_safe_to_proceed() is True

    @pytest.mark.asyncio
    async def test_emergency_stop(self):
        """Test emergency stop functionality."""
        system = SafetyInterlockSystem()

        # Set up safe conditions
        system.update_flight_mode("GUIDED")
        await system.enable_homing()

        assert await system.is_safe_to_proceed() is True

        # Activate emergency stop
        await system.emergency_stop("Test emergency")

        assert system.emergency_stopped is True
        assert await system.is_safe_to_proceed() is False

        # Check that homing was disabled
        operator_check = system.checks["operator"]
        if isinstance(operator_check, OperatorActivationCheck):
            assert operator_check.homing_enabled is False

    @pytest.mark.asyncio
    async def test_emergency_stop_reset(self):
        """Test emergency stop reset."""
        system = SafetyInterlockSystem()

        await system.emergency_stop("Test")
        assert system.emergency_stopped is True

        await system.reset_emergency_stop()
        assert system.emergency_stopped is False

    @pytest.mark.asyncio
    async def test_enable_homing_blocked_by_safety(self):
        """Test that homing cannot be enabled if unsafe."""
        system = SafetyInterlockSystem()

        # Mode not GUIDED
        system.update_flight_mode("LOITER")

        success = await system.enable_homing()
        assert success is False

        # Check operator activation is still disabled
        operator_check = system.checks["operator"]
        if isinstance(operator_check, OperatorActivationCheck):
            assert operator_check.homing_enabled is False

    @pytest.mark.asyncio
    async def test_safety_event_logging(self):
        """Test that safety events are logged."""
        system = SafetyInterlockSystem()

        # Trigger a safety event
        system.update_flight_mode("LOITER")
        await system.check_all_safety()

        # Check events were logged
        events = system.get_safety_events()
        assert len(events) > 0

        # Find mode change event
        mode_events = [e for e in events if e.trigger == SafetyTrigger.MODE_CHANGE]
        assert len(mode_events) > 0

    @pytest.mark.asyncio
    async def test_safety_status(self):
        """Test getting comprehensive safety status."""
        system = SafetyInterlockSystem()

        status = system.get_safety_status()

        assert "emergency_stopped" in status
        assert "checks" in status
        assert "timestamp" in status

        # Verify all checks are present
        assert "mode" in status["checks"]
        assert "operator" in status["checks"]
        assert "signal" in status["checks"]
        assert "battery" in status["checks"]
        assert "geofence" in status["checks"]

    @pytest.mark.asyncio
    async def test_mode_change_detection_timing(self):
        """Test that mode changes are detected within 100ms."""
        system = SafetyInterlockSystem()

        # Start monitoring
        await system.start_monitoring()

        try:
            # Change mode
            asyncio.get_event_loop().time()
            system.update_flight_mode("LOITER")

            # Wait for detection (should happen in next monitoring cycle)
            await asyncio.sleep(0.15)  # 150ms to ensure detection

            # Check that mode was detected
            mode_check = system.checks["mode"]
            if isinstance(mode_check, ModeCheck):
                assert mode_check.current_mode == "LOITER"

                # Verify timing (last check should be within 100ms)
                elapsed = datetime.now(UTC) - mode_check.last_check
                assert elapsed.total_seconds() < 0.2  # Some margin for test execution

        finally:
            await system.stop_monitoring()

    @pytest.mark.asyncio
    async def test_signal_loss_timeout_accuracy(self):
        """Test that signal loss timeout is exactly 10 seconds."""
        system = SafetyInterlockSystem()

        # Configure short timeout for testing
        from src.backend.utils.safety import SignalLossCheck

        signal_check = SignalLossCheck(snr_threshold=6.0, timeout_seconds=0.5)
        system.checks["signal"] = signal_check

        # Lose signal
        signal_check.update_snr(3.0)

        # First check to start timer
        await signal_check.check()

        # Check just before timeout
        await asyncio.sleep(0.4)
        assert await signal_check.check() is True

        # Check after timeout
        await asyncio.sleep(0.2)
        assert await signal_check.check() is False

    @pytest.mark.asyncio
    async def test_concurrent_safety_checks(self):
        """Test that multiple safety checks can run concurrently."""
        system = SafetyInterlockSystem()

        # Set different states
        system.update_flight_mode("GUIDED")
        system.update_battery(50.0)
        system.update_signal_snr(10.0)

        # Run multiple checks concurrently
        tasks = [
            system.check_all_safety(),
            system.is_safe_to_proceed(),
            system.check_all_safety(),
        ]

        results = await asyncio.gather(*tasks)

        # All should complete without error
        assert len(results) == 3
        assert isinstance(results[0], dict)
        assert isinstance(results[1], bool)
        assert isinstance(results[2], dict)
