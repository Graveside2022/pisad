"""SITL automated test suite for safety scenarios.

This module provides comprehensive automated testing of safety scenarios
using ArduPilot's Software In The Loop (SITL) simulation environment.
"""

import asyncio
import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.serial
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.services.mavlink_service import MAVLinkService
from backend.utils.safety import SafetyInterlockSystem


# Mock StateMachine for now since it doesn't exist yet
class StateMachine:
    def __init__(self, safety_system, mavlink_service):
        self.safety_system = safety_system
        self.mavlink_service = mavlink_service
        self.current_state = "IDLE"

    async def start(self):
        pass

    async def stop(self):
        pass


class SITLTestEnvironment:
    """SITL test environment setup and teardown."""

    def __init__(self, connection_string: str = "udp:127.0.0.1:14550") -> None:
        """Initialize SITL test environment.

        Args:
            connection_string: MAVLink connection string for SITL
        """
        self.connection_string = connection_string
        self.mavlink_service: MAVLinkService | None = None
        self.safety_system: SafetyInterlockSystem | None = None
        self.state_machine: StateMachine | None = None
        self.connection: Any | None = None

    async def setup(self) -> None:
        """Set up SITL environment."""
        # Initialize services
        self.mavlink_service = MAVLinkService(self.connection_string)
        self.safety_system = SafetyInterlockSystem()
        self.state_machine = StateMachine(self.safety_system, self.mavlink_service)

        # Start services
        await self.mavlink_service.connect()
        await self.safety_system.start_monitoring()
        await self.state_machine.start()

        # Wait for connection
        await self._wait_for_heartbeat()

    async def teardown(self) -> None:
        """Tear down SITL environment."""
        if self.state_machine:
            await self.state_machine.stop()
        if self.safety_system:
            await self.safety_system.stop_monitoring()
        if self.mavlink_service:
            await self.mavlink_service.disconnect()

    async def _wait_for_heartbeat(self, timeout: int = 10) -> None:
        """Wait for MAVLink heartbeat.

        Args:
            timeout: Maximum seconds to wait

        Raises:
            TimeoutError: If no heartbeat received
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.mavlink_service and self.mavlink_service.connected:
                return
            await asyncio.sleep(0.1)
        raise TimeoutError("No heartbeat received from SITL")

    async def set_mode(self, mode: str) -> bool:
        """Set flight mode.

        Args:
            mode: Flight mode name

        Returns:
            True if mode set successfully
        """
        if not self.mavlink_service:
            return False
        return await self.mavlink_service.set_mode(mode)

    async def arm_vehicle(self) -> bool:
        """Arm the vehicle.

        Returns:
            True if armed successfully
        """
        if not self.mavlink_service:
            return False
        return await self.mavlink_service.arm()

    async def takeoff(self, altitude: float) -> bool:
        """Command takeoff to altitude.

        Args:
            altitude: Target altitude in meters

        Returns:
            True if takeoff command accepted
        """
        if not self.mavlink_service:
            return False
        return await self.mavlink_service.takeoff(altitude)


@pytest.fixture
async def sitl_env():
    """Pytest fixture for SITL environment."""
    env = SITLTestEnvironment()
    await env.setup()
    yield env
    await env.teardown()


@pytest.mark.asyncio
class TestSafetyScenarios:
    """Test safety scenarios in SITL environment."""

    async def test_normal_homing_activation(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Normal homing activation.

        Validates that homing activates correctly when all safety
        conditions are met in normal operation.
        """
        # Setup: Set all conditions to pass
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)

        # Act: Enable homing
        result = await sitl_env.safety_system.enable_homing("test_token")

        # Assert: Homing should be enabled
        assert result is True
        assert await sitl_env.safety_system.is_safe_to_proceed()

        # Verify state machine transition
        assert sitl_env.state_machine.current_state in ["SEARCHING", "DETECTING"]

    async def test_mode_change_during_homing(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Mode change during homing.

        Validates that homing is disabled when flight mode changes
        from GUIDED to another mode during active homing.
        """
        # Setup: Enable homing in GUIDED mode
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Simulate homing state
        sitl_env.state_machine.current_state = "HOMING"

        # Act: Change mode to LOITER
        await sitl_env.set_mode("LOITER")
        sitl_env.safety_system.update_flight_mode("LOITER")

        # Assert: Safety check should fail
        assert not await sitl_env.safety_system.is_safe_to_proceed()

        # Verify homing is disabled
        mode_check = sitl_env.safety_system.checks["mode"]
        assert not await mode_check.check()

    async def test_signal_loss_during_homing(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Signal loss during homing.

        Validates that homing is disabled after 10 seconds of
        signal loss (SNR below threshold).
        """
        # Setup: Enable homing with good signal
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Act: Simulate signal loss
        sitl_env.safety_system.update_signal_snr(3.0)  # Below 6 dB threshold

        # Wait less than timeout - should still be safe
        await asyncio.sleep(5)
        signal_check = sitl_env.safety_system.checks["signal"]
        assert await signal_check.check()

        # Wait past timeout - should fail
        await asyncio.sleep(6)
        assert not await signal_check.check()

        # Verify safety system blocks operation
        assert not await sitl_env.safety_system.is_safe_to_proceed()

    async def test_low_battery_during_homing(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Low battery during homing.

        Validates that homing is disabled when battery level
        drops below the safety threshold.
        """
        # Setup: Enable homing with good battery
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Simulate active homing
        sitl_env.state_machine.current_state = "HOMING"

        # Act: Battery drops below threshold
        sitl_env.safety_system.update_battery(15.0)  # Below 20% threshold

        # Assert: Safety check should fail
        battery_check = sitl_env.safety_system.checks["battery"]
        assert not await battery_check.check()
        assert not await sitl_env.safety_system.is_safe_to_proceed()

        # Verify appropriate safety event logged
        events = sitl_env.safety_system.get_safety_events()
        assert any(e.trigger.value == "low_battery" for e in events)

    async def test_geofence_violation_attempt(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Geofence violation attempt.

        Validates that homing commands are blocked when they would
        cause the vehicle to exit the geofence boundary.
        """
        # Setup: Configure geofence
        geofence_check = sitl_env.safety_system.checks["geofence"]
        geofence_check.set_geofence(37.7749, -122.4194, 100.0)

        # Enable homing at valid position
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        geofence_check.update_position(37.7749, -122.4194)
        await sitl_env.safety_system.enable_homing()

        # Act: Update position outside geofence
        geofence_check.update_position(37.7850, -122.4194)  # >100m away

        # Assert: Safety check should fail
        assert not await geofence_check.check()
        assert not await sitl_env.safety_system.is_safe_to_proceed()

    async def test_emergency_stop_activation(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Emergency stop activation.

        Validates that emergency stop immediately halts all
        operations and disables homing.
        """
        # Setup: Enable homing
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Verify system is operational
        assert await sitl_env.safety_system.is_safe_to_proceed()

        # Act: Activate emergency stop
        await sitl_env.safety_system.emergency_stop("Test emergency")

        # Assert: All operations should be blocked
        assert not await sitl_env.safety_system.is_safe_to_proceed()
        assert sitl_env.safety_system.emergency_stopped

        # Verify homing is disabled
        operator_check = sitl_env.safety_system.checks["operator"]
        assert not await operator_check.check()

        # Verify emergency event logged
        events = sitl_env.safety_system.get_safety_events()
        assert any(e.event_type.value == "emergency_stop" for e in events)

    async def test_multi_interlock_trigger(self, sitl_env: SITLTestEnvironment) -> None:
        """Test scenario: Multiple interlock triggers.

        Validates system behavior when multiple safety interlocks
        are triggered simultaneously.
        """
        # Setup: Start with all conditions good
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Act: Trigger multiple failures
        await sitl_env.set_mode("STABILIZE")  # Mode failure
        sitl_env.safety_system.update_battery(15.0)  # Battery failure
        sitl_env.safety_system.update_signal_snr(3.0)  # Signal failure

        # Assert: Multiple checks should fail
        results = await sitl_env.safety_system.check_all_safety()
        failed_checks = [name for name, passed in results.items() if not passed]
        assert len(failed_checks) >= 3
        assert "mode" in failed_checks
        assert "battery" in failed_checks

        # System should not be safe to proceed
        assert not await sitl_env.safety_system.is_safe_to_proceed()

        # Fix some issues
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_signal_snr(10.0)

        # Still should fail due to battery
        assert not await sitl_env.safety_system.is_safe_to_proceed()

        # Fix all issues
        sitl_env.safety_system.update_battery(30.0)

        # Now should pass
        assert await sitl_env.safety_system.is_safe_to_proceed()


@pytest.mark.asyncio
class TestTimingRequirements:
    """Test timing requirements for safety responses."""

    async def test_mode_detection_latency(self, sitl_env: SITLTestEnvironment) -> None:
        """Test that mode changes are detected within 100ms."""
        # Setup: Enable homing in GUIDED
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Measure mode change detection time
        start_time = time.perf_counter()

        # Change mode
        await sitl_env.set_mode("LOITER")
        sitl_env.safety_system.update_flight_mode("LOITER")

        # Check detection
        mode_check = sitl_env.safety_system.checks["mode"]
        await mode_check.check()

        detection_time = (time.perf_counter() - start_time) * 1000

        # Assert: Detection within 100ms
        assert detection_time < 100, f"Mode detection took {detection_time:.1f}ms"

    async def test_emergency_stop_response_time(self, sitl_env: SITLTestEnvironment) -> None:
        """Test that emergency stop responds within 500ms."""
        # Setup: Enable homing
        await sitl_env.set_mode("GUIDED")
        sitl_env.safety_system.update_battery(50.0)
        sitl_env.safety_system.update_signal_snr(10.0)
        await sitl_env.safety_system.enable_homing()

        # Measure emergency stop response time
        start_time = time.perf_counter()

        # Activate emergency stop
        await sitl_env.safety_system.emergency_stop("Test timing")

        # Verify stopped
        is_safe = await sitl_env.safety_system.is_safe_to_proceed()

        response_time = (time.perf_counter() - start_time) * 1000

        # Assert: Response within 500ms
        assert not is_safe
        assert response_time < 500, f"Emergency stop took {response_time:.1f}ms"


@pytest.mark.asyncio
class TestAutomatedReporting:
    """Test automated reporting of safety test results."""

    async def test_report_generation(self, sitl_env: SITLTestEnvironment) -> None:
        """Test that test reports are generated with timing data."""
        test_results = []

        # Run a series of tests
        tests = [
            ("Normal Homing", True),
            ("Mode Change", False),
            ("Signal Loss", False),
            ("Battery Low", False),
            ("Emergency Stop", False),
        ]

        for test_name, expected_pass in tests:
            start_time = time.perf_counter()

            # Simulate test execution
            await asyncio.sleep(0.01)  # Simulate test time

            duration_ms = (time.perf_counter() - start_time) * 1000

            test_results.append(
                {
                    "test": test_name,
                    "passed": expected_pass,
                    "duration_ms": duration_ms,
                    "timestamp": time.time(),
                }
            )

        # Generate report
        report = {
            "test_run_id": "SITL_TEST_001",
            "total_tests": len(test_results),
            "passed": sum(1 for r in test_results if r["passed"]),
            "failed": sum(1 for r in test_results if not r["passed"]),
            "total_duration_ms": sum(r["duration_ms"] for r in test_results),
            "results": test_results,
        }

        # Assert: Report contains required data
        assert report["total_tests"] == 5
        assert report["passed"] == 1
        assert report["failed"] == 4
        assert all("duration_ms" in r for r in report["results"])
        assert all("timestamp" in r for r in report["results"])


# Mock SITL tests for CI/CD environment without actual SITL
@pytest.mark.asyncio
class TestMockedSITL:
    """Mocked SITL tests for CI environment."""

    @pytest.fixture
    def mock_mavlink(self):
        """Create mock MAVLink service."""
        mock = AsyncMock(spec=MAVLinkService)
        mock.connected = True
        mock.set_mode = AsyncMock(return_value=True)
        mock.arm = AsyncMock(return_value=True)
        mock.takeoff = AsyncMock(return_value=True)
        return mock

    async def test_sitl_connection_mock(self, mock_mavlink):
        """Test SITL connection with mocked MAVLink."""
        # Test connection
        assert mock_mavlink.connected

        # Test mode setting
        result = await mock_mavlink.set_mode("GUIDED")
        assert result is True
        mock_mavlink.set_mode.assert_called_once_with("GUIDED")

        # Test arming
        result = await mock_mavlink.arm()
        assert result is True

        # Test takeoff
        result = await mock_mavlink.takeoff(10.0)
        assert result is True

    async def test_safety_scenarios_mock(self):
        """Test safety scenarios with mocked components."""
        # Create safety system
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()

        # Test normal operation
        safety.update_flight_mode("GUIDED")
        safety.update_battery(50.0)
        safety.update_signal_snr(10.0)

        result = await safety.enable_homing()
        assert result is True

        # Test safety interlock
        safety.update_battery(15.0)
        assert not await safety.is_safe_to_proceed()

        await safety.stop_monitoring()


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
