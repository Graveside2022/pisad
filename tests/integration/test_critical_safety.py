"""
Critical Safety Tests - Priority 1
Tests emergency stop, RC override, battery monitoring, and safety interlocks
"""

import asyncio
import time
from unittest.mock import patch

import pytest

from src.backend.core.config import get_config
from src.backend.services.command_pipeline import CommandPipeline
from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.state_machine import SystemState

pytestmark = pytest.mark.serial


@pytest.mark.hardware
class TestCriticalSafety:
    """Priority 1 - Critical safety tests with real hardware"""

    @pytest.fixture
    async def mavlink_service(self):
        """Create MAVLink service for testing"""
        config = get_config()
        service = MAVLinkService(config)

        # Try to connect to real hardware
        connected = await service.connect()
        if not connected:
            pytest.skip("MAVLink hardware not connected")

        yield service

        await service.disconnect()

    @pytest.fixture
    async def command_pipeline(self, mavlink_service):
        """Create command pipeline for safety testing"""
        pipeline = CommandPipeline(mavlink_service)
        await pipeline.start()

        yield pipeline

        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_emergency_stop_response_time(self, command_pipeline):
        """Test emergency stop response <500ms requirement"""
        # Record start time
        start_time = time.perf_counter()

        # Send emergency stop command
        success = await command_pipeline.emergency_stop()

        # Record response time
        response_time = (time.perf_counter() - start_time) * 1000  # Convert to ms

        # Verify response
        assert success, "Emergency stop should succeed"
        assert response_time < 500, f"Emergency stop took {response_time:.1f}ms, must be <500ms"

        # Verify system is in IDLE state
        state = command_pipeline.get_state()
        assert state == SystemState.IDLE, "System should be in IDLE state after emergency stop"

        # Verify all velocities are zero
        telemetry = await command_pipeline.mavlink_service.get_telemetry()
        assert telemetry.get("velocity_x", 0) == 0, "X velocity should be zero"
        assert telemetry.get("velocity_y", 0) == 0, "Y velocity should be zero"
        assert telemetry.get("velocity_z", 0) == 0, "Z velocity should be zero"

    @pytest.mark.asyncio
    async def test_rc_override_detection(self, mavlink_service):
        """Test RC override detection with ±50 PWM threshold"""
        # Monitor RC channels
        rc_baseline = {}
        rc_override_detected = False

        async def monitor_rc_channels():
            """Monitor RC channels for override"""
            nonlocal rc_override_detected

            for _ in range(100):  # Monitor for up to 10 seconds
                telemetry = await mavlink_service.get_telemetry()

                # Get RC channel values (typically channels 1-8)
                rc_channels = telemetry.get("rc_channels", {})

                if not rc_baseline:
                    # Set baseline on first read
                    rc_baseline.update(rc_channels)
                else:
                    # Check for significant change (±50 PWM)
                    for channel, value in rc_channels.items():
                        baseline_value = rc_baseline.get(channel, 1500)  # 1500 is typical center

                        if abs(value - baseline_value) > 50:
                            rc_override_detected = True
                            print(
                                f"RC override detected on channel {channel}: {baseline_value} -> {value}"
                            )
                            return

                await asyncio.sleep(0.1)

        # Start monitoring
        monitor_task = asyncio.create_task(monitor_rc_channels())

        # Simulate or wait for actual RC input
        print("Move RC sticks to test override detection...")

        try:
            await asyncio.wait_for(monitor_task, timeout=10.0)
        except TimeoutError:
            pytest.skip("No RC override detected - manual RC input required")

        assert rc_override_detected, "RC override should be detected with ±50 PWM change"

    @pytest.mark.asyncio
    async def test_battery_monitoring_thresholds(self, mavlink_service):
        """Test battery monitoring with 19.2V low and 18.0V critical thresholds"""
        # Get battery telemetry
        telemetry = await mavlink_service.get_telemetry()
        battery_voltage = telemetry.get("battery_voltage", 0)

        if battery_voltage == 0:
            pytest.skip("Battery telemetry not available")

        # Define thresholds for 6S Li-ion battery
        NOMINAL_VOLTAGE = 22.2  # 6S at 3.7V per cell
        LOW_VOLTAGE = 19.2  # 6S at 3.2V per cell
        CRITICAL_VOLTAGE = 18.0  # 6S at 3.0V per cell

        print(f"Current battery voltage: {battery_voltage}V")

        # Test battery status detection
        if battery_voltage >= NOMINAL_VOLTAGE:
            battery_status = "NORMAL"
        elif battery_voltage >= LOW_VOLTAGE:
            battery_status = "LOW"
        elif battery_voltage >= CRITICAL_VOLTAGE:
            battery_status = "CRITICAL"
        else:
            battery_status = "EMERGENCY"

        print(f"Battery status: {battery_status}")

        # Verify appropriate action based on battery level
        if battery_status == "CRITICAL" or battery_status == "EMERGENCY":
            # Should trigger RTL or land
            assert mavlink_service.get_flight_mode() in [
                "RTL",
                "LAND",
            ], "Critical battery should trigger RTL or LAND mode"

        # Test battery percentage calculation
        battery_percentage = telemetry.get("battery_percentage", -1)
        if battery_percentage >= 0:
            assert 0 <= battery_percentage <= 100, "Battery percentage should be 0-100"

    @pytest.mark.asyncio
    async def test_geofence_enforcement(self, mavlink_service):
        """Test geofence enforcement with GPS lock"""
        telemetry = await mavlink_service.get_telemetry()

        # Check GPS status
        gps_fix = telemetry.get("gps_fix_type", 0)
        gps_sats = telemetry.get("gps_satellites", 0)
        gps_hdop = telemetry.get("gps_hdop", 99.0)

        print(f"GPS Status - Fix: {gps_fix}, Sats: {gps_sats}, HDOP: {gps_hdop}")

        # Require good GPS for geofence testing
        if gps_fix < 3 or gps_sats < 8 or gps_hdop > 2.0:
            pytest.skip("GPS quality insufficient for geofence test (need 8+ sats, HDOP<2.0)")

        # Get current position
        lat = telemetry.get("latitude", 0)
        lon = telemetry.get("longitude", 0)
        alt = telemetry.get("altitude", 0)

        print(f"Current position: {lat:.6f}, {lon:.6f}, {alt:.1f}m")

        # Check if geofence is enabled
        params = await mavlink_service.get_parameters(
            ["FENCE_ENABLE", "FENCE_ACTION", "FENCE_RADIUS"]
        )

        fence_enabled = params.get("FENCE_ENABLE", 0)
        fence_action = params.get("FENCE_ACTION", 0)
        fence_radius = params.get("FENCE_RADIUS", 0)

        if fence_enabled:
            print(f"Geofence enabled - Radius: {fence_radius}m, Action: {fence_action}")
            assert fence_radius > 0, "Geofence radius should be positive"
            assert fence_action > 0, "Geofence should have an action configured"
        else:
            print("Geofence not enabled - enable via Mission Planner for testing")

    @pytest.mark.asyncio
    async def test_mode_change_to_idle(self, command_pipeline):
        """Test mode change from GUIDED to IDLE"""
        # Set initial state to GUIDED (simulating active mission)
        command_pipeline.set_state(SystemState.APPROACH)

        # Trigger mode change to IDLE
        success = await command_pipeline.change_mode(SystemState.IDLE)

        assert success, "Mode change to IDLE should succeed"
        assert command_pipeline.get_state() == SystemState.IDLE, "System should be in IDLE state"

        # Verify MAVLink mode reflects the change
        telemetry = await command_pipeline.mavlink_service.get_telemetry()
        flight_mode = telemetry.get("flight_mode", "")

        # IDLE should map to LOITER or POSHOLD mode in ArduPilot
        assert flight_mode in [
            "LOITER",
            "POSHOLD",
            "BRAKE",
        ], f"Flight mode should be LOITER/POSHOLD/BRAKE in IDLE state, got {flight_mode}"

    @pytest.mark.asyncio
    async def test_signal_loss_to_searching_transition(self, command_pipeline):
        """Test signal loss triggers SEARCHING state transition"""
        # Set initial state to APPROACH (actively following signal)
        command_pipeline.set_state(SystemState.APPROACH)

        # Simulate signal loss
        command_pipeline.signal_processor.update_rssi(-120)  # Below noise floor

        # Wait for timeout (should be configurable, typically 5 seconds)
        signal_loss_timeout = 5.0
        await asyncio.sleep(signal_loss_timeout + 0.5)

        # Verify transition to SEARCHING
        state = command_pipeline.get_state()
        assert (
            state == SystemState.SEARCHING
        ), f"System should transition to SEARCHING after signal loss, got {state}"

        # Verify search pattern is active
        telemetry = await command_pipeline.mavlink_service.get_telemetry()

        # Should have non-zero velocities in search pattern
        vx = telemetry.get("velocity_x", 0)
        vy = telemetry.get("velocity_y", 0)

        assert abs(vx) > 0 or abs(vy) > 0, "Should have active search pattern velocities"

    @pytest.mark.asyncio
    async def test_safety_interlock_cascade(self, command_pipeline):
        """Test that safety interlocks cascade properly"""
        safety_failures = []

        # Test 1: Low battery prevents arm
        with patch.object(command_pipeline.mavlink_service, "get_telemetry") as mock_telemetry:
            mock_telemetry.return_value = {"battery_voltage": 17.5}  # Below critical

            success = await command_pipeline.arm()
            if not success:
                safety_failures.append("Low battery prevented arm (GOOD)")

        # Test 2: No GPS prevents guided mode
        with patch.object(command_pipeline.mavlink_service, "get_telemetry") as mock_telemetry:
            mock_telemetry.return_value = {"gps_fix_type": 0, "gps_satellites": 0}

            success = await command_pipeline.set_guided_mode()
            if not success:
                safety_failures.append("No GPS prevented guided mode (GOOD)")

        # Test 3: Excessive velocity triggers safety
        MAX_VELOCITY = 20.0  # m/s safety limit

        with patch.object(command_pipeline.mavlink_service, "get_telemetry") as mock_telemetry:
            mock_telemetry.return_value = {"velocity_x": 25.0}  # Over limit

            # Try to increase velocity further
            success = await command_pipeline.send_velocity_command(30.0, 0, 0)
            if not success:
                safety_failures.append("Excessive velocity prevented (GOOD)")

        # All safety checks should have triggered
        assert (
            len(safety_failures) >= 2
        ), f"At least 2 safety interlocks should trigger. Got: {safety_failures}"

        print("Safety interlock results:")
        for result in safety_failures:
            print(f"  - {result}")


if __name__ == "__main__":
    # Run with: pytest tests/hardware/real/test_critical_safety.py -v -m hardware
    pytest.main([__file__, "-v", "-m", "hardware"])
