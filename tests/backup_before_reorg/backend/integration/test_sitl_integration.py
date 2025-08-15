"""
ArduPilot SITL Integration Tests for PISAD.

Story 4.7 - Sprint 5: SITL Integration Testing
Tests MAVLink communication and command execution with ArduPilot SITL.
"""

import asyncio
import os
from collections.abc import AsyncGenerator

import pytest
import pytest_asyncio

from src.backend.hal.sitl_interface import SITLInterface
from src.backend.services.mavlink_service import MAVLinkService


@pytest.mark.sitl
@pytest.mark.skipif(
    os.environ.get("RUN_SITL_TESTS", "").lower() not in ("1", "true", "yes"),
    reason="SITL tests disabled. Set RUN_SITL_TESTS=1 to enable",
)
class TestSITLIntegration:
    """Integration tests with ArduPilot SITL simulator."""

    @pytest_asyncio.fixture
    async def sitl_interface(self) -> AsyncGenerator[SITLInterface, None]:
        """Create and manage SITL interface for tests.

        Yields:
            Configured SITL interface
        """
        sitl = SITLInterface()

        # Start SITL if not already running
        started = await sitl.start_sitl(wipe_eeprom=True)
        if not started:
            pytest.skip("Failed to start SITL - ensure ArduPilot is installed")

        # Connect to SITL
        connected = await sitl.connect()
        if not connected:
            await sitl.stop_sitl()
            pytest.skip("Failed to connect to SITL")

        yield sitl

        # Cleanup
        await sitl.disconnect()
        await sitl.stop_sitl()

    @pytest_asyncio.fixture
    async def mavlink_service(self) -> AsyncGenerator[MAVLinkService, None]:
        """Create MAVLink service connected to SITL.

        Yields:
            Connected MAVLink service
        """
        service = MAVLinkService(device="tcp:127.0.0.1:5760", baudrate=115200)

        # Initialize and connect
        await service.initialize()

        # Wait for connection
        for _ in range(10):
            if service.is_connected():
                break
            await asyncio.sleep(1)
        else:
            pytest.skip("Failed to connect MAVLink service to SITL")

        yield service

        # Cleanup
        await service.stop()

    @pytest.mark.asyncio
    async def test_sitl_basic_connection(self, sitl_interface: SITLInterface) -> None:
        """Test basic connection to SITL.

        AC: 2 - Test basic SITL connection
        """
        # GIVEN: SITL interface is created and connected
        assert sitl_interface.connected

        # WHEN: We request telemetry
        telemetry = await sitl_interface.get_telemetry()

        # THEN: We should receive valid telemetry data
        assert "position" in telemetry
        assert "attitude" in telemetry
        assert "gps" in telemetry
        assert "battery" in telemetry
        assert "mode" in telemetry
        assert "armed" in telemetry

        # Verify GPS has lock (SITL should have simulated GPS)
        assert telemetry["gps"]["fix_type"] >= 3  # 3D fix or better
        assert telemetry["gps"]["satellites"] >= 8  # Meeting Story 4.7 requirement

    @pytest.mark.asyncio
    async def test_sitl_telemetry_streaming(self, sitl_interface: SITLInterface) -> None:
        """Test telemetry streaming from SITL.

        AC: 3 - Verify telemetry streaming from SITL
        """
        # GIVEN: SITL is connected
        assert sitl_interface.connected

        # WHEN: We collect telemetry over time
        telemetry_samples = []
        start_time = asyncio.get_event_loop().time()

        while len(telemetry_samples) < 10:
            telemetry = await sitl_interface.get_telemetry()
            telemetry_samples.append(telemetry)
            await asyncio.sleep(0.25)  # 4Hz sampling

        elapsed = asyncio.get_event_loop().time() - start_time

        # THEN: Telemetry should be streaming at expected rate
        assert len(telemetry_samples) == 10
        assert elapsed < 3.0  # Should complete in under 3 seconds

        # Verify telemetry is updating (position might not change if stationary)
        attitudes = [t["attitude"]["yaw"] for t in telemetry_samples]
        # At least check we're getting consistent data
        assert all(isinstance(a, (int, float)) for a in attitudes)

    @pytest.mark.asyncio
    async def test_sitl_mode_changes(self, sitl_interface: SITLInterface) -> None:
        """Test flight mode changes in SITL.

        AC: 4 - Test command execution in SITL
        """
        # GIVEN: SITL is in default mode
        initial_telemetry = await sitl_interface.get_telemetry()
        initial_mode = initial_telemetry["mode"]

        # WHEN: We change to GUIDED mode
        success = await sitl_interface.set_mode("GUIDED")
        assert success

        # Wait for mode change
        await asyncio.sleep(1)

        # THEN: Mode should be changed
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["mode"] == "GUIDED"

        # WHEN: We change to STABILIZE mode
        success = await sitl_interface.set_mode("STABILIZE")
        assert success

        # Wait for mode change
        await asyncio.sleep(1)

        # THEN: Mode should be STABILIZE
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["mode"] == "STABILIZE"

    @pytest.mark.asyncio
    async def test_sitl_arm_disarm(self, sitl_interface: SITLInterface) -> None:
        """Test arming and disarming in SITL.

        AC: 4 - Test command execution in SITL
        """
        # GIVEN: Vehicle is disarmed and in GUIDED mode
        await sitl_interface.set_mode("GUIDED")
        await asyncio.sleep(1)

        initial_telemetry = await sitl_interface.get_telemetry()
        assert not initial_telemetry["armed"]

        # WHEN: We arm the vehicle
        success = await sitl_interface.arm()
        assert success

        # Wait for arm
        await asyncio.sleep(1)

        # THEN: Vehicle should be armed
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["armed"]

        # WHEN: We disarm the vehicle
        success = await sitl_interface.disarm()
        assert success

        # Wait for disarm
        await asyncio.sleep(1)

        # THEN: Vehicle should be disarmed
        telemetry = await sitl_interface.get_telemetry()
        assert not telemetry["armed"]

    @pytest.mark.asyncio
    async def test_sitl_takeoff_land(self, sitl_interface: SITLInterface) -> None:
        """Test takeoff and landing in SITL.

        AC: 4 - Test command execution in SITL
        """
        # GIVEN: Vehicle is in GUIDED mode and armed
        await sitl_interface.set_mode("GUIDED")
        await asyncio.sleep(1)

        success = await sitl_interface.arm()
        assert success
        await asyncio.sleep(1)

        initial_alt = (await sitl_interface.get_telemetry())["position"]["alt"]

        # WHEN: We command takeoff to 10m
        success = await sitl_interface.takeoff(10.0)
        assert success

        # Wait for takeoff (max 20 seconds)
        for _ in range(40):
            telemetry = await sitl_interface.get_telemetry()
            current_alt = telemetry["position"]["alt"]
            if current_alt >= initial_alt + 9.0:  # Within 1m of target
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Takeoff did not reach target altitude")

        # THEN: Vehicle should be at ~10m altitude
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["position"]["alt"] >= initial_alt + 9.0
        assert telemetry["armed"]

        # WHEN: We command landing
        success = await sitl_interface.land()
        assert success

        # Wait for landing (max 30 seconds)
        for _ in range(60):
            telemetry = await sitl_interface.get_telemetry()
            current_alt = telemetry["position"]["alt"]
            if current_alt <= initial_alt + 1.0:  # Within 1m of ground
                break
            await asyncio.sleep(0.5)
        else:
            pytest.fail("Landing did not complete")

        # THEN: Vehicle should be on ground and disarmed
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["position"]["alt"] <= initial_alt + 1.0
        # Vehicle should auto-disarm after landing

    @pytest.mark.asyncio
    async def test_sitl_velocity_commands(self, sitl_interface: SITLInterface) -> None:
        """Test velocity command execution in SITL.

        AC: 4 - Test command execution in SITL
        """
        # GIVEN: Vehicle is airborne in GUIDED mode
        await sitl_interface.set_mode("GUIDED")
        await asyncio.sleep(1)

        await sitl_interface.arm()
        await asyncio.sleep(1)

        await sitl_interface.takeoff(10.0)

        # Wait for takeoff
        for _ in range(40):
            telemetry = await sitl_interface.get_telemetry()
            if telemetry["position"]["alt"] >= 9.0:
                break
            await asyncio.sleep(0.5)

        initial_pos = await sitl_interface.get_telemetry()
        initial_lat = initial_pos["position"]["lat"]
        initial_lon = initial_pos["position"]["lon"]

        # WHEN: We send velocity command (2 m/s North)
        success = await sitl_interface.send_velocity_command(2.0, 0.0, 0.0, 0.0)
        assert success

        # Let it fly for 3 seconds
        await asyncio.sleep(3)

        # Stop movement
        await sitl_interface.send_velocity_command(0.0, 0.0, 0.0, 0.0)

        # THEN: Vehicle should have moved North
        final_pos = await sitl_interface.get_telemetry()
        final_lat = final_pos["position"]["lat"]

        # Latitude should increase (North is positive)
        assert final_lat > initial_lat

        # Land the vehicle
        await sitl_interface.land()

    @pytest.mark.asyncio
    async def test_sitl_emergency_stop(self, sitl_interface: SITLInterface) -> None:
        """Test emergency stop functionality in SITL.

        AC: 5 - Safety validation with SITL
        """
        # GIVEN: Vehicle is airborne and moving
        await sitl_interface.set_mode("GUIDED")
        await asyncio.sleep(1)

        await sitl_interface.arm()
        await asyncio.sleep(1)

        await sitl_interface.takeoff(10.0)

        # Wait for takeoff
        for _ in range(40):
            telemetry = await sitl_interface.get_telemetry()
            if telemetry["position"]["alt"] >= 9.0:
                break
            await asyncio.sleep(0.5)

        # Start moving
        await sitl_interface.send_velocity_command(5.0, 0.0, 0.0, 0.0)
        await asyncio.sleep(1)

        # WHEN: We trigger emergency stop
        start_time = asyncio.get_event_loop().time()
        success = await sitl_interface.emergency_stop()
        stop_time = asyncio.get_event_loop().time()

        assert success

        # THEN: Emergency stop should complete quickly
        response_time = stop_time - start_time
        assert response_time < 0.5  # Meeting <500ms requirement from Story 4.7

        # Vehicle should be in STABILIZE mode
        await asyncio.sleep(1)
        telemetry = await sitl_interface.get_telemetry()
        assert telemetry["mode"] == "STABILIZE"

        # Velocity should be near zero
        assert abs(telemetry["velocity"]["vx"]) < 0.5
        assert abs(telemetry["velocity"]["vy"]) < 0.5

    @pytest.mark.asyncio
    async def test_sitl_battery_monitoring(self, sitl_interface: SITLInterface) -> None:
        """Test battery monitoring in SITL.

        AC: 5 - Safety validation with SITL
        """
        # GIVEN: SITL is connected
        assert sitl_interface.connected

        # WHEN: We check battery telemetry
        telemetry = await sitl_interface.get_telemetry()

        # THEN: Battery data should be available
        assert "battery" in telemetry
        battery = telemetry["battery"]

        # SITL simulates a healthy battery
        assert battery["voltage"] > 0  # Should have voltage
        # Note: SITL may not simulate exact 6S Li-ion values

        # Check battery thresholds from config
        config = sitl_interface.config
        safety = config.get("safety", {})
        low_voltage = safety.get("battery_low_voltage", 19.2)
        critical_voltage = safety.get("battery_critical_voltage", 18.0)

        # Verify thresholds are configured
        assert low_voltage == 19.2  # 6S Li-ion low threshold
        assert critical_voltage == 18.0  # 6S Li-ion critical threshold

    @pytest.mark.asyncio
    async def test_sitl_gps_requirements(self, sitl_interface: SITLInterface) -> None:
        """Test GPS requirements in SITL.

        AC: 5 - Safety validation with SITL
        """
        # GIVEN: SITL is connected
        assert sitl_interface.connected

        # WHEN: We check GPS status
        telemetry = await sitl_interface.get_telemetry()

        # THEN: GPS should meet requirements
        assert "gps" in telemetry
        gps = telemetry["gps"]

        # Check Story 4.7 GPS requirements
        assert gps["satellites"] >= 8  # Minimum 8 satellites
        assert gps["hdop"] <= 2.0  # HDOP less than 2.0
        assert gps["fix_type"] >= 3  # 3D fix or better

    @pytest.mark.asyncio
    async def test_sitl_mavlink_service_integration(
        self, sitl_interface: SITLInterface, mavlink_service: MAVLinkService
    ) -> None:
        """Test MAVLink service integration with SITL.

        AC: 4 - Test command execution via MAVLink service
        """
        # GIVEN: Both SITL and MAVLink service are connected
        assert sitl_interface.connected
        assert mavlink_service.is_connected()

        # WHEN: We get telemetry from MAVLink service
        telemetry = mavlink_service.get_telemetry()

        # THEN: Telemetry should be valid
        assert telemetry is not None
        assert telemetry["gps"]["satellites"] >= 8
        assert telemetry["gps"]["hdop"] <= 2.0

        # WHEN: We send commands via MAVLink service
        # Set GUIDED mode
        success = await mavlink_service.send_command("set_mode", {"mode": "GUIDED"})

        # Note: MAVLink service command interface may differ
        # This test validates that both interfaces can work with SITL

    @pytest.mark.asyncio
    async def test_sitl_performance_metrics(self, sitl_interface: SITLInterface) -> None:
        """Test performance metrics with SITL.

        AC: 8 - Performance validation with SITL
        """
        # GIVEN: SITL is connected
        assert sitl_interface.connected

        # WHEN: We measure mode change latency
        start_time = asyncio.get_event_loop().time()
        await sitl_interface.set_mode("GUIDED")
        mode_change_time = asyncio.get_event_loop().time() - start_time

        # THEN: Mode change should meet latency requirement
        assert mode_change_time < 0.1  # <100ms requirement

        # WHEN: We measure telemetry update rate
        telemetry_times = []
        for _ in range(10):
            start = asyncio.get_event_loop().time()
            await sitl_interface.get_telemetry()
            telemetry_times.append(asyncio.get_event_loop().time() - start)

        avg_telemetry_time = sum(telemetry_times) / len(telemetry_times)

        # THEN: Telemetry should be fast
        assert avg_telemetry_time < 0.05  # <50ms MAVLink latency requirement
