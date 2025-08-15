"""Integration tests with ArduPilot SITL simulator for flight dynamics."""

import asyncio
import os
import subprocess
import time

import pytest
from pymavlink import mavutil

from src.backend.services.mavlink_service import MAVLinkService
from src.backend.services.state_machine import StateMachine, SystemState


@pytest.mark.sitl
@pytest.mark.skipif(
    not os.environ.get("RUN_SITL_TESTS"),
    reason="SITL tests disabled. Set RUN_SITL_TESTS=1 to enable",
)
class TestArduPilotSITLIntegration:
    """Integration tests with actual ArduPilot SITL simulator."""

    @classmethod
    def setup_class(cls):
        """Start ArduPilot SITL simulator."""
        cls.sitl_process = None
        cls.mavproxy_process = None

        try:
            # Start SITL
            cls.sitl_process = subprocess.Popen(
                ["sim_vehicle.py", "-v", "ArduCopter", "--no-mavproxy", "-L", "BMAC"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for SITL to start
            time.sleep(10)

            # Start MAVProxy for easier interaction
            cls.mavproxy_process = subprocess.Popen(
                ["mavproxy.py", "--master", "tcp:127.0.0.1:5760", "--out", "udp:127.0.0.1:14550"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            time.sleep(5)
        except FileNotFoundError:
            pytest.skip("ArduPilot SITL not installed")

    @classmethod
    def teardown_class(cls):
        """Stop SITL simulator."""
        if cls.mavproxy_process:
            cls.mavproxy_process.terminate()
            cls.mavproxy_process.wait(timeout=5)

        if cls.sitl_process:
            cls.sitl_process.terminate()
            cls.sitl_process.wait(timeout=5)

    @pytest.fixture
    async def mavlink_service(self):
        """Create MAVLink service connected to SITL."""
        service = MAVLinkService()
        await service.connect("udp:127.0.0.1:14550")

        # Wait for connection
        for _ in range(10):
            if service.connected:
                break
            await asyncio.sleep(1)

        yield service

        await service.disconnect()

    @pytest.fixture
    def state_machine(self):
        """Create state machine for testing."""
        return StateMachine()

    @pytest.mark.asyncio
    async def test_sitl_connection(self, mavlink_service):
        """Test basic connection to SITL."""
        assert mavlink_service.connected

        # Get heartbeat
        heartbeat = await mavlink_service.wait_heartbeat()
        assert heartbeat is not None

        # Check system info
        assert mavlink_service.target_system > 0
        assert mavlink_service.target_component >= 0

    @pytest.mark.asyncio
    async def test_arm_and_takeoff(self, mavlink_service):
        """Test arming and takeoff in SITL."""
        # Set mode to GUIDED
        await mavlink_service.set_mode("GUIDED")
        await asyncio.sleep(1)

        # Arm the vehicle
        armed = await mavlink_service.arm()
        assert armed

        # Takeoff to 10 meters
        takeoff_success = await mavlink_service.takeoff(10)
        assert takeoff_success

        # Wait for altitude
        start_time = time.time()
        while time.time() - start_time < 30:
            position = await mavlink_service.get_position()
            if position and position.get("alt", 0) > 9:
                break
            await asyncio.sleep(1)

        # Verify altitude reached
        position = await mavlink_service.get_position()
        assert position["alt"] >= 9

        # Land
        await mavlink_service.land()

        # Disarm
        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_velocity_control(self, mavlink_service):
        """Test velocity-based control in SITL."""
        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(20)

        # Wait for altitude
        await asyncio.sleep(15)

        # Test forward velocity
        await mavlink_service.send_velocity_command(vx=2, vy=0, vz=0)
        await asyncio.sleep(5)

        # Test lateral velocity
        await mavlink_service.send_velocity_command(vx=0, vy=2, vz=0)
        await asyncio.sleep(5)

        # Test vertical velocity
        await mavlink_service.send_velocity_command(vx=0, vy=0, vz=1)
        await asyncio.sleep(5)

        # Stop
        await mavlink_service.send_velocity_command(vx=0, vy=0, vz=0)

        # Land and disarm
        await mavlink_service.land()
        await asyncio.sleep(20)
        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_waypoint_navigation(self, mavlink_service):
        """Test waypoint mission in SITL."""
        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(30)
        await asyncio.sleep(20)

        # Create mission waypoints
        home = await mavlink_service.get_position()
        waypoints = [
            {
                "seq": 0,
                "lat": home["lat"],
                "lon": home["lon"],
                "alt": 30,
                "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            },
            {
                "seq": 1,
                "lat": home["lat"] + 0.0001,
                "lon": home["lon"],
                "alt": 30,
                "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            },
            {
                "seq": 2,
                "lat": home["lat"] + 0.0001,
                "lon": home["lon"] + 0.0001,
                "alt": 30,
                "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            },
            {
                "seq": 3,
                "lat": home["lat"],
                "lon": home["lon"] + 0.0001,
                "alt": 30,
                "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            },
            {
                "seq": 4,
                "lat": home["lat"],
                "lon": home["lon"],
                "alt": 30,
                "command": mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
            },
        ]

        # Upload mission
        upload_success = await mavlink_service.upload_mission(waypoints)
        assert upload_success

        # Start mission
        await mavlink_service.set_mode("AUTO")

        # Monitor mission progress
        start_time = time.time()
        while time.time() - start_time < 60:
            progress = await mavlink_service.get_mission_progress()
            if progress and progress.get("current", 0) >= len(waypoints) - 1:
                break
            await asyncio.sleep(2)

        # Return to GUIDED mode
        await mavlink_service.set_mode("GUIDED")

        # Land and disarm
        await mavlink_service.land()
        await asyncio.sleep(20)
        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_emergency_procedures(self, mavlink_service):
        """Test emergency procedures in SITL."""
        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(25)
        await asyncio.sleep(15)

        # Test emergency stop (hold position)
        await mavlink_service.hold_position()
        initial_pos = await mavlink_service.get_position()

        await asyncio.sleep(5)

        held_pos = await mavlink_service.get_position()

        # Position should be relatively stable
        lat_diff = abs(held_pos["lat"] - initial_pos["lat"])
        lon_diff = abs(held_pos["lon"] - initial_pos["lon"])
        assert lat_diff < 0.00001  # Very small movement
        assert lon_diff < 0.00001

        # Test RTL
        await mavlink_service.return_to_launch()
        await mavlink_service.set_mode("RTL")

        # Wait for RTL to complete
        await asyncio.sleep(30)

        # Should be back near home and landed
        final_pos = await mavlink_service.get_position()
        assert final_pos["alt"] < 2  # On ground

        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_telemetry_streaming(self, mavlink_service):
        """Test telemetry data streaming from SITL."""
        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(15)
        await asyncio.sleep(10)

        # Collect telemetry for 10 seconds
        telemetry_data = []
        start_time = time.time()

        while time.time() - start_time < 10:
            telemetry = await mavlink_service.get_telemetry()
            if telemetry:
                telemetry_data.append(telemetry)
            await asyncio.sleep(0.5)

        # Verify telemetry data
        assert len(telemetry_data) > 15  # Should have ~20 samples

        # Check telemetry fields
        for data in telemetry_data:
            assert "position" in data
            assert "attitude" in data
            assert "velocity" in data
            assert "battery" in data
            assert "gps" in data
            assert "mode" in data

        # Land and disarm
        await mavlink_service.land()
        await asyncio.sleep(15)
        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_simulated_homing_behavior(self, mavlink_service, state_machine):
        """Test simulated homing behavior in SITL."""
        # Initialize state
        state_machine.current_state = SystemState.IDLE

        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(40)
        await asyncio.sleep(20)

        # Enable homing
        state_machine.homing_enabled = True
        state_machine.current_state = SystemState.SEARCHING

        # Simulate beacon detection (would come from signal processor)
        beacon_bearing = 45  # degrees
        beacon_distance = 100  # meters (estimated)

        # Calculate velocity vector towards beacon
        import math

        speed = 3.0  # m/s
        vx = speed * math.cos(math.radians(beacon_bearing))
        vy = speed * math.sin(math.radians(beacon_bearing))

        # Start homing
        state_machine.current_state = SystemState.HOMING

        # Move towards beacon for 20 seconds
        start_time = time.time()
        while time.time() - start_time < 20:
            # Send velocity commands
            await mavlink_service.send_velocity_command(vx=vx, vy=vy, vz=0)

            # In real scenario, would check RSSI and adjust
            await asyncio.sleep(1)

        # Stop at beacon
        await mavlink_service.send_velocity_command(vx=0, vy=0, vz=0)
        state_machine.current_state = SystemState.BEACON_LOCATED

        # Hold position for 5 seconds
        await mavlink_service.hold_position()
        await asyncio.sleep(5)

        # Return home
        state_machine.current_state = SystemState.RETURNING
        await mavlink_service.return_to_launch()
        await mavlink_service.set_mode("RTL")

        # Wait for RTL
        await asyncio.sleep(40)

        # Verify landed
        position = await mavlink_service.get_position()
        assert position["alt"] < 2

        state_machine.current_state = SystemState.IDLE
        state_machine.homing_enabled = False

        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_battery_failsafe(self, mavlink_service):
        """Test battery failsafe behavior in SITL."""
        # Note: SITL battery simulation may be limited
        # This test demonstrates the pattern

        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(30)
        await asyncio.sleep(15)

        # Monitor battery (SITL usually simulates slow discharge)
        battery_data = []
        for _ in range(10):
            telemetry = await mavlink_service.get_telemetry()
            if telemetry and "battery" in telemetry:
                battery_data.append(telemetry["battery"])
            await asyncio.sleep(1)

        # Check battery monitoring works
        assert len(battery_data) > 0
        assert all("voltage" in b and "percentage" in b for b in battery_data)

        # In real scenario, would trigger RTL on low battery
        # For SITL, just demonstrate the logic
        min_battery = min(b["percentage"] for b in battery_data)
        if min_battery < 20:  # Would trigger failsafe
            await mavlink_service.return_to_launch()
            await mavlink_service.set_mode("RTL")

        # Land and disarm
        await mavlink_service.land()
        await asyncio.sleep(20)
        await mavlink_service.disarm()

    @pytest.mark.asyncio
    async def test_geofence_enforcement(self, mavlink_service):
        """Test geofence enforcement in SITL."""
        # Note: Requires geofence to be configured in SITL

        # Arm and takeoff
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.arm()
        await mavlink_service.takeoff(25)
        await asyncio.sleep(15)

        # Get current position (home)
        home = await mavlink_service.get_position()

        # Try to fly outside geofence (if configured)
        # Move 200m north (may trigger geofence)
        await mavlink_service.send_velocity_command(vx=5, vy=0, vz=0)

        # Monitor for 20 seconds
        positions = []
        for _ in range(20):
            pos = await mavlink_service.get_position()
            positions.append(pos)

            # Check mode - might switch to RTL if geofence triggered
            mode = await mavlink_service.get_mode()
            if mode == "RTL":
                print("Geofence triggered - RTL activated")
                break

            await asyncio.sleep(1)

        # Stop movement
        await mavlink_service.send_velocity_command(vx=0, vy=0, vz=0)

        # Return home
        await mavlink_service.set_mode("GUIDED")
        await mavlink_service.return_to_launch()

        await asyncio.sleep(30)

        # Land and disarm
        await mavlink_service.land()
        await asyncio.sleep(20)
        await mavlink_service.disarm()
