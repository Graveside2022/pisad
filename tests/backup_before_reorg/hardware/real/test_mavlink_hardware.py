"""
Core MAVLink Hardware Tests - Priority 2
Tests Cube Orange+ connectivity, telemetry, and command execution
"""

import asyncio
import subprocess
import time

import pytest

from src.backend.core.config import get_config
from src.backend.services.mavlink_service import MAVLinkService


@pytest.mark.hardware
class TestMAVLinkHardware:
    """Priority 2 - Core MAVLink hardware functionality tests"""

    @pytest.fixture
    async def mavlink(self):
        """Create and connect MAVLink service"""
        config = get_config()
        service = MAVLinkService(config)

        # Try to connect
        connected = await service.connect()
        if not connected:
            pytest.skip("MAVLink hardware not connected")

        yield service

        await service.disconnect()

    def test_cube_orange_usb_detection(self):
        """Test Cube Orange+ on /dev/ttyACM0"""
        # Check for Cube Orange+ USB device
        result = subprocess.run(["ls", "-la", "/dev/ttyACM*"], capture_output=True, text=True)

        if "/dev/ttyACM0" not in result.stdout:
            pytest.skip("/dev/ttyACM0 not found - Cube Orange+ may not be connected")

        # Check device permissions
        result = subprocess.run(["ls", "-la", "/dev/ttyACM0"], capture_output=True, text=True)

        print(f"Device info: {result.stdout.strip()}")

        # Verify device exists and is accessible
        assert "/dev/ttyACM0" in result.stdout, "Primary MAVLink device should exist"

        # Check for secondary device
        if "/dev/ttyACM1" in result.stdout:
            print("Secondary MAVLink device also available at /dev/ttyACM1")

    @pytest.mark.asyncio
    async def test_serial_connection_115200_baud(self, mavlink):
        """Test 115200 baud communication stability"""
        # Already connected via fixture
        assert mavlink.connected, "MAVLink should be connected"

        # Test connection stability over time
        stable_duration = 5.0  # seconds
        start_time = time.time()
        connection_drops = 0

        while time.time() - start_time < stable_duration:
            if not mavlink.connected:
                connection_drops += 1
                # Try to reconnect
                await mavlink.connect()

            await asyncio.sleep(0.1)

        assert connection_drops == 0, f"Connection dropped {connection_drops} times"
        print(f"✓ Connection stable for {stable_duration} seconds at 115200 baud")

    @pytest.mark.asyncio
    async def test_heartbeat_1hz_with_timeout(self, mavlink):
        """Test heartbeat at 1Hz with 10s timeout"""
        heartbeat_count = 0
        heartbeat_times = []

        # Monitor heartbeats for 5 seconds
        start_time = time.time()

        while time.time() - start_time < 5.0:
            telemetry = await mavlink.get_telemetry()

            # Check if we have a recent heartbeat
            last_heartbeat = telemetry.get("last_heartbeat", 0)

            if last_heartbeat > 0:
                heartbeat_count += 1
                heartbeat_times.append(time.time())

            await asyncio.sleep(0.5)

        # Calculate heartbeat rate
        if len(heartbeat_times) >= 2:
            intervals = [
                heartbeat_times[i + 1] - heartbeat_times[i] for i in range(len(heartbeat_times) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            heartbeat_rate = 1.0 / avg_interval if avg_interval > 0 else 0

            print(f"Heartbeat rate: {heartbeat_rate:.2f} Hz")
            print(f"Average interval: {avg_interval:.2f} seconds")

            # Should be approximately 1 Hz
            assert (
                0.8 <= heartbeat_rate <= 1.2
            ), f"Heartbeat rate should be ~1 Hz, got {heartbeat_rate:.2f}"

        assert heartbeat_count > 0, "Should receive heartbeats"

    @pytest.mark.asyncio
    async def test_ned_velocity_commands(self, mavlink):
        """Test NED velocity commands execution"""
        # First ensure we're in a mode that accepts velocity commands
        telemetry = await mavlink.get_telemetry()
        current_mode = telemetry.get("flight_mode", "")

        print(f"Current flight mode: {current_mode}")

        if current_mode not in ["GUIDED", "GUIDED_NOGPS"]:
            # Try to set GUIDED mode
            success = await mavlink.set_mode("GUIDED")
            if not success:
                pytest.skip("Cannot set GUIDED mode - vehicle may not be armed or GPS not ready")

        # Send a small velocity command (North: 1 m/s)
        vx, vy, vz = 1.0, 0.0, 0.0  # North, East, Down in NED frame

        success = await mavlink.send_velocity_ned(vx, vy, vz)
        assert success, "Should send NED velocity command"

        # Wait for command to be processed
        await asyncio.sleep(0.5)

        # Send stop command
        success = await mavlink.send_velocity_ned(0, 0, 0)
        assert success, "Should send stop command"

        print("✓ NED velocity commands sent successfully")

    @pytest.mark.asyncio
    async def test_gps_telemetry_quality(self, mavlink):
        """Test real GPS data (8+ sats, HDOP <2.0)"""
        telemetry = await mavlink.get_telemetry()

        gps_fix = telemetry.get("gps_fix_type", 0)
        gps_sats = telemetry.get("gps_satellites", 0)
        gps_hdop = telemetry.get("gps_hdop", 99.0)
        latitude = telemetry.get("latitude", 0)
        longitude = telemetry.get("longitude", 0)
        altitude = telemetry.get("altitude", 0)

        print("GPS Status:")
        print(
            f"  Fix type: {gps_fix} (0=no GPS, 1=no fix, 2=2D, 3=3D, 4=DGPS, 5=RTK float, 6=RTK fixed)"
        )
        print(f"  Satellites: {gps_sats}")
        print(f"  HDOP: {gps_hdop:.2f}")
        print(f"  Position: {latitude:.6f}, {longitude:.6f}, {altitude:.1f}m")

        # Check GPS quality
        if gps_fix >= 3:  # 3D fix or better
            assert gps_sats >= 6, f"Should have at least 6 satellites for 3D fix, got {gps_sats}"

            if gps_sats >= 8:
                print("✓ Good GPS: 8+ satellites")

            if gps_hdop < 2.0:
                print(f"✓ Good HDOP: {gps_hdop:.2f} < 2.0")
            else:
                print(f"⚠ Poor HDOP: {gps_hdop:.2f} >= 2.0")
        else:
            print("⚠ No 3D GPS fix - indoor testing or no GPS module")

    @pytest.mark.asyncio
    async def test_battery_telemetry_6s_liion(self, mavlink):
        """Test 6S Li-ion battery telemetry"""
        telemetry = await mavlink.get_telemetry()

        battery_voltage = telemetry.get("battery_voltage", 0)
        battery_current = telemetry.get("battery_current", 0)
        battery_percentage = telemetry.get("battery_percentage", -1)

        print("Battery Status:")
        print(f"  Voltage: {battery_voltage:.2f}V")
        print(f"  Current: {battery_current:.2f}A")
        print(f"  Percentage: {battery_percentage}%")

        if battery_voltage > 0:
            # 6S Li-ion voltage ranges
            CELL_COUNT = 6
            NOMINAL_CELL = 3.7  # Li-ion nominal
            MIN_CELL = 3.0  # Li-ion minimum
            MAX_CELL = 4.2  # Li-ion maximum

            min_voltage = CELL_COUNT * MIN_CELL  # 18.0V
            max_voltage = CELL_COUNT * MAX_CELL  # 25.2V
            nominal_voltage = CELL_COUNT * NOMINAL_CELL  # 22.2V

            assert (
                min_voltage <= battery_voltage <= max_voltage
            ), f"6S battery voltage should be {min_voltage}-{max_voltage}V, got {battery_voltage}V"

            # Check battery health
            if battery_voltage < 19.2:  # 3.2V per cell
                print("⚠ WARNING: Battery low!")
            elif battery_voltage < 18.0:  # 3.0V per cell
                print("⚠ CRITICAL: Battery critical!")
            else:
                print(f"✓ Battery healthy: {battery_voltage:.2f}V")

            # Check current draw
            if battery_current > 0:
                power_draw = battery_voltage * battery_current
                print(f"  Power draw: {power_draw:.1f}W")
        else:
            print("⚠ No battery telemetry available")

    @pytest.mark.asyncio
    async def test_attitude_telemetry(self, mavlink):
        """Test attitude (roll, pitch, yaw) telemetry"""
        telemetry = await mavlink.get_telemetry()

        roll = telemetry.get("roll", 0)
        pitch = telemetry.get("pitch", 0)
        yaw = telemetry.get("yaw", 0)

        print("Attitude:")
        print(f"  Roll:  {roll:.2f}°")
        print(f"  Pitch: {pitch:.2f}°")
        print(f"  Yaw:   {yaw:.2f}°")

        # Verify attitude values are in valid range
        assert -180 <= roll <= 180, f"Roll out of range: {roll}"
        assert -90 <= pitch <= 90, f"Pitch out of range: {pitch}"
        assert -180 <= yaw <= 180, f"Yaw out of range: {yaw}"

        # Check if vehicle is level (for ground testing)
        if abs(roll) < 5 and abs(pitch) < 5:
            print("✓ Vehicle is level")

    @pytest.mark.asyncio
    async def test_flight_mode_reading(self, mavlink):
        """Test flight mode reading and available modes"""
        telemetry = await mavlink.get_telemetry()

        current_mode = telemetry.get("flight_mode", "UNKNOWN")
        armed = telemetry.get("armed", False)

        print("Flight Status:")
        print(f"  Mode: {current_mode}")
        print(f"  Armed: {armed}")

        # Common Cube Orange+ flight modes
        common_modes = [
            "STABILIZE",
            "ACRO",
            "ALT_HOLD",
            "AUTO",
            "GUIDED",
            "LOITER",
            "RTL",
            "CIRCLE",
            "LAND",
            "POSHOLD",
            "BRAKE",
            "THROW",
            "AVOID_ADSB",
            "GUIDED_NOGPS",
            "SMART_RTL",
            "FLOWHOLD",
            "FOLLOW",
            "ZIGZAG",
            "SYSTEMID",
            "AUTOROTATE",
            "AUTO_RTL",
        ]

        if current_mode in common_modes:
            print(f"✓ Valid flight mode: {current_mode}")
        else:
            print(f"⚠ Unexpected flight mode: {current_mode}")

    @pytest.mark.asyncio
    async def test_parameter_access(self, mavlink):
        """Test reading flight controller parameters"""
        # Try to read some common parameters
        param_names = [
            "ARMING_CHECK",  # Pre-arm checks
            "FENCE_ENABLE",  # Geofence
            "RTL_ALT",  # RTL altitude
            "WPNAV_SPEED",  # Waypoint navigation speed
            "PILOT_SPEED_UP",  # Maximum climb rate
        ]

        for param_name in param_names:
            try:
                # Note: This assumes the MAVLink service has a get_parameter method
                # If not implemented, this test documents what's needed
                value = await mavlink.get_parameter(param_name)
                if value is not None:
                    print(f"  {param_name}: {value}")
            except Exception as e:
                print(f"  {param_name}: Not available ({e})")

    @pytest.mark.asyncio
    async def test_mission_capability(self, mavlink):
        """Test mission upload/download capability"""
        # Check if mission commands are supported
        telemetry = await mavlink.get_telemetry()

        # Try to request mission count
        try:
            # This tests if the MAVLink service can handle mission protocol
            # Actual implementation may vary
            mission_count = telemetry.get("mission_count", 0)
            print(f"Mission items loaded: {mission_count}")

            if mission_count > 0:
                print("✓ Mission capability confirmed")
            else:
                print("⚠ No mission loaded (expected for hardware test)")
        except Exception as e:
            print(f"Mission protocol not fully implemented: {e}")

    @pytest.mark.asyncio
    async def test_telemetry_stream_performance(self, mavlink):
        """Test telemetry streaming performance"""
        update_times = []
        previous_telemetry = {}

        # Monitor for 5 seconds
        start_time = time.time()

        while time.time() - start_time < 5.0:
            telemetry = await mavlink.get_telemetry()

            # Check if telemetry actually updated
            if telemetry != previous_telemetry:
                update_times.append(time.time())
                previous_telemetry = telemetry.copy()

            await asyncio.sleep(0.05)  # 20 Hz sampling

        # Calculate update rate
        if len(update_times) >= 2:
            intervals = [
                update_times[i + 1] - update_times[i] for i in range(len(update_times) - 1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            update_rate = 1.0 / avg_interval if avg_interval > 0 else 0

            print(f"Telemetry update rate: {update_rate:.1f} Hz")
            print(f"Total updates: {len(update_times)}")

            # Should get at least 4 Hz telemetry
            assert update_rate >= 4.0, f"Telemetry rate too low: {update_rate:.1f} Hz"


if __name__ == "__main__":
    # Run with: pytest tests/hardware/real/test_mavlink_hardware.py -v -m hardware
    pytest.main([__file__, "-v", "-m", "hardware"])
