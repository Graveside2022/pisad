"""Mock MAVLink Telemetry Tests.

Tests for MAVLink GPS, battery, and attitude telemetry
without requiring real hardware.
"""

import time
from unittest.mock import Mock

import pytest

from backend.hal.mavlink_interface import MAVLinkInterface


@pytest.mark.mock_hardware
@pytest.mark.mavlink
class TestMockMAVLinkTelemetry:
    """Test MAVLink telemetry data with mock hardware."""

    @pytest.fixture
    def mock_gps_message(self) -> Mock:
        """Create mock GPS message."""
        msg = Mock()
        msg.get_type = Mock(return_value="GPS_RAW_INT")
        msg.lat = 377749000  # 37.7749 degrees * 1e7
        msg.lon = -1224194000  # -122.4194 degrees * 1e7
        msg.alt = 100000  # 100m * 1000
        msg.eph = 150  # HDOP * 100
        msg.epv = 200  # VDOP * 100
        msg.vel = 100  # Ground speed cm/s
        msg.cog = 9000  # Course over ground * 100
        msg.fix_type = 3  # 3D fix
        msg.satellites_visible = 12
        return msg

    @pytest.fixture
    def mock_battery_message(self) -> Mock:
        """Create mock battery message."""
        msg = Mock()
        msg.get_type = Mock(return_value="BATTERY_STATUS")
        msg.voltages = [3700, 3700, 3700, 3700, 3700, 3700]  # 6S battery, mV per cell
        msg.current_battery = 15000  # 15A * 1000
        msg.battery_remaining = 75  # 75%
        msg.temperature = 35  # 35°C
        return msg

    @pytest.fixture
    def mock_attitude_message(self) -> Mock:
        """Create mock attitude message."""
        msg = Mock()
        msg.get_type = Mock(return_value="ATTITUDE")
        msg.roll = 0.05  # radians
        msg.pitch = -0.02  # radians
        msg.yaw = 1.57  # radians (90 degrees)
        msg.rollspeed = 0.01  # rad/s
        msg.pitchspeed = -0.005  # rad/s
        msg.yawspeed = 0.02  # rad/s
        return msg

    @pytest.fixture
    def mock_interface_with_telemetry(
        self, mock_gps_message: Mock, mock_battery_message: Mock, mock_attitude_message: Mock
    ) -> MAVLinkInterface:
        """Create interface with telemetry messages."""
        mock_conn = Mock()

        # Cycle through different message types
        messages = [mock_gps_message, mock_battery_message, mock_attitude_message]
        mock_conn.recv_match = Mock(side_effect=messages * 10)  # Repeat messages

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        return interface

    def test_gps_telemetry_parsing(self, mock_gps_message: Mock) -> None:
        """Test GPS telemetry parsing."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=mock_gps_message)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process GPS message
        interface.process_telemetry()

        telemetry = interface.get_telemetry()
        assert telemetry["gps"]["lat"] == 37.7749
        assert telemetry["gps"]["lon"] == -122.4194
        assert telemetry["gps"]["alt"] == 100.0
        assert telemetry["gps"]["hdop"] == 1.5
        assert telemetry["gps"]["vdop"] == 2.0
        assert telemetry["gps"]["fix_type"] == 3
        assert telemetry["gps"]["satellites"] == 12
        assert telemetry["gps"]["ground_speed"] == 1.0  # m/s

    def test_battery_telemetry_parsing(self, mock_battery_message: Mock) -> None:
        """Test battery telemetry parsing."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=mock_battery_message)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process battery message
        interface.process_telemetry()

        telemetry = interface.get_telemetry()
        assert telemetry["battery"]["voltage"] == 22.2  # 6S * 3.7V
        assert telemetry["battery"]["current"] == 15.0  # Amps
        assert telemetry["battery"]["percentage"] == 75
        assert telemetry["battery"]["temperature"] == 35
        assert telemetry["battery"]["cells"] == 6

    def test_attitude_telemetry_parsing(self, mock_attitude_message: Mock) -> None:
        """Test attitude telemetry parsing."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=mock_attitude_message)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process attitude message
        interface.process_telemetry()

        telemetry = interface.get_telemetry()
        assert abs(telemetry["attitude"]["roll"] - 2.86) < 0.1  # degrees
        assert abs(telemetry["attitude"]["pitch"] + 1.15) < 0.1  # degrees
        assert abs(telemetry["attitude"]["yaw"] - 90.0) < 0.1  # degrees
        assert telemetry["attitude"]["roll_rate"] == 0.01
        assert telemetry["attitude"]["pitch_rate"] == -0.005
        assert telemetry["attitude"]["yaw_rate"] == 0.02

    def test_telemetry_update_rate(self, mock_interface_with_telemetry: MAVLinkInterface) -> None:
        """Test telemetry update rate."""
        # Start telemetry thread
        mock_interface_with_telemetry.start_telemetry_thread()

        # Collect telemetry for 1 second
        time.sleep(1.0)

        # Should have received multiple updates
        call_count = mock_interface_with_telemetry.connection.recv_match.call_count
        assert call_count >= 4  # At least 4 Hz

    def test_telemetry_data_validation(self) -> None:
        """Test telemetry data validation."""
        # Create message with invalid data
        invalid_gps = Mock()
        invalid_gps.get_type = Mock(return_value="GPS_RAW_INT")
        invalid_gps.lat = 999999999  # Invalid latitude
        invalid_gps.lon = 999999999  # Invalid longitude
        invalid_gps.fix_type = 0  # No fix
        invalid_gps.satellites_visible = 0

        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=invalid_gps)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process invalid message
        interface.process_telemetry()

        telemetry = interface.get_telemetry()
        # Should handle invalid data gracefully
        assert telemetry["gps"]["fix_type"] == 0
        assert telemetry["gps"]["satellites"] == 0

    def test_telemetry_history(self, mock_interface_with_telemetry: MAVLinkInterface) -> None:
        """Test telemetry history tracking."""
        # Process multiple messages
        for _ in range(10):
            mock_interface_with_telemetry.process_telemetry()

        history = mock_interface_with_telemetry.get_telemetry_history()

        # Should maintain history
        assert len(history) > 0
        assert "timestamp" in history[0]
        assert "data" in history[0]


@pytest.mark.mock_hardware
@pytest.mark.mavlink
class TestMockMAVLinkFlightModes:
    """Test flight mode telemetry."""

    @pytest.fixture
    def mock_heartbeat_message(self) -> Mock:
        """Create mock heartbeat with flight mode."""
        msg = Mock()
        msg.get_type = Mock(return_value="HEARTBEAT")
        msg.type = 2  # Quadrotor
        msg.autopilot = 3  # ArduPilot
        msg.base_mode = 81  # Armed + Guided
        msg.custom_mode = 4  # GUIDED mode
        msg.system_status = 4  # Active
        return msg

    def test_flight_mode_parsing(self, mock_heartbeat_message: Mock) -> None:
        """Test flight mode parsing from heartbeat."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=mock_heartbeat_message)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process heartbeat
        interface.process_telemetry()

        telemetry = interface.get_telemetry()
        assert telemetry["flight_mode"] == "GUIDED"
        assert telemetry["armed"] is True
        assert telemetry["system_status"] == "ACTIVE"

    @pytest.mark.parametrize(
        "custom_mode,expected_mode",
        [
            (0, "STABILIZE"),
            (1, "ACRO"),
            (2, "ALT_HOLD"),
            (3, "AUTO"),
            (4, "GUIDED"),
            (5, "LOITER"),
            (6, "RTL"),
            (7, "CIRCLE"),
            (9, "LAND"),
            (16, "POSHOLD"),
        ],
    )
    def test_flight_mode_mapping(self, custom_mode: int, expected_mode: str) -> None:
        """Test flight mode mapping for ArduPilot."""
        msg = Mock()
        msg.get_type = Mock(return_value="HEARTBEAT")
        msg.custom_mode = custom_mode
        msg.base_mode = 81  # Armed + Guided

        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=msg)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        interface.process_telemetry()
        telemetry = interface.get_telemetry()

        assert telemetry["flight_mode"] == expected_mode


@pytest.mark.mock_hardware
@pytest.mark.mavlink
class TestMockMAVLinkPosition:
    """Test position telemetry."""

    @pytest.fixture
    def mock_position_message(self) -> Mock:
        """Create mock position message."""
        msg = Mock()
        msg.get_type = Mock(return_value="GLOBAL_POSITION_INT")
        msg.lat = 377749000  # 37.7749 degrees * 1e7
        msg.lon = -1224194000  # -122.4194 degrees * 1e7
        msg.alt = 100000  # 100m * 1000
        msg.relative_alt = 50000  # 50m * 1000
        msg.vx = 100  # North velocity cm/s
        msg.vy = 50  # East velocity cm/s
        msg.vz = -10  # Down velocity cm/s
        msg.hdg = 9000  # Heading * 100
        return msg

    def test_position_parsing(self, mock_position_message: Mock) -> None:
        """Test position message parsing."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=mock_position_message)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        interface.process_telemetry()
        telemetry = interface.get_telemetry()

        assert telemetry["position"]["lat"] == 37.7749
        assert telemetry["position"]["lon"] == -122.4194
        assert telemetry["position"]["alt"] == 100.0
        assert telemetry["position"]["relative_alt"] == 50.0
        assert telemetry["velocity"]["north"] == 1.0  # m/s
        assert telemetry["velocity"]["east"] == 0.5  # m/s
        assert telemetry["velocity"]["down"] == -0.1  # m/s
        assert telemetry["position"]["heading"] == 90.0  # degrees

    def test_home_position(self) -> None:
        """Test home position tracking."""
        home_msg = Mock()
        home_msg.get_type = Mock(return_value="HOME_POSITION")
        home_msg.latitude = 377749000
        home_msg.longitude = -1224194000
        home_msg.altitude = 50000

        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=home_msg)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        interface.process_telemetry()
        telemetry = interface.get_telemetry()

        assert telemetry["home"]["lat"] == 37.7749
        assert telemetry["home"]["lon"] == -122.4194
        assert telemetry["home"]["alt"] == 50.0


@pytest.mark.mock_hardware
@pytest.mark.mavlink
@pytest.mark.integration
class TestMockMAVLinkIntegration:
    """Integration tests for MAVLink telemetry."""

    def test_full_telemetry_stream(self) -> None:
        """Test processing full telemetry stream."""
        # Create various messages
        messages = []

        # GPS
        gps = Mock()
        gps.get_type = Mock(return_value="GPS_RAW_INT")
        gps.lat = 377749000
        gps.lon = -1224194000
        gps.fix_type = 3
        gps.satellites_visible = 12
        messages.append(gps)

        # Battery
        battery = Mock()
        battery.get_type = Mock(return_value="BATTERY_STATUS")
        battery.voltages = [3700] * 6
        battery.current_battery = 10000
        battery.battery_remaining = 80
        messages.append(battery)

        # Attitude
        attitude = Mock()
        attitude.get_type = Mock(return_value="ATTITUDE")
        attitude.roll = 0.0
        attitude.pitch = 0.0
        attitude.yaw = 0.0
        messages.append(attitude)

        # Heartbeat
        heartbeat = Mock()
        heartbeat.get_type = Mock(return_value="HEARTBEAT")
        heartbeat.custom_mode = 4  # GUIDED
        heartbeat.base_mode = 81  # Armed
        messages.append(heartbeat)

        mock_conn = Mock()
        mock_conn.recv_match = Mock(side_effect=messages)

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        # Process all messages
        for _ in range(len(messages)):
            interface.process_telemetry()

        # Get complete telemetry
        telemetry = interface.get_telemetry()

        # Verify all data present
        assert telemetry["gps"]["satellites"] == 12
        assert telemetry["battery"]["percentage"] == 80
        assert telemetry["attitude"]["roll"] == 0.0
        assert telemetry["flight_mode"] == "GUIDED"
        assert telemetry["armed"] is True
