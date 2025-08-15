"""Unit tests for MAVLink service."""

import asyncio
import contextlib
import logging
from unittest.mock import MagicMock, patch

import pytest
from pymavlink import mavutil

from src.backend.services.mavlink_service import (
    ConnectionState,
    LogLevel,
    MAVLinkService,
)


@pytest.fixture
def mavlink_service():
    """Create a MAVLink service instance for testing."""
    return MAVLinkService(
        device_path="tcp:127.0.0.1:5760", baud_rate=115200, log_level=LogLevel.INFO
    )


@pytest.fixture
def mock_connection():
    """Create a mock MAVLink connection."""
    mock = MagicMock()
    mock.wait_heartbeat = MagicMock()
    mock.recv_match = MagicMock()
    mock.mav = MagicMock()
    mock.close = MagicMock()
    return mock


class TestMAVLinkService:
    """Test MAVLink service functionality."""

    def test_initialization(self, mavlink_service):
        """Test service initialization."""
        assert mavlink_service.device_path == "tcp:127.0.0.1:5760"
        assert mavlink_service.baud_rate == 115200
        assert mavlink_service.state == ConnectionState.DISCONNECTED
        assert mavlink_service.connection is None
        assert not mavlink_service._velocity_commands_enabled

    def test_state_callback(self, mavlink_service):
        """Test state change callbacks."""
        callback_called = False
        new_state = None

        def callback(state):
            nonlocal callback_called, new_state
            callback_called = True
            new_state = state

        mavlink_service.add_state_callback(callback)
        mavlink_service._set_state(ConnectionState.CONNECTED)

        assert callback_called
        assert new_state == ConnectionState.CONNECTED
        assert mavlink_service.state == ConnectionState.CONNECTED

    @pytest.mark.asyncio
    async def test_start_stop(self, mavlink_service):
        """Test starting and stopping the service."""
        # Start the service
        await mavlink_service.start()
        assert mavlink_service._running
        assert len(mavlink_service._tasks) == 5  # Updated to 5 tasks (includes telemetry_sender)

        # Stop the service
        await mavlink_service.stop()
        assert not mavlink_service._running
        assert len(mavlink_service._tasks) == 0
        assert mavlink_service.state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_connect_tcp(self, mavlink_service, mock_connection):
        """Test TCP connection to SITL."""
        with patch(
            "src.backend.services.mavlink_service.mavutil.mavlink_connection",
            return_value=mock_connection,
        ):
            # Mock successful heartbeat
            mock_connection.wait_heartbeat.return_value = MagicMock(
                get_srcSystem=MagicMock(return_value=1)
            )

            mavlink_service._running = True
            await mavlink_service._connect()

            assert mavlink_service.state == ConnectionState.CONNECTED
            assert mavlink_service.connection == mock_connection

    @pytest.mark.asyncio
    async def test_connect_serial(self, mock_connection):
        """Test serial connection to hardware."""
        service = MAVLinkService(device_path="/dev/ttyACM0", baud_rate=921600)

        with patch(
            "src.backend.services.mavlink_service.mavutil.mavlink_connection",
            return_value=mock_connection,
        ):
            # Mock successful heartbeat
            mock_connection.wait_heartbeat.return_value = MagicMock(
                get_srcSystem=MagicMock(return_value=1)
            )

            service._running = True
            await service._connect()

            assert service.state == ConnectionState.CONNECTED
            assert service.connection == mock_connection

    @pytest.mark.asyncio
    async def test_connect_failure(self, mavlink_service):
        """Test connection failure handling."""
        with patch(
            "src.backend.services.mavlink_service.mavutil.mavlink_connection",
            side_effect=Exception("Connection failed"),
        ):
            mavlink_service._running = True
            await mavlink_service._connect()

            assert mavlink_service.state == ConnectionState.DISCONNECTED
            assert mavlink_service.connection is None

    @pytest.mark.asyncio
    async def test_heartbeat_sender(self, mavlink_service, mock_connection):
        """Test heartbeat sending."""
        mavlink_service.connection = mock_connection
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service._running = True

        # Run one iteration - the sleep will raise CancelledError after heartbeat is sent
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError
            with contextlib.suppress(asyncio.CancelledError):
                await mavlink_service._heartbeat_sender()

        # Verify heartbeat was sent
        mock_connection.mav.heartbeat_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_message_receiver(self, mavlink_service, mock_connection):
        """Test message receiving and processing."""
        mavlink_service.connection = mock_connection
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service._running = True

        # Mock received message
        mock_msg = MagicMock()
        mock_msg.get_type.return_value = "HEARTBEAT"
        mock_msg.custom_mode = 4  # GUIDED mode
        mock_msg.base_mode = mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED

        mock_connection.recv_match.return_value = mock_msg

        # Run one iteration
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError
            with contextlib.suppress(asyncio.CancelledError):
                await mavlink_service._message_receiver()

        # Verify telemetry was updated
        assert mavlink_service.telemetry["flight_mode"] == "GUIDED"
        assert mavlink_service.telemetry["armed"] is True

    def test_process_heartbeat(self, mavlink_service):
        """Test heartbeat message processing."""
        mock_msg = MagicMock()
        mock_msg.custom_mode = 5  # LOITER mode
        mock_msg.base_mode = 0  # Not armed

        mavlink_service._process_heartbeat(mock_msg)

        assert mavlink_service.telemetry["flight_mode"] == "LOITER"
        assert mavlink_service.telemetry["armed"] is False

    def test_process_global_position(self, mavlink_service):
        """Test global position message processing."""
        mock_msg = MagicMock()
        mock_msg.lat = -353632610  # -35.363261 * 1e7
        mock_msg.lon = 1491652300  # 149.165230 * 1e7
        mock_msg.alt = 584000  # 584m * 1000

        mavlink_service._process_global_position(mock_msg)

        assert mavlink_service.telemetry["position"]["lat"] == pytest.approx(-35.363261)
        assert mavlink_service.telemetry["position"]["lon"] == pytest.approx(149.165230)
        assert mavlink_service.telemetry["position"]["alt"] == 584.0

    def test_process_attitude(self, mavlink_service):
        """Test attitude message processing."""
        import math

        mock_msg = MagicMock()
        mock_msg.roll = math.radians(10)
        mock_msg.pitch = math.radians(-5)
        mock_msg.yaw = math.radians(90)

        mavlink_service._process_attitude(mock_msg)

        assert mavlink_service.telemetry["attitude"]["roll"] == pytest.approx(10)
        assert mavlink_service.telemetry["attitude"]["pitch"] == pytest.approx(-5)
        assert mavlink_service.telemetry["attitude"]["yaw"] == pytest.approx(90)

    def test_process_sys_status(self, mavlink_service):
        """Test system status message processing."""
        mock_msg = MagicMock()
        mock_msg.voltage_battery = 12600  # 12.6V in mV
        mock_msg.current_battery = 1500  # 15A in cA
        mock_msg.battery_remaining = 75  # 75%

        mavlink_service._process_sys_status(mock_msg)

        assert mavlink_service.telemetry["battery"]["voltage"] == 12.6
        assert mavlink_service.telemetry["battery"]["current"] == 15.0
        assert mavlink_service.telemetry["battery"]["percentage"] == 75

    def test_process_gps_raw(self, mavlink_service):
        """Test GPS raw message processing."""
        mock_msg = MagicMock()
        mock_msg.fix_type = 3  # 3D fix
        mock_msg.satellites_visible = 12
        mock_msg.eph = 150  # 1.5m HDOP in cm

        mavlink_service._process_gps_raw(mock_msg)

        assert mavlink_service.telemetry["gps"]["fix_type"] == 3
        assert mavlink_service.telemetry["gps"]["satellites"] == 12
        assert mavlink_service.telemetry["gps"]["hdop"] == 1.5

    def test_get_flight_mode_name(self, mavlink_service):
        """Test flight mode name conversion."""
        assert mavlink_service._get_flight_mode_name(0) == "STABILIZE"
        assert mavlink_service._get_flight_mode_name(4) == "GUIDED"
        assert mavlink_service._get_flight_mode_name(6) == "RTL"
        assert mavlink_service._get_flight_mode_name(999) == "UNKNOWN"

    def test_get_gps_status_string(self, mavlink_service):
        """Test GPS status string conversion."""
        mavlink_service.telemetry["gps"]["fix_type"] = 0
        assert mavlink_service.get_gps_status_string() == "NO_FIX"

        mavlink_service.telemetry["gps"]["fix_type"] = 2
        assert mavlink_service.get_gps_status_string() == "2D_FIX"

        mavlink_service.telemetry["gps"]["fix_type"] = 3
        assert mavlink_service.get_gps_status_string() == "3D_FIX"

        mavlink_service.telemetry["gps"]["fix_type"] = 6
        assert mavlink_service.get_gps_status_string() == "RTK"

    @pytest.mark.asyncio
    async def test_connection_monitor(self, mavlink_service):
        """Test connection health monitoring."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.last_heartbeat_received = 0  # Force timeout
        mavlink_service._running = True

        mock_connection = MagicMock()
        mavlink_service.connection = mock_connection

        # Run one iteration
        with patch("time.time", return_value=10), patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError
            with contextlib.suppress(asyncio.CancelledError):
                await mavlink_service._connection_monitor()

        # Verify disconnection due to timeout
        assert mavlink_service.state == ConnectionState.DISCONNECTED
        mock_connection.close.assert_called_once()

    def test_enable_velocity_commands(self, mavlink_service):
        """Test enabling/disabling velocity commands."""
        assert not mavlink_service._velocity_commands_enabled

        mavlink_service.enable_velocity_commands(True)
        assert mavlink_service._velocity_commands_enabled

        mavlink_service.enable_velocity_commands(False)
        assert not mavlink_service._velocity_commands_enabled

    @pytest.mark.asyncio
    async def test_send_velocity_command_disabled(self, mavlink_service):
        """Test velocity command when disabled."""
        result = await mavlink_service.send_velocity_command(1.0, 0.0, 0.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_velocity_command_disconnected(self, mavlink_service):
        """Test velocity command when disconnected."""
        mavlink_service.enable_velocity_commands(True)
        result = await mavlink_service.send_velocity_command(1.0, 0.0, 0.0)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_velocity_command_success(self, mavlink_service, mock_connection):
        """Test successful velocity command sending."""
        mavlink_service.enable_velocity_commands(True)
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        result = await mavlink_service.send_velocity_command(vx=2.0, vy=1.0, vz=-0.5, yaw_rate=0.1)

        assert result is True
        mock_connection.mav.set_position_target_local_ned_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_velocity_command_rate_limiting(self, mavlink_service, mock_connection):
        """Test velocity command rate limiting."""
        mavlink_service.enable_velocity_commands(True)
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        # Send first command
        result1 = await mavlink_service.send_velocity_command(1.0, 0.0, 0.0)
        assert result1 is True

        # Immediate second command should be rate limited
        result2 = await mavlink_service.send_velocity_command(1.0, 0.0, 0.0)
        assert result2 is False

    @pytest.mark.asyncio
    async def test_send_velocity_command_bounds_checking(self, mavlink_service, mock_connection):
        """Test velocity command bounds checking."""
        mavlink_service.enable_velocity_commands(True)
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        # Send command with excessive velocity
        await mavlink_service.send_velocity_command(
            vx=10.0,
            vy=-10.0,
            vz=3.0,  # Should be clamped to 5.0  # Should be clamped to -5.0
        )

        # Check that velocities were clamped
        call_args = mock_connection.mav.set_position_target_local_ned_send.call_args
        assert call_args[0][8] == 5.0  # vx
        assert call_args[0][9] == -5.0  # vy
        assert call_args[0][10] == 3.0  # vz

    def test_get_telemetry(self, mavlink_service):
        """Test telemetry data retrieval."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.telemetry["flight_mode"] = "GUIDED"

        telemetry = mavlink_service.get_telemetry()

        assert telemetry["connected"] is True
        assert telemetry["connection_state"] == "connected"
        assert telemetry["flight_mode"] == "GUIDED"

    def test_is_connected(self, mavlink_service):
        """Test connection status check."""
        assert not mavlink_service.is_connected()

        mavlink_service.state = ConnectionState.CONNECTED
        assert mavlink_service.is_connected()

        mavlink_service.state = ConnectionState.CONNECTING
        assert not mavlink_service.is_connected()

    def test_set_log_level(self, mavlink_service):
        """Test logging level configuration."""
        mavlink_service.set_log_level(LogLevel.DEBUG)
        assert mavlink_service.log_level == LogLevel.DEBUG

        mavlink_service.set_log_level(LogLevel.TRACE)
        assert mavlink_service.log_level == LogLevel.TRACE

    def test_set_log_filters(self, mavlink_service):
        """Test message type filtering for logging."""
        mavlink_service.set_log_filters(["HEARTBEAT", "GPS_RAW_INT"])
        assert mavlink_service.log_messages == ["HEARTBEAT", "GPS_RAW_INT"]

        mavlink_service.set_log_filters(None)
        assert mavlink_service.log_messages == []

    @pytest.mark.asyncio
    async def test_process_message_with_trace_logging(self, mavlink_service):
        """Test message processing with TRACE level logging."""
        mavlink_service.log_level = LogLevel.TRACE
        mavlink_service.log_messages = ["HEARTBEAT"]

        mock_msg = MagicMock()
        mock_msg.get_type.return_value = "HEARTBEAT"
        mock_msg.to_dict.return_value = {"type": "HEARTBEAT", "custom_mode": 4}
        mock_msg.custom_mode = 4
        mock_msg.base_mode = 0

        with patch.object(
            logging.getLogger("src.backend.services.mavlink_service"), "log"
        ) as mock_log:
            await mavlink_service._process_message(mock_msg)

            # Verify TRACE logging was called
            mock_log.assert_called_with(
                LogLevel.TRACE.value, "Received HEARTBEAT: {'type': 'HEARTBEAT', 'custom_mode': 4}"
            )

    def test_send_statustext_success(self, mavlink_service, mock_connection):
        """Test successful STATUSTEXT message sending."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        result = mavlink_service.send_statustext("Test message", severity=6)

        assert result is True
        mock_connection.mav.statustext_send.assert_called_once()

    def test_send_statustext_disconnected(self, mavlink_service):
        """Test STATUSTEXT when disconnected."""
        result = mavlink_service.send_statustext("Test message")
        assert result is False

    def test_send_statustext_truncation(self, mavlink_service, mock_connection):
        """Test STATUSTEXT message truncation to 50 chars."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        long_message = "This is a very long message that exceeds the maximum 50 character limit"
        mavlink_service.send_statustext(long_message)

        call_args = mock_connection.mav.statustext_send.call_args
        sent_text = call_args[0][1].decode("utf-8")
        assert len(sent_text) <= 50

    def test_send_named_value_float_success(self, mavlink_service, mock_connection):
        """Test successful NAMED_VALUE_FLOAT message sending."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        result = mavlink_service.send_named_value_float("RSSI", -75.5, 1234567890.0)

        assert result is True
        mock_connection.mav.named_value_float_send.assert_called_once()

    def test_send_named_value_float_disconnected(self, mavlink_service):
        """Test NAMED_VALUE_FLOAT when disconnected."""
        result = mavlink_service.send_named_value_float("RSSI", -75.5)
        assert result is False

    def test_send_named_value_float_name_truncation(self, mavlink_service, mock_connection):
        """Test NAMED_VALUE_FLOAT name truncation to 10 chars."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        mavlink_service.send_named_value_float("VERY_LONG_PARAMETER_NAME", 42.0)

        call_args = mock_connection.mav.named_value_float_send.call_args
        sent_name = call_args[0][1].decode("utf-8")
        assert len(sent_name) <= 10

    def test_send_state_change(self, mavlink_service, mock_connection):
        """Test state change notification."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        # First state change should send
        result = mavlink_service.send_state_change("SEARCHING")
        assert result is True

        # Same state should not resend
        result = mavlink_service.send_state_change("SEARCHING")
        assert result is True
        assert mock_connection.mav.statustext_send.call_count == 1

    def test_send_detection_event_throttling(self, mavlink_service, mock_connection):
        """Test detection event throttling."""
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        # First event should send
        with patch("time.time", return_value=1000.0):
            result = mavlink_service.send_detection_event(-70.0, 95.0)
            assert result is True

        # Immediate second event should be throttled
        with patch("time.time", return_value=1000.1):
            result = mavlink_service.send_detection_event(-69.0, 96.0)
            assert result is False

        # After throttle period, should send again
        with patch("time.time", return_value=1001.0):
            result = mavlink_service.send_detection_event(-68.0, 97.0)
            assert result is True

    def test_update_rssi_value(self, mavlink_service):
        """Test RSSI value update."""
        mavlink_service.update_rssi_value(-85.5)
        assert mavlink_service._rssi_value == -85.5

    def test_update_telemetry_config(self, mavlink_service):
        """Test telemetry configuration update."""
        config = {"rssi_rate_hz": 5.0, "health_interval_seconds": 30, "detection_throttle_ms": 1000}

        mavlink_service.update_telemetry_config(config)

        assert mavlink_service._telemetry_config["rssi_rate_hz"] == 5.0
        assert mavlink_service._telemetry_config["health_interval_seconds"] == 30
        assert mavlink_service._telemetry_config["detection_throttle_ms"] == 1000

    def test_update_telemetry_config_bounds(self, mavlink_service):
        """Test telemetry configuration bounds checking."""
        config = {
            "rssi_rate_hz": 20.0,  # Should be clamped to 10.0
            "health_interval_seconds": 100,  # Should be clamped to 60
            "detection_throttle_ms": 50,  # Should be clamped to 100
        }

        mavlink_service.update_telemetry_config(config)

        assert mavlink_service._telemetry_config["rssi_rate_hz"] == 10.0
        assert mavlink_service._telemetry_config["health_interval_seconds"] == 60
        assert mavlink_service._telemetry_config["detection_throttle_ms"] == 100

    def test_get_telemetry_config(self, mavlink_service):
        """Test getting telemetry configuration."""
        config = mavlink_service.get_telemetry_config()

        assert "rssi_rate_hz" in config
        assert "health_interval_seconds" in config
        assert "detection_throttle_ms" in config

    def test_mode_callback(self, mavlink_service):
        """Test flight mode change callbacks."""
        callback_called = False
        new_mode = None

        def callback(mode):
            nonlocal callback_called, new_mode
            callback_called = True
            new_mode = mode

        mavlink_service.add_mode_callback(callback)

        # Simulate mode change via heartbeat
        mock_msg = MagicMock()
        mock_msg.custom_mode = 3  # AUTO mode
        mock_msg.base_mode = 0

        mavlink_service._process_heartbeat(mock_msg)

        assert callback_called
        assert new_mode == "AUTO"

    def test_battery_callback(self, mavlink_service):
        """Test battery percentage callbacks."""
        callback_called = False
        battery_percentage = None

        def callback(percentage):
            nonlocal callback_called, battery_percentage
            callback_called = True
            battery_percentage = percentage

        mavlink_service.add_battery_callback(callback)

        # Simulate battery update
        mock_msg = MagicMock()
        mock_msg.voltage_battery = 12000
        mock_msg.current_battery = 1000
        mock_msg.battery_remaining = 80

        mavlink_service._process_sys_status(mock_msg)

        assert callback_called
        assert battery_percentage == 80

    def test_position_callback(self, mavlink_service):
        """Test position update callbacks."""
        callback_called = False
        received_lat = None
        received_lon = None

        def callback(lat, lon):
            nonlocal callback_called, received_lat, received_lon
            callback_called = True
            received_lat = lat
            received_lon = lon

        mavlink_service.add_position_callback(callback)

        # Simulate position update
        mock_msg = MagicMock()
        mock_msg.lat = -353632610
        mock_msg.lon = 1491652300
        mock_msg.alt = 500000

        mavlink_service._process_global_position(mock_msg)

        assert callback_called
        assert received_lat == pytest.approx(-35.363261)
        assert received_lon == pytest.approx(149.165230)

    def test_safety_check_callback(self, mavlink_service, mock_connection):
        """Test safety check callback for velocity commands."""
        mavlink_service.enable_velocity_commands(True)
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service.connection = mock_connection

        # Set safety callback that returns False
        mavlink_service.set_safety_check_callback(lambda: False)

        # Velocity command should be blocked
        result = asyncio.run(mavlink_service.send_velocity_command(1.0, 0.0, 0.0))
        assert result is False

        # Set safety callback that returns True
        mavlink_service.set_safety_check_callback(lambda: True)

        # Velocity command should succeed
        result = asyncio.run(mavlink_service.send_velocity_command(1.0, 0.0, 0.0))
        assert result is True

    @pytest.mark.asyncio
    async def test_upload_mission_success(self, mavlink_service, mock_connection):
        """Test successful mission upload."""
        mavlink_service.connection = mock_connection

        # Mock ACK responses
        mock_ack = MagicMock()
        mock_ack.type = mavutil.mavlink.MAV_MISSION_ACCEPTED

        # Mock mission requests for each waypoint
        mock_req1 = MagicMock()
        mock_req1.seq = 0

        mock_req2 = MagicMock()
        mock_req2.seq = 1

        mock_connection.recv_match.side_effect = [mock_ack, mock_req1, mock_req2, mock_ack]

        waypoints = [
            {"lat": -35.363261, "lon": 149.165230, "alt": 50.0},
            {"lat": -35.364000, "lon": 149.166000, "alt": 60.0},
        ]

        result = await mavlink_service.upload_mission(waypoints)

        assert result is True
        assert mock_connection.mav.mission_clear_all_send.called
        assert mock_connection.mav.mission_count_send.called
        assert mock_connection.mav.mission_item_send.call_count == 2  # Called twice for 2 waypoints

    @pytest.mark.asyncio
    async def test_upload_mission_no_connection(self, mavlink_service):
        """Test mission upload with no connection."""
        result = await mavlink_service.upload_mission([])
        assert result is False

    @pytest.mark.asyncio
    async def test_start_mission_success(self, mavlink_service, mock_connection):
        """Test successful mission start."""
        mavlink_service.connection = mock_connection
        mavlink_service.telemetry["armed"] = False

        # Mock mode mapping
        mock_connection.mode_mapping.return_value = {"AUTO": 3}

        result = await mavlink_service.start_mission()

        assert result is True
        assert mock_connection.mav.set_mode_send.called
        assert mock_connection.mav.command_long_send.called

    @pytest.mark.asyncio
    async def test_pause_mission(self, mavlink_service, mock_connection):
        """Test mission pause."""
        mavlink_service.connection = mock_connection

        result = await mavlink_service.pause_mission()

        assert result is True
        mock_connection.mav.command_long_send.assert_called_once()
        call_args = mock_connection.mav.command_long_send.call_args[0]
        assert call_args[2] == mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE
        assert call_args[4] == 0  # 0 = pause

    @pytest.mark.asyncio
    async def test_resume_mission(self, mavlink_service, mock_connection):
        """Test mission resume."""
        mavlink_service.connection = mock_connection

        result = await mavlink_service.resume_mission()

        assert result is True
        mock_connection.mav.command_long_send.assert_called_once()
        call_args = mock_connection.mav.command_long_send.call_args[0]
        assert call_args[2] == mavutil.mavlink.MAV_CMD_DO_PAUSE_CONTINUE
        assert call_args[4] == 1  # 1 = continue

    @pytest.mark.asyncio
    async def test_stop_mission(self, mavlink_service, mock_connection):
        """Test mission stop."""
        mavlink_service.connection = mock_connection

        # Mock mode mapping
        mock_connection.mode_mapping.return_value = {"LOITER": 5, "GUIDED": 4}

        result = await mavlink_service.stop_mission()

        assert result is True
        mock_connection.mav.set_mode_send.assert_called_once()

    def test_get_mission_progress(self, mavlink_service, mock_connection):
        """Test getting mission progress."""
        mavlink_service.connection = mock_connection

        # Mock current waypoint response
        mock_current = MagicMock()
        mock_current.seq = 3

        # Mock mission count response
        mock_count = MagicMock()
        mock_count.count = 10

        mock_connection.recv_match.side_effect = [mock_current, mock_count]

        current, total = mavlink_service.get_mission_progress()

        assert current == 3
        assert total == 10

    @pytest.mark.asyncio
    async def test_telemetry_sender(self, mavlink_service, mock_connection):
        """Test telemetry sender task."""
        mavlink_service.connection = mock_connection
        mavlink_service.state = ConnectionState.CONNECTED
        mavlink_service._running = True
        mavlink_service._rssi_value = -75.0

        # Run one iteration
        with patch("time.time", return_value=1000.0), patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError
            with contextlib.suppress(asyncio.CancelledError):
                await mavlink_service.telemetry_sender()

        # Verify RSSI was sent
        mock_connection.mav.named_value_float_send.assert_called()

    @pytest.mark.asyncio
    async def test_send_health_status(self, mavlink_service, mock_connection):
        """Test health status sending."""
        mavlink_service.connection = mock_connection
        mavlink_service.state = ConnectionState.CONNECTED

        with (
            patch("psutil.cpu_percent", return_value=45.0),
            patch("psutil.virtual_memory") as mock_mem,
        ):
            mock_mem.return_value.percent = 60.0

            await mavlink_service._send_health_status()

        mock_connection.mav.statustext_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_running(self, mavlink_service):
        """Test starting service when already running."""
        mavlink_service._running = True

        with patch.object(mavlink_service, "_connection_manager") as mock_manager:
            await mavlink_service.start()
            mock_manager.assert_not_called()
