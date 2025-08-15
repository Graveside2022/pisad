"""Integration tests for MAVLink service with SITL."""

import asyncio
import logging
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from src.backend.services.mavlink_service import (
    ConnectionState,

pytestmark = pytest.mark.serial
    LogLevel,
    MAVLinkService,
)

# Skip integration tests if SITL is not available
pytestmark = pytest.mark.skipif(
    subprocess.run(["which", "sim_vehicle.py"], capture_output=True).returncode != 0,
    reason="SITL not installed",
)


class TestMAVLinkSITLIntegration:
    """Integration tests with SITL."""

    @pytest.fixture(scope="class")
    def sitl_process(self):
        """Start SITL for testing."""
        # Check if SITL is available
        try:
            # Start SITL in background
            proc = subprocess.Popen(
                [
                    "sim_vehicle.py",
                    "-v",
                    "ArduCopter",
                    "--no-mavproxy",
                    "--out",
                    "tcp:127.0.0.1:5760",
                    "-L",
                    "-35.363261,149.165230,584,90",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            # Wait for SITL to start
            time.sleep(10)

            yield proc

            # Clean up
            proc.terminate()
            proc.wait(timeout=5)

        except FileNotFoundError:
            pytest.skip("SITL not available")
        except Exception as e:
            pytest.skip(f"Failed to start SITL: {e}")

    @pytest.mark.asyncio
    async def test_sitl_connection(self, sitl_process):
        """Test connecting to SITL."""
        service = MAVLinkService(device_path="tcp:127.0.0.1:5760", baud_rate=115200)

        try:
            await service.start()

            # Wait for connection
            max_attempts = 10
            for _ in range(max_attempts):
                if service.is_connected():
                    break
                await asyncio.sleep(1)

            assert service.is_connected()
            assert service.state == ConnectionState.CONNECTED

            # Check telemetry is being received
            telemetry = service.get_telemetry()
            assert telemetry["connected"] is True

        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_telemetry_reception(self, sitl_process):
        """Test receiving telemetry from SITL."""
        service = MAVLinkService(device_path="tcp:127.0.0.1:5760", baud_rate=115200)

        try:
            await service.start()

            # Wait for connection
            await asyncio.sleep(2)

            if not service.is_connected():
                pytest.skip("Could not connect to SITL")

            # Wait for telemetry updates
            await asyncio.sleep(3)

            telemetry = service.get_telemetry()

            # Check telemetry data
            assert telemetry["flight_mode"] != "UNKNOWN"
            assert "position" in telemetry
            assert "battery" in telemetry
            assert "gps" in telemetry

            # GPS should have some fix in SITL
            assert telemetry["gps"]["fix_type"] >= 2

        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_velocity_command_safety(self, sitl_process):
        """Test velocity command safety features."""
        service = MAVLinkService(device_path="tcp:127.0.0.1:5760", baud_rate=115200)

        try:
            await service.start()
            await asyncio.sleep(2)

            if not service.is_connected():
                pytest.skip("Could not connect to SITL")

            # Commands should be disabled by default
            result = await service.send_velocity_command(1.0, 0.0, 0.0)
            assert result is False

            # Enable commands
            service.enable_velocity_commands(True)

            # Now command should succeed
            result = await service.send_velocity_command(1.0, 0.0, 0.0)
            assert result is True

            # Test rate limiting
            result1 = await service.send_velocity_command(1.0, 0.0, 0.0)
            result2 = await service.send_velocity_command(1.0, 0.0, 0.0)
            assert result1 is True
            assert result2 is False  # Should be rate limited

        finally:
            await service.stop()

    @pytest.mark.asyncio
    async def test_connection_recovery(self, sitl_process):
        """Test automatic reconnection after connection loss."""
        service = MAVLinkService(device_path="tcp:127.0.0.1:5760", baud_rate=115200)

        try:
            await service.start()
            await asyncio.sleep(2)

            if not service.is_connected():
                pytest.skip("Could not connect to SITL")

            # Simulate connection loss
            if service.connection:
                service.connection.close()
                service._set_state(ConnectionState.DISCONNECTED)

            # Wait for automatic reconnection
            await asyncio.sleep(5)

            # Should reconnect automatically
            assert service.is_connected()

        finally:
            await service.stop()


class TestMAVLinkWebSocketIntegration:
    """Integration tests for MAVLink with WebSocket."""

    @pytest.mark.asyncio
    async def test_websocket_telemetry_broadcast(self):
        """Test telemetry broadcasting via WebSocket."""
        # This would require running the full FastAPI app
        # For now, we'll test the broadcasting function directly

        from src.backend.api.websocket import broadcast_telemetry_updates

        # Mock the MAVLink service
        mock_service = MagicMock()
        mock_service.is_connected.return_value = True
        mock_service.get_telemetry.return_value = {
            "position": {"lat": -35.363261, "lon": 149.165230, "alt": 584},
            "battery": {"percentage": 75},
            "flight_mode": "GUIDED",
            "armed": False,
        }
        mock_service.get_gps_status_string.return_value = "3D_FIX"

        with (
            patch("src.backend.api.websocket.mavlink_service", mock_service),
            patch("src.backend.api.websocket.manager.broadcast_json") as mock_broadcast,
            patch("asyncio.sleep", side_effect=asyncio.CancelledError),
            pytest.raises(asyncio.CancelledError),
        ):
            await broadcast_telemetry_updates()

        # Verify broadcast was called with telemetry
        mock_broadcast.assert_called_once()
        message = mock_broadcast.call_args[0][0]
        assert message["type"] == "telemetry"
        assert "position" in message["data"]
        assert "battery" in message["data"]
        assert message["data"]["flightMode"] == "GUIDED"

    @pytest.mark.asyncio
    async def test_mavlink_status_broadcast(self):
        """Test MAVLink connection status broadcasting."""
        from src.backend.api.websocket import broadcast_mavlink_status

        with patch("src.backend.api.websocket.manager.broadcast_json") as mock_broadcast:
            await broadcast_mavlink_status(ConnectionState.CONNECTED)

            mock_broadcast.assert_called_once()
            message = mock_broadcast.call_args[0][0]
            assert message["type"] == "mavlink_status"
            assert message["data"]["connected"] is True
            assert message["data"]["state"] == "connected"


class TestMAVLinkLogging:
    """Test MAVLink logging functionality."""

    def test_logging_levels(self):
        """Test different logging verbosity levels."""

        # Test INFO level
        service = MAVLinkService(log_level=LogLevel.INFO)
        assert service.log_level == LogLevel.INFO

        # Test DEBUG level
        service = MAVLinkService(log_level=LogLevel.DEBUG)
        assert service.log_level == LogLevel.DEBUG

        # Test TRACE level
        service = MAVLinkService(log_level=LogLevel.TRACE)
        assert service.log_level == LogLevel.TRACE

    @pytest.mark.asyncio
    async def test_message_logging(self):
        """Test message logging at different levels."""
        service = MAVLinkService(
            log_level=LogLevel.TRACE, log_messages=["HEARTBEAT", "GPS_RAW_INT"]
        )

        # Create mock message
        mock_msg = MagicMock()
        mock_msg.get_type.return_value = "HEARTBEAT"
        mock_msg.to_dict.return_value = {"type": "HEARTBEAT"}
        mock_msg.custom_mode = 4
        mock_msg.base_mode = 0

        with patch.object(
            logging.getLogger("src.backend.services.mavlink_service"), "log"
        ) as mock_log:
            await service._process_message(mock_msg)

            # Should log at TRACE level
            assert mock_log.called

        # Test filtered message (not in log_messages)
        mock_msg.get_type.return_value = "ATTITUDE"

        with patch.object(
            logging.getLogger("src.backend.services.mavlink_service"), "log"
        ) as mock_log:
            await service._process_message(mock_msg)

            # Should not log (filtered out)
            assert not mock_log.called


class TestMAVLinkMessageValidation:
    """Test MAVLink message validation and processing."""

    def test_gps_status_conversion(self):
        """Test GPS status string conversion."""
        service = MAVLinkService()

        test_cases = [
            (0, "NO_FIX"),
            (1, "NO_FIX"),
            (2, "2D_FIX"),
            (3, "3D_FIX"),
            (4, "RTK"),
            (5, "RTK"),
            (6, "RTK"),
        ]

        for fix_type, expected in test_cases:
            service.telemetry["gps"]["fix_type"] = fix_type
            assert service.get_gps_status_string() == expected

    def test_flight_mode_conversion(self):
        """Test flight mode name conversion."""
        service = MAVLinkService()

        test_cases = [
            (0, "STABILIZE"),
            (3, "AUTO"),
            (4, "GUIDED"),
            (5, "LOITER"),
            (6, "RTL"),
            (9, "LAND"),
            (999, "UNKNOWN"),
        ]

        for mode, expected in test_cases:
            assert service._get_flight_mode_name(mode) == expected

    def test_telemetry_data_structure(self):
        """Test telemetry data structure."""
        service = MAVLinkService()
        telemetry = service.get_telemetry()

        # Check required fields
        assert "position" in telemetry
        assert "lat" in telemetry["position"]
        assert "lon" in telemetry["position"]
        assert "alt" in telemetry["position"]

        assert "attitude" in telemetry
        assert "roll" in telemetry["attitude"]
        assert "pitch" in telemetry["attitude"]
        assert "yaw" in telemetry["attitude"]

        assert "battery" in telemetry
        assert "voltage" in telemetry["battery"]
        assert "current" in telemetry["battery"]
        assert "percentage" in telemetry["battery"]

        assert "gps" in telemetry
        assert "fix_type" in telemetry["gps"]
        assert "satellites" in telemetry["gps"]
        assert "hdop" in telemetry["gps"]

        assert "flight_mode" in telemetry
        assert "armed" in telemetry
        assert "connected" in telemetry
        assert "connection_state" in telemetry
