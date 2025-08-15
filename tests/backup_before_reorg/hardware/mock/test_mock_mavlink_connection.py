"""Mock MAVLink Connection Tests.

Tests for MAVLink serial and UDP connection modes
without requiring real hardware.
"""

import threading
import time
from typing import Any
from unittest.mock import Mock, patch

import pytest

from backend.hal.mavlink_interface import MAVLinkInterface


@pytest.mark.mock_hardware
@pytest.mark.mavlink
class TestMockMAVLinkConnection:
    """Test MAVLink connection with mock hardware."""

    @pytest.fixture
    def mock_connection(self) -> Mock:
        """Create a mock MAVLink connection."""
        conn = Mock()
        conn.target_system = 1
        conn.target_component = 1
        conn.close = Mock()
        conn.mav = Mock()

        # Mock heartbeat message
        heartbeat_msg = Mock()
        heartbeat_msg.type = 2  # Quadrotor
        heartbeat_msg.autopilot = 3  # ArduPilot
        heartbeat_msg.base_mode = 81  # Armed + Guided
        heartbeat_msg.system_status = 4  # Active
        heartbeat_msg.get_type = Mock(return_value="HEARTBEAT")

        conn.recv_match = Mock(return_value=heartbeat_msg)
        return conn

    @pytest.fixture
    def mock_interface(self, mock_connection: Mock) -> MAVLinkInterface:
        """Create MAVLink interface with mock connection."""
        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_connection

            interface = MAVLinkInterface()
            interface.connection = mock_connection
            interface.connected = True

            return interface

    def test_serial_connection(self, mock_connection: Mock) -> None:
        """Test serial port connection."""
        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_connection

            interface = MAVLinkInterface()
            result = interface.connect(connection_string="serial:///dev/ttyACM0:115200")

            assert result is True
            assert interface.connected is True
            mock_mavutil.assert_called_once_with(
                "serial:///dev/ttyACM0:115200", source_system=255, source_component=190
            )

    def test_udp_connection(self, mock_connection: Mock) -> None:
        """Test UDP connection for SITL."""
        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_connection

            interface = MAVLinkInterface()
            result = interface.connect(connection_string="udp:127.0.0.1:14550")

            assert result is True
            assert interface.connected is True
            mock_mavutil.assert_called_once_with(
                "udp:127.0.0.1:14550", source_system=255, source_component=190
            )

    def test_tcp_connection(self, mock_connection: Mock) -> None:
        """Test TCP connection."""
        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_connection

            interface = MAVLinkInterface()
            result = interface.connect(connection_string="tcp:127.0.0.1:5760")

            assert result is True
            mock_mavutil.assert_called_once()

    def test_connection_with_custom_ids(self, mock_connection: Mock) -> None:
        """Test connection with custom system/component IDs."""
        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_connection

            interface = MAVLinkInterface()
            result = interface.connect(
                connection_string="serial:///dev/ttyACM0:115200",
                source_system=254,
                source_component=1,
            )

            assert result is True
            mock_mavutil.assert_called_with(
                "serial:///dev/ttyACM0:115200", source_system=254, source_component=1
            )

    def test_heartbeat_detection(self, mock_interface: MAVLinkInterface) -> None:
        """Test heartbeat detection on connection."""
        # Start heartbeat thread
        mock_interface.start_heartbeat()
        time.sleep(0.1)

        # Should have received heartbeat
        mock_interface.connection.recv_match.assert_called()
        assert mock_interface.last_heartbeat > 0

    def test_connection_timeout(self) -> None:
        """Test connection timeout handling."""
        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=None)  # No heartbeat

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_conn

            interface = MAVLinkInterface()
            interface.heartbeat_timeout = 0.1  # Short timeout for test

            result = interface.connect("serial:///dev/ttyACM0:115200")

            # Should timeout and fail
            assert result is False or interface.connected is False

    def test_reconnection_logic(self, mock_interface: MAVLinkInterface) -> None:
        """Test automatic reconnection."""
        # Simulate disconnect
        mock_interface.connected = False
        mock_interface.last_heartbeat = 0

        # Trigger reconnection check
        mock_interface.check_connection()

        # Should attempt to reconnect
        assert mock_interface.reconnect_attempts > 0

    def test_multiple_connection_strings(self) -> None:
        """Test fallback connection strings."""
        connections_tried = []

        def mock_connect(conn_str: str, **kwargs: Any) -> Mock:
            connections_tried.append(conn_str)
            if "ttyACM0" in conn_str:
                raise Exception("Primary failed")
            conn = Mock()
            conn.recv_match = Mock(return_value=Mock(get_type=Mock(return_value="HEARTBEAT")))
            return conn

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection", mock_connect):
            interface = MAVLinkInterface()

            # Try primary then fallback
            result = interface.connect_with_fallback(
                [
                    "serial:///dev/ttyACM0:115200",
                    "serial:///dev/ttyACM1:115200",
                ]
            )

            assert len(connections_tried) == 2
            assert result is True


@pytest.mark.mock_hardware
@pytest.mark.mavlink
class TestMockMAVLinkDisconnection:
    """Test MAVLink disconnection scenarios."""

    @pytest.fixture
    def mock_interface(self) -> MAVLinkInterface:
        """Create interface with mock connection."""
        mock_conn = Mock()
        mock_conn.close = Mock()

        interface = MAVLinkInterface()
        interface.connection = mock_conn
        interface.connected = True

        return interface

    def test_clean_disconnect(self, mock_interface: MAVLinkInterface) -> None:
        """Test clean disconnection."""
        mock_interface.disconnect()

        assert mock_interface.connected is False
        mock_interface.connection.close.assert_called_once()
        assert mock_interface.connection is None

    def test_disconnect_with_active_threads(self, mock_interface: MAVLinkInterface) -> None:
        """Test disconnection with active threads."""
        # Start heartbeat thread
        mock_interface.heartbeat_thread = threading.Thread(target=lambda: time.sleep(0.1))
        mock_interface.heartbeat_thread.start()

        # Disconnect should stop threads
        mock_interface.disconnect()

        assert mock_interface.connected is False
        assert (
            mock_interface.heartbeat_thread is None
            or not mock_interface.heartbeat_thread.is_alive()
        )

    def test_disconnect_error_handling(self, mock_interface: MAVLinkInterface) -> None:
        """Test disconnection with errors."""
        # Make close raise exception
        mock_interface.connection.close.side_effect = Exception("Close failed")

        # Should handle error gracefully
        mock_interface.disconnect()

        assert mock_interface.connected is False
        assert mock_interface.connection is None

    def test_reconnect_after_disconnect(self) -> None:
        """Test reconnection after disconnection."""
        mock_conn1 = Mock()
        mock_conn1.close = Mock()
        mock_conn1.recv_match = Mock(return_value=Mock(get_type=Mock(return_value="HEARTBEAT")))

        mock_conn2 = Mock()
        mock_conn2.recv_match = Mock(return_value=Mock(get_type=Mock(return_value="HEARTBEAT")))

        connections = [mock_conn1, mock_conn2]

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.side_effect = connections

            interface = MAVLinkInterface()

            # First connection
            assert interface.connect("serial:///dev/ttyACM0:115200") is True
            assert interface.connection == mock_conn1

            # Disconnect
            interface.disconnect()
            assert interface.connected is False

            # Reconnect
            assert interface.connect("serial:///dev/ttyACM0:115200") is True
            assert interface.connection == mock_conn2


@pytest.mark.mock_hardware
@pytest.mark.mavlink
@pytest.mark.integration
class TestMockMAVLinkWithDetector:
    """Test MAVLink with hardware detector."""

    def test_mavlink_detection(self) -> None:
        """Test MAVLink auto-detection."""
        from backend.services.hardware_detector import HardwareDetector

        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=Mock(get_type=Mock(return_value="HEARTBEAT")))
        mock_conn.close = Mock()

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_conn

            detector = HardwareDetector()
            detector.check_hardware()

            status = detector.get_status()
            assert "mavlink" in status

    def test_mavlink_detection_failure(self) -> None:
        """Test MAVLink detection when unavailable."""
        from backend.services.hardware_detector import HardwareDetector

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.side_effect = Exception("No device")

            detector = HardwareDetector()
            detector.check_hardware()

            assert detector.mavlink_available is False
            status = detector.get_status()
            assert status["mavlink"]["available"] is False

    def test_concurrent_mavlink_checks(self) -> None:
        """Test concurrent MAVLink availability checks."""
        from backend.services.hardware_detector import HardwareDetector

        mock_conn = Mock()
        mock_conn.recv_match = Mock(return_value=Mock(get_type=Mock(return_value="HEARTBEAT")))
        mock_conn.close = Mock()

        with patch("backend.hal.mavlink_interface.mavutil.mavlink_connection") as mock_mavutil:
            mock_mavutil.return_value = mock_conn

            detector = HardwareDetector()

            # Run concurrent checks
            threads = []
            results = []

            def check() -> None:
                detector._check_mavlink()
                results.append(detector.mavlink_available)

            for _ in range(5):
                t = threading.Thread(target=check)
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            # All should have same result
            assert all(r == results[0] for r in results)
