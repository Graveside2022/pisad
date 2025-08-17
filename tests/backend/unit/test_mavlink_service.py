"""Unit tests for MAVLinkService.

Tests the MAVLink communication layer including telemetry streaming,
message sending, and connection management functionality.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.backend.services.mavlink_service import ConnectionState, LogLevel, MAVLinkService


class TestMAVLinkService:
    """Test MAVLink communication service."""

    @pytest.fixture
    def mavlink_config(self):
        """Provide test MAVLink configuration."""
        return {
            "device_path": "/dev/ttyACM0",
            "baud_rate": 115200,
            "source_system": 1,
            "source_component": 191,
            "target_system": 1,
            "target_component": 1,
        }

    @pytest.fixture
    def mavlink_service(self, mavlink_config):
        """Provide MAVLinkService instance."""
        return MAVLinkService(**mavlink_config)

    def test_mavlink_service_initialization(self, mavlink_service):
        """Test MAVLinkService initializes with correct parameters."""
        assert mavlink_service.device_path == "/dev/ttyACM0"
        assert mavlink_service.baud_rate == 115200
        assert mavlink_service.source_system == 1
        assert mavlink_service.source_component == 191
        assert mavlink_service.connection_state == ConnectionState.DISCONNECTED

    @pytest.mark.asyncio
    async def test_telemetry_sender_streams_data(self, mavlink_service):
        """Test telemetry sender streams RSSI and status data."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_connection:
            mock_conn = MagicMock()
            mock_connection.return_value = mock_conn

            # Mock RSSI value update
            mavlink_service.update_rssi_value(-45.5)

            # This should fail initially as telemetry streaming needs implementation
            await mavlink_service.telemetry_sender()

            # Verify NAMED_VALUE_FLOAT message sent
            assert mock_conn.mav.named_value_float_send.called

    def test_send_named_value_float_formats_correctly(self, mavlink_service):
        """Test NAMED_VALUE_FLOAT message formatting."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_connection:
            mock_conn = MagicMock()
            mock_connection.return_value = mock_conn
            mavlink_service.connection = mock_conn

            result = mavlink_service.send_named_value_float(
                name="RSSI_DBM", value=-42.5, time_ms=1000
            )

            assert result is True
            mock_conn.mav.named_value_float_send.assert_called_once()

    def test_send_state_change_broadcasts_status(self, mavlink_service):
        """Test state change broadcasts via STATUSTEXT."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_connection:
            mock_conn = MagicMock()
            mock_connection.return_value = mock_conn
            mavlink_service.connection = mock_conn

            result = mavlink_service.send_state_change("HOMING")

            assert result is True
            mock_conn.mav.statustext_send.assert_called_once()

    def test_send_detection_event_includes_confidence(self, mavlink_service):
        """Test detection event includes RSSI and confidence data."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_connection:
            mock_conn = MagicMock()
            mock_connection.return_value = mock_conn
            mavlink_service.connection = mock_conn

            result = mavlink_service.send_detection_event(rssi=-38.2, confidence=0.85)

            assert result is True
            mock_conn.mav.statustext_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_health_status_periodic(self, mavlink_service):
        """Test periodic health status reporting."""
        with patch("pymavlink.mavutil.mavlink_connection") as mock_connection:
            mock_conn = MagicMock()
            mock_connection.return_value = mock_conn
            mavlink_service.connection = mock_conn

            # This should fail initially as health monitoring needs implementation
            await mavlink_service._send_health_status()

            # Verify health data transmitted
            assert mock_conn.mav.named_value_float_send.called

    def test_update_rssi_value_stores_correctly(self, mavlink_service):
        """Test RSSI value update and storage."""
        test_rssi = -55.3
        mavlink_service.update_rssi_value(test_rssi)

        # This should fail as RSSI storage needs implementation
        assert hasattr(mavlink_service, "_current_rssi")
        assert mavlink_service._current_rssi == test_rssi

    def test_update_telemetry_config_applies_settings(self, mavlink_service):
        """Test telemetry configuration updates."""
        new_config = {"rate": 5, "precision": 2}  # 5 Hz

        mavlink_service.update_telemetry_config(new_config)

        stored_config = mavlink_service.get_telemetry_config()
        assert stored_config["rate"] == 5
        assert stored_config["precision"] == 2

    def test_get_telemetry_config_returns_current(self, mavlink_service):
        """Test telemetry configuration retrieval."""
        config = mavlink_service.get_telemetry_config()

        assert isinstance(config, dict)
        assert "rate" in config
        assert "precision" in config

    def test_log_level_enum_values(self):
        """Test LogLevel enum contains expected values."""
        assert LogLevel.ERROR.value == 40
        assert LogLevel.INFO.value == 20
        assert LogLevel.DEBUG.value == 10
        assert LogLevel.TRACE.value == 5

    def test_connection_state_enum_values(self):
        """Test ConnectionState enum contains expected states."""
        assert ConnectionState.DISCONNECTED.value == "disconnected"
        assert ConnectionState.CONNECTING.value == "connecting"
        assert ConnectionState.CONNECTED.value == "connected"
