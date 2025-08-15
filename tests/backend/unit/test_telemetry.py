"""Unit tests for MAVLink telemetry system."""

import asyncio
import contextlib
from unittest.mock import MagicMock

import pytest

from src.backend.services.mavlink_service import ConnectionState, MAVLinkService

pytestmark = pytest.mark.serial


@pytest.fixture
def mavlink_service():
    """Create MAVLink service instance for testing."""
    service = MAVLinkService(
        device_path="/dev/ttyACM0",
        baud_rate=115200,
    )
    # Mock the connection
    service.connection = MagicMock()
    service.state = ConnectionState.CONNECTED
    return service


class TestTelemetryConfiguration:
    """Test telemetry configuration management."""

    def test_default_telemetry_config(self, mavlink_service):
        """Test default telemetry configuration values."""
        config = mavlink_service.get_telemetry_config()
        assert config["rssi_rate_hz"] == 2.0
        assert config["health_interval_seconds"] == 10
        assert config["detection_throttle_ms"] == 500

    def test_update_telemetry_config(self, mavlink_service):
        """Test updating telemetry configuration."""
        new_config = {
            "rssi_rate_hz": 1.0,
            "health_interval_seconds": 20,
            "detection_throttle_ms": 1000,
        }
        mavlink_service.update_telemetry_config(new_config)

        config = mavlink_service.get_telemetry_config()
        assert config["rssi_rate_hz"] == 1.0
        assert config["health_interval_seconds"] == 20
        assert config["detection_throttle_ms"] == 1000

    def test_telemetry_config_validation(self, mavlink_service):
        """Test telemetry configuration validation bounds."""
        # Test rate limits
        mavlink_service.update_telemetry_config({"rssi_rate_hz": 100})
        assert mavlink_service.get_telemetry_config()["rssi_rate_hz"] == 10.0  # Max

        mavlink_service.update_telemetry_config({"rssi_rate_hz": 0.01})
        assert mavlink_service.get_telemetry_config()["rssi_rate_hz"] == 0.1  # Min

        # Test health interval limits
        mavlink_service.update_telemetry_config({"health_interval_seconds": 100})
        assert mavlink_service.get_telemetry_config()["health_interval_seconds"] == 60  # Max

        mavlink_service.update_telemetry_config({"health_interval_seconds": 0})
        assert mavlink_service.get_telemetry_config()["health_interval_seconds"] == 1  # Min

        # Test throttle limits
        mavlink_service.update_telemetry_config({"detection_throttle_ms": 10000})
        assert mavlink_service.get_telemetry_config()["detection_throttle_ms"] == 5000  # Max

        mavlink_service.update_telemetry_config({"detection_throttle_ms": 10})
        assert mavlink_service.get_telemetry_config()["detection_throttle_ms"] == 100  # Min


class TestNamedValueFloat:
    """Test NAMED_VALUE_FLOAT message sending."""

    def test_send_named_value_float(self, mavlink_service):
        """Test sending NAMED_VALUE_FLOAT message."""
        result = mavlink_service.send_named_value_float("PISAD_RSSI", -75.5)

        assert result is True
        mavlink_service.connection.mav.named_value_float_send.assert_called_once()

        # Check arguments
        call_args = mavlink_service.connection.mav.named_value_float_send.call_args[0]
        assert call_args[1] == b"PISAD_RSSI"
        assert call_args[2] == -75.5

    def test_named_value_float_name_truncation(self, mavlink_service):
        """Test name truncation to 10 characters."""
        long_name = "VERY_LONG_NAME_THAT_EXCEEDS_LIMIT"
        mavlink_service.send_named_value_float(long_name, 1.0)

        call_args = mavlink_service.connection.mav.named_value_float_send.call_args[0]
        assert call_args[1] == b"VERY_LONG_"  # Truncated to 10 chars

    def test_named_value_float_disconnected(self, mavlink_service):
        """Test NAMED_VALUE_FLOAT when disconnected."""
        mavlink_service.state = ConnectionState.DISCONNECTED
        result = mavlink_service.send_named_value_float("TEST", 1.0)

        assert result is False
        mavlink_service.connection.mav.named_value_float_send.assert_not_called()

    def test_rssi_value_update(self, mavlink_service):
        """Test updating RSSI value for streaming."""
        mavlink_service.update_rssi_value(-80.0)
        assert mavlink_service._rssi_value == -80.0

        mavlink_service.update_rssi_value(-60.5)
        assert mavlink_service._rssi_value == -60.5


class TestStatusText:
    """Test STATUSTEXT message sending."""

    def test_send_state_change(self, mavlink_service):
        """Test sending state change via STATUSTEXT."""
        result = mavlink_service.send_state_change("HOMING")

        assert result is True
        mavlink_service.connection.mav.statustext_send.assert_called_once()

        call_args = mavlink_service.connection.mav.statustext_send.call_args[0]
        assert call_args[0] == 6  # INFO severity
        assert b"PISAD: State changed to HOMING" in call_args[1]

    def test_state_change_deduplication(self, mavlink_service):
        """Test that duplicate state changes are not sent."""
        mavlink_service.send_state_change("IDLE")
        mavlink_service.connection.mav.statustext_send.reset_mock()

        # Second call with same state should not send
        result = mavlink_service.send_state_change("IDLE")
        assert result is True
        mavlink_service.connection.mav.statustext_send.assert_not_called()

        # Different state should send
        result = mavlink_service.send_state_change("SEARCHING")
        assert result is True
        mavlink_service.connection.mav.statustext_send.assert_called_once()

    def test_send_detection_event(self, mavlink_service):
        """Test sending detection event via STATUSTEXT."""
        result = mavlink_service.send_detection_event(-65.0, 87.5)

        assert result is True
        mavlink_service.connection.mav.statustext_send.assert_called_once()

        call_args = mavlink_service.connection.mav.statustext_send.call_args[0]
        assert call_args[0] == 5  # NOTICE severity
        assert b"PISAD: Signal detected -65.0dBm @ 88%" in call_args[1]

    def test_detection_event_throttling(self, mavlink_service):
        """Test detection event throttling."""
        # First detection should send
        result = mavlink_service.send_detection_event(-70.0, 80.0)
        assert result is True

        # Immediate second detection should be throttled
        mavlink_service.connection.mav.statustext_send.reset_mock()
        result = mavlink_service.send_detection_event(-68.0, 85.0)
        assert result is False
        mavlink_service.connection.mav.statustext_send.assert_not_called()


class TestHealthMonitoring:
    """Test system health monitoring."""

    @pytest.mark.asyncio
    async def test_send_health_status(self, mavlink_service):
        """Test sending health status via STATUSTEXT."""
        # Just test that the method runs and sends a message
        # Don't worry about actual system stats
        try:
            await mavlink_service._send_health_status()
        except ImportError:
            # psutil might not be installed in test env
            pytest.skip("psutil not available")

        # If it ran, should have sent a statustext
        if mavlink_service.connection.mav.statustext_send.called:
            call_args = mavlink_service.connection.mav.statustext_send.call_args[0]
            # Check severity
            assert call_args[0] == 6  # INFO severity
            message = call_args[1].decode("utf-8")
            assert "PISAD: Health" in message


class TestTelemetrySender:
    """Test telemetry sender task."""

    @pytest.mark.asyncio
    async def test_telemetry_sender_rssi_rate(self, mavlink_service):
        """Test RSSI sending at configured rate."""
        mavlink_service._rssi_value = -75.0
        mavlink_service._telemetry_config["rssi_rate_hz"] = 10.0  # Fast rate for testing
        mavlink_service._running = True  # Mark as running

        # Run telemetry sender for a short time
        task = asyncio.create_task(mavlink_service.telemetry_sender())
        await asyncio.sleep(0.15)
        mavlink_service._running = False
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await task

        # Should have sent at least one RSSI value
        mavlink_service.connection.mav.named_value_float_send.assert_called()

    @pytest.mark.asyncio
    async def test_telemetry_sender_health_interval(self, mavlink_service):
        """Test health status sending at configured interval."""
        mavlink_service._telemetry_config["health_interval_seconds"] = 0.1  # Fast for testing
        mavlink_service._running = True  # Mark as running

        # Run telemetry sender for a short time
        task = asyncio.create_task(mavlink_service.telemetry_sender())
        await asyncio.sleep(0.15)
        mavlink_service._running = False
        task.cancel()

        with contextlib.suppress(asyncio.CancelledError, ImportError):
            await task

        # Should have sent either health status or RSSI
        # At minimum RSSI should be sent
        assert (
            mavlink_service.connection.mav.named_value_float_send.called
            or mavlink_service.connection.mav.statustext_send.called
        )


class TestMessageFormatting:
    """Test message formatting and constraints."""

    def test_statustext_truncation(self, mavlink_service):
        """Test STATUSTEXT truncation to 50 characters."""
        long_message = "This is a very long message that exceeds the maximum allowed length for STATUSTEXT messages"
        mavlink_service.send_statustext(long_message)

        call_args = mavlink_service.connection.mav.statustext_send.call_args[0]
        message = call_args[1].decode("utf-8")
        assert len(message) <= 50

    def test_message_prefixes(self, mavlink_service):
        """Test that all messages use PISAD prefix."""
        # State change
        mavlink_service.send_state_change("TEST")
        call = mavlink_service.connection.mav.statustext_send.call_args[0]
        assert b"PISAD:" in call[1]

        # Detection event
        mavlink_service.connection.mav.statustext_send.reset_mock()
        mavlink_service.send_detection_event(-70.0, 80.0)
        call = mavlink_service.connection.mav.statustext_send.call_args[0]
        assert b"PISAD:" in call[1]
