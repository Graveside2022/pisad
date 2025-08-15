"""Integration tests for GCS telemetry communication."""

import asyncio
import contextlib
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.serial
# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.backend.services.mavlink_service import MAVLinkService  # noqa: E402
from src.backend.services.signal_processor import SignalProcessor  # noqa: E402
from src.backend.services.state_machine import StateMachine, SystemState  # noqa: E402


@pytest.fixture
def mock_mavlink_connection():
    """Create mock MAVLink connection."""
    connection = MagicMock()
    connection.mav = MagicMock()
    connection.recv_match = MagicMock(return_value=None)
    connection.wait_heartbeat = MagicMock(
        return_value=MagicMock(get_srcSystem=MagicMock(return_value=1))
    )
    return connection


@pytest.fixture
async def mavlink_service(mock_mavlink_connection):
    """Create MAVLink service with mock connection."""
    with patch("src.backend.services.mavlink_service.mavutil.mavlink_connection") as mock_conn:
        mock_conn.return_value = mock_mavlink_connection
        service = MAVLinkService(device_path="tcp:127.0.0.1:5760")
        await service.start()
        yield service
        await service.stop()


@pytest.fixture
def signal_processor():
    """Create signal processor instance."""
    return SignalProcessor()


@pytest.fixture
def state_machine():
    """Create state machine instance."""
    return StateMachine()


class TestGCSTelemetryIntegration:
    """Test GCS telemetry integration."""

    @pytest.mark.asyncio
    async def test_telemetry_flow_integration(
        self, mavlink_service, signal_processor, state_machine
    ):
        """Test complete telemetry flow from signal processor to MAVLink."""
        # Connect components
        state_machine.set_mavlink_service(mavlink_service)
        state_machine.set_signal_processor(signal_processor)
        signal_processor.set_mavlink_service(mavlink_service)

        # Verify initial state
        assert state_machine.get_current_state() == SystemState.IDLE

        # Simulate state transition
        await state_machine.transition_to(SystemState.SEARCHING)

        # Verify state change was sent via MAVLink
        mavlink_service.connection.mav.statustext_send.assert_called()
        call_args = mavlink_service.connection.mav.statustext_send.call_args
        assert b"PISAD: State changed to SEARCHING" in call_args[0]

        # Simulate RSSI update
        signal_processor._current_rssi = -75.0
        signal_processor._mavlink_service.update_rssi_value(-75.0)

        # Verify RSSI was updated
        assert mavlink_service._rssi_value == -75.0

        # Simulate detection event
        await state_machine.handle_detection(-65.0, 85.0)

        # Verify detection was sent
        assert mavlink_service.connection.mav.statustext_send.call_count >= 2

    @pytest.mark.asyncio
    async def test_rssi_streaming_rate(self, mavlink_service):
        """Test RSSI streaming at configured rate."""
        # Set RSSI rate to 5Hz for faster testing
        mavlink_service.update_telemetry_config({"rssi_rate_hz": 5.0})

        # Update RSSI value
        mavlink_service.update_rssi_value(-70.0)

        # Start telemetry sender task
        telemetry_task = asyncio.create_task(mavlink_service.telemetry_sender())

        # Let it run for 1 second
        await asyncio.sleep(1.1)

        # Cancel task
        telemetry_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await telemetry_task

        # Verify approximately 5 RSSI messages were sent (±1 for timing)
        named_value_calls = [
            call
            for call in mavlink_service.connection.mav.named_value_float_send.call_args_list
            if b"PISAD_RSSI" in call[0]
        ]
        assert 4 <= len(named_value_calls) <= 6, f"Expected ~5 calls, got {len(named_value_calls)}"

    @pytest.mark.asyncio
    async def test_health_status_reporting(self, mavlink_service):
        """Test health status reporting at configured interval."""
        # Set health interval to 1 second for faster testing
        mavlink_service.update_telemetry_config({"health_interval_seconds": 1})

        # Mock psutil
        with patch("src.backend.services.mavlink_service.psutil") as mock_psutil:
            mock_psutil.cpu_percent.return_value = 45.5
            mock_psutil.virtual_memory.return_value = MagicMock(percent=62.3)

            # Start telemetry sender
            telemetry_task = asyncio.create_task(mavlink_service.telemetry_sender())

            # Let it run for 2.1 seconds
            await asyncio.sleep(2.1)

            # Cancel task
            telemetry_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await telemetry_task

            # Verify health messages were sent
            health_calls = [
                call
                for call in mavlink_service.connection.mav.statustext_send.call_args_list
                if b"PISAD: Health" in call[0]
            ]
            assert (
                len(health_calls) >= 2
            ), f"Expected at least 2 health messages, got {len(health_calls)}"

            # Verify health content
            for call in health_calls:
                text = call[0][1].decode("utf-8")
                assert "cpu" in text.lower()
                assert "mem" in text.lower()

    @pytest.mark.asyncio
    async def test_detection_throttling(self, mavlink_service):
        """Test detection event throttling."""
        # Set throttle to 200ms for testing
        mavlink_service.update_telemetry_config({"detection_throttle_ms": 200})

        # Send multiple detection events rapidly
        results = []
        for i in range(5):
            result = mavlink_service.send_detection_event(-60.0 + i, 80.0 + i)
            results.append(result)
            await asyncio.sleep(0.05)  # 50ms between attempts

        # First should succeed, next 3 should be throttled, last might succeed
        assert results[0] is True
        assert results[1] is False
        assert results[2] is False
        # results[3] and results[4] depend on exact timing

    @pytest.mark.asyncio
    async def test_state_transition_telemetry(self, state_machine, mavlink_service):
        """Test state transition telemetry messages."""
        state_machine.set_mavlink_service(mavlink_service)

        # Test all valid transitions
        transitions = [
            (SystemState.IDLE, SystemState.SEARCHING),
            (SystemState.SEARCHING, SystemState.DETECTING),
            (SystemState.DETECTING, SystemState.HOMING),
            (SystemState.HOMING, SystemState.HOLDING),
            (SystemState.HOLDING, SystemState.SEARCHING),
        ]

        for from_state, to_state in transitions:
            # Set current state
            state_machine._current_state = from_state

            # Transition
            await state_machine.transition_to(to_state)

            # Verify telemetry was sent
            mavlink_service.connection.mav.statustext_send.assert_called()
            last_call = mavlink_service.connection.mav.statustext_send.call_args
            expected_text = f"PISAD: State changed to {to_state.value}"
            assert expected_text.encode() in last_call[0]

    @pytest.mark.asyncio
    async def test_bandwidth_management(self, mavlink_service):
        """Test bandwidth usage stays within limits."""
        # Configure high rates
        mavlink_service.update_telemetry_config(
            {
                "rssi_rate_hz": 10.0,  # Maximum rate
                "health_interval_seconds": 1,
                "detection_throttle_ms": 100,
            }
        )

        # Track message sizes
        message_sizes = []

        def track_message(severity, text):
            message_sizes.append(len(text))

        mavlink_service.connection.mav.statustext_send.side_effect = track_message

        def track_named_value(time_ms, name, value):
            message_sizes.append(17)  # NAMED_VALUE_FLOAT is 17 bytes

        mavlink_service.connection.mav.named_value_float_send.side_effect = track_named_value

        # Run for 1 second
        telemetry_task = asyncio.create_task(mavlink_service.telemetry_sender())
        await asyncio.sleep(1.0)
        telemetry_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await telemetry_task

        # Calculate bandwidth (rough estimate)
        total_bytes = sum(message_sizes)
        bandwidth_bps = total_bytes * 8  # bits per second

        # Should be under 10kbps for telemetry
        assert bandwidth_bps < 10000, f"Bandwidth too high: {bandwidth_bps} bps"

    @pytest.mark.asyncio
    async def test_gcs_message_format_compatibility(self, mavlink_service):
        """Test message formats for GCS compatibility."""
        # Test NAMED_VALUE_FLOAT format
        success = mavlink_service.send_named_value_float("PISAD_RSSI", -72.5)
        assert success

        call_args = mavlink_service.connection.mav.named_value_float_send.call_args[0]
        # Check name is properly encoded and truncated
        assert len(call_args[1]) <= 10  # Max 10 chars
        assert call_args[2] == -72.5  # Value preserved

        # Test STATUSTEXT format
        success = mavlink_service.send_statustext("PISAD: Test message", severity=6)
        assert success

        call_args = mavlink_service.connection.mav.statustext_send.call_args[0]
        assert call_args[0] == 6  # Severity preserved
        assert len(call_args[1]) <= 50  # Max 50 chars

    @pytest.mark.asyncio
    async def test_connection_recovery(self, mavlink_service):
        """Test telemetry continues after connection recovery."""
        # Import ConnectionState from the correct module
        from src.backend.services.mavlink_service import ConnectionState

        # Simulate connection loss
        mavlink_service._set_state(ConnectionState.DISCONNECTED)

        # Try to send telemetry (should fail gracefully)
        result = mavlink_service.send_state_change("SEARCHING")
        assert result is False

        # Simulate reconnection
        mavlink_service._set_state(ConnectionState.CONNECTED)

        # Telemetry should work again
        result = mavlink_service.send_state_change("DETECTING")
        assert result is True

    @pytest.mark.asyncio
    async def test_sitl_compatibility(self):
        """Test compatibility with SITL (Software In The Loop)."""
        # This test would connect to actual SITL if available
        # For unit testing, we mock the connection
        with patch("src.backend.services.mavlink_service.mavutil.mavlink_connection") as mock_conn:
            mock_connection = MagicMock()
            mock_connection.wait_heartbeat = MagicMock(
                return_value=MagicMock(get_srcSystem=MagicMock(return_value=1))
            )
            mock_conn.return_value = mock_connection

            service = MAVLinkService(device_path="tcp:127.0.0.1:5760")
            await service.start()

            # Verify SITL connection string was used
            mock_conn.assert_called_with(
                "tcp:127.0.0.1:5760", source_system=1, source_component=191
            )

            await service.stop()


class TestTelemetryConfiguration:
    """Test telemetry configuration management."""

    def test_telemetry_config_validation(self):
        """Test telemetry configuration validation."""
        service = MAVLinkService()

        # Test valid config
        service.update_telemetry_config(
            {"rssi_rate_hz": 5.0, "health_interval_seconds": 20, "detection_throttle_ms": 1000}
        )

        config = service.get_telemetry_config()
        assert config["rssi_rate_hz"] == 5.0
        assert config["health_interval_seconds"] == 20
        assert config["detection_throttle_ms"] == 1000

        # Test boundary validation
        service.update_telemetry_config(
            {
                "rssi_rate_hz": 15.0,  # Should be clamped to 10
                "health_interval_seconds": 100,  # Should be clamped to 60
                "detection_throttle_ms": 10,  # Should be clamped to 100
            }
        )

        config = service.get_telemetry_config()
        assert config["rssi_rate_hz"] == 10.0
        assert config["health_interval_seconds"] == 60
        assert config["detection_throttle_ms"] == 100

    def test_telemetry_rate_limits(self):
        """Test telemetry rate limiting."""
        service = MAVLinkService()

        # Configure minimum rates
        service.update_telemetry_config(
            {"rssi_rate_hz": 0.05, "health_interval_seconds": 0, "detection_throttle_ms": 50}
        )

        config = service.get_telemetry_config()
        # Should be clamped to minimum safe values
        assert config["rssi_rate_hz"] >= 0.1
        assert config["health_interval_seconds"] >= 1
        assert config["detection_throttle_ms"] >= 100
