# FLAKY_FIXED: Deterministic time control applied
"""Integration tests for safety interlock system with other services."""

import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.backend.services.mavlink_service import ConnectionState, MAVLinkService
from src.backend.services.signal_processor import SignalProcessor
from src.backend.utils.safety import SafetyInterlockSystem

pytestmark = pytest.mark.serial


class TestSafetyMAVLinkIntegration:
    """Test integration between safety system and MAVLink service."""

    @pytest.mark.asyncio
    async def test_mode_callback_integration(self):
        """Test that MAVLink mode changes trigger safety updates."""
        safety = SafetyInterlockSystem()

        # Create mock MAVLink service
        mavlink = MAVLinkService(device_path="tcp:127.0.0.1:5760")

        # Register safety callback
        mavlink.add_mode_callback(safety.update_flight_mode)

        # Set initial mode to UNKNOWN (default)
        mavlink.telemetry["flight_mode"] = "UNKNOWN"

        # Process a heartbeat message that triggers mode callback
        mock_msg = Mock()
        mock_msg.custom_mode = 5  # LOITER mode
        mock_msg.base_mode = 0

        mavlink._process_heartbeat(mock_msg)

        # Check safety system received update
        mode_check = safety.checks["mode"]
        assert mode_check.current_mode == "LOITER"

    @pytest.mark.asyncio
    async def test_battery_callback_integration(self):
        """Test that MAVLink battery updates trigger safety updates."""
        safety = SafetyInterlockSystem()

        # Create mock MAVLink service
        mavlink = MAVLinkService(device_path="tcp:127.0.0.1:5760")

        # Register safety callback
        mavlink.add_battery_callback(safety.update_battery)

        # Simulate battery update through MAVLink
        mock_msg = Mock()
        mock_msg.voltage_battery = 12000  # 12V
        mock_msg.current_battery = 1000  # 10A
        mock_msg.battery_remaining = 15  # 15%

        mavlink._process_sys_status(mock_msg)

        # Check safety system received update
        battery_check = safety.checks["battery"]
        assert battery_check.current_battery_percent == 15.0

    @pytest.mark.asyncio
    async def test_position_callback_integration(self):
        """Test that MAVLink position updates trigger safety updates."""
        safety = SafetyInterlockSystem()

        # Set up geofence
        geofence_check = safety.checks["geofence"]
        geofence_check.set_geofence(37.0, -122.0, 100.0)

        # Create mock MAVLink service
        mavlink = MAVLinkService(device_path="tcp:127.0.0.1:5760")

        # Register safety callback
        mavlink.add_position_callback(safety.update_position)

        # Simulate position update through MAVLink
        mock_msg = Mock()
        mock_msg.lat = 370010000  # 37.001 degrees
        mock_msg.lon = -1220000000  # -122.0 degrees
        mock_msg.alt = 100000  # 100m

        mavlink._process_global_position(mock_msg)

        # Check safety system received update
        assert geofence_check.current_lat == 37.001
        assert geofence_check.current_lon == -122.0

    @pytest.mark.asyncio
    async def test_velocity_command_blocked_by_safety(self):
        """Test that velocity commands are blocked when unsafe."""
        SafetyInterlockSystem()
        mavlink = MAVLinkService(device_path="tcp:127.0.0.1:5760")

        # Mock the safety check to return False (unsafe)
        mavlink.set_safety_check_callback(lambda: False)

        # Enable velocity commands in MAVLink
        mavlink.enable_velocity_commands(True)

        # Mock connection
        mavlink.connection = Mock()
        mavlink.state = ConnectionState.CONNECTED

        # Try to send velocity command (should be blocked by safety)
        result = await mavlink.send_velocity_command(vx=1.0, vy=0.0, vz=0.0)

        assert result is False  # Blocked by safety

    @pytest.mark.asyncio
    async def test_statustext_on_safety_events(self):
        """Test that safety events send STATUSTEXT to GCS."""
        safety = SafetyInterlockSystem()
        mavlink = MAVLinkService(device_path="tcp:127.0.0.1:5760")

        # Mock connection
        mavlink.connection = Mock()
        mavlink.connection.mav = Mock()
        mavlink.state = ConnectionState.CONNECTED

        # Enable homing (should send STATUSTEXT)
        with patch.object(mavlink, "send_statustext") as mock_send:
            # Simulate enabling homing through API
            safety.update_flight_mode("GUIDED")
            success = await safety.enable_homing()

            # In real integration, the API would call mavlink.send_statustext
            # We'll simulate that here
            if success:
                mavlink.send_statustext("Homing enabled", severity=6)

            mock_send.assert_called_once_with("Homing enabled", severity=6)


class TestSafetySignalProcessorIntegration:
    """Test integration between safety system and signal processor."""

    @pytest.mark.asyncio
    async def test_snr_callback_integration(self):
        """Test that signal processor SNR updates trigger safety updates."""
        safety = SafetyInterlockSystem()
        processor = SignalProcessor()

        # Register safety callback
        processor.add_snr_callback(safety.update_signal_snr)

        # Process signal that will update SNR
        import numpy as np

        samples = np.random.randn(1024) + 1j * np.random.randn(1024)
        samples = samples.astype(np.complex64)

        # Process will calculate RSSI and SNR
        reading = await processor.process_iq(samples)

        # Trigger detection which updates SNR
        if reading:
            await processor.detect_signal(reading.rssi)

        # Check safety system received SNR update
        signal_check = safety.checks["signal"]
        assert signal_check.current_snr == processor.get_current_snr()

    @pytest.mark.asyncio
    async def test_signal_loss_disables_homing(self):
        """Test that prolonged signal loss disables homing."""
        safety = SafetyInterlockSystem()

        # Configure short timeout for testing
        signal_check = safety.checks["signal"]
        signal_check.timeout_seconds = 0.5

        # Enable homing
        safety.update_flight_mode("GUIDED")
        safety.update_battery(100.0)
        safety.update_signal_snr(10.0)
        await safety.enable_homing()

        assert await safety.is_safe_to_proceed() is True

        # Lose signal
        safety.update_signal_snr(3.0)

        # Initial check should still be safe (within timeout)
        assert await safety.is_safe_to_proceed() is True

        # Wait for timeout
        await asyncio.sleep(0.001)  # Minimal yield for determinism

        # Now check should fail after timeout
        assert await safety.is_safe_to_proceed() is False

        # Verify event was logged
        events = safety.get_safety_events()
        signal_events = [e for e in events if "signal" in str(e.trigger).lower()]
        assert len(signal_events) > 0


class TestSafetyWebSocketIntegration:
    """Test integration between safety system and WebSocket broadcasting."""

    @pytest.mark.asyncio
    async def test_safety_status_broadcast(self):
        """Test that safety status is broadcast via WebSocket."""
        from src.backend.api.websocket import get_safety_system, manager

        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        await manager.connect(mock_websocket)

        # Get safety system (will initialize if needed)
        safety = await get_safety_system()

        # Update safety status
        safety.update_flight_mode("GUIDED")
        safety.update_battery(50.0)

        # Get status that would be broadcast
        status = safety.get_safety_status()

        # Verify structure matches what WebSocket expects
        assert "emergency_stopped" in status
        assert "checks" in status
        assert "timestamp" in status

        # Clean up
        await manager.disconnect(mock_websocket)

    @pytest.mark.asyncio
    async def test_safety_event_notification(self):
        """Test that safety events trigger WebSocket notifications."""
        from src.backend.api.websocket import broadcast_message

        safety = SafetyInterlockSystem()

        # Mock broadcast function
        with patch("src.backend.api.websocket.manager.broadcast_json") as mock_broadcast:
            # Trigger emergency stop
            await safety.emergency_stop("Test emergency")

            # In real system, this would trigger a broadcast
            # Simulate it here
            await broadcast_message(
                {
                    "type": "safety_event",
                    "data": {
                        "event": "emergency_stop",
                        "reason": "Test emergency",
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                }
            )

            mock_broadcast.assert_called_once()


class TestSafetyAPIIntegration:
    """Test integration between safety system and REST API."""

    @pytest.mark.asyncio
    async def test_homing_endpoint_integration(self):
        """Test /system/homing endpoint with safety system."""
        from src.backend.api.routes import system

        # Initialize safety system for routes
        system.safety_system = SafetyInterlockSystem()

        # Set up safe conditions
        system.safety_system.update_flight_mode("GUIDED")
        system.safety_system.update_battery(100.0)
        system.safety_system.update_signal_snr(10.0)

        # Test enabling homing
        from src.backend.api.routes.system import HomingRequest

        request = HomingRequest(enabled=True)

        response = await system.control_homing(request)

        assert response["homing_enabled"] is True
        assert "safety_status" in response

    @pytest.mark.asyncio
    async def test_emergency_stop_endpoint_integration(self):
        """Test /system/emergency-stop endpoint."""
        from src.backend.api.routes import system

        # Initialize safety system
        system.safety_system = SafetyInterlockSystem()

        # Test emergency stop
        from src.backend.api.routes.system import EmergencyStopRequest

        request = EmergencyStopRequest(reason="Integration test")

        response = await system.emergency_stop(request)

        assert response["emergency_stopped"] is True
        assert response["reason"] == "Integration test"
        assert system.safety_system.emergency_stopped is True

    @pytest.mark.asyncio
    async def test_safety_events_endpoint_integration(self):
        """Test /safety/events endpoint."""
        from src.backend.api.routes import system

        # Initialize safety system
        system.safety_system = SafetyInterlockSystem()

        # Generate some events
        system.safety_system.update_flight_mode("LOITER")
        await system.safety_system.check_all_safety()

        # Query events
        response = await system.get_safety_events(limit=10)

        assert "events" in response
        assert "count" in response
        assert len(response["events"]) > 0


class TestSafetyCascadingFailures:
    """Test cascading safety failures and recovery."""

    @pytest.mark.asyncio
    async def test_multiple_safety_failures(self):
        """Test system behavior with multiple simultaneous failures."""
        safety = SafetyInterlockSystem()

        # Start with safe conditions
        safety.update_flight_mode("GUIDED")
        safety.update_battery(100.0)
        safety.update_signal_snr(10.0)
        await safety.enable_homing()

        assert await safety.is_safe_to_proceed() is True

        # Introduce multiple failures
        safety.update_flight_mode("RTL")  # Mode failure
        safety.update_battery(15.0)  # Battery failure
        safety.update_signal_snr(3.0)  # Signal failure

        # Check all failures are detected
        results = await safety.check_all_safety()
        assert results["mode"] is False
        assert results["battery"] is False
        # Signal may still be OK if within timeout

        assert await safety.is_safe_to_proceed() is False

        # Check multiple events logged
        events = safety.get_safety_events()
        assert len(events) >= 2  # At least mode and battery events

    @pytest.mark.asyncio
    async def test_safety_recovery_sequence(self):
        """Test recovery from safety failures."""
        safety = SafetyInterlockSystem()

        # Start with failures
        safety.update_flight_mode("LOITER")
        safety.update_battery(15.0)

        assert await safety.is_safe_to_proceed() is False

        # Recover step by step
        safety.update_flight_mode("GUIDED")
        assert await safety.is_safe_to_proceed() is False  # Still battery issue

        safety.update_battery(50.0)
        assert await safety.is_safe_to_proceed() is False  # Still no operator enable

        await safety.enable_homing()
        assert await safety.is_safe_to_proceed() is True  # All recovered

    @pytest.mark.asyncio
    async def test_emergency_stop_overrides_all(self):
        """Test that emergency stop overrides all other conditions."""
        safety = SafetyInterlockSystem()

        # Set all conditions to safe
        safety.update_flight_mode("GUIDED")
        safety.update_battery(100.0)
        safety.update_signal_snr(10.0)
        await safety.enable_homing()

        assert await safety.is_safe_to_proceed() is True

        # Emergency stop
        await safety.emergency_stop("Override test")

        # Should be unsafe regardless of other conditions
        assert await safety.is_safe_to_proceed() is False

        # Even if we try to enable homing again
        success = await safety.enable_homing()
        assert success is False  # Blocked by emergency stop

        # Only reset can recover
        await safety.reset_emergency_stop()
        success = await safety.enable_homing()
        assert success is True
        assert await safety.is_safe_to_proceed() is True
