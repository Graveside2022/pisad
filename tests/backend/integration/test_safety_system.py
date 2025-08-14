"""Integration tests for complete safety system."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from backend.utils.safety import SafetyInterlockSystem


@pytest.mark.asyncio
class TestSafetySystemIntegration:
    """Integration tests for safety system."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety system fixture."""
        system = SafetyInterlockSystem()
        await system.start_monitoring()
        yield system
        await system.stop_monitoring()

    async def test_full_safety_interlock_chain(self, safety_system):
        """Test full safety interlock chain end-to-end."""
        # Set all conditions to pass
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(50.0)
        safety_system.update_signal_snr(10.0)

        # Enable homing
        result = await safety_system.enable_homing()
        assert result is True

        # Verify all checks pass
        checks = await safety_system.check_all_safety()
        assert all(checks.values())

    async def test_concurrent_safety_triggers(self, safety_system):
        """Test concurrent safety triggers."""
        # Enable system
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(50.0)
        safety_system.update_signal_snr(10.0)
        await safety_system.enable_homing()

        # Trigger multiple failures concurrently
        tasks = [
            asyncio.create_task(safety_system.check_all_safety()),
            asyncio.create_task(safety_system.emergency_stop("Test")),
            asyncio.create_task(safety_system.check_all_safety()),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        # System should handle concurrent operations
        assert safety_system.emergency_stopped

    async def test_state_machine_transitions(self, safety_system):
        """Test state machine transitions under safety events."""
        # Test transition blocking when unsafe
        safety_system.update_flight_mode("STABILIZE")
        assert not await safety_system.is_safe_to_proceed()

        # Test transition allowed when safe
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(50.0)
        safety_system.update_signal_snr(10.0)
        await safety_system.enable_homing()  # Enable operator check
        assert await safety_system.is_safe_to_proceed()

    async def test_websocket_safety_notifications(self, safety_system):
        """Test WebSocket notifications for safety events."""
        # Test event logging
        await safety_system.emergency_stop("Test notification")

        events = safety_system.get_safety_events()
        assert len(events) > 0
        assert any(e.event_type.value == "emergency_stop" for e in events)

    async def test_logging_during_safety_activation(self, safety_system):
        """Test logging and telemetry during safety activation."""
        # Enable and then trigger safety
        safety_system.update_flight_mode("GUIDED")
        safety_system.update_battery(50.0)
        safety_system.update_signal_snr(10.0)
        await safety_system.enable_homing()

        # Trigger safety event
        safety_system.update_battery(15.0)
        await safety_system.check_all_safety()

        # Verify event logged
        events = safety_system.get_safety_events()
        assert any(e.trigger.value == "low_battery" for e in events)
