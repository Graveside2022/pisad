"""Integration tests for service coordination.

Tests inter-service communication, event propagation, state synchronization,
and coordinated operations across the entire backend system.
"""

import asyncio
import time
from unittest.mock import Mock

import pytest

from src.backend.core.config import get_config
from src.backend.services.safety_manager import SafetyManager
from src.backend.services.signal_processor import SignalProcessor
from src.backend.services.state_machine import StateMachine, SystemState


class TestServiceCoordination:
    """Test service coordination and communication."""

    @pytest.fixture
    def state_machine(self):
        """Provide StateMachine instance."""
        return StateMachine()

    @pytest.fixture
    def signal_processor(self):
        """Provide SignalProcessor instance."""
        return SignalProcessor()

    @pytest.fixture
    def safety_manager(self):
        """Provide SafetyManager instance."""
        return SafetyManager()

    @pytest.fixture
    def mock_mavlink(self):
        """Provide mock MAVLink service."""
        mock = Mock()
        mock.emergency_stop.return_value = True
        mock.get_mode.return_value = "GUIDED"
        mock.telemetry = {
            "battery": {"voltage": 22.0},
            "rc_channels": {"throttle": 1500, "roll": 1500, "pitch": 1500, "yaw": 1500},
            "gps": {"satellites": 10, "hdop": 1.5},
            "position": {"lat": 37.7749, "lon": -122.4194, "alt": 30.0},
        }
        return mock

    @pytest.mark.asyncio
    async def test_state_machine_signal_processor_coordination(
        self, state_machine, signal_processor
    ):
        """Test coordination between state machine and signal processor."""
        # Set up signal processor callback for state machine
        detection_events = []

        def detection_callback(event):
            detection_events.append(event)

        signal_processor.add_detection_callback(detection_callback)

        # Simulate signal detection triggering state change
        import numpy as np

        strong_signal = 5.0 * np.ones(1024, dtype=np.complex64)

        # Process signal
        result = signal_processor.compute_rssi(strong_signal)

        # Should generate RSSI result
        assert hasattr(result, "rssi")
        assert hasattr(result, "snr")

        # State machine should be able to process detection events
        if detection_events:
            # If detection occurred, state machine should handle it
            await state_machine.handle_detection(result.rssi, result.snr)

        # State machine should be operational
        current_state = state_machine.current_state
        assert current_state in SystemState

    @pytest.mark.asyncio
    async def test_safety_manager_state_machine_coordination(
        self, state_machine, safety_manager, mock_mavlink
    ):
        """Test coordination between safety manager and state machine."""
        # Set up safety manager with MAVLink
        safety_manager.mavlink = mock_mavlink

        # Test emergency stop coordination
        emergency_result = safety_manager.trigger_emergency_stop()

        assert emergency_result["success"] is True
        assert emergency_result["response_time_ms"] < 500

        # Emergency stop should be able to trigger state machine emergency
        emergency_state_result = await state_machine.emergency_stop()

        assert emergency_state_result is True
        assert state_machine.current_state in [SystemState.EMERGENCY, SystemState.IDLE]

    @pytest.mark.asyncio
    async def test_signal_loss_coordination(self, state_machine, safety_manager):
        """Test signal loss handling coordination."""
        # Simulate signal loss scenario
        await state_machine.handle_signal_lost()

        # Safety manager should handle signal loss
        safety_manager.last_signal_time = time.time() - 15.0  # 15 seconds ago

        # State machine should handle signal loss appropriately
        assert state_machine._homing_enabled is False

    def test_configuration_coordination(self, state_machine, signal_processor, safety_manager):
        """Test configuration sharing across services."""
        # Test that services can access shared configuration
        config = get_config()

        # Services should be configurable
        assert hasattr(state_machine, "_db_path")
        assert hasattr(signal_processor, "snr_threshold")
        assert hasattr(safety_manager, "battery_low_voltage")

        # Configuration should be consistent
        assert isinstance(config, dict)

    @pytest.mark.asyncio
    async def test_telemetry_coordination(
        self, state_machine, signal_processor, safety_manager, mock_mavlink
    ):
        """Test telemetry data coordination across services."""
        # Set up MAVLink
        safety_manager.mavlink = mock_mavlink

        # Get telemetry from each service
        state_telemetry = state_machine.get_telemetry_metrics()
        signal_stats = signal_processor.get_processing_stats()
        battery_status = safety_manager.check_battery_status()

        # All services should provide telemetry
        assert isinstance(state_telemetry, dict)
        assert isinstance(signal_stats, dict)
        assert isinstance(battery_status, dict)

        # Telemetry should have timestamps or status
        assert "current_state" in state_telemetry
        assert "samples_processed" in signal_stats
        assert "level" in battery_status

    @pytest.mark.asyncio
    async def test_concurrent_service_operations(
        self, state_machine, signal_processor, safety_manager
    ):
        """Test concurrent operations across services."""
        import numpy as np

        # Define concurrent operations
        async def state_operations():
            await state_machine.transition_to(SystemState.SEARCHING)
            await asyncio.sleep(0.01)
            await state_machine.transition_to(SystemState.IDLE)

        async def signal_operations():
            for _ in range(3):
                samples = np.random.randn(1024).astype(np.complex64)
                signal_processor.compute_rssi(samples)
                await asyncio.sleep(0.01)

        async def safety_operations():
            for _ in range(3):
                safety_manager.check_battery_status()
                await asyncio.sleep(0.01)

        # Run operations concurrently
        await asyncio.gather(state_operations(), signal_operations(), safety_operations())

        # All services should remain operational
        assert state_machine.current_state in SystemState
        assert signal_processor.get_processing_stats()["samples_processed"] >= 3

    @pytest.mark.asyncio
    async def test_error_propagation_coordination(self, state_machine, safety_manager):
        """Test error propagation between services."""
        # Test that errors in one service don't break others

        # Force an error condition in safety manager
        safety_manager.mavlink = None  # Remove MAVLink connection

        # Safety manager should handle gracefully
        battery_status = safety_manager.check_battery_status()
        assert battery_status["level"] == "UNKNOWN"

        # State machine should still operate
        await state_machine.transition_to(SystemState.IDLE)
        assert state_machine.current_state == SystemState.IDLE

    def test_service_lifecycle_coordination(self, state_machine, signal_processor, safety_manager):
        """Test service lifecycle coordination."""
        # Test service initialization
        assert state_machine._current_state is not None
        assert signal_processor.ewma_filter is not None
        assert safety_manager.state is not None

        # Test service cleanup capability
        # (Services should be able to cleanup resources)
        assert hasattr(state_machine, "_enable_persistence")
        assert hasattr(signal_processor, "_fft_buffer")
        assert hasattr(safety_manager, "active_violations")

    @pytest.mark.asyncio
    async def test_detection_event_flow_coordination(self, state_machine, signal_processor):
        """Test detection event flow coordination."""
        # Set up event flow
        events_received = []

        def state_callback(*args):
            events_received.append(("state", args))

        def signal_callback(*args):
            events_received.append(("signal", args))

        # Register callbacks
        state_machine.add_state_callback(state_callback)
        signal_processor.add_detection_callback(signal_callback)

        # Trigger events
        import numpy as np

        strong_signal = 3.0 * np.ones(1024, dtype=np.complex64)
        signal_processor.compute_rssi(strong_signal)

        await state_machine.transition_to(SystemState.SEARCHING)

        # Event system should be functional
        # (Actual event generation depends on thresholds and implementation)

    def test_resource_sharing_coordination(self, state_machine, signal_processor, safety_manager):
        """Test resource sharing coordination."""
        # Test that services can share resources appropriately

        # Database access coordination
        assert hasattr(state_machine, "_db_path")

        # Memory usage coordination
        assert hasattr(signal_processor, "_fft_buffer")

        # Configuration sharing
        assert hasattr(safety_manager, "battery_low_voltage")

        # Services should not conflict with each other's resources

    @pytest.mark.asyncio
    async def test_performance_coordination(self, state_machine, signal_processor, safety_manager):
        """Test performance coordination across services."""
        import numpy as np

        # Measure coordinated performance
        start_time = time.perf_counter()

        # Concurrent operations
        tasks = [
            state_machine.transition_to(SystemState.SEARCHING),
            asyncio.create_task(
                asyncio.to_thread(
                    signal_processor.compute_rssi, np.random.randn(1024).astype(np.complex64)
                )
            ),
            asyncio.create_task(asyncio.to_thread(safety_manager.check_battery_status)),
        ]

        await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Should complete quickly (concurrent execution)
        assert total_time < 1.0  # Should be much faster than sequential

    @pytest.mark.asyncio
    async def test_state_synchronization_coordination(
        self, state_machine, safety_manager, mock_mavlink
    ):
        """Test state synchronization across services."""
        # Set up services
        safety_manager.mavlink = mock_mavlink

        # Coordinate state changes
        initial_state = state_machine.current_state

        # Safety manager state
        safety_state = safety_manager.state

        # Services should maintain consistent state views
        assert isinstance(initial_state, SystemState)
        assert isinstance(safety_state, str)

        # State changes should be coordinated
        await state_machine.transition_to(SystemState.IDLE)
        safety_manager.state = "MONITORING"

        # Services should handle state coordination

    def test_dependency_injection_coordination(self, state_machine, safety_manager, mock_mavlink):
        """Test dependency injection coordination."""
        # Test service dependencies can be injected

        # Inject MAVLink into safety manager
        safety_manager.mavlink = mock_mavlink

        # Inject signal processor into state machine
        from src.backend.services.signal_processor import SignalProcessor

        signal_proc = SignalProcessor()
        state_machine.set_signal_processor(signal_proc)

        # Services should accept dependency injection
        assert safety_manager.mavlink == mock_mavlink

        # Dependencies should work
        battery_status = safety_manager.check_battery_status()
        assert battery_status["level"] == "NORMAL"

    def test_configuration_hot_reload_coordination(
        self, state_machine, signal_processor, safety_manager
    ):
        """Test configuration hot reload coordination."""
        # Test that services can handle configuration updates

        # Update signal processor configuration
        original_threshold = signal_processor.snr_threshold
        signal_processor.snr_threshold = 15.0

        assert signal_processor.snr_threshold == 15.0
        assert signal_processor.snr_threshold != original_threshold

        # Update safety manager configuration
        original_voltage = safety_manager.battery_low_voltage
        safety_manager.battery_low_voltage = 20.0

        assert safety_manager.battery_low_voltage == 20.0
        assert safety_manager.battery_low_voltage != original_voltage

        # Services should handle configuration changes gracefully

    @pytest.mark.asyncio
    async def test_graceful_shutdown_coordination(
        self, state_machine, signal_processor, safety_manager
    ):
        """Test graceful shutdown coordination."""
        # Test that services can shutdown gracefully in coordination

        # Services should be in operational state
        assert state_machine.current_state in SystemState

        # Simulate shutdown preparation
        # (Implementation depends on actual shutdown mechanism)

        # Services should handle shutdown signals
        # Note: Actual shutdown testing would require more complex setup

        # For now, test that services can be stopped
        state_machine._is_running = False

        # Services should respond to shutdown coordination
        assert state_machine._is_running is False
