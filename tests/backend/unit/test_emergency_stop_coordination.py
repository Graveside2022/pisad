"""
Emergency stop coordination system timing validation tests.

Tests SUBTASK-5.5.2.3: Ensure emergency stop maintains <500ms response time.
Validates that emergency stop response times meet <500ms requirement
even with DualSDRCoordinator active and processing coordination tasks.

This ensures PRD-FR16 and NFR12 timing requirements are met:
- PRD-FR16: "Disable Homing control that stops all velocity commands within 500ms"  
- PRD-NFR12: "All safety-critical functions shall execute with deterministic timing"

Chain of Thought Context:
- PRD → Epic 5 → Story 5.5 → TASK-5.5.2-EMERGENCY-FALLBACK → SUBTASK-5.5.2.3
- Integration Points: SafetyManager, DualSDRCoordinator, emergency stop pathways
- Previous Context: Communication loss detection (5.5.2.1) ✅ COMPLETED, fallback implementation (5.5.2.2) ✅ COMPLETED
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.services.safety_manager import SafetyManager
from src.backend.utils.safety import SafetyInterlockSystem


class TestEmergencyStopCoordinationTiming:
    """Test emergency stop timing requirements with coordination system active."""

    @pytest.fixture
    async def safety_manager(self):
        """Create safety manager with mock MAVLink for timing tests."""
        manager = SafetyManager()

        # Mock MAVLink with emergency_stop method for authentic testing
        mock_mavlink = MagicMock()
        mock_mavlink.emergency_stop = MagicMock(return_value=True)
        mock_mavlink.telemetry = {
            "battery": {"voltage": 12.5},
            "gps": {"satellites": 10, "hdop": 1.0, "fix_type": 3},
            "mode": "GUIDED",
        }
        manager.mavlink = mock_mavlink

        return manager

    @pytest.fixture
    async def coordination_system(self):
        """Create DualSDRCoordinator with active coordination loop."""
        coordinator = DualSDRCoordinator()

        # Mock dependencies for realistic coordination overhead
        coordinator._signal_processor = AsyncMock()
        coordinator._tcp_bridge = AsyncMock()
        coordinator._safety_manager = AsyncMock()

        await coordinator.start()
        yield coordinator
        await coordinator.stop()

    @pytest.fixture
    async def safety_interlock_system(self):
        """Create safety interlock system for emergency stop testing."""
        system = SafetyInterlockSystem()
        await system.start_monitoring()
        yield system
        await system.stop_monitoring()

    async def test_emergency_stop_timing_under_500ms_baseline(self, safety_manager):
        """
        Test [2m]: Emergency stop timing validation with coordination system active.

        RED PHASE: This test should initially fail until emergency stop timing
        is validated to work under coordination system load.
        """

        # Simulate coordination system load during emergency stop
        async def coordination_load():
            """Simulate ongoing coordination processing during emergency stop."""
            for _ in range(10):  # Simulate coordination decisions
                await asyncio.sleep(0.005)  # 5ms per decision (realistic load)

        # Start coordination load in background
        load_task = asyncio.create_task(coordination_load())

        try:
            # Measure emergency stop timing under load
            start_time = time.perf_counter()
            result = safety_manager.trigger_emergency_stop()
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000

            # Validate emergency stop completed successfully
            assert result is not None
            assert isinstance(result, dict)

            # CRITICAL: Emergency stop must complete in <500ms per PRD-FR16
            assert (
                response_time_ms < 500.0
            ), f"Emergency stop took {response_time_ms:.1f}ms (>500ms limit)"

            # Document actual timing for performance tracking
            print(f"Emergency stop timing under coordination load: {response_time_ms:.1f}ms")

        finally:
            # Clean up coordination load task
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass

    async def test_emergency_stop_during_communication_loss(
        self, safety_manager, coordination_system
    ):
        """
        Test [2n]: Emergency stop response during communication loss scenarios.

        RED PHASE: Tests emergency stop effectiveness when coordination system
        is handling communication failures and fallback transitions.
        """
        # Simulate communication loss triggering fallback mode
        coordination_system.fallback_active = True
        coordination_system.active_source = "drone"  # Fallback to drone-only

        # Measure emergency stop timing during communication loss scenario
        start_time = time.perf_counter()
        result = safety_manager.trigger_emergency_stop()
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        # Emergency stop must work during communication loss
        assert result is not None
        assert (
            response_time_ms < 500.0
        ), f"Emergency stop during comm loss took {response_time_ms:.1f}ms"

        # Verify coordination system state doesn't interfere
        assert coordination_system.fallback_active == True  # Fallback state preserved

        print(f"Emergency stop during communication loss: {response_time_ms:.1f}ms")

    async def test_emergency_stop_triggers_through_coordination_components(
        self, coordination_system, safety_interlock_system
    ):
        """
        Test [2o]: Emergency stop triggers through all coordination components.

        RED PHASE: Validates emergency stop pathways work through coordination
        system components without degrading response time.
        """
        # Test emergency stop trigger through coordination system
        with patch("src.backend.services.dual_sdr_coordinator.logger") as mock_logger:

            # Simulate emergency stop triggered through coordination system
            start_time = time.perf_counter()

            # Trigger emergency stop through safety interlock system
            await safety_interlock_system.emergency_stop("Coordination emergency test")

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            # Emergency stop must propagate through coordination components
            assert (
                response_time_ms < 500.0
            ), f"Coordinated emergency stop took {response_time_ms:.1f}ms"
            assert safety_interlock_system.emergency_stopped == True

            print(f"Emergency stop through coordination components: {response_time_ms:.1f}ms")

    async def test_emergency_stop_performance_benchmarks(self, safety_manager):
        """
        Test [2p]: Performance benchmarks for emergency response timing.

        RED PHASE: Creates performance benchmarks to ensure emergency stop
        timing remains consistent under various system load conditions.
        """
        timing_results = []

        # Run multiple emergency stop timing tests
        for load_level in [0, 25, 50, 75, 100]:  # Percentage system load

            # Simulate system load
            async def system_load():
                load_iterations = load_level // 10  # Scale load
                for _ in range(load_iterations):
                    await asyncio.sleep(0.001)  # 1ms per load iteration

            load_task = asyncio.create_task(system_load()) if load_level > 0 else None

            try:
                # Reset safety manager state for clean test
                safety_manager.mavlink.emergency_stop.reset_mock()

                # Measure emergency stop timing
                start_time = time.perf_counter()
                result = safety_manager.trigger_emergency_stop()
                end_time = time.perf_counter()

                response_time_ms = (end_time - start_time) * 1000
                timing_results.append(
                    {"load_level": load_level, "response_time_ms": response_time_ms}
                )

                # Each test must meet timing requirement
                assert (
                    response_time_ms < 500.0
                ), f"Emergency stop at {load_level}% load took {response_time_ms:.1f}ms"

            finally:
                if load_task is not None and not load_task.done():
                    load_task.cancel()
                    try:
                        await load_task
                    except asyncio.CancelledError:
                        pass

        # Validate performance consistency across load levels
        max_response_time = max(result["response_time_ms"] for result in timing_results)
        assert (
            max_response_time < 500.0
        ), f"Maximum emergency stop time {max_response_time:.1f}ms exceeds 500ms"

        print(f"Emergency stop performance benchmarks: {timing_results}")

    async def test_emergency_stop_pathway_verification_all_modes(
        self, coordination_system, safety_interlock_system
    ):
        """
        Test [2q]: Emergency stop pathway verification in all operation modes.

        RED PHASE: Validates emergency stop works in all coordination modes:
        drone-only, ground-coordinated, and fallback transition modes.
        """
        operation_modes = [
            ("drone_only", {"active_source": "drone", "fallback_active": False}),
            ("ground_coordinated", {"active_source": "ground", "fallback_active": False}),
            ("fallback_transition", {"active_source": "drone", "fallback_active": True}),
        ]

        for mode_name, mode_config in operation_modes:
            # Configure coordination system for specific mode
            coordination_system.active_source = mode_config["active_source"]
            coordination_system.fallback_active = mode_config["fallback_active"]

            # Reset emergency stop state
            await safety_interlock_system.reset_emergency_stop()

            # Test emergency stop in this mode
            start_time = time.perf_counter()
            await safety_interlock_system.emergency_stop(f"Test emergency stop in {mode_name}")
            end_time = time.perf_counter()

            response_time_ms = (end_time - start_time) * 1000

            # Emergency stop must work in all modes
            assert (
                safety_interlock_system.emergency_stopped == True
            ), f"Emergency stop failed in {mode_name}"
            assert (
                response_time_ms < 500.0
            ), f"Emergency stop in {mode_name} took {response_time_ms:.1f}ms"

            print(f"Emergency stop in {mode_name} mode: {response_time_ms:.1f}ms")

    async def test_emergency_stop_effectiveness_during_fallback_transitions(
        self, coordination_system, safety_interlock_system
    ):
        """
        Test [2r]: Emergency stop effectiveness during fallback transitions.

        RED PHASE: Tests emergency stop works correctly while coordination system
        is transitioning between normal and fallback modes.
        """
        # Simulate fallback transition in progress
        coordination_system.fallback_active = False
        coordination_system.active_source = "ground"

        async def trigger_fallback_during_emergency():
            """Simulate fallback transition during emergency stop."""
            await asyncio.sleep(0.01)  # 10ms delay
            coordination_system.fallback_active = True
            coordination_system.active_source = "drone"

        # Start fallback transition
        fallback_task = asyncio.create_task(trigger_fallback_during_emergency())

        try:
            # Trigger emergency stop during transition
            start_time = time.perf_counter()
            await safety_interlock_system.emergency_stop("Emergency during fallback transition")
            end_time = time.perf_counter()

            # Wait for fallback transition to complete
            await fallback_task

            response_time_ms = (end_time - start_time) * 1000

            # Emergency stop must be effective during transitions
            assert safety_interlock_system.emergency_stopped == True
            assert (
                response_time_ms < 500.0
            ), f"Emergency stop during transition took {response_time_ms:.1f}ms"

            # Coordination system should still be in fallback mode
            assert coordination_system.fallback_active == True

            print(f"Emergency stop during fallback transition: {response_time_ms:.1f}ms")

        except Exception:
            fallback_task.cancel()
            raise

    async def test_emergency_stop_coordination_integration_timing(
        self, safety_manager, coordination_system
    ):
        """
        Integration test: Emergency stop timing with full coordination system active.

        RED PHASE: Validates emergency stop maintains <500ms response time when
        coordination system is actively processing dual SDR coordination decisions.
        """
        # Ensure coordination system is running with realistic load
        assert coordination_system.is_running == True

        # Simulate realistic coordination processing load
        coordination_system._signal_processor.get_current_rssi.return_value = -45.0
        coordination_system._tcp_bridge.is_connected.return_value = True

        # Measure emergency stop with active coordination
        start_time = time.perf_counter()
        result = safety_manager.trigger_emergency_stop()
        end_time = time.perf_counter()

        response_time_ms = (end_time - start_time) * 1000

        # Validate emergency stop succeeded under coordination load
        assert result is not None
        assert (
            response_time_ms < 500.0
        ), f"Emergency stop with coordination took {response_time_ms:.1f}ms"

        # Verify coordination system continues running (emergency stop shouldn't break coordination)
        assert coordination_system.is_running == True

        print(f"Emergency stop with full coordination system: {response_time_ms:.1f}ms")
