"""
Safety system performance validation tests.

Tests SUBTASK-5.5.1.4 implementation with step [1w].
Validates that safety systems maintain timing requirements during
coordination operations and meet all PRD performance specifications.

This ensures NFR12 deterministic timing and PRD safety requirements.
"""

import asyncio
import statistics
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.backend.services.dual_sdr_coordinator import DualSDRCoordinator
from src.backend.utils.safety import SafetyInterlockSystem


class TestSafetyPerformanceValidation:
    """Test safety system performance requirements."""

    @pytest.fixture
    async def safety_system(self):
        """Create safety interlock system."""
        safety = SafetyInterlockSystem()
        await safety.start_monitoring()
        yield safety
        await safety.stop_monitoring()

    @pytest.fixture
    def dual_coordinator(self):
        """Create dual coordinator for performance testing."""
        coordinator = DualSDRCoordinator()
        
        # Mock with realistic but consistent timing
        coordinator.get_health_status = AsyncMock(return_value={
            "coordination_latency_ms": 35.0,
            "ground_connection_status": 0.85,
            "coordination_active": True,
        })
        
        async def mock_emergency_override():
            await asyncio.sleep(0.08)  # 80ms processing time
            return {
                "emergency_override_active": True,
                "response_time_ms": 80.0,
                "source_switched_to": "drone"
            }
        
        coordinator.trigger_emergency_override = mock_emergency_override
        coordinator.get_ground_rssi = MagicMock(return_value=12.0)
        coordinator.get_drone_rssi = MagicMock(return_value=9.0)
        
        return coordinator

    @pytest.mark.asyncio
    async def test_safety_check_latency_requirement(self, safety_system, dual_coordinator):
        """Test [1w] - Safety checks meet <100ms latency requirement per NFR12."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test multiple safety check cycles
        latencies = []
        
        for _ in range(10):
            start_time = time.perf_counter()
            
            await safety_system.check_all_safety()
            
            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
        
        # Calculate statistics
        avg_latency = statistics.mean(latencies)
        max_latency = max(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
        
        # Performance requirements
        assert avg_latency < 50.0, f"Average latency {avg_latency:.1f}ms should be under 50ms"
        assert max_latency < 100.0, f"Max latency {max_latency:.1f}ms should be under 100ms"
        assert p95_latency < 80.0, f"95th percentile {p95_latency:.1f}ms should be under 80ms"

    @pytest.mark.asyncio
    async def test_emergency_stop_performance_requirement(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Emergency stop meets <500ms requirement under all conditions."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test emergency stop performance multiple times
        response_times = []
        
        for i in range(5):
            # Reset emergency state
            if safety_system.emergency_stopped:
                await safety_system.reset_emergency_stop()
            
            start_time = time.perf_counter()
            
            await safety_system.trigger_coordination_emergency_stop(
                f"Performance test {i+1}"
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            response_times.append(response_time_ms)
        
        # Performance analysis
        avg_response = statistics.mean(response_times)
        max_response = max(response_times)
        
        # Critical requirements
        assert max_response < 500.0, f"Max emergency response {max_response:.1f}ms exceeds 500ms limit"
        assert avg_response < 200.0, f"Average response {avg_response:.1f}ms should be under 200ms"
        
        # Consistency check
        std_dev = statistics.stdev(response_times)
        assert std_dev < 50.0, f"Response time std dev {std_dev:.1f}ms indicates inconsistent performance"

    @pytest.mark.asyncio
    async def test_coordination_health_check_performance(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Coordination health checks have minimal overhead."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test health check performance
        health_check_times = []
        
        for _ in range(20):
            start_time = time.perf_counter()
            
            await safety_system.check_coordination_health()
            
            end_time = time.perf_counter()
            check_time_ms = (end_time - start_time) * 1000
            health_check_times.append(check_time_ms)
        
        # Performance analysis
        avg_time = statistics.mean(health_check_times)
        max_time = max(health_check_times)
        
        # Health checks should be very fast
        assert avg_time < 20.0, f"Average health check {avg_time:.1f}ms should be under 20ms"
        assert max_time < 50.0, f"Max health check {max_time:.1f}ms should be under 50ms"

    @pytest.mark.asyncio
    async def test_dual_source_signal_check_performance(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Dual source signal checks are efficient."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test signal check performance
        signal_check_times = []
        
        for _ in range(15):
            start_time = time.perf_counter()
            
            await safety_system.check_dual_source_signals()
            
            end_time = time.perf_counter()
            check_time_ms = (end_time - start_time) * 1000
            signal_check_times.append(check_time_ms)
        
        # Performance analysis
        avg_time = statistics.mean(signal_check_times)
        max_time = max(signal_check_times)
        
        # Signal checks should be very fast
        assert avg_time < 15.0, f"Average signal check {avg_time:.1f}ms should be under 15ms"
        assert max_time < 30.0, f"Max signal check {max_time:.1f}ms should be under 30ms"

    @pytest.mark.asyncio
    async def test_safety_monitoring_loop_performance(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Safety monitoring loop maintains performance."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Monitor several monitoring cycles
        cycle_times = []
        
        # Run monitoring for a short period
        start_monitoring = time.perf_counter()
        
        for _ in range(10):  # 10 monitoring cycles
            cycle_start = time.perf_counter()
            
            await safety_system.check_all_safety()
            await asyncio.sleep(safety_system._check_interval)  # Simulate monitoring interval
            
            cycle_end = time.perf_counter()
            cycle_time_ms = (cycle_end - cycle_start) * 1000
            cycle_times.append(cycle_time_ms)
        
        end_monitoring = time.perf_counter()
        total_time = end_monitoring - start_monitoring
        
        # Performance analysis
        avg_cycle_time = statistics.mean(cycle_times)
        target_cycle_time = safety_system._check_interval * 1000  # Convert to ms
        
        # Monitoring should not significantly exceed target cycle time
        overhead_ratio = avg_cycle_time / target_cycle_time
        assert overhead_ratio < 2.0, f"Monitoring overhead {overhead_ratio:.1f}x too high"
        
        # Total monitoring should complete in reasonable time
        expected_total_time = 10 * safety_system._check_interval
        assert total_time < expected_total_time * 1.5, "Monitoring loop too slow"

    @pytest.mark.asyncio
    async def test_concurrent_safety_operations_performance(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Performance under concurrent safety operations."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test concurrent operations
        async def concurrent_safety_checks():
            tasks = []
            
            # Run multiple concurrent safety operations
            for i in range(5):
                tasks.append(safety_system.check_all_safety())
                tasks.append(safety_system.check_coordination_health())
                tasks.append(safety_system.check_dual_source_signals())
            
            start_time = time.perf_counter()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            end_time = time.perf_counter()
            
            return results, (end_time - start_time) * 1000
        
        results, total_time_ms = await concurrent_safety_checks()
        
        # All operations should complete successfully
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert len(exceptions) == 0, f"Concurrent operations failed: {exceptions}"
        
        # Concurrent operations should complete in reasonable time
        assert total_time_ms < 500.0, f"Concurrent operations took {total_time_ms:.1f}ms"

    @pytest.mark.asyncio
    async def test_memory_usage_during_coordination(self, safety_system, dual_coordinator):
        """Test [1w] - Memory usage remains stable during coordination."""
        import gc
        import psutil
        import os
        
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run safety operations for a while
        for _ in range(100):
            await safety_system.check_all_safety()
            await safety_system.check_coordination_health()
            await safety_system.check_dual_source_signals()
        
        # Force garbage collection
        gc.collect()
        
        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not grow excessively
        assert memory_increase < 10.0, f"Memory increased by {memory_increase:.1f}MB"

    @pytest.mark.asyncio
    async def test_safety_system_throughput(self, safety_system, dual_coordinator):
        """Test [1w] - Safety system throughput meets requirements."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Measure throughput over time
        duration_seconds = 2.0
        operation_count = 0
        
        start_time = time.perf_counter()
        end_time = start_time + duration_seconds
        
        while time.perf_counter() < end_time:
            await safety_system.check_all_safety()
            operation_count += 1
        
        actual_duration = time.perf_counter() - start_time
        throughput = operation_count / actual_duration  # operations per second
        
        # Should be able to perform at least 20 safety checks per second
        assert throughput >= 20.0, f"Throughput {throughput:.1f} ops/sec too low"

    @pytest.mark.asyncio
    async def test_safety_response_under_load(self, safety_system, dual_coordinator):
        """Test [1w] - Safety response times under system load."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Create background load
        async def background_load():
            while True:
                await safety_system.check_all_safety()
                await asyncio.sleep(0.01)  # 10ms interval for load
        
        load_task = asyncio.create_task(background_load())
        
        try:
            # Let load run for a bit
            await asyncio.sleep(0.1)
            
            # Test emergency stop under load
            start_time = time.perf_counter()
            
            await safety_system.trigger_coordination_emergency_stop(
                "Emergency under load test"
            )
            
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            
            # Emergency stop should still meet timing requirement under load
            assert response_time_ms < 500.0, f"Emergency stop under load took {response_time_ms:.1f}ms"
            
        finally:
            load_task.cancel()
            try:
                await load_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_deterministic_timing_consistency(
        self, safety_system, dual_coordinator
    ):
        """Test [1w] - Deterministic timing per NFR12."""
        # Setup coordination
        safety_system.set_coordination_system(dual_coordinator, active=True)
        
        # Test timing consistency across multiple runs
        emergency_times = []
        health_check_times = []
        signal_check_times = []
        
        for i in range(10):
            # Reset emergency state
            if safety_system.emergency_stopped:
                await safety_system.reset_emergency_stop()
            
            # Measure emergency stop timing
            start = time.perf_counter()
            await safety_system.trigger_coordination_emergency_stop(f"Deterministic test {i}")
            emergency_times.append((time.perf_counter() - start) * 1000)
            
            await safety_system.reset_emergency_stop()
            
            # Measure health check timing
            start = time.perf_counter()
            await safety_system.check_coordination_health()
            health_check_times.append((time.perf_counter() - start) * 1000)
            
            # Measure signal check timing
            start = time.perf_counter()
            await safety_system.check_dual_source_signals()
            signal_check_times.append((time.perf_counter() - start) * 1000)
        
        # Analyze timing consistency (coefficient of variation)
        def coefficient_of_variation(data):
            return statistics.stdev(data) / statistics.mean(data)
        
        emergency_cv = coefficient_of_variation(emergency_times)
        health_cv = coefficient_of_variation(health_check_times)
        signal_cv = coefficient_of_variation(signal_check_times)
        
        # Timing should be deterministic (low coefficient of variation)
        assert emergency_cv < 0.3, f"Emergency stop timing inconsistent: CV={emergency_cv:.2f}"
        assert health_cv < 0.5, f"Health check timing inconsistent: CV={health_cv:.2f}"
        assert signal_cv < 0.5, f"Signal check timing inconsistent: CV={signal_cv:.2f}"