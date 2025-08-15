"""
Performance tests for homing algorithm execution time.
Ensures the algorithm meets the <50ms requirement for 10Hz operation.
"""

import time

import numpy as np
import pytest

from src.backend.services.homing_algorithm import HomingAlgorithm


class TestHomingAlgorithmPerformance:
    """Performance tests for homing algorithm."""

    @pytest.fixture
    def algorithm(self):
        """Create homing algorithm instance."""
        return HomingAlgorithm()

    def test_gradient_calculation_performance(self, algorithm):
        """Test gradient calculation completes within 50ms."""
        # Pre-populate with samples
        current_time = time.time()
        for i in range(10):
            algorithm.add_rssi_sample(
                rssi=-70 + i * 0.5,
                position_x=i * 10,
                position_y=i * 5,
                heading=i * 10,
                timestamp=current_time + i,
            )

        # Measure gradient calculation time
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            algorithm.calculate_gradient()

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Should complete in less than 10ms (leaving margin for other operations)
        assert avg_time_ms < 10, f"Gradient calculation took {avg_time_ms:.2f}ms (limit: 10ms)"
        print(f"Gradient calculation average time: {avg_time_ms:.2f}ms")

    def test_velocity_command_generation_performance(self, algorithm):
        """Test velocity command generation completes quickly."""
        # Pre-populate with samples
        current_time = time.time()
        for i in range(10):
            algorithm.add_rssi_sample(
                rssi=-70 + i * 0.5,
                position_x=i * 10,
                position_y=i * 5,
                heading=i * 10,
                timestamp=current_time + i,
            )

        gradient = algorithm.calculate_gradient()

        # Measure command generation time
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            algorithm.generate_velocity_command(
                gradient=gradient, current_heading=45.0, current_time=current_time + 10
            )

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Should complete in less than 5ms
        assert avg_time_ms < 5, f"Velocity command generation took {avg_time_ms:.2f}ms (limit: 5ms)"
        print(f"Velocity command generation average time: {avg_time_ms:.2f}ms")

    def test_full_update_cycle_performance(self, algorithm):
        """Test complete update cycle (add sample + gradient + command) within 50ms."""
        current_time = time.time()

        # Pre-populate with some samples
        for i in range(5):
            algorithm.add_rssi_sample(
                rssi=-75 + i,
                position_x=i * 10,
                position_y=i * 5,
                heading=i * 20,
                timestamp=current_time + i,
            )

        # Measure full update cycle
        iterations = 100
        start_time = time.perf_counter()

        for i in range(iterations):
            # Add new sample
            algorithm.add_rssi_sample(
                rssi=-70 + (i % 10) * 0.5,
                position_x=(5 + i) * 10,
                position_y=(5 + i) * 5,
                heading=(5 + i) * 20,
                timestamp=current_time + 5 + i,
            )

            # Calculate gradient
            gradient = algorithm.calculate_gradient()

            # Generate velocity command
            algorithm.generate_velocity_command(
                gradient=gradient, current_heading=45.0 + i, current_time=current_time + 5 + i
            )

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Should complete full cycle in less than 50ms for 10Hz operation
        assert avg_time_ms < 50, f"Full update cycle took {avg_time_ms:.2f}ms (limit: 50ms)"
        print(f"Full update cycle average time: {avg_time_ms:.2f}ms")

    def test_large_history_performance(self, algorithm):
        """Test performance with maximum history size."""
        # Fill to maximum capacity
        current_time = time.time()
        max_samples = 50  # Test with larger than configured window

        for i in range(max_samples):
            algorithm.add_rssi_sample(
                rssi=-80 + np.random.normal(0, 2),
                position_x=i * 5 + np.random.normal(0, 1),
                position_y=i * 3 + np.random.normal(0, 1),
                heading=i * 7.2,
                timestamp=current_time + i * 0.1,
            )

        # Measure performance with full history
        iterations = 100
        start_time = time.perf_counter()

        for _ in range(iterations):
            gradient = algorithm.calculate_gradient()
            algorithm.generate_velocity_command(
                gradient=gradient,
                current_heading=180.0,
                current_time=current_time + max_samples * 0.1,
            )

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Should still be fast even with full history
        assert avg_time_ms < 30, f"Large history processing took {avg_time_ms:.2f}ms (limit: 30ms)"
        print(f"Large history processing average time: {avg_time_ms:.2f}ms")

    def test_sampling_maneuver_performance(self, algorithm):
        """Test performance during sampling maneuvers."""
        current_time = time.time()

        # Create scenario requiring sampling (low gradient confidence)
        for i in range(3):  # Minimal samples
            algorithm.add_rssi_sample(
                rssi=-75, position_x=i, position_y=i, heading=0, timestamp=current_time + i
            )

        # Measure sampling command generation
        iterations = 100
        start_time = time.perf_counter()

        for i in range(iterations):
            # Force sampling by providing None gradient
            algorithm.generate_velocity_command(
                gradient=None, current_heading=45.0, current_time=current_time + 3 + i * 0.1
            )

        end_time = time.perf_counter()
        avg_time_ms = ((end_time - start_time) / iterations) * 1000

        # Sampling should be fast (mostly trigonometry)
        assert (
            avg_time_ms < 5
        ), f"Sampling maneuver generation took {avg_time_ms:.2f}ms (limit: 5ms)"
        print(f"Sampling maneuver generation average time: {avg_time_ms:.2f}ms")

    def test_memory_usage(self, algorithm):
        """Test memory usage remains bounded with continuous operation."""
        import sys

        initial_size = sys.getsizeof(algorithm.rssi_history)

        # Add many samples (should be limited by deque maxlen)
        current_time = time.time()
        for i in range(1000):
            algorithm.add_rssi_sample(
                rssi=-70 + np.random.normal(0, 5),
                position_x=i * 0.1,
                position_y=i * 0.05,
                heading=i % 360,
                timestamp=current_time + i * 0.1,
            )

        final_size = sys.getsizeof(algorithm.rssi_history)

        # Memory should be bounded (not growing indefinitely)
        # Allow for some overhead but should not grow with samples beyond window
        assert (
            final_size < initial_size * 20
        ), f"Memory usage grew too much: {initial_size} -> {final_size}"
        print(f"Memory usage: initial={initial_size} bytes, final={final_size} bytes")

    def test_concurrent_operations_performance(self, algorithm):
        """Test performance with rapid concurrent updates."""
        current_time = time.time()

        # Simulate rapid updates at 20Hz (50ms intervals)
        update_rate = 20  # Hz
        duration = 2  # seconds
        total_updates = update_rate * duration

        start_time = time.perf_counter()

        for i in range(total_updates):
            # Add sample
            algorithm.add_rssi_sample(
                rssi=-65 + np.random.normal(0, 3),
                position_x=i * 2,
                position_y=i * 1,
                heading=i * 9,
                timestamp=current_time + i / update_rate,
            )

            # Calculate and generate command
            gradient = algorithm.calculate_gradient()
            algorithm.generate_velocity_command(
                gradient=gradient,
                current_heading=i * 9,
                current_time=current_time + i / update_rate,
            )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        avg_time_ms = (total_time / total_updates) * 1000

        # Should handle rapid updates efficiently
        assert avg_time_ms < 25, f"Rapid updates took {avg_time_ms:.2f}ms average (limit: 25ms)"
        print(
            f"Rapid update handling: {total_updates} updates in {total_time:.2f}s, avg {avg_time_ms:.2f}ms"
        )
