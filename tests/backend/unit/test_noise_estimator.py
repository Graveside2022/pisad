"""
Unit tests for NoiseEstimator optimization.

BACKWARDS ANALYSIS:
- User Action: Operating drone with RF beacon detection at 100Hz
- Expected Result: Real-time signal processing without CPU overload
- Failure Impact: Missed beacon detections, system unresponsive, mission failure

REQUIREMENT TRACE:
- User Story: #4.9 Task 5 - Signal Processing Optimization
- Performance Target: <0.5ms per update (from 45ms)

TEST VALUE: Prevents CPU overload that would cause system failure in field operations
"""

import time

import numpy as np
import pytest

from src.backend.utils.noise_estimator import NoiseEstimator


class TestNoiseEstimator:
    """Test suite for optimized noise floor estimation."""

    def test_initialization(self):
        """Test estimator initialization with valid parameters."""
        estimator = NoiseEstimator(window_size=100, percentile=10)
        assert estimator.window_size == 100
        assert estimator.percentile == 10
        assert len(estimator.window) == 0
        assert len(estimator.sorted_window) == 0

    def test_invalid_initialization(self):
        """Test error handling for invalid parameters."""
        with pytest.raises(ValueError, match="Percentile must be 0-100"):
            NoiseEstimator(percentile=101)

        with pytest.raises(ValueError, match="Percentile must be 0-100"):
            NoiseEstimator(percentile=-1)

        with pytest.raises(ValueError, match="Window size must be >= 10"):
            NoiseEstimator(window_size=5)

    def test_add_samples(self):
        """Test adding samples to window."""
        estimator = NoiseEstimator(window_size=10)

        # Add samples
        for i in range(5):
            estimator.add_sample(float(-70 - i))

        assert len(estimator.window) == 5
        assert len(estimator.sorted_window) == 5
        assert estimator.sorted_window == sorted(estimator.sorted_window)

    def test_window_overflow(self):
        """Test sliding window behavior when full."""
        estimator = NoiseEstimator(window_size=10)

        # Fill window
        for i in range(15):
            estimator.add_sample(float(-70 - i))

        # Window should maintain size limit
        assert len(estimator.window) == 10
        assert len(estimator.sorted_window) == 10

        # Oldest samples should be removed
        assert -70.0 not in estimator.window  # First sample removed
        assert -84.0 in estimator.window  # Last sample present

    def test_percentile_calculation(self):
        """Test percentile calculation accuracy."""
        estimator = NoiseEstimator(window_size=100, percentile=10)

        # Add known distribution
        np.random.seed(42)
        samples = np.random.normal(-75, 5, 100)
        for sample in samples:
            estimator.add_sample(sample)

        # Compare with numpy
        np_p10 = np.percentile(samples, 10)
        est_p10 = estimator.get_percentile()

        # Should be close (within 1 dB due to different methods)
        assert abs(est_p10 - np_p10) < 1.0

    def test_insufficient_samples(self):
        """Test behavior with insufficient samples."""
        estimator = NoiseEstimator()

        # Less than 10 samples
        for i in range(5):
            estimator.add_sample(-70.0)

        # Should return default noise floor
        assert estimator.get_percentile() == -85.0

        # Add more samples
        for i in range(10):
            estimator.add_sample(-70.0)

        # Now should return actual percentile
        assert estimator.get_percentile() != -85.0

    def test_cache_behavior(self):
        """Test caching for performance."""
        estimator = NoiseEstimator(window_size=100)

        # Fill window
        for i in range(100):
            estimator.add_sample(-70.0 - i / 10)

        # First call calculates
        result1 = estimator.get_percentile()
        assert estimator._cache_valid

        # Second call uses cache
        result2 = estimator.get_percentile()
        assert result1 == result2
        assert estimator._cache_valid

        # Adding sample invalidates cache
        estimator.add_sample(-75.0)
        assert not estimator._cache_valid

    def test_different_percentiles(self):
        """Test getting different percentiles."""
        estimator = NoiseEstimator(window_size=100, percentile=10)

        # Add uniform distribution
        for i in range(100):
            estimator.add_sample(-80.0 + i * 0.1)

        p10 = estimator.get_percentile(10)
        p50 = estimator.get_percentile(50)
        p90 = estimator.get_percentile(90)

        # Verify ordering
        assert p10 < p50 < p90

        # Verify approximate values
        assert abs(p10 - (-80.0 + 1.0)) < 0.5  # ~10th percentile
        assert abs(p50 - (-80.0 + 5.0)) < 0.5  # ~50th percentile
        assert abs(p90 - (-80.0 + 9.0)) < 0.5  # ~90th percentile

    def test_reset(self):
        """Test reset functionality."""
        estimator = NoiseEstimator(window_size=50)

        # Add samples
        for i in range(50):
            estimator.add_sample(-70.0)

        assert len(estimator.window) == 50

        # Reset
        estimator.reset()

        assert len(estimator.window) == 0
        assert len(estimator.sorted_window) == 0
        assert estimator._cached_percentile is None
        assert not estimator._cache_valid

    def test_statistics(self):
        """Test statistics reporting."""
        estimator = NoiseEstimator(window_size=10)

        # Empty stats
        stats = estimator.get_statistics()
        assert stats["samples"] == 0
        assert stats["min"] is None

        # Add samples
        for i in range(10):
            estimator.add_sample(-70.0 - i)

        stats = estimator.get_statistics()
        assert stats["samples"] == 10
        assert stats["min"] == -79.0
        assert stats["max"] == -70.0
        assert stats["percentile"] is not None
        assert stats["median"] is not None

    @pytest.mark.benchmark
    def test_performance_improvement(self):
        """
        Benchmark performance improvement over numpy.percentile.

        PERFORMANCE TARGET: <0.5ms per update (from 45ms)
        EXPECTED IMPROVEMENT: >99% CPU reduction
        """
        window_size = 1000
        num_updates = 1000

        # Generate test data
        np.random.seed(42)
        samples = np.random.normal(-75, 5, num_updates + window_size).tolist()

        # Benchmark old method (numpy percentile)
        old_window: list[float] = []
        start = time.perf_counter()
        for i in range(num_updates):
            old_window.append(samples[i])
            if len(old_window) > window_size:
                old_window.pop(0)
            if len(old_window) >= 10:
                _ = float(np.percentile(old_window, 10))
        old_time = time.perf_counter() - start

        # Benchmark new method (NoiseEstimator)
        estimator = NoiseEstimator(window_size=window_size, percentile=10)
        start = time.perf_counter()
        for i in range(num_updates):
            estimator.add_sample(samples[i])
            if len(estimator.window) >= 10:
                _ = estimator.get_percentile()
        new_time = time.perf_counter() - start

        # Calculate improvement
        improvement = (old_time - new_time) / old_time * 100
        speedup = old_time / new_time

        # Log results
        print("\nPerformance Benchmark Results:")
        print(
            f"Old method (numpy): {old_time*1000:.2f}ms total, {old_time/num_updates*1000:.3f}ms per update"
        )
        print(
            f"New method (NoiseEstimator): {new_time*1000:.2f}ms total, {new_time/num_updates*1000:.3f}ms per update"
        )
        print(f"Improvement: {improvement:.1f}%")
        print(f"Speedup: {speedup:.1f}x faster")

        # Verify performance target met
        ms_per_update = (new_time / num_updates) * 1000
        assert ms_per_update < 0.5, f"Performance target not met: {ms_per_update:.3f}ms > 0.5ms"
        assert improvement > 85, f"Insufficient improvement: {improvement:.1f}% < 85%"

    def test_concurrent_updates(self):
        """Test thread safety considerations."""
        # NoiseEstimator is designed for single-threaded async use
        # This test documents that it's NOT thread-safe
        estimator = NoiseEstimator(window_size=100)

        # Add samples in sequence (async-safe)
        for i in range(100):
            estimator.add_sample(-70.0 - i / 10)

        # Get percentile (async-safe)
        result = estimator.get_percentile()
        assert result < -70.0

        # Note: For multi-threaded use, would need locks
        # Current design assumes asyncio single-thread model
