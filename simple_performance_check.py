#!/usr/bin/env python3
"""Simple performance verification script to check if the actual implementation
can achieve the claimed <0.5ms performance metrics."""

import time

import numpy as np


def measure_time(func, *args, **kwargs):
    """Measure execution time in milliseconds."""
    times = []
    for _ in range(10):  # Reduced from 100 for faster testing
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)
        except Exception as e:
            print(f"Error in {func.__name__}: {e}")
            return 999.0  # Return failure time
    return sorted(times)[len(times) // 2]  # Median


def test_numpy_performance():
    """Test basic numpy operations that form the core of signal processing."""
    print("=== Basic NumPy Performance Tests ===")

    # Generate test data similar to what SignalProcessor would use
    samples = np.random.randn(1024) + 1j * np.random.randn(1024)

    # Test FFT performance (core operation)
    time_fft = measure_time(np.fft.fft, samples)
    print(f"FFT (1024 samples): {time_fft:.3f}ms")

    # Test power calculation
    def power_calc(x):
        return np.mean(np.abs(x) ** 2)

    time_power = measure_time(power_calc, samples)
    print(f"Power calculation: {time_power:.3f}ms")

    # Test log conversion
    def log_convert(x):
        power = np.mean(np.abs(x) ** 2)
        return 10 * np.log10(power + 1e-12)

    time_log = measure_time(log_convert, samples)
    print(f"Log conversion: {time_log:.3f}ms")

    # Test percentile calculation (noise floor estimation)
    rssi_data = np.random.uniform(-100, -60, 100)
    time_percentile = measure_time(np.percentile, rssi_data, 10)
    print(f"Percentile calculation: {time_percentile:.3f}ms")

    # Combined operation similar to compute_rssi
    def combined_rssi_operation(x):
        power = np.mean(np.abs(x) ** 2)
        if power > 0:
            return 10 * np.log10(power) - 10.0  # calibration offset
        return -120.0

    time_combined = measure_time(combined_rssi_operation, samples)
    print(f"Combined RSSI operation: {time_combined:.3f}ms")

    # Summary
    total_time = time_fft + time_power + time_log + time_percentile
    print(f"\nTotal typical processing time: {total_time:.3f}ms")
    print(f"Meets <0.5ms requirement: {'✓' if total_time < 0.5 else '✗'}")


def test_simple_operations():
    """Test simple operations claimed in the story."""
    print("\n=== Simple Operations Performance ===")

    # Test threshold comparison (is_signal_detected equivalent)
    def threshold_check(rssi, noise_floor, threshold):
        snr = rssi - noise_floor
        return snr > threshold

    time_threshold = measure_time(threshold_check, -70.0, -85.0, 12.0)
    print(f"Threshold check: {time_threshold:.3f}ms")

    # Test confidence calculation
    def confidence_calc(snr, rssi):
        snr_weight = 0.7
        rssi_weight = 0.3
        snr_normalized = max(0.0, min(1.0, snr / 30.0))
        rssi_normalized = max(0.0, min(1.0, (rssi + 100.0) / 70.0))
        return snr_weight * snr_normalized + rssi_weight * rssi_normalized

    time_confidence = measure_time(confidence_calc, 15.0, -70.0)
    print(f"Confidence calculation: {time_confidence:.3f}ms")

    # Test adaptive threshold
    def adaptive_threshold(noise_samples):
        if len(noise_samples) < 2:
            return 12.0
        noise_std = np.std(noise_samples)
        base_threshold = 12.0
        if noise_std < 1.0:
            return base_threshold - 2.0
        elif noise_std < 3.0:
            return base_threshold
        else:
            return base_threshold + min(6.0, noise_std)

    noise_data = np.random.uniform(-95, -85, 50)
    time_adaptive = measure_time(adaptive_threshold, noise_data)
    print(f"Adaptive threshold: {time_adaptive:.3f}ms")


def check_story_claims():
    """Check the specific claims made in the 4.5_Story.md file."""
    print("\n=== Story 4.5 Claims Verification ===")

    # The story claims:
    # - Average 0.021ms latency
    # - Max 0.065ms latency
    # - 42x faster than 0.5ms requirement

    print("Story claims:")
    print("- Average: 0.021ms")
    print("- Max: 0.065ms")
    print("- All operations <0.5ms")

    # Run the actual performance tests we did above
    test_numpy_performance()
    test_simple_operations()

    print("\n=== Analysis ===")
    print("Based on basic NumPy operations, the claimed performance")
    print("metrics appear to be plausible for simple operations like")
    print("threshold checks and confidence calculations.")
    print("\nHowever, complex operations like FFT and percentile")
    print("calculations may take longer than the claimed 0.021ms average.")


if __name__ == "__main__":
    check_story_claims()
