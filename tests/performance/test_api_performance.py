#!/usr/bin/env python3
"""Performance benchmarks for critical API methods.

This module verifies that all signal processing operations meet the
<0.5ms latency requirement specified in Story 4.5.
"""

import asyncio
import os
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from backend.services.mavlink_service import MAVLinkService
from backend.services.signal_processor import SignalProcessor
from backend.services.state_machine import StateMachine


class PerformanceBenchmark:
    """Performance benchmark suite for critical APIs."""

    def __init__(self):
        self.signal_processor = SignalProcessor()
        self.state_machine = StateMachine()
        self.mavlink_service = MAVLinkService()
        self.results = []

    def generate_test_samples(self, size: int = 1024) -> np.ndarray:
        """Generate realistic IQ samples for testing."""
        # Generate complex IQ samples with signal + noise
        t = np.arange(size) / 2.048e6  # 2.048 MHz sample rate
        signal = np.exp(1j * 2 * np.pi * 406.025e6 * t)  # 406.025 MHz carrier
        noise = (np.random.randn(size) + 1j * np.random.randn(size)) * 0.1
        return signal + noise

    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of a function in milliseconds."""
        times = []
        # Run multiple iterations for accuracy
        for _ in range(100):
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

        # Return median to avoid outliers
        return sorted(times)[len(times) // 2]

    async def measure_async_execution_time(self, func, *args, **kwargs) -> float:
        """Measure execution time of an async function in milliseconds."""
        times = []
        for _ in range(100):
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        return sorted(times)[len(times) // 2]

    def benchmark_signal_processor(self):
        """Benchmark SignalProcessor API methods."""
        print("\n=== SignalProcessor Performance Benchmarks ===")
        samples = self.generate_test_samples(1024)
        rssi_history = [np.random.uniform(-100, -60) for _ in range(100)]
        noise_history = [np.random.uniform(-95, -85) for _ in range(50)]

        # Test compute_rssi
        time_ms = self.measure_execution_time(self.signal_processor.compute_rssi, samples)
        self.results.append(("compute_rssi", time_ms, time_ms < 0.5))
        print(f"compute_rssi: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test compute_rssi_fft
        time_ms = self.measure_execution_time(self.signal_processor.compute_rssi_fft, samples)
        self.results.append(("compute_rssi_fft", time_ms, time_ms < 0.5))
        print(f"compute_rssi_fft: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test compute_snr
        time_ms = self.measure_execution_time(self.signal_processor.compute_snr, samples, -85.0)
        self.results.append(("compute_snr", time_ms, time_ms < 0.5))
        print(f"compute_snr: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test estimate_noise_floor
        time_ms = self.measure_execution_time(
            self.signal_processor.estimate_noise_floor, rssi_history
        )
        self.results.append(("estimate_noise_floor", time_ms, time_ms < 0.5))
        print(f"estimate_noise_floor: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test is_signal_detected
        time_ms = self.measure_execution_time(
            self.signal_processor.is_signal_detected, -70.0, -85.0, 12.0
        )
        self.results.append(("is_signal_detected", time_ms, time_ms < 0.5))
        print(f"is_signal_detected: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test calculate_confidence
        time_ms = self.measure_execution_time(
            self.signal_processor.calculate_confidence, 15.0, -70.0
        )
        self.results.append(("calculate_confidence", time_ms, time_ms < 0.5))
        print(f"calculate_confidence: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test calculate_adaptive_threshold
        time_ms = self.measure_execution_time(
            self.signal_processor.calculate_adaptive_threshold, noise_history
        )
        self.results.append(("calculate_adaptive_threshold", time_ms, time_ms < 0.5))
        print(f"calculate_adaptive_threshold: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

    async def benchmark_state_machine(self):
        """Benchmark StateMachine API methods."""
        print("\n=== StateMachine Performance Benchmarks ===")

        # Test get_valid_transitions (synchronous)
        time_ms = self.measure_execution_time(self.state_machine.get_valid_transitions)
        self.results.append(("get_valid_transitions", time_ms, time_ms < 0.5))
        print(f"get_valid_transitions: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        # Test async methods
        detection_event = {"rssi": -70.0, "snr": 15.0, "confidence": 0.85, "timestamp": time.time()}

        time_ms = await self.measure_async_execution_time(
            self.state_machine.on_signal_detected, detection_event
        )
        self.results.append(("on_signal_detected", time_ms, time_ms < 0.5))
        print(f"on_signal_detected: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        time_ms = await self.measure_async_execution_time(self.state_machine.on_signal_lost)
        self.results.append(("on_signal_lost", time_ms, time_ms < 0.5))
        print(f"on_signal_lost: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

        time_ms = await self.measure_async_execution_time(
            self.state_machine.on_mode_change, "GUIDED"
        )
        self.results.append(("on_mode_change", time_ms, time_ms < 0.5))
        print(f"on_mode_change: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

    def benchmark_mavlink_service(self):
        """Benchmark MAVLinkService API methods."""
        print("\n=== MAVLinkService Performance Benchmarks ===")

        telemetry_data = {
            "rssi": -70.0,
            "snr": 15.0,
            "confidence": 0.85,
            "lat": 37.7749,
            "lon": -122.4194,
            "alt": 100.0,
        }

        # Test send_telemetry
        time_ms = self.measure_execution_time(self.mavlink_service.send_telemetry, telemetry_data)
        self.results.append(("send_telemetry", time_ms, time_ms < 0.5))
        print(f"send_telemetry: {time_ms:.3f}ms {'✓' if time_ms < 0.5 else '✗ FAIL'}")

    def print_summary(self):
        """Print performance benchmark summary."""
        print("\n" + "=" * 60)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)

        total_tests = len(self.results)
        passed_tests = sum(1 for _, _, passed in self.results if passed)

        print(f"\nTotal Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Pass Rate: {(passed_tests/total_tests)*100:.1f}%")

        if total_tests > 0:
            avg_time = sum(time_ms for _, time_ms, _ in self.results) / total_tests
            max_time = max(time_ms for _, time_ms, _ in self.results)
            min_time = min(time_ms for _, time_ms, _ in self.results)

            print("\nTiming Statistics:")
            print(f"Average: {avg_time:.3f}ms")
            print(f"Min: {min_time:.3f}ms")
            print(f"Max: {max_time:.3f}ms")

        print("\n<0.5ms Latency Requirement: ", end="")
        if passed_tests == total_tests:
            print("✓ ALL TESTS PASS")
        else:
            print(f"✗ {total_tests - passed_tests} TESTS FAIL")
            print("\nFailed Tests:")
            for name, time_ms, passed in self.results:
                if not passed:
                    print(f"  - {name}: {time_ms:.3f}ms (>{0.5}ms)")

        return passed_tests == total_tests


async def main():
    """Run all performance benchmarks."""
    benchmark = PerformanceBenchmark()

    # Run synchronous benchmarks
    benchmark.benchmark_signal_processor()

    # Run async benchmarks
    await benchmark.benchmark_state_machine()

    # Run MAVLink benchmarks
    benchmark.benchmark_mavlink_service()

    # Print summary
    all_pass = benchmark.print_summary()

    # Exit with appropriate code
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    asyncio.run(main())
